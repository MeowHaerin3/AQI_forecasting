import pandas as pd
import numpy as np
import aqi
from fancyimpute import KNN
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import matplotlib.pyplot as plt
import warnings
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ARIMAForecaster:
    def __init__(self, data_path):
        self.data_path = data_path
        self.aqi_data = pd.read_csv(data_path)
        self.aqi_cleaned = None
        self.forecast_df = None
    
    def remove_whitespace_header(self, df):
        df.columns = df.columns.str.strip()
        return df
    
    def cleaning_data(self, df):
        df = self.remove_whitespace_header(df)
        if 'date' not in df.columns:
            raise KeyError("'date' column is missing in the DataFrame.")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in df.columns:
            if col != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def impute_missing_values(self, df):
        # Separate the date column
        date_col = df['date']
        # Drop the date column for imputation
        df_numeric = df.drop(columns=['date'])
        
        # Impute missing pollutant values
        imputer = KNN(k=3)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
        
        # Reattach the date column
        df_imputed['date'] = date_col
        return df_imputed
    
    def extract_aqi(self, df):
        aqi_list = []
        df = df.replace({'NaT': np.nan})
        col_name = df.columns
        for idx, row in df.iterrows():
            aqi_val = row['aqi'] if 'aqi' in col_name else np.nan
            pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
            input_list = [
                (pollutant, row[pollutant]) 
                for pollutant in pollutants 
                if pollutant in df.columns and not np.isnan(row[pollutant])
            ]
            if np.isnan(aqi_val) and len(input_list) > 1:
                try:
                    calc_aqi = aqi.to_aqi(input_list, algo=aqi.ALGO_MEP)
                    aqi_list.append(float(calc_aqi))
                except ValueError:
                    aqi_list.append(np.nan)
            elif np.isnan(aqi_val) and len(input_list) == 1:
                val = input_list[0]
                try:
                    calc_aqi = aqi.to_aqi([val], algo=aqi.ALGO_MEP)
                    aqi_list.append(calc_aqi)
                except ValueError:
                    aqi_list.append(np.nan)
            elif len(input_list) < 1:
                aqi_list.append(np.nan)
            else:
                aqi_list.append(float(aqi_val))
        df['aqi'] = aqi_list
        return df

    def preprocess_data(self):
        self.aqi_cleaned = self.cleaning_data(self.aqi_data)
        self.aqi_cleaned = self.impute_missing_values(self.aqi_cleaned)
        self.aqi_cleaned = self.extract_aqi(self.aqi_cleaned)
        
        # Further cleaning and transformation
        cols = ['date', 'aqi']
        aqi_complete = self.aqi_cleaned[cols]
        aqi_complete['aqi'] = pd.to_numeric(aqi_complete['aqi'], errors='coerce')
        aqi_complete = aqi_complete[aqi_complete.date >= '2016-01-01'].reset_index(drop=True)
        aqi_complete = aqi_complete.rename(columns={'aqi': 'bangkok_aqi'}).set_index('date')
        
        # Reset index to access 'date' column again
        aqi_complete = aqi_complete.reset_index()

        # Feature engineering
        aqi_complete = self.add_features(aqi_complete)
        return aqi_complete

    def add_features(self, df):
        # Lagged AQI (1-day lag)
        df['aqi_lag1'] = df['bangkok_aqi'].shift(1)
        # 7-day rolling average
        df['aqi_rolling7'] = df['bangkok_aqi'].rolling(7).mean().fillna(method='bfill')
        # Weekend flag
        df['is_weekend'] = df['date'].dt.weekday >= 5
        return df.dropna()

    def adf_test(self, data_cleaned):
        adf_res = adfuller(data_cleaned, autolag='AIC')
        print('p-Values:', adf_res[1])

    def fit_arima_model(self, data):
        # Prepare data for StatsForecast
        data_sf = data[['date', 'bangkok_aqi']].rename(columns={'date': 'ds', 'bangkok_aqi': 'y'})
        data_sf['unique_id'] = 1  # Add unique_id column
        
        # Initialize StatsForecast with AutoARIMA
        sf = StatsForecast(
            models=[AutoARIMA(season_length=7)],  # Weekly seasonality
            freq='D'
        )
        
        # Fit the model
        sf.fit(data_sf)
        return sf

    def generate_forecast(self, sf, data_2025):
        # Forecast the next 30 days
        forecast = sf.predict(h=30)
        forecast_dates = pd.date_range(start=data_2025['date'].max() + pd.Timedelta(days=1), periods=30)
        
        # Forecasted DataFrame
        self.forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecasted_aqi': forecast['AutoARIMA']
        })
        return self.forecast_df

    def analyze_residuals(self, sf, data):
        # Use cross-validation with a larger step size
        step_size = 14  # Increase step size to reduce the number of folds
        residuals = []
        for i in range(30, len(data), step_size):
            train = data.iloc[:i]
            test = data.iloc[i:i+step_size]
            
            # Prepare training data
            train_sf = train[['date', 'bangkok_aqi']].rename(columns={'date': 'ds', 'bangkok_aqi': 'y'})
            train_sf['unique_id'] = 1
            
            # Fit model on training data
            sf.fit(train_sf)
            
            # Forecast one step ahead
            forecast = sf.predict(h=step_size)
            residuals.extend(test['bangkok_aqi'].values - forecast['AutoARIMA'].values)
        
        # Plot residuals
        plt.figure(figsize=(12, 4))
        plt.plot(residuals)
        plt.title('Residuals Plot')
        plt.show()
        
        # ACF of residuals
        plot_acf(residuals, lags=20)
        plt.show()

    def validate_model(self, data):
        # Split data into train/test
        train = data[:-30]
        test = data[-30:]
        
        # Fit model on training data
        sf = self.fit_arima_model(train)
        
        # Forecast and evaluate
        forecast = sf.predict(h=30)
        rmse = sqrt(mean_squared_error(test['bangkok_aqi'], forecast['AutoARIMA']))
        print(f'RMSE: {rmse:.2f}')

    def run_forecast(self):
        # Step 1: Preprocess the data
        data = self.preprocess_data()
        
        # Step 2: Perform ADF test
        data_cleaned = data['bangkok_aqi'].dropna()
        self.adf_test(data_cleaned)
        
        # Step 3: Validate model performance
        self.validate_model(data)
        
        # Step 4: Fit the final model
        sf = self.fit_arima_model(data)
        
        # Step 5: Analyze residuals
        self.analyze_residuals(sf, data)
        
        # Step 6: Forecast for 2025
        data_2025 = data[data['Year'] == 2025].reset_index()
        forecast_df = self.generate_forecast(sf, data_2025)
        
        return forecast_df

# Usage Example
data_path = r"D:\AQI_forecasting\backend\data\bangkok-air-quality.csv"
arima_forecaster = ARIMAForecaster(data_path)
forecast_df = arima_forecaster.run_forecast()
print(forecast_df)