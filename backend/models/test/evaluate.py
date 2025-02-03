import pandas as pd
import numpy as np
import aqi
from fancyimpute import SimpleFill, KNN, IterativeImputer
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings

warnings.filterwarnings('ignore')

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
        self.aqi_cleaned = self.extract_aqi(self.aqi_cleaned)
        
        # Further cleaning and transformation
        cols = ['date', 'aqi']
        aqi_complete = self.aqi_cleaned[cols]
        aqi_complete['aqi'] = pd.to_numeric(aqi_complete['aqi'], errors='coerce')
        aqi_complete = aqi_complete[aqi_complete.date >= '2016-01-01'].reset_index(drop=True)
        aqi_complete = aqi_complete.rename(columns={'aqi': 'bangkok_aqi'}).set_index('date')
        
        # Reset index to access 'date' column again
        aqi_complete = aqi_complete.reset_index()

        # Extracting year, month, and day
        aqi_complete['Year'] = aqi_complete['date'].dt.year
        aqi_complete['Month'] = aqi_complete['date'].dt.month
        aqi_complete['Day'] = aqi_complete['date'].dt.day
        return aqi_complete


    def adf_test(self, data_cleaned):
        adf_res = adfuller(data_cleaned, autolag='AIC')
        print('p-Values:', adf_res[1])

    def fit_arima_model(self, data):
        # ARIMA model order selection
        p = range(1, 2)
        d = range(1, 2)
        q = range(0, 4)
        pdq = list(itertools.product(p, d, q))
        print(pdq)

        # Fitting ARIMA model
        aic = []
        for param in pdq:
            try:
                model = sm.tsa.arima.ARIMA(data['bangkok_aqi'].dropna(), order=param)
                results = model.fit()
                print('Order = {}'.format(param))
                print('AIC = {}'.format(results.aic))
                a = 'Order: '+str(param) +' AIC: ' + str(results.aic)
                aic.append(a)
            except:
                continue

        # Fit ARIMA model with selected order
        best_order = pdq[0]  # Select the best order from the printed list
        model = sm.tsa.arima.ARIMA(data['bangkok_aqi'], order=best_order)
        results = model.fit()
        print(results.summary())
        return results

    def generate_forecast(self, results, data_2025):
        # Forecast the next 30 days
        forecast = results.get_forecast(steps=30)
        forecast_dates = pd.date_range(start=data_2025['date'].max() + pd.Timedelta(days=1), periods=30)
        forecast_values = forecast.predicted_mean
        forecast_std_errors = forecast.se_mean
        exact_forecast_values = np.random.normal(loc=forecast_values, scale=forecast_std_errors)
        conf_int = forecast.conf_int(alpha=0.05)

        # Forecasted DataFrame
        self.forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecasted_aqi': exact_forecast_values,
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1]
        })
        return self.forecast_df

    def run_forecast(self):
        # Step 1: Preprocess the data
        data = self.preprocess_data()
        
        # Step 2: Perform ADF test
        data_cleaned = data['bangkok_aqi'].dropna()
        self.adf_test(data_cleaned)
        
        # Step 3: Fit the ARIMA model
        results = self.fit_arima_model(data)
        
        # Step 4: Forecast for 2025
        data_2025 = data[data['Year'] == 2025].reset_index()
        forecast_df = self.generate_forecast(results, data_2025)
        
        return forecast_df

# Usage Example
data_path = r"D:\AQI_forecasting\backend\data\bangkok-air-quality.csv"
arima_forecaster = ARIMAForecaster(data_path)
forecast_df = arima_forecaster.run_forecast()
print(forecast_df)


