import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
from math import sqrt

warnings.filterwarnings('ignore')

class ARIMAForecaster:
    def __init__(self, data_path):
        self.data_path = data_path
        self.aqi_data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
        self.aqi_cleaned = None
        self.forecast_df = None
    
    def remove_whitespace_header(self, df):
        df.columns = df.columns.str.strip()
        return df
    
    def cleaning_data(self, df):
        df = self.remove_whitespace_header(df)
        df = df.fillna(method='ffill')  # Forward fill missing values
        return df
    
    def fit_arima_model(self, series):
        # Automatically determine the ARIMA order
        model = auto_arima(series, seasonal=False, trace=True)
        return model
    
    def forecast(self, steps=30):
        # Clean the data
        self.aqi_cleaned = self.cleaning_data(self.aqi_data)
        
        # Select the column to forecast
        if 'pm25' not in self.aqi_cleaned.columns:
            raise ValueError("Column 'pm25' does not exist in the dataset.")
        pm25_series = self.aqi_cleaned['pm25']
        
        # Split data into training and testing sets
        train_size = int(len(pm25_series) * 0.8)
        train, test = pm25_series[:train_size], pm25_series[train_size:]
        
        # Fit the ARIMA model
        model_fit = self.fit_arima_model(train)
        
        # Forecast
        forecast = model_fit.predict(n_periods=len(test))
        self.forecast_df = pd.DataFrame(forecast, index=test.index, columns=['forecast'])
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(train, label='Training Data')
        plt.plot(test, label='Test Data')
        plt.plot(self.forecast_df, label='Forecast')
        plt.legend(loc='best')
        plt.title('PM2.5 Forecasting with ARIMA')
        plt.show()
        
        # Print the forecasted values
        print(self.forecast_df)
        
        # Calculate and print the RMSE
        rmse = sqrt(mean_squared_error(test, forecast))
        print(f'Root Mean Squared Error: {rmse}')

# Usage
data_path = 'd:/AQI_forecasting/backend/models/bangkok-air-quality.csv'
forecaster = ARIMAForecaster(data_path)
forecaster.forecast()