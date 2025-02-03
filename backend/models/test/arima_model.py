import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# Data loading and initial inspection
data_path = r"D:\AQI_forecasting\backend\data\bangkok-air-quality.csv"
aqi_data = pd.read_csv(data_path)

# Cleaning and processing the AQI data
def remove_whitespace_header(df):
    df.columns = df.columns.str.strip()
    return df

def cleaning_data(df):
    df = remove_whitespace_header(df)
    if 'date' not in df.columns:
        raise KeyError("'date' column is missing in the DataFrame.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    return df

def extract_aqi(df):
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

aqi_cleaning = cleaning_data(aqi_data)
aqi_cleaning = extract_aqi(aqi_cleaning)

# # Further cleaning and transformation
cols = ['date', 'aqi']
aqi_complete = aqi_cleaning[cols]
aqi_complete['aqi'] = pd.to_numeric(aqi_complete['aqi'], errors='coerce')
aqi_complete = aqi_complete[aqi_complete.date >= '2016-01-01'].reset_index(drop=True)
aqi_complete = aqi_complete.rename(columns={'aqi': 'bangkok_aqi'}).set_index('date')

# Extracting year, month, and day
def extract_ymd(df):
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    return df

data = extract_ymd(aqi_complete.reset_index())

data_cleaned = data['bangkok_aqi'].dropna()

data_cleaned = data_cleaned.replace([np.inf, -np.inf], np.nan).dropna()

adf_res = adfuller(data_cleaned, autolag='AIC')
print('p-Values:', adf_res[1])

data['bangkok_aqi_diff'] = data['bangkok_aqi'].diff()
adf_res = adfuller(data['bangkok_aqi_diff'].dropna(), autolag='AIC')
print('p-Values after differencing:', adf_res[1])

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
model = sm.tsa.arima.ARIMA(data['bangkok_aqi'], order=param)
results = model.fit()
print(results.summary())

# # Forecast for 2025
data_2025 = data[data['Year'] == 2025].reset_index()

# Forecast the next 30 days
forecast = results.get_forecast(steps=30)
forecast_dates = pd.date_range(start=data_2025['date'].max() + pd.Timedelta(days=1), periods=30)
forecast_values = forecast.predicted_mean
forecast_std_errors = forecast.se_mean
exact_forecast_values = np.random.normal(loc=forecast_values, scale=forecast_std_errors)
conf_int = forecast.conf_int(alpha=0.05)

# Forecasted DataFrame
forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'forecasted_aqi': exact_forecast_values,
    'lower_bound': conf_int.iloc[:, 0],
    'upper_bound': conf_int.iloc[:, 1]
})

# Plot forecasted data
plt.figure(figsize=(15, 6))
sns.lineplot(data=data_2025, x='date', y='bangkok_aqi', label='Actual')
sns.lineplot(data=forecast_df, x='date', y='forecasted_aqi', label='Forecasted', color='orange')
plt.fill_between(forecast_df['date'], forecast_df['lower_bound'], forecast_df['upper_bound'], color='orange', alpha=0.3)
plt.title('Forecasted AQI for the Next 30 Days')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()

