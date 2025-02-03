import pandas as pd
import numpy as np
import aqi
from decimal import Decimal
from fancyimpute import KNN
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# Load Data
data_path = r"D:\AQI_forecasting\backend\data\bangkok-air-quality.csv"
df = pd.read_csv(data_path)

# Remove whitespace in column names
df.columns = df.columns.str.strip()

# Convert date column
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Convert all columns (except 'date') to numeric
for col in df.columns:
    if col != 'date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.sort_values('date').reset_index(drop=True)

# Calculate AQI
def calc_aqi(row):
    pollutants = {
        'pm25': aqi.POLLUTANT_PM25,
        'pm10': aqi.POLLUTANT_PM10,
        'o3': aqi.POLLUTANT_O3_8H,
        'no2': aqi.POLLUTANT_NO2_1H,
        'so2': aqi.POLLUTANT_SO2_1H,
        'co': aqi.POLLUTANT_CO_8H
    }
    aqi_values = []
    for pollutant, aqi_pollutant in pollutants.items():
        if not np.isnan(row[pollutant]):
            try:
                aqi_values.append(aqi.to_aqi([(aqi_pollutant, row[pollutant])]))
            except TypeError:
                continue
    return np.nan if len(aqi_values) == 0 else np.max(aqi_values)

df['aqi'] = df.apply(calc_aqi, axis=1)

# Preprocess Data
def preprocess_data(df):
    df = df[['date', 'aqi']].rename(columns={'date': 'ds', 'aqi': 'y'})
    df = df.set_index('ds').asfreq('D')
    df['y'] = df['y'].interpolate(method='time')
    
    # Convert Decimal to Float
    df['y'] = df['y'].astype(float)

    df['unique_id'] = 1
    return df.reset_index()

data = preprocess_data(df)

# Impute missing values using KNN
def impute_missing_data(df):
    ds = df['ds']
    df = df.drop(columns=['ds'])
    imputer = KNN(k=3)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    df['ds'] = ds
    return df

data = impute_missing_data(data)

# Split Data
train_size = int(len(data) * 0.7)
train_df, test_df = data.iloc[:train_size], data.iloc[train_size:]

# Ensure test_df has unique_id before merging
test_df['unique_id'] = 1

# -------------------------------
# Forecasting using AutoARIMA
# -------------------------------

# Define the forecast horizon (number of days to forecast)
h = len(test_df)

# Initialize StatsForecast with the AutoARIMA model.
sf = StatsForecast(
    models=[AutoARIMA(season_length=7)],
    freq='D',
    n_jobs=-1
)

# Prepare training data (original AQI values)
train_data = train_df[['unique_id', 'ds', 'y']]

# Fit the model and forecast
forecast_df = sf.forecast(h, train_data)

# -------------------------------
# Filter Forecast and Test Data for 2025 Only
# -------------------------------
test_df_2025 = test_df[test_df['ds'].dt.year == 2025]
forecast_df_2025 = forecast_df[forecast_df['ds'].dt.year == 2025]

# -------------------------------
# Plotting the Results
# -------------------------------
fig = go.Figure()

# Plot the actual AQI values for 2025
fig.add_trace(go.Scatter(
    x=test_df_2025['ds'], 
    y=test_df_2025['y'], 
    mode='lines', 
    name='Actual (2025)',
    line=dict(color='black')
))

# Plot the forecasted AQI values for 2025
fig.add_trace(go.Scatter(
    x=forecast_df_2025['ds'], 
    y=forecast_df_2025['AutoARIMA'], 
    mode='lines', 
    name='Forecast (2025)',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='AQI Forecasting for 2025 (AutoARIMA)',
    xaxis_title='Date',
    yaxis_title='AQI',
    template='plotly_white'
)

fig.show()
