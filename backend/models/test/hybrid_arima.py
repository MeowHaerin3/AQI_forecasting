import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import aqi  # Make sure you have an 'aqi' module installed

# Suppress warnings and TensorFlow logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##############################
# 1. Data Loading and Cleaning
##############################

data_path = r"D:\AQI_forecasting\backend\data\bangkok-air-quality.csv"
df = pd.read_csv(data_path)
print("Raw Data Sample:")
print(df.head())

# Utility functions
def remove_whitespace_cols(df):
    df.columns = df.columns.str.strip()
    return df

def cleaning_data(df):
    df = remove_whitespace_cols(df)
    if 'date' not in df.columns:
        raise ValueError('date column not found')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Convert all non-date columns to numeric
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    return df

def calc_aqi(row: pd.Series):
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
        if pollutant in row and pd.notna(row[pollutant]):
            try:
                aqi_values.append(aqi.to_aqi([(aqi_pollutant, row[pollutant])]))
            except TypeError:
                continue
    if len(aqi_values) == 0:
        return np.nan
    return np.max(aqi_values)

def preprocess_data(df):
    # Check required columns
    if 'date' not in df.columns or 'aqi' not in df.columns:
        raise ValueError('Required columns not found')
    aqi_df = df[['date', 'aqi']].copy()
    aqi_df['aqi'] = pd.to_numeric(aqi_df['aqi'], errors='coerce')
    aqi_df.set_index('date', inplace=True)
    aqi_df['aqi'] = aqi_df['aqi'].interpolate(method='time')
    aqi_df = aqi_df.reset_index()
    return aqi_df

# Clean and preprocess
df_clean = cleaning_data(df)
# Compute AQI from available pollutant columns
df_clean['aqi'] = df_clean.apply(calc_aqi, axis=1)
aqi_df = preprocess_data(df_clean)
print("Cleaned and Preprocessed Data Sample:")
print(aqi_df.head())

##################################
# 2. SARIMA Modeling and Residuals
##################################

# For this example, we assume daily data.
# If your data shows yearly seasonality, you might choose seasonality period = 365;
# if monthly then seasonality = 12. Adjust accordingly.
# Here, we use a seasonal period of 365.
aqi_df.set_index('date', inplace=True)

# (Optional) Visualize the series
plt.figure(figsize=(10,4))
plt.plot(aqi_df['aqi'], label="AQI")
plt.title("AQI Time Series")
plt.legend()
plt.show()

# Check stationarity (ADF test)
result_adf = adfuller(aqi_df['aqi'].dropna())
print("ADF Statistic:", result_adf[0])
print("p-value:", result_adf[1])
# If non-stationary, SARIMA with differencing will be applied automatically.

# Fit a SARIMAX model.
# Change orders based on your data. Here, we use order=(2,1,2) and seasonal_order=(1,1,1,365).
sarima_model = SARIMAX(aqi_df['aqi'], order=(2,1,2), seasonal_order=(1,1,1,365))
sarima_fit = sarima_model.fit(disp=False)

# Get in-sample fitted values and compute residuals.
sarima_pred = sarima_fit.fittedvalues
residuals = aqi_df['aqi'] - sarima_pred

plt.figure(figsize=(10,4))
plt.plot(residuals, label="SARIMA Residuals")
plt.title("SARIMA In-sample Residuals")
plt.legend()
plt.show()

#######################################
# 3. Prepare Data and Train the LSTM
#######################################

# We will train the LSTM on the residuals.
# Reset the index so that we have a numeric index for sequence creation.
resid_df = residuals.reset_index(drop=True)

# Normalize the residuals
scaler = MinMaxScaler(feature_range=(-1, 1))
resid_scaled = scaler.fit_transform(resid_df.values.reshape(-1, 1))

# Function to create sequences for supervised learning
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Using the past 30 time steps to predict the next residual value
X_train, y_train = create_sequences(resid_scaled, seq_length)

# Build the LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict residual corrections using the trained LSTM
lstm_pred_scaled = lstm_model.predict(X_train)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

#######################################
# 4. Combine SARIMA and LSTM Predictions
#######################################

# For simplicity, we will align the last len(lstm_pred) in-sample SARIMA predictions with the LSTM predictions.
# Note: X_train was created from resid_scaled, so its predictions correspond to time steps starting from index seq_length.
sarima_pred_aligned = sarima_pred.iloc[seq_length:]  # Drop the first 'seq_length' points so lengths match

# Convert LSTM predictions to a pandas Series with the same index as the aligned SARIMA predictions.
lstm_pred_series = pd.Series(lstm_pred.flatten(), index=sarima_pred_aligned.index)

# The hybrid prediction is the SARIMA in-sample prediction plus the LSTM residual correction.
hybrid_pred = sarima_pred_aligned + lstm_pred_series

# Plot the actual AQI, SARIMA prediction, and hybrid prediction.
plt.figure(figsize=(12,6))
plt.plot(aqi_df['aqi'], label="Actual AQI", color="blue")
plt.plot(sarima_pred, label="SARIMA Fitted", color="orange", alpha=0.7)
plt.plot(hybrid_pred, label="Hybrid (SARIMA + LSTM)", linestyle="dashed", color="red")
plt.title("AQI Forecast: Hybrid SARIMA + LSTM")
plt.legend()
plt.show()

###############################
# 5. Evaluate the Hybrid Model
###############################

# Compute error metrics for the period where hybrid predictions are available.
actual = aqi_df['aqi'].iloc[seq_length:]
mae = mean_absolute_error(actual, hybrid_pred)
mse = mean_squared_error(actual, hybrid_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
