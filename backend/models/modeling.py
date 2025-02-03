import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
import os

# Set up logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_sequences(data, seq_length=30):
    logger.info(f"Creating sequences with length {seq_length}")
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    logger.info(f"Created {len(X)} sequences.")
    return np.array(X), np.array(y)

def fit_sarima_model(aqi_df):
    logger.info("Fitting SARIMA model...")
    # Fit a SARIMAX model using the datetime-indexed series
    sarima_model = SARIMAX(aqi_df['aqi'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.fittedvalues
    logger.info("SARIMA model fit complete.")
    return sarima_fit, sarima_pred

def train_lstm_model(residuals, seq_length=30):
    logger.info("Training LSTM model on residuals...")
    # Normalize the residuals
    scaler = MinMaxScaler(feature_range=(-1, 1))
    resid_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

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

    # Predict residual corrections using the trained LSTM model
    lstm_pred_scaled = lstm_model.predict(X_train)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    logger.info("LSTM model training complete.")
    return lstm_pred

def hybrid_prediction(aqi_df, sarima_pred, lstm_pred, seq_length=30):
    logger.info("Creating hybrid predictions...")
    # Align SARIMA and LSTM predictions by dropping the first 'seq_length' points from SARIMA
    sarima_pred_aligned = sarima_pred.iloc[seq_length:]
    lstm_pred_series = pd.Series(lstm_pred.flatten(), index=sarima_pred_aligned.index)

    # The hybrid prediction is the SARIMA fitted value plus the LSTM correction
    hybrid_pred = sarima_pred_aligned + lstm_pred_series
    logger.info("Hybrid prediction created.")
    return hybrid_pred

def evaluate_model(aqi_df, hybrid_pred, seq_length=30):
    logger.info("Evaluating the hybrid model...")
    # Evaluate only the period where hybrid predictions are available
    actual = aqi_df['aqi'].iloc[seq_length:]
    mae = mean_absolute_error(actual, hybrid_pred)
    mse = mean_squared_error(actual, hybrid_pred)
    rmse = np.sqrt(mse)
    logger.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, mse, rmse

def plot_results(aqi_df, sarima_pred, hybrid_pred, forecast_steps=None, forecast_values=None):
    """
    Plots:
      - Actual AQI (using the datetime index)
      - SARIMA in-sample fitted values
      - Hybrid (SARIMA + LSTM) predictions
      - (Optional) Out-of-sample forecast values
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual AQI using the datetime index
    plt.plot(aqi_df.index, aqi_df['aqi'], label="Actual AQI", color="blue")
    
    # Plot SARIMA fitted values
    plt.plot(sarima_pred.index, sarima_pred, label="SARIMA Fitted", color="orange", alpha=0.7)
    
    # Plot Hybrid predictions
    plt.plot(hybrid_pred.index, hybrid_pred, label="Hybrid (SARIMA + LSTM)", linestyle="dashed", color="red")
    
    # Optionally plot out-of-sample forecasts
    if forecast_steps is not None and forecast_values is not None:
        plt.plot(forecast_values.index, forecast_values, label="Out-of-Sample Forecast", color="green")
    
    plt.title("AQI Forecast: Hybrid SARIMA + LSTM")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.show()

def plot_2025_actual(aqi_df):
    """
    Filters and plots the actual AQI data for the year 2025.
    """
    # Filter actual data for 2025. This assumes your index is a DatetimeIndex.
    actual_2025 = aqi_df.loc['2025']
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual_2025.index, actual_2025['aqi'], label="Actual AQI 2025", color="blue")
    plt.title("Actual AQI Data for 2025")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.show()

def plot_7_days_forecast(forecast_values):
    """
    Plots a 7-day forecast from the model starting from the day after the last available date.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_values.index, forecast_values, label="7-Day Forecast", marker="o", color="green")
    plt.title("7-Day Forecast from Model")
    plt.xlabel("Date")
    plt.ylabel("Forecasted AQI")
    plt.legend()
    plt.show()

def predict_full_dataset(aqi_df):
    """
    Predict AQI from the beginning of the dataset to the latest date using the Hybrid Model (SARIMA + LSTM).

    Parameters:
    - aqi_df: DataFrame containing the AQI data with a DatetimeIndex

    Returns:
    - hybrid_predictions: Pandas Series of predicted AQI values from start to last available date
    """
    logger.info("Predicting AQI for the entire dataset...")

    # Fit SARIMA model
    sarima_fit, sarima_pred = fit_sarima_model(aqi_df)

    # Compute residuals
    residuals = aqi_df['aqi'] - sarima_pred

    # Train LSTM on residuals
    lstm_pred = train_lstm_model(residuals)

    # Generate hybrid predictions
    hybrid_pred = hybrid_prediction(aqi_df, sarima_pred, lstm_pred)

    logger.info("Full dataset prediction complete.")
    return hybrid_pred

def predict_2025(aqi_df):
    """
    Predict AQI for 2025 using the Hybrid Model (SARIMA + LSTM).

    Parameters:
    - aqi_df: DataFrame containing the AQI data with a DatetimeIndex

    Returns:
    - hybrid_predictions_2025: Pandas Series of predicted AQI values for 2025
    """
    logger.info("Predicting AQI for 2025...")

    # Fit SARIMA model
    sarima_fit, sarima_pred = fit_sarima_model(aqi_df)

    # Compute residuals
    residuals = aqi_df['aqi'] - sarima_pred

    # Train LSTM on residuals
    lstm_pred = train_lstm_model(residuals)

    # Generate hybrid predictions
    hybrid_pred = hybrid_prediction(aqi_df, sarima_pred, lstm_pred)

    # Filter only 2025 predictions
    hybrid_pred_2025 = hybrid_pred.loc['2025']

    logger.info("2025 prediction complete.")
    return hybrid_pred_2025

import matplotlib.pyplot as plt

def plot_2025_comparison(aqi_df, hybrid_pred_2025):
    """
    Plots actual AQI vs hybrid forecast for the year 2025.
    
    Parameters:
    - aqi_df: DataFrame containing actual AQI values.
    - hybrid_pred_2025: Series containing predicted AQI values for 2025.
    """
    logger.info("Plotting actual vs predicted AQI for 2025...")
    
    # Filter actual data for 2025
    actual_2025 = aqi_df.loc['2025']
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual_2025.index, actual_2025['aqi'], label='Actual AQI 2025', color='blue')
    plt.plot(hybrid_pred_2025.index, hybrid_pred_2025, label='Hybrid Forecast 2025', linestyle='dashed', color='red')
    
    plt.title('AQI Actual vs Hybrid Forecast for 2025')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.show()
    
    logger.info("Plot completed.")


