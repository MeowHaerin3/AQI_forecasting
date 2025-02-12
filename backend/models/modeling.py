import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable
import itertools
import json
# Set up logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model paths
MODEL_DIR = "backend/models"
SARIMA_MODEL_PATH = os.path.join(MODEL_DIR, "sarima_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
BEST_ORDER_PATH = "best_sarima_order.json"
os.makedirs(MODEL_DIR, exist_ok=True)


def save_sarima_model(model, path=SARIMA_MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("SARIMA model saved.")

def load_sarima_model(path=SARIMA_MODEL_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            logger.info("Loaded SARIMA model from disk.")
            return pickle.load(f)
    return None

def save_lstm_model(model, path=LSTM_MODEL_PATH):
    model.save(path)
    logger.info("LSTM model saved.")

@register_keras_serializable()
def custom_mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

def load_lstm_model(path=LSTM_MODEL_PATH):
    if os.path.exists(path):
        logger.info("Loaded LSTM model from disk.")
        return load_model(path, custom_objects={"mse": custom_mse})
    return None

def create_sequences(data, seq_length=30):
    logger.info(f"Creating sequences with length {seq_length}")
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    logger.info(f"Created {len(X)} sequences.")
    return np.array(X), np.array(y)

def save_best_order(order, seasonal_order, filename=BEST_ORDER_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if len(seasonal_order) != 4:
        logger.error(f"Invalid seasonal_order format: {seasonal_order}. Must have 4 elements.")
        raise ValueError("`seasonal_order` must be a tuple with four elements.")
    
    best_order_dict = {
        "order": order,
        "seasonal_order": seasonal_order
    }
    with open(filename, "w") as f:
        json.dump(best_order_dict, f)
    
    logger.info(f"Best SARIMA order and seasonal_order saved to {filename}")

def load_best_order(filename=BEST_ORDER_PATH):
    filename = os.path.join(MODEL_DIR, "best_sarima_order.json")

    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {filename}. Returning None.")
        return None


def find_best_order_sarima(aqi_df, seasonal_period=12):
    logger.info("Finding best SARIMA model using AIC...")
    saved_order = load_best_order()
    if saved_order:
        logger.info(f"Using saved best SARIMA order: {saved_order}")
        return saved_order["order"], saved_order["seasonal_order"]
    
    best_aic = np.inf
    best_order, best_seasonal_order = None, None
    
    for p, d, q, P, D, Q in itertools.product(range(3), range(2), range(3), range(3), range(2), range(3)):
        try:
            model = SARIMAX(aqi_df['aqi'], order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period))
            result = model.fit(disp=False)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order, best_seasonal_order = (p, d, q), (P, D, Q)
        except:
            continue
    
    save_best_order(best_order, best_seasonal_order)
    logger.info(f"Best seasonal order: {best_seasonal_order}")

    return best_order, best_seasonal_order

def fit_sarima_model(aqi_df,best_order,best_seasonal_order):
    logger.info("Fitting SARIMA model...")
    # Fit a SARIMAX model using the datetime-indexed series
    sarima_model = SARIMAX(aqi_df['aqi'], order=best_order, seasonal_order=best_seasonal_order)
    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.fittedvalues
    save_sarima_model(sarima_fit)
    
    summary_path = os.path.join(MODEL_DIR, "sarima_summary.txt")
    with open(summary_path, "w") as f:
        f.write(sarima_fit.summary().as_text())
        
    logger.info("SARIMA model fit complete.")
    return sarima_fit, sarima_pred

def train_lstm_model(residuals, seq_length=30, retrain=False):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    resid_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
    model = None if retrain else load_lstm_model()
    if model:
        logger.info("Using pre-trained LSTM model.")
        X_train, _ = create_sequences(resid_scaled, seq_length)
        return scaler.inverse_transform(model.predict(X_train)), model  # Return predictions
    
    logger.info("Training LSTM model on residuals...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    resid_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
    X_train, y_train = create_sequences(resid_scaled, seq_length)
    
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    # Predict residual corrections using the trained LSTM model
    lstm_pred_scaled = lstm_model.predict(X_train)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    save_lstm_model(lstm_model)
    logger.info("LSTM model training complete.")
    return lstm_pred, lstm_model

def hybrid_prediction(aqi_df, sarima_pred, lstm_pred, seq_length=30):
    logger.info("Creating hybrid predictions...")
    
    sarima_pred_aligned = sarima_pred.iloc[seq_length:]
    lstm_pred_series = pd.Series(lstm_pred.flatten(), index=sarima_pred_aligned.index)

    hybrid_pred = sarima_pred_aligned + lstm_pred_series
    logger.info("Hybrid prediction created.")
    return hybrid_pred

def evaluate_model(aqi_df, hybrid_pred, seq_length=30):
    logger.info("Evaluating the hybrid model...")
    actual = aqi_df['aqi'].iloc[seq_length:]
    mae, mse = mean_absolute_error(actual, hybrid_pred), mean_squared_error(actual, hybrid_pred)
    return mae, mse, np.sqrt(mse)

def hybrid_forecast(aqi_df, sarima_fit, lstm_model, forecast_steps, seq_length=30, noise_std=0.1):
    logger.info(f"Generating hybrid forecast for {forecast_steps} steps.")

    try:
        forecast_obj = sarima_fit.get_forecast(steps=forecast_steps)
        forecast_values_sarima = forecast_obj.predicted_mean.values
    except Exception as e:
        logger.error(f"Error in SARIMA forecast: {e}")
        return pd.Series([np.nan] * forecast_steps)

    # Ensure SARIMA fitted values align with aqi_df['aqi']
    sarima_pred = sarima_fit.fittedvalues.reindex(aqi_df.index).dropna()
    if len(sarima_pred) != len(aqi_df['aqi']):
        logger.warning("SARIMA fitted values length mismatch. Adjusting...")
        sarima_pred = sarima_pred.reindex(aqi_df.index, fill_value=0)

    residuals = aqi_df['aqi'] - sarima_pred

    scaler = MinMaxScaler(feature_range=(-1, 1))
    resid_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

    if len(resid_scaled) < seq_length:
        logger.warning("Not enough residuals for LSTM input. Padding with zeros.")
        resid_scaled = np.pad(resid_scaled, ((seq_length - len(resid_scaled), 0), (0, 0)), mode='constant')

    last_sequence = resid_scaled[-seq_length:].reshape(1, seq_length, 1)

    try:
        lstm_residuals_scaled = lstm_model.predict(last_sequence, verbose=0)
        lstm_residuals = scaler.inverse_transform(lstm_residuals_scaled).flatten()
    except Exception as e:
        logger.error(f"LSTM prediction error: {e}")
        lstm_residuals = np.zeros(forecast_steps)

    if len(lstm_residuals) != forecast_steps:
        logger.warning(f"LSTM residuals shape mismatch. Expected {forecast_steps}, got {len(lstm_residuals)}")
        lstm_residuals = np.interp(np.linspace(0, 1, forecast_steps), np.linspace(0, 1, len(lstm_residuals)), lstm_residuals)

    # Add noise to the forecast values
    noise = np.random.normal(0, noise_std, forecast_steps)
    hybrid_forecast_values = forecast_values_sarima + lstm_residuals + noise

    last_date = aqi_df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

    hybrid_forecast_series = pd.Series(hybrid_forecast_values, index=forecast_dates)

    if hybrid_forecast_series.isna().all():
        logger.error("Hybrid forecast series is empty! Returning NaN values.")
        hybrid_forecast_series = pd.Series([np.nan] * forecast_steps, index=forecast_dates)

    logger.info("Hybrid forecast generation complete.")
    return hybrid_forecast_series
