import logging
import os
import pickle
import itertools
import json
import warnings
from typing import Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_DIR = "backend/models"
SARIMA_MODEL_PATH = os.path.join(MODEL_DIR, "sarima_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
BEST_ORDER_PATH = os.path.join(MODEL_DIR, "best_sarima_order.json")
os.makedirs(MODEL_DIR, exist_ok=True)


def save_sarima_model(model: Any, path: str = SARIMA_MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("SARIMA model saved.")


def load_sarima_model(path: str = SARIMA_MODEL_PATH) -> Optional[Any]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            logger.info("Loaded SARIMA model from disk.")
            return pickle.load(f)
    return None


def save_lstm_model(model: Model, path: str = LSTM_MODEL_PATH) -> None:
    model.save(path)
    logger.info("LSTM model saved.")


@register_keras_serializable()
def custom_mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)


def load_lstm_model(path: str = LSTM_MODEL_PATH) -> Optional[Model]:
    if os.path.exists(path):
        logger.info("Loaded LSTM model from disk.")
        return load_model(path, custom_objects={"mse": custom_mse})
    return None


def create_sequences(data: np.ndarray, seq_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    logger.info(f"Created {len(X)} sequences.")
    return np.array(X), np.array(y)


def save_best_order(order: Tuple[int, int, int],
                    seasonal_order: Tuple[int, int, int, int],
                    filename: str = BEST_ORDER_PATH) -> None:
    if len(seasonal_order) != 4:
        logger.error(f"Invalid seasonal_order format: {seasonal_order}. Must have 4 elements.")
        raise ValueError("`seasonal_order` must be a tuple with four elements.")
    best_order_dict = {"order": order, "seasonal_order": seasonal_order}
    with open(filename, "w") as f:
        json.dump(best_order_dict, f)
    logger.info(f"Best SARIMA order and seasonal_order saved to {filename}")


def load_best_order(filename: str = BEST_ORDER_PATH) -> Optional[dict]:
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    logger.warning(f"File not found: {filename}. Returning None.")
    return None


def find_best_order_sarima(aqi_df: pd.DataFrame, seasonal_period: int = 12) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    logger.info("Finding best SARIMA model using AIC...")
    saved_order = load_best_order()
    if saved_order:
        logger.info(f"Using saved best SARIMA order: {saved_order}")
        return tuple(saved_order["order"]), tuple(saved_order["seasonal_order"])
    best_aic = np.inf
    best_order, best_seasonal_order = None, None
    for p, d, q, P, D, Q in itertools.product(range(3), range(2), range(3), range(3), range(2), range(3)):
        seasonal_ord = (P, D, Q, seasonal_period)
        try:
            model = SARIMAX(aqi_df['aqi'], order=(p, d, q), seasonal_order=seasonal_ord)
            result = model.fit(disp=False)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order, best_seasonal_order = (p, d, q), seasonal_ord
        except Exception:
            continue
    if best_order is None or best_seasonal_order is None:
        raise ValueError("No valid SARIMA model found.")
    save_best_order(best_order, best_seasonal_order)
    logger.info(f"Best SARIMA order: {best_order}, seasonal order: {best_seasonal_order}")
    return best_order, best_seasonal_order


def fit_sarima_model(aqi_df: pd.DataFrame, best_order: Tuple[int, int, int],
                     best_seasonal_order: Tuple[int, int, int, int]) -> Tuple[Any, pd.Series]:
    logger.info("Fitting SARIMA model...")
    sarima_model = SARIMAX(aqi_df['aqi'], order=best_order, seasonal_order=best_seasonal_order)
    sarima_fit = sarima_model.fit(disp=False)
    
    sarima_pred = sarima_fit.fittedvalues
   
    save_sarima_model(sarima_fit)
    summary_path = os.path.join(MODEL_DIR, "sarima_summary.txt")
    
    with open(summary_path, "w") as f:
        f.write(sarima_fit.summary().as_text())
    logger.info("SARIMA model fit complete.")
    return sarima_fit, sarima_pred


def train_lstm_model(residuals: pd.Series, seq_length: int = 30, retrain: bool = False) -> Tuple[np.ndarray, Model]:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    resid_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
    model = None if retrain else load_lstm_model()
    
    if model is not None:
        logger.info("Using pre-trained LSTM model.")
        X_train, _ = create_sequences(resid_scaled, seq_length)
        pred_scaled = model.predict(X_train)
        return scaler.inverse_transform(pred_scaled), model
    
    logger.info("Training LSTM model on residuals...")
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
    lstm_pred_scaled = lstm_model.predict(X_train)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    
    save_lstm_model(lstm_model)
    logger.info("LSTM model training complete.")
    return lstm_pred, lstm_model


def hybrid_prediction(aqi_df: pd.DataFrame, sarima_pred: pd.Series,
                      lstm_pred: np.ndarray, seq_length: int = 30) -> pd.Series:
    sarima_pred_aligned = sarima_pred.iloc[seq_length:]
    lstm_pred_series = pd.Series(lstm_pred.flatten(), index=sarima_pred_aligned.index)
    hybrid_pred = sarima_pred_aligned + lstm_pred_series
    logger.info("Hybrid prediction created.")
    return hybrid_pred


def evaluate_model(aqi_df: pd.DataFrame, hybrid_pred: pd.Series, seq_length: int = 30) -> Tuple[float, float, float]:
    actual = aqi_df['aqi'].iloc[seq_length:]
    mae = mean_absolute_error(actual, hybrid_pred)
    mse = mean_squared_error(actual, hybrid_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def hybrid_forecast(
    aqi_df: pd.DataFrame,
    sarima_fit: Any,
    lstm_model: Any,
    forecast_steps: int,
    seq_length: int = 30,
    noise_std: Optional[float] = None,
    residual_weight: float = 1.0
) -> pd.Series:

    try:
        forecast_values_sarima = sarima_fit.forecast(steps=forecast_steps).values
    except Exception as e:
        logger.error(f"Error in SARIMA forecast: {e}")
        return pd.Series([np.nan] * forecast_steps)
    
    sarima_fitted = sarima_fit.fittedvalues.reindex(aqi_df.index).fillna(0)
    residuals = aqi_df['aqi'] - sarima_fitted
    
    # Scale residuals for LSTM
    scaler = MinMaxScaler(feature_range=(-1, 1))
    resid_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
    
    if len(resid_scaled) < seq_length:
        logger.warning("Not enough residuals for LSTM input. Padding with zeros.")
        resid_scaled = np.pad(resid_scaled, ((seq_length - len(resid_scaled), 0), (0, 0)), mode='constant')
    
    current_sequence = resid_scaled[-seq_length:].copy()
    predicted_residuals = []
    
    for step in range(forecast_steps):
        input_seq = current_sequence.reshape(1, seq_length, 1)
        try:
            pred_scaled = lstm_model.predict(input_seq, verbose=0)
            pred_residual = scaler.inverse_transform(pred_scaled)[0, 0]
        except Exception as e:
            logger.error(f"LSTM prediction error at step {step}: {e}")
            pred_residual = 0.0
        
        predicted_residuals.append(pred_residual)
        
        # Update sequence
        current_sequence = np.append(current_sequence[1:], [[pred_scaled[0, 0]]], axis=0)
    
    predicted_residuals = np.array(predicted_residuals)
    
    if noise_std is None:
        noise_std = np.std(residuals)
    noise = np.random.normal(0, noise_std, forecast_steps)
    
    # hybrid_forecast_values = forecast_values_sarima + residual_weight * predicted_residuals + noise
    hybrid_forecast_values = forecast_values_sarima + residual_weight * predicted_residuals + 30
    forecast_dates = pd.date_range(start=aqi_df.index[-1] + pd.Timedelta(days=1),
                                   periods=forecast_steps, freq='D')
    
    hybrid_forecast_series = pd.Series(hybrid_forecast_values, index=forecast_dates)
    
    if hybrid_forecast_series.isna().all():
        logger.error("Hybrid forecast series is empty! Returning NaN values.")
        hybrid_forecast_series = pd.Series([np.nan] * forecast_steps, index=forecast_dates)
    
    logger.info("Hybrid iterative forecast generation complete.")
    return hybrid_forecast_series


