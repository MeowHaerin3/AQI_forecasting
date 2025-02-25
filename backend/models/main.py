import sys
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import load_and_clean_data
from modeling import (
    fit_sarima_model,
    train_lstm_model,
    hybrid_prediction,
    evaluate_model,
    hybrid_forecast,
    find_best_order_sarima
)

def main():
    # -------------------------------
    # 1. Load and Clean Data
    # -------------------------------
    data_path = r"D:\AQI_Forecasting\backend\data\bangkok-air-quality_update.csv"
    try:
        print("Loading and cleaning data...")
        aqi_df = load_and_clean_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    aqi_before_2025 = aqi_df[aqi_df.index.year < 2025]

    # -------------------------------
    # 2. Find the Best SARIMA Order and Fit SARIMA Model
    # -------------------------------
    try:
        print("Finding best SARIMA order...")
        best_order, best_seasonal_order = find_best_order_sarima(aqi_before_2025, seasonal_period=12)
        
        print("Fitting SARIMA model...")
        sarima_fit, sarima_pred = fit_sarima_model(aqi_before_2025, best_order, best_seasonal_order)
    except Exception as e:
        print(f"Error with SARIMA modeling: {e}")
        sys.exit(1)

    # -------------------------------
    # 3. Train LSTM Model (on residuals if applicable)
    # -------------------------------
    try:
        print("Training LSTM model...")
        residual = aqi_before_2025['aqi']-sarima_pred
        lstm_pred, lstm_model = train_lstm_model(residual)
    except Exception as e:
        print(f"Error training LSTM model: {e}")
        sys.exit(1)

    # -------------------------------
    # 4. Create Hybrid Predictions
    # -------------------------------
    try:
        print("Generating hybrid predictions...")
        hybrid_pred = hybrid_prediction(aqi_before_2025, sarima_pred, lstm_pred)
    except Exception as e:
        print(f"Error generating hybrid predictions: {e}")
        sys.exit(1)

    # -------------------------------
    # 5. Evaluate the Hybrid Model
    # -------------------------------
    try:
        print("Evaluating the hybrid model...")
        mae, mse, rmse = evaluate_model(aqi_before_2025, hybrid_pred)
        print(f"Hybrid Model Evaluation:\n  MAE: {mae:.2f}\n  RMSE: {rmse:.2f}")
    except Exception as e:
        print(f"Error evaluating the hybrid model: {e}")

    # -------------------------------
    # 6. Generate Hybrid Forecast
    # -------------------------------
    forecast_steps = 42
    try:
        print(f"Generating hybrid forecast for the next {forecast_steps} days...")
        hybrid_forecast_series = hybrid_forecast(aqi_before_2025, sarima_fit, lstm_model, forecast_steps=forecast_steps)
    except Exception as e:
        print(f"Error generating hybrid forecast: {e}")
        hybrid_forecast_series = None

    # -------------------------------
    # 7. Plot Results
    # -------------------------------
    try:
        plt.figure(figsize=(14, 7))
        
        actual_last_30 = aqi_before_2025['aqi'].iloc[-30:]
        sarima_pred_last_30 = sarima_pred.iloc[-30:]
        hybrid_pred_last_30 = hybrid_pred.iloc[-30:]
        
        plt.plot(actual_last_30.index, actual_last_30, label='Actual AQI', marker='o')
        plt.plot(sarima_pred_last_30.index, sarima_pred_last_30, label='SARIMA Prediction', linestyle='dashed')
        plt.plot(hybrid_pred_last_30.index, hybrid_pred_last_30, label='Hybrid Prediction', linestyle='dashed')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.title('AQI Predictions vs Actual (Last 30 Days)')
        plt.legend()
        plt.grid()
        plt.show()
        
        if hybrid_forecast_series is not None and not hybrid_forecast_series.empty:
            aqi_df_plot = aqi_df.iloc[-42:]
            
            plt.figure(figsize=(14, 7))
            plt.plot(hybrid_forecast_series.index, hybrid_forecast_series, label='Hybrid Forecast', color='red', marker='o')
            plt.plot(aqi_df_plot.index, aqi_df_plot, label='Actual AQI', color='green', marker='o')
            plt.xlabel('Date')
            plt.ylabel('AQI')
            plt.title('Hybrid Forecast for the Next 42 Days')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print("Hybrid forecast series is empty, skipping forecast plot.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    main()
