import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    data_path = r"D:\AQI_Forecasting\backend\data\bangkok-air-quality_update.csv"
    try:
        print("Loading and cleaning data...")
        aqi_df = load_and_clean_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    aqi_before_2025 = aqi_df[aqi_df.index.year < 2025]
    aqi_test_df = aqi_df[len(aqi_before_2025):]

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
    try:
        print("Training LSTM model...")
        residual = aqi_before_2025['aqi']-sarima_pred
        lstm_pred, lstm_model = train_lstm_model(residual)
    except Exception as e:
        print(f"Error training LSTM model: {e}")
        sys.exit(1)

    # -------------------------------
    try:
        print("Generating hybrid predictions...")
        hybrid_pred = hybrid_prediction(aqi_before_2025, sarima_pred, lstm_pred)
    except Exception as e:
        print(f"Error generating hybrid predictions: {e}")
        sys.exit(1)

    # -------------------------------
    try:
        print("Evaluating the hybrid model...")
        mae, mse, rmse = evaluate_model(aqi_before_2025, hybrid_pred)
        print(f"Hybrid Model Evaluation:\n  MAE: {mae:.2f}\n  RMSE: {rmse:.2f}")
    except Exception as e:
        print(f"Error evaluating the hybrid model: {e}")

    # -------------------------------
    forecast_steps = 42
    try:
        print(f"Generating hybrid forecast for the next {forecast_steps} days...")
        hybrid_forecast_series = hybrid_forecast(aqi_before_2025, sarima_fit, lstm_model, forecast_steps=forecast_steps)
    except Exception as e:
        print(f"Error generating hybrid forecast: {e}")
        hybrid_forecast_series = None

    
    
    
    
    
    
    sns.set(style="whitegrid")
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(aqi_df.index, aqi_df, label='Actual AQI', color='b', linewidth=2)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('AQI', fontsize=12)
        plt.title('AQI Data', fontsize=14)
        plt.legend()
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(True, linestyle='--', alpha=0.6)  # Customize grid
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(aqi_test_df.index, aqi_test_df, label='Actual AQI after 2024 until 2025', color='b', linewidth=2)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('AQI', fontsize=12)
        plt.title('AQI Data', fontsize=14)
        plt.legend()
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(True, linestyle='--', alpha=0.6)  # Customize grid
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        
    try:
        plt.figure(figsize=(14, 7))
        
        plt.plot(aqi_df.index, aqi_df, label='Actual AQI', color='b', linewidth=2)
        plt.plot(sarima_pred.index, sarima_pred, label='predict', color='r', linewidth=2)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('AQI', fontsize=12)
        plt.title('AQI Data', fontsize=14)
        plt.legend()
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(True, linestyle='--', alpha=0.6)  # Customize grid
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(aqi_test_df.index, aqi_test_df, label='actual', color='r', linewidth=2)
        plt.plot(hybrid_forecast_series.index, hybrid_forecast_series, label='forecast AQI', color='b', linewidth=2)

        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('AQI', fontsize=12)
        plt.title('AQI Data', fontsize=14)
        plt.legend()
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(True, linestyle='--', alpha=0.6)  # Customize grid
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()
    except Exception as e:
        print(f"An error occurred during plotting: {e}")


if __name__ == "__main__":
    main()
