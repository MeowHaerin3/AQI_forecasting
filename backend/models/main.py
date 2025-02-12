import pandas as pd
from data_preprocessing import load_and_clean_data
from modeling import (fit_sarima_model, train_lstm_model, hybrid_prediction,
                      evaluate_model,hybrid_forecast,find_best_order_sarima)
import matplotlib.pyplot as plt

# File path to your CSV data file
data_path = r"D:\AQI_Forecasting\backend\data\bangkok-air-quality_update.csv"

# 1. Load and Clean Data
aqi_df = load_and_clean_data(data_path)

aqi_before_2025 = aqi_df[aqi_df.index.year < 2025]

best_order, best_seasonal_order = find_best_order_sarima(aqi_before_2025, seasonal_period=12)
# 2. Fit SARIMA Model
sarima_fit, sarima_pred = fit_sarima_model(aqi_before_2025,best_order,best_seasonal_order)

# 3. Train LSTM on Residuals (difference between actual and SARIMA fitted values)
residuals = aqi_before_2025['aqi'] - sarima_pred
lstm_pred, lstm_model = train_lstm_model(residuals)

# 4. Combine SARIMA and LSTM Predictions (Hybrid)
hybrid_pred = hybrid_prediction(aqi_before_2025, sarima_pred, lstm_pred)

# 5. Evaluate the Hybrid Model
mae, mse, rmse = evaluate_model(aqi_before_2025, hybrid_pred)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

hybrid_forecast_series = hybrid_forecast(aqi_before_2025,sarima_fit,lstm_model,forecast_steps=42)

try:
    plt.figure(figsize=(14, 7))

    # Ensure index alignment
    aqi_before_2025 = aqi_before_2025['aqi'].iloc[-30:]
    sarima_pred_plot = sarima_pred.iloc[-30:]  
    hybrid_pred_plot = hybrid_pred.iloc[-30:]  

    plt.plot(aqi_before_2025.index, aqi_before_2025, label='Actual AQI', marker='o')
    plt.plot(sarima_pred_plot.index, sarima_pred_plot, label='SARIMA Prediction', linestyle='dashed')
    plt.plot(hybrid_pred_plot.index, hybrid_pred_plot, label='Hybrid Prediction', linestyle='dashed')
    
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('AQI Predictions vs Actual (Last 30 Days)')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Hybrid Forecasting Series if not empty
    if not hybrid_forecast_series.empty:
        aqi_df_plot = aqi_df.iloc[-42:]
        plt.figure(figsize=(14, 7))
        plt.plot(hybrid_forecast_series.index, hybrid_forecast_series, label='Hybrid Forecast', color='red', marker='o')
        plt.plot(aqi_df_plot.index, aqi_df_plot, label='ACtual AQI', color='blue', marker='o')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.title('Hybrid Forecasting Series 30 days')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("Hybrid forecast series is empty, skipping plot.")

except Exception as e:
    print(f"An error occurred while plotting: {e}")