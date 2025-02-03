import pandas as pd
from data_preprocessing import load_and_clean_data
from modeling import (fit_sarima_model, train_lstm_model, hybrid_prediction,
                      evaluate_model, plot_results, plot_2025_actual, plot_7_days_forecast)

# File path to your CSV data file
data_path = r"D:\AQI_forecasting\backend\data\bangkok-air-quality.csv"

# 1. Load and Clean Data
aqi_df = load_and_clean_data(data_path)

# 2. Fit SARIMA Model
sarima_fit, sarima_pred = fit_sarima_model(aqi_df)

# 3. Train LSTM on Residuals (difference between actual and SARIMA fitted values)
residuals = aqi_df['aqi'] - sarima_pred
lstm_pred = train_lstm_model(residuals)

# 4. Combine SARIMA and LSTM Predictions (Hybrid)
hybrid_pred = hybrid_prediction(aqi_df, sarima_pred, lstm_pred)

# 5. Evaluate the Hybrid Model
mae, mse, rmse = evaluate_model(aqi_df, hybrid_pred)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# --- Additional Plots ---

# Plot Actual Data for 2025 only
plot_2025_actual(aqi_df)

# 6. Forecasting future AQI (7 days forecast after the last available date)
forecast_steps = 7  # Forecast for 7 days

# Use get_forecast instead of forecast to obtain a prediction results object
forecast_obj = sarima_fit.get_forecast(steps=forecast_steps)
forecast_values = forecast_obj.predicted_mean  # This should be a pandas Series

# Debug print: check if forecast_values is non-empty and look at its index
print("Forecast Values:")
print(forecast_values)
print("Forecast Values Index:", forecast_values.index)

# Get the last date from the dataset (as a datetime)
last_date = aqi_df.index[-1]
print("Last Date in Data:", last_date)

# Create forecast dates starting from the next day after the last date.
# In some cases, the forecast object may already have an index,
# but if not, you can reassign it using forecast_dates.
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                               periods=forecast_steps, freq='D')

# If forecast_values has an empty or non-datetime index, assign forecast_dates:
if forecast_values.empty or not isinstance(forecast_values.index, pd.DatetimeIndex):
    forecast_values.index = forecast_dates

# Plot the 7-Day Forecast only
plot_7_days_forecast(forecast_values)
