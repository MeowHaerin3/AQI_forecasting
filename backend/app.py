from flask import Flask, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Fix the path for importing the module
sys.path.append(r'D:\AQI_forecasting')

from backend.models.evaluate import ARIMAForecaster  # Correct import

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the AQI Forecast API! Use /forecast to get the AQI forecast."

@app.route('/forecast', methods=['GET'])
def get_forecast():
    try:
        # Initialize the ARIMAForecaster with the data path
        data_path = r"D:\\AQI_forecasting\\backend\\data\\bangkok-air-quality.csv"
        arima_forecaster = ARIMAForecaster(data_path)

        # Run the forecast and get the DataFrame
        forecast_df = arima_forecaster.run_forecast()

        # Convert forecast_df to a list of dictionaries
        forecast_list = forecast_df.to_dict(orient="records")

        # Return the forecast data as JSON
        return jsonify(forecast_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot', methods=['GET'])
def plot_forecast():
    try:
        # Initialize the ARIMAForecaster with the data path
        data_path = r"D:\\AQI_forecasting\\backend\\data\\bangkok-air-quality.csv"
        arima_forecaster = ARIMAForecaster(data_path)

        # Run the forecast and get the DataFrame
        forecast_df = arima_forecaster.run_forecast()

        # Load and preprocess the actual data
        actual_data = pd.read_csv(data_path)
        actual_data['date'] = pd.to_datetime(actual_data['date'], errors='coerce')
        actual_data = actual_data.sort_values('date')

        # Filter only the data from 2025
        actual_data_2025 = actual_data[actual_data['date'].dt.year == 2025]

        # Plot the actual and forecasted data
        plt.figure(figsize=(12, 6))

        # Plot actual AQI data
        if not actual_data_2025.empty:
            plt.plot(
                actual_data_2025['date'], 
                actual_data_2025['aqi'], 
                label='Actual AQI (2025)', 
                color='green'
            )

        # Plot forecasted AQI data
        plt.plot(
            forecast_df['date'], 
            forecast_df['forecasted_aqi'], 
            label='Forecasted AQI', 
            color='blue'
        )

        # Plot confidence intervals
        plt.fill_between(
            forecast_df['date'], 
            forecast_df['lower_bound'], 
            forecast_df['upper_bound'], 
            color='lightblue', 
            alpha=0.5, 
            label='Confidence Interval'
        )

        # Add titles and labels
        plt.title('AQI Forecast vs Actual Data (2025)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('AQI', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return jsonify({"message": "Plot displayed successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
