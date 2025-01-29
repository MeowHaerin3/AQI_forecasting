# %% ----------------------------
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
from typing import Dict
import aqi
import warnings

# Suppress warnings globally
warnings.filterwarnings("ignore")

plt.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')


# Main Forecaster Class
class AQIForecaster:
    """End-to-end AQI forecasting pipeline using AutoARIMA with improved accuracy."""
    
    def __init__(self, data_path: str, season_length: int = 7, horizon: int = 30):
        """
        Initialize forecaster.
        
        :param data_path: Path to raw data CSV
        :param season_length: Seasonal period length (7=daily seasonality)
        :param horizon: Number of periods to forecast
        """
        self.data_path = data_path
        self.season_length = season_length
        self.horizon = horizon
        self.raw_df = None
        self.processed_df = None
        self.train_df = None
        self.test_df = None
        self.forecaster = None
        self.forecast = None

    def load_data(self) -> None:
        """Load and clean raw data with numeric validation."""
        try:
            self.raw_df = pd.read_csv(self.data_path)
            self.raw_df = self.raw_df.rename(columns=str.strip)
            
            # Convert pollutant columns to numeric
            pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
            for col in pollutants:
                if col in self.raw_df.columns:
                    self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce')
            
            self.raw_df['date'] = pd.to_datetime(self.raw_df['date'], errors='coerce')
            self.raw_df = self.raw_df.sort_values('date').reset_index(drop=True)
            
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def _calculate_aqi(self, row: pd.Series) -> float:
        """Calculate AQI from pollutants with validation."""
        pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        valid_pollutants = []
        
        for p in pollutants:
            if p in row and not pd.isna(row[p]):
                try:
                    value = float(row[p])
                    if value >= 0:  # Filter negative values
                        valid_pollutants.append((p, value))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pollutants:
            return np.nan
        
        try:
            return aqi.to_aqi(valid_pollutants, algo=aqi.ALGO_MEP)
        except (ValueError, KeyError):
            return np.nan

    def preprocess_data(self) -> None:
        """Preprocess data with proper type handling."""
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        try:
            # Calculate AQI
            self.raw_df['aqi'] = self.raw_df.apply(self._calculate_aqi, axis=1)
            
            # Process time series
            aqi_df = self.raw_df[['date', 'aqi']].rename(columns={'aqi': 'y'})
            aqi_df = aqi_df.set_index('date').asfreq('D')
            
            # Ensure numeric type
            aqi_df['y'] = pd.to_numeric(aqi_df['y'], errors='coerce')
            
            # Temporal imputation
            aqi_df['y'] = aqi_df['y'].ffill().interpolate(method='time')
            
            self.processed_df = aqi_df.reset_index().rename(columns={'date': 'ds'})
            
        except Exception as e:
            raise RuntimeError(f"Data preprocessing failed: {str(e)}")

    def train_test_split(self) -> None:
        """Create time-aware train-test split."""
        if self.processed_df is None:
            raise ValueError("Data not processed. Call preprocess_data() first.")
            
        try:
            split_idx = -self.horizon
            self.train_df = self.processed_df.iloc[:split_idx]
            self.test_df = self.processed_df.iloc[split_idx:]
        except IndexError:
            raise ValueError("Not enough data for specified horizon")

    def setup_model(self) -> None:
        """Initialize AutoARIMA with optimized parameters."""
        if self.train_df is None:
            raise ValueError("Train data not available. Call train_test_split() first.")
        
        try:
            models = [AutoARIMA(
                season_length=self.season_length,
                approximation=False,  # Disable approximation for better accuracy
                trace=True,           # Show model selection process
                stepwise=False        # Perform exhaustive search for best parameters
            )]
            
            self.forecaster = StatsForecast(
                models=models,
                freq='D',
                n_jobs=-1
            )
        except Exception as e:
            raise RuntimeError(f"Model setup failed: {str(e)}")

    # def fit_predict(self) -> pd.DataFrame:
    #     """Train model and generate forecasts."""
    #     if self.forecaster is None:
    #         raise ValueError("Model not initialized. Call setup_model() first.")
            
    #     try:
    #         # Prepare training data
    #         sf_df = self.train_df.assign(unique_id=1)[['unique_id', 'ds', 'y']]
            
    #         # Generate forecasts (with confidence intervals)
    #         forecast = self.forecaster.forecast(
    #             df=sf_df,  # Pass the training data
    #             h=self.horizon,
    #             level=[90]  # 90% confidence interval
    #         )
            
    #         # Generate exact forecasted values using normal distribution
    #         forecast_values = forecast['AutoARIMA'].values
    #         forecast_std_errors = (forecast['AutoARIMA-hi-90'] - forecast['AutoARIMA-lo-90']) / (2 * 1.645)  # Convert CI to std
    #         exact_forecast_values = np.random.normal(loc=forecast_values, scale=forecast_std_errors)
            
    #         # Add exact forecasted values to the forecast DataFrame
    #         forecast['ExactForecast'] = exact_forecast_values
            
    #         self.forecast = forecast.reset_index().merge(
    #             self.test_df.assign(unique_id=1),
    #             on=['ds', 'unique_id'],
    #             how='left'
    #         )
    #         return self.forecast
            
    #     except Exception as e:
    #         raise RuntimeError(f"Forecasting failed: {str(e)}")
    
    def fit_predict(self) -> pd.DataFrame:
        """Train model and generate forecasts with accuracy-tuned noise."""
        if self.forecaster is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
            
        try:
            # Prepare training data
            sf_df = self.train_df.assign(unique_id=1)[['unique_id', 'ds', 'y']]
            
            # Generate forecasts with confidence intervals
            forecast = self.forecaster.forecast(
                df=sf_df,
                h=self.horizon,
                level=[90]
            )
            
            # --- Improved noise generation ---
            # 1. Calculate scaled standard errors
            ci_range = forecast['AutoARIMA-hi-90'] - forecast['AutoARIMA-lo-90']
            std_errors = ci_range / (2 * 1.645)  # Convert 90% CI to std
            
            # 2. Reduce noise impact by 40% for stability
            scale_factor = 0.6
            scaled_std = std_errors * scale_factor
            
            # 3. Generate constrained normal noise
            exact_forecast_values = np.random.normal(
                loc=forecast['AutoARIMA'],
                scale=scaled_std,
                size=len(forecast)
            )
            
            # 4. Clip values to CI boundaries as safety
            forecast['ExactForecast'] = np.clip(
                exact_forecast_values,
                forecast['AutoARIMA-lo-90'],
                forecast['AutoARIMA-hi-90']
            )
            # --- End of improvements ---
            
            self.forecast = forecast.reset_index().merge(
                self.test_df.assign(unique_id=1),
                on=['ds', 'unique_id'],
                how='left'
            )
            return self.forecast
            
        except Exception as e:
            raise RuntimeError(f"Forecasting failed: {str(e)}")
        
    def evaluate(self) -> Dict[str, float]:  # PROPERLY INDENTED METHOD
        """Calculate forecast accuracy metrics."""
        if self.forecast is None:
            raise ValueError("No forecast available. Call fit_predict() first.")
            
        try:
            y_true = self.forecast['y'].values
            y_pred = self.forecast['ExactForecast'].values
            
            return {
                'RMSE': sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
            }
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")

    def plot_forecast(self) -> None:
        """Visualize forecast results."""
        if self.forecast is None:
            raise ValueError("No forecast available. Call fit_predict() first.")
            
        try:
            plt.figure(figsize=(15, 6))
            
            plt.plot(self.forecast['ds'], self.forecast['y'], 
                    label='Actual', marker='o')
            plt.plot(self.forecast['ds'], self.forecast['ExactForecast'], 
                    label='Exact Forecast', marker='o')
            
            plt.title('AQI Forecast vs Actual')
            plt.xlabel('Date')
            plt.ylabel('AQI')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            raise RuntimeError(f"Plotting failed: {str(e)}")


# Usage Example
if __name__ == "__main__":
    try:
        # Initialize pipeline
        forecaster = AQIForecaster(
            data_path=r"D:\AQI_forecasting\backend\data\bangkok-air-quality.csv",
            season_length=7,
            horizon=30
        )
        
        # Execute workflow
        forecaster.load_data()
        forecaster.preprocess_data()
        forecaster.train_test_split()
        forecaster.setup_model()
        forecast_df = forecaster.fit_predict()
        
        # Show results
        metrics = forecaster.evaluate()
        print("\nForecast Accuracy:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}{'%' if metric == 'MAPE' else ''}")
        
        forecaster.plot_forecast()
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")