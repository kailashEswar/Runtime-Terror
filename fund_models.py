import pandas as pd
import numpy as np

DATASET_PATH = "processed_daily_revenue.csv"  
REVENUE_COLUMN = "revenue"                        
FORECAST_HORIZON = 30                             
WINDOW_SIZE = 7                                 

df = pd.read_csv(DATASET_PATH)

if REVENUE_COLUMN not in df.columns:
    raise ValueError(f"'{REVENUE_COLUMN}' column not found in dataset")

revenue_series = df[REVENUE_COLUMN].dropna().reset_index(drop=True)


def naive_forecast(series, horizon):
    last_value = series.iloc[-1]
    return np.full(horizon, last_value)

naive_predictions = naive_forecast(revenue_series, FORECAST_HORIZON)

def moving_average_forecast(series, horizon, window_size):
    avg_value = series.tail(window_size).mean()
    return np.full(horizon, avg_value)

ma_predictions = moving_average_forecast(
    revenue_series, FORECAST_HORIZON, WINDOW_SIZE
)

print("\nBaseline Forecasting Results")
print("-" * 40)

print("\nNaive Forecast (Next 30 Values):")
print(naive_predictions)

print("\nMoving Average Forecast (Next 30 Values):")
print(ma_predictions)
