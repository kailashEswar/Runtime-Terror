import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATASET_PATH = "processed_daily_revenue.csv"
REVENUE_COLUMN = "revenue"
FORECAST_HORIZON = 30

df = pd.read_csv(DATASET_PATH)

if REVENUE_COLUMN not in df.columns:
    raise ValueError(f"Column '{REVENUE_COLUMN}' not found in dataset")

series = df[REVENUE_COLUMN].dropna().reset_index(drop=True)


def is_stationary(series):
    p_value = adfuller(series)[1]
    return p_value < 0.05

d = 0
stationary_series = series.copy()

if not is_stationary(series):
    stationary_series = series.diff().dropna()
    d = 1


train = stationary_series[:-FORECAST_HORIZON]
test = stationary_series[-FORECAST_HORIZON:]


p = 1
q = 1


model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

test_predictions = model_fit.forecast(steps=FORECAST_HORIZON)

future_forecast = model_fit.forecast(steps=FORECAST_HORIZON)


def evaluate(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    return mae, rmse

mae, rmse = evaluate(test, test_predictions)



print("\nARIMA Model Results")
print("-" * 45)
print(f"Dataset        : {DATASET_PATH}")
print(f"Revenue column : {REVENUE_COLUMN}")
print(f"ARIMA(p={p}, d={d}, q={q})")
print(f"MAE            : {mae:.2f}")
print(f"RMSE           : {rmse:.2f}")

print("\nNext 30-Day Revenue Forecast:")
print(future_forecast.values)
