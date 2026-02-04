import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATASET_PATH = "processed_daily_revenue.csv"
REVENUE_COLUMN = "revenue"
FORECAST_HORIZON = 30


df = pd.read_csv(DATASET_PATH)
series = df[REVENUE_COLUMN].dropna().reset_index(drop=True)

actual_values = series.tail(FORECAST_HORIZON).values


p_value = adfuller(series)[1]
d = 0
stationary_series = series.copy()

if p_value > 0.05:
    stationary_series = series.diff().dropna()
    d = 1


train_arima = stationary_series[:-FORECAST_HORIZON]


arima_model = ARIMA(train_arima, order=(1, d, 1))
arima_fit = arima_model.fit()

arima_predictions = arima_fit.forecast(steps=FORECAST_HORIZON)


df_ml = pd.DataFrame()
df_ml["revenue"] = series

df_ml["lag_1"] = df_ml["revenue"].shift(1)
df_ml["lag_7"] = df_ml["revenue"].shift(7)
df_ml["lag_14"] = df_ml["revenue"].shift(14)

df_ml["roll_mean_7"] = df_ml["revenue"].rolling(7).mean()
df_ml["roll_mean_14"] = df_ml["revenue"].rolling(14).mean()
df_ml["roll_std_7"] = df_ml["revenue"].rolling(7).std()
df_ml["roll_std_14"] = df_ml["revenue"].rolling(14).std()


def rolling_slope(series, window=7):
    slopes = [np.nan] * len(series)
    for i in range(window, len(series)):
        y = series[i-window:i]
        x = np.arange(window)
        slopes[i] = np.polyfit(x, y, 1)[0]
    return slopes


df_ml["trend_slope"] = rolling_slope(df_ml["revenue"], 7)
df_ml = df_ml.dropna().reset_index(drop=True)


train_ml = df_ml.iloc[:-FORECAST_HORIZON]
test_ml = df_ml.iloc[-FORECAST_HORIZON:]

FEATURES = [
    "lag_1", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_14",
    "roll_std_7", "roll_std_14",
    "trend_slope"
]

X_train = train_ml[FEATURES]
y_train = train_ml["revenue"]

X_test = test_ml[FEATURES]
y_test = test_ml["revenue"]


ml_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
ml_model.fit(X_train, y_train)

ml_predictions = ml_model.predict(X_test)


def evaluate(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    return mae, rmse

def mape(true, predicted):
    true, predicted = np.array(true), np.array(predicted)
    mask = true != 0
    return np.mean(np.abs((true[mask] - predicted[mask]) / true[mask])) * 100

def wmape(true, predicted):
    true, predicted = np.array(true), np.array(predicted)
    return np.sum(np.abs(true - predicted)) / np.sum(np.abs(true)) * 100


arima_mae, arima_rmse = evaluate(actual_values, arima_predictions)
ml_mae, ml_rmse = evaluate(actual_values, ml_predictions)


w_arima = (1 / arima_mae) / ((1 / arima_mae) + (1 / ml_mae))
w_ml = (1 / ml_mae) / ((1 / arima_mae) + (1 / ml_mae))

hybrid_forecast = (w_arima * arima_predictions) + (w_ml * ml_predictions)

hybrid_mae, hybrid_rmse = evaluate(actual_values, hybrid_forecast)


arima_mape = mape(actual_values, arima_predictions)
ml_mape = mape(actual_values, ml_predictions)
hybrid_mape = mape(actual_values, hybrid_forecast)

arima_wmape = wmape(actual_values, arima_predictions)
ml_wmape = wmape(actual_values, ml_predictions)
hybrid_wmape = wmape(actual_values, hybrid_forecast)


print("\n===== Adaptive Hybrid Forecasting Results =====")
print(
    f"ARIMA  -> MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}, "
    f"MAPE: {arima_mape:.2f}%, wMAPE: {arima_wmape:.2f}%"
)

print(
    f"ML     -> MAE: {ml_mae:.2f}, RMSE: {ml_rmse:.2f}, "
    f"MAPE: {ml_mape:.2f}%, wMAPE: {ml_wmape:.2f}%"
)

print(
    f"Hybrid -> MAE: {hybrid_mae:.2f}, RMSE: {hybrid_rmse:.2f}, "
    f"MAPE: {hybrid_mape:.2f}%, wMAPE: {hybrid_wmape:.2f}%"
)

print("\nModel Weights:")
print(f"Weight (ARIMA): {w_arima:.2f}")
print(f"Weight (ML)   : {w_ml:.2f}")

print("\nHybrid Forecast (First 10 Values):")
print(hybrid_forecast[:10])
