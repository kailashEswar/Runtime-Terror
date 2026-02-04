import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------
# 1. CONFIGURATION
# ----------------------------------

DATASET_PATH = "processed_daily_revenue.csv"
REVENUE_COLUMN = "revenue"
FORECAST_HORIZON = 30

# ----------------------------------
# 2. LOAD DATA
# ----------------------------------

df = pd.read_csv(DATASET_PATH)
series = df[REVENUE_COLUMN].dropna().reset_index(drop=True)

actual_values = series.tail(FORECAST_HORIZON).values

# ----------------------------------
# 3. -------- ARIMA MODEL ----------
# ----------------------------------

# Stationarity check
p_value = adfuller(series)[1]
d = 0
stationary_series = series.copy()

if p_value > 0.05:
    stationary_series = series.diff().dropna()
    d = 1

# Train–test split
train_arima = stationary_series[:-FORECAST_HORIZON]

# Train ARIMA
arima_model = ARIMA(train_arima, order=(1, d, 1))
arima_fit = arima_model.fit()

# ARIMA predictions
arima_predictions = arima_fit.forecast(steps=FORECAST_HORIZON)

# ----------------------------------
# 4. ------ ML REGRESSION MODEL -----
# ----------------------------------

df_ml = pd.DataFrame()
df_ml["revenue"] = series

# Feature engineering
df_ml["lag_1"] = df_ml["revenue"].shift(1)
df_ml["lag_7"] = df_ml["revenue"].shift(7)
df_ml["lag_14"] = df_ml["revenue"].shift(14)

df_ml["roll_mean_7"] = df_ml["revenue"].rolling(7).mean()
df_ml["roll_mean_14"] = df_ml["revenue"].rolling(14).mean()
df_ml["roll_std_7"] = df_ml["revenue"].rolling(7).std()
df_ml["roll_std_14"] = df_ml["revenue"].rolling(14).std()

# Trend slope
def rolling_slope(series, window=7):
    slopes = [np.nan] * len(series)
    for i in range(window, len(series)):
        y = series[i-window:i]
        x = np.arange(window)
        slopes[i] = np.polyfit(x, y, 1)[0]
    return slopes

df_ml["trend_slope"] = rolling_slope(df_ml["revenue"], 7)

df_ml = df_ml.dropna().reset_index(drop=True)

# Train–test split
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

# Train ML model
ml_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
ml_model.fit(X_train, y_train)

# ML predictions
ml_predictions = ml_model.predict(X_test)

# ----------------------------------
# 5. ---- EVALUATION (MAE & RMSE) ---
# ----------------------------------

def evaluate(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    return mae, rmse

arima_mae, arima_rmse = evaluate(actual_values, arima_predictions)
ml_mae, ml_rmse = evaluate(actual_values, ml_predictions)

# ----------------------------------
# 6. ---- ADAPTIVE HYBRID MODEL -----
# ----------------------------------

# Dynamic weights (inverse MAE)
w_arima = (1 / arima_mae) / ((1 / arima_mae) + (1 / ml_mae))
w_ml = (1 / ml_mae) / ((1 / arima_mae) + (1 / ml_mae))

hybrid_forecast = (w_arima * arima_predictions) + (w_ml * ml_predictions)

hybrid_mae, hybrid_rmse = evaluate(actual_values, hybrid_forecast)

# ----------------------------------
# 7. RESULTS
# ----------------------------------

print("\n===== Adaptive Hybrid Forecasting Results =====")
print(f"ARIMA  → MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}")
print(f"ML     → MAE: {ml_mae:.2f}, RMSE: {ml_rmse:.2f}")
print(f"Hybrid → MAE: {hybrid_mae:.2f}, RMSE: {hybrid_rmse:.2f}")

print("\nModel Weights:")
print(f"Weight (ARIMA): {w_arima:.2f}")
print(f"Weight (ML)   : {w_ml:.2f}")

print("\nHybrid Forecast (First 10 Values):")
print(hybrid_forecast[:10])
