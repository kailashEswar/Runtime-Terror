import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATASET_PATH = "processed_daily_revenue.csv"
DATE_COLUMN = "transaction_date"
REVENUE_COLUMN = "revenue"
FORECAST_HORIZON = 30

df = pd.read_csv(DATASET_PATH)
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)


df["lag_1"] = df[REVENUE_COLUMN].shift(1)
df["lag_7"] = df[REVENUE_COLUMN].shift(7)
df["lag_14"] = df[REVENUE_COLUMN].shift(14)


df["roll_mean_7"] = df[REVENUE_COLUMN].rolling(7).mean()
df["roll_mean_14"] = df[REVENUE_COLUMN].rolling(14).mean()
df["roll_std_7"] = df[REVENUE_COLUMN].rolling(7).std()
df["roll_std_14"] = df[REVENUE_COLUMN].rolling(14).std()


def rolling_slope(series, window=7):
    slopes = [np.nan] * len(series)
    for i in range(window, len(series)):
        y = series[i-window:i]
        x = np.arange(window)
        slope = np.polyfit(x, y, 1)[0]
        slopes[i] = slope
    return slopes

df["trend_slope"] = rolling_slope(df[REVENUE_COLUMN], window=7)


df["day_of_week"] = df[DATE_COLUMN].dt.weekday
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


df = df.dropna().reset_index(drop=True)


train_df = df.iloc[:-FORECAST_HORIZON]
test_df = df.iloc[-FORECAST_HORIZON:]

FEATURES = [
    "lag_1", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_14",
    "roll_std_7", "roll_std_14",
    "trend_slope",
    "day_of_week", "is_weekend"
]

X_train = train_df[FEATURES]
y_train = train_df[REVENUE_COLUMN]

X_test = test_df[FEATURES]
y_test = test_df[REVENUE_COLUMN]


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


test_predictions = model.predict(X_test)

def evaluate(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    return mae, rmse

mae, rmse = evaluate(y_test, test_predictions)

print("\nMachine Learning Regression Model Results")
print("-" * 55)
print(f"Dataset        : {DATASET_PATH}")
print(f"Model          : Random Forest Regressor")
print(f"MAE            : {mae:.2f}")
print(f"RMSE           : {rmse:.2f}")

ml_predictions = test_predictions
