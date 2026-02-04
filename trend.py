import pandas as pd
import numpy as np

# -------------------------------
# CONFIGURATION
# -------------------------------

DATASET_PATH = "processed_daily_revenue.csv"
REVENUE_COLUMN = "revenue"
ROLLING_WINDOW = 7
SLOPE_THRESHOLD = 0.01   # controls sensitivity

# -------------------------------
# LOAD DATA
# -------------------------------

df = pd.read_csv(DATASET_PATH)
revenue_series = df[REVENUE_COLUMN].dropna().reset_index(drop=True)

# -------------------------------
# ROLLING MEAN
# -------------------------------

rolling_mean = revenue_series.rolling(ROLLING_WINDOW).mean()

# -------------------------------
# ROLLING SLOPE COMPUTATION
# -------------------------------

def rolling_slope(series, window):
    slopes = [np.nan] * len(series)
    for i in range(window, len(series)):
        y = series[i-window:i]
        x = np.arange(window)
        slopes[i] = np.polyfit(x, y, 1)[0]
    return slopes

rolling_mean_slope = rolling_slope(rolling_mean, ROLLING_WINDOW)

# -------------------------------
# TREND REGIME CLASSIFICATION
# -------------------------------

def classify_trend(slope, threshold):
    if slope > threshold:
        return "Uptrend"
    elif slope < -threshold:
        return "Downtrend"
    else:
        return "Stable"

trend_regime = [
    classify_trend(s, SLOPE_THRESHOLD) if not np.isnan(s) else np.nan
    for s in rolling_mean_slope
]

# -------------------------------
# STORE RESULTS
# -------------------------------

trend_df = pd.DataFrame({
    "revenue": revenue_series,
    "rolling_mean": rolling_mean,
    "rolling_slope": rolling_mean_slope,
    "trend_regime": trend_regime
})

print("\nTrend Regime Detection Sample")
print(trend_df.tail(15))
