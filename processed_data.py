
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = "revenue_transactions.csv"
ID_COL = "customer_id"
DATE_COL = "transaction_date"
REVENUE_COL = "revenue"
OUTPUT_FILE = "processed_daily_revenue.csv"

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv(CSV_FILE)

print("Original columns:", df.columns.tolist())

# -----------------------------
# 2. Clean and standardize column names
# -----------------------------
df.columns = df.columns.str.strip()

# Rename to standard names (EDIT if needed)
df = df.rename(columns={
    "Date": "date",
    "date": "date",
    "Revenue": "revenue",
    "revenue": "revenue",
    "Amount": "revenue",
    "Total": "revenue"
})

print("Columns after renaming:", df.columns.tolist())

# -----------------------------
# 3. Check required columns
# -----------------------------
if DATE_COL not in df.columns or REVENUE_COL not in df.columns:
    raise ValueError(
        f"❌ Required columns not found!\n"
        f"Found: {df.columns.tolist()}\n"
        f"Expected: '{DATE_COL}', '{REVENUE_COL}'"
    )

# -----------------------------
# 4. Convert date column to datetime
# -----------------------------
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])

# -----------------------------
# 5. Sort by date
# -----------------------------
df = df.sort_values(DATE_COL)

# -----------------------------
# 6. Aggregate to daily level (3 columns)
# -----------------------------
daily_df = (
    df.groupby(DATE_COL)
      .agg(
          revenue=(REVENUE_COL, "sum"),
          transactions=(REVENUE_COL, "count")
      )
      .reset_index()
)

# -----------------------------
# 7. Resample to daily frequency (fill missing days)
# -----------------------------
daily_df = daily_df.set_index(DATE_COL).resample("D").sum()

# Fill missing values
daily_df["revenue"] = daily_df["revenue"].ffill().interpolate()
daily_df["transactions"] = daily_df["transactions"].fillna(0)

# -----------------------------
# 8. Outlier Detection using IQR
# -----------------------------
Q1 = daily_df["revenue"].quantile(0.25)
Q3 = daily_df["revenue"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

daily_df["is_outlier"] = (
    (daily_df["revenue"] < lower_bound) |
    (daily_df["revenue"] > upper_bound)
)

print("\nOutlier summary:")
print(daily_df["is_outlier"].value_counts())

# -----------------------------
# 9. Plot with outliers highlighted
# -----------------------------
plt.figure()
plt.plot(daily_df.index, daily_df["revenue"], label="Revenue")

outliers = daily_df[daily_df["is_outlier"]]
plt.scatter(outliers.index, outliers["revenue"], color="red", label="Outliers")

plt.title("Daily Revenue with Outliers Highlighted")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Final checks
# -----------------------------
print("\nProcessed Data Info:")
print(daily_df.info())

print("\nMissing Values Check:")
print(daily_df.isna().sum())

# -----------------------------
# 11. Save final CSV
# -----------------------------
daily_df.reset_index().to_csv(OUTPUT_FILE, index=False)

print("\n Data processing complete.")
print(f" Saved file: {OUTPUT_FILE}")
