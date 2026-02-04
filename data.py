import pandas as pd
import numpy as np
from datetime import datetime

START_DATE = "2024-01-01"
NUM_DAYS = 120
NUM_CUSTOMERS = 20
MAX_TXNS_PER_DAY = 4

MEAN_REVENUE = 500
STD_REVENUE = 150

NEGATIVE_PROB = 0.05   # refund/error probability
MISSING_PROB = 0.05    # missing value probability

CSV_PATH = "revenue_transactions.csv"
XLSX_PATH = "revenue_transactions.xlsx"

np.random.seed(int(datetime.now().timestamp()))

# ----------------------------------
# DATA GENERATION
# ----------------------------------

dates = pd.date_range(START_DATE, periods=NUM_DAYS, freq="D")
customers = [f"CUST_{i:03d}" for i in range(1, NUM_CUSTOMERS + 1)]

data = []

for date in dates:
    num_txns = np.random.randint(1, MAX_TXNS_PER_DAY + 1)
    chosen_customers = np.random.choice(customers, size=num_txns, replace=False)

    for cust in chosen_customers:
        revenue = np.random.normal(MEAN_REVENUE, STD_REVENUE)

        # Inject error cases
        if np.random.rand() < NEGATIVE_PROB:
            revenue = -abs(revenue)

        if np.random.rand() < MISSING_PROB:
            revenue = None

        data.append([cust, date, revenue])


df = pd.DataFrame(
    data,
    columns=["customer_id", "transaction_date", "revenue"]
)

df.to_csv(CSV_PATH, index=False)
df.to_excel(XLSX_PATH, index=False)

print("Dataset generated successfully:")
print(f"- {CSV_PATH}")
print(f"- {XLSX_PATH}")
