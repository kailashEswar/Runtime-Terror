import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range("2024-01-01", periods=120, freq="D")
customers = [f"CUST_{i:03d}" for i in range(1, 21)]

data = []
for d in dates:
    for c in np.random.choice(customers, size=np.random.randint(1, 5)):
        revenue = np.random.normal(500, 150)
        if np.random.rand() < 0.05:
            revenue = -abs(revenue)  # error data
        if np.random.rand() < 0.05:
            revenue = None  # missing value
        data.append([c, d, revenue])

df = pd.DataFrame(data, columns=["customer_id", "transaction_date", "revenue"])

csv_path = "revenue_transactions.csv"
xlsx_path = "revenue_transactions.xlsx"

df.to_csv(csv_path, index=False)
df.to_excel(xlsx_path, index=False)

csv_path, xlsx_path