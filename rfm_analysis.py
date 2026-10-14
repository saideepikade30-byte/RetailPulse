
import pandas as pd
from pathlib import Path

# 1. Load the cleaned retail dataset
file_path = Path("retail_cleaned.csv")
df = pd.read_csv(file_path)

# 2. Convert transaction dates to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

# 3. Keep valid customer transactions
# Exclude cancellations and non-positive quantities/prices
sales = df[
    df["Customer ID"].notna()
    & (~df["IsCancellation"].astype(str).str.lower().eq("true"))
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (df["InvoiceDate"].notna())
].copy()

# 4. Make sure Customer ID has a consistent format
sales["Customer ID"] = sales["Customer ID"].astype(str)

# 5. Define the reference date for Recency
reference_date = sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

# 6. Calculate RFM for each customer
rfm = sales.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda dates:
             (reference_date - dates.max().normalize()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Sales", "sum")
).reset_index()

# 7. Save the customer-level RFM table
output_path = Path("rfm_customers.csv")
rfm.to_csv(output_path, index=False)

# 8. Print a summary
print("RFM ANALYSIS COMPLETED")
print("Customers:", len(rfm))
print("\nFirst 10 customers:")
print(rfm.head(10).to_string(index=False))
print("\nRFM summary:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2))
print("\nSaved to:", output_path.resolve())
