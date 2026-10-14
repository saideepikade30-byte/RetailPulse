
import pandas as pd
from pathlib import Path


DATA_FILE = Path(__file__).parent / "retail_cleaned.csv"
OUTPUT_FILE = Path(__file__).parent / "churn_customer_data.csv"

df = pd.read_csv(
    DATA_FILE,
    usecols=[
        "Invoice",
        "InvoiceDate",
        "Customer ID",
        "Quantity",
        "Price",
        "IsCancellation",
    ],
)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Keep valid customer purchase records
df = df.dropna(subset=["Customer ID", "InvoiceDate"])
df = df[
    (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (~df["Invoice"].astype(str).str.startswith("C"))
].copy()

# Create customer-level features
df["Sales"] = df["Quantity"] * df["Price"]
reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

customers = df.groupby("Customer ID").agg(
    Last_Purchase=("InvoiceDate", "max"),
    Frequency=("Invoice", "nunique"),
    Monetary=("Sales", "sum"),
    Total_Quantity=("Quantity", "sum"),
).reset_index()

customers["Recency"] = (
    reference_date - customers["Last_Purchase"].dt.normalize()
).dt.days

# Working label: no purchase in the final 90 days of the dataset
customers["Churn"] = (customers["Recency"] > 90).astype(int)

customers = customers.drop(columns=["Last_Purchase"])
customers.to_csv(OUTPUT_FILE, index=False)

print("CUSTOMER CHURN DATA CREATED")
print("Customers:", len(customers))
print("\nChurn labels (0 = active, 1 = potentially churned):")
print(customers["Churn"].value_counts().sort_index())
print("\nChurn percentage:", round(customers["Churn"].mean() * 100, 2), "%")
print("Saved:", OUTPUT_FILE)
