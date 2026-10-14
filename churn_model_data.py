
import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).parent / "retail_cleaned.csv"
OUTPUT_FILE = Path(__file__).parent / "churn_model_data.csv"

# Load required columns
df = pd.read_csv(
    DATA_FILE,
    usecols=[
        "Invoice",
        "InvoiceDate",
        "Customer ID",
        "Quantity",
        "Price",
    ],
)

# Clean data types
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Keep valid purchases only
df = df.dropna(subset=["Customer ID", "InvoiceDate"])
df = df[
    (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (~df["Invoice"].astype(str).str.startswith("C"))
].copy()

df["Sales"] = df["Quantity"] * df["Price"]

# Use the final 90 days as the prediction window
end_date = df["InvoiceDate"].max().normalize()
cutoff_date = end_date - pd.Timedelta(days=90)

history = df[df["InvoiceDate"].dt.normalize() <= cutoff_date].copy()
future = df[
    (df["InvoiceDate"].dt.normalize() > cutoff_date)
    & (df["InvoiceDate"].dt.normalize() <= end_date)
].copy()

# Build customer features using history only
customers = history.groupby("Customer ID").agg(
    Last_Purchase=("InvoiceDate", "max"),
    Frequency=("Invoice", "nunique"),
    Monetary=("Sales", "sum"),
    Total_Quantity=("Quantity", "sum"),
).reset_index()

customers["Recency"] = (
    cutoff_date - customers["Last_Purchase"].dt.normalize()
).dt.days

# Label: 1 = no purchase in the next 90 days; 0 = purchased
future_customers = set(future["Customer ID"].unique())
customers["Churn"] = (
    ~customers["Customer ID"].isin(future_customers)
).astype(int)

customers = customers.drop(columns=["Last_Purchase"])
customers.to_csv(OUTPUT_FILE, index=False)

print("\nCHURN MODEL DATA CREATED")
print("Observation cutoff:", cutoff_date.date())
print("Prediction window:", (cutoff_date + pd.Timedelta(days=1)).date(),
      "to", end_date.date())
print("Customers:", len(customers))
print("\nChurn labels (0 = purchased, 1 = no purchase):")
print(customers["Churn"].value_counts().sort_index())
print("\nChurn percentage:", round(customers["Churn"].mean() * 100, 2), "%")
print("Saved:", OUTPUT_FILE)
