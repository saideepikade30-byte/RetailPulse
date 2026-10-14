
import pandas as pd
from pathlib import Path

print("RETAILPULSE - DATA CLEANING")

file_path = Path("online_retail_ii.xlsx")

if not file_path.exists():
    print("ERROR: online_retail_ii.xlsx not found.")
else:
    sheets = pd.read_excel(file_path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)

    print("Original shape:", df.shape)

    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )
    df["Price"] = pd.to_numeric(
        df["Price"], errors="coerce"
    )
    df["Customer ID"] = pd.to_numeric(
        df["Customer ID"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Invoice", "StockCode", "InvoiceDate",
            "Quantity", "Price"
        ]
    ).copy()

    # Extra columns
    df["Sales"] = df["Quantity"] * df["Price"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["MonthYear"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayName"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    df["IsCancellation"] = (
        df["Invoice"].astype(str).str.startswith("C")
    )
    df["OrderType"] = df["IsCancellation"].map({
        True: "Cancellation",
        False: "Sale"
    })

    output_path = Path("retail_cleaned.csv")
    df.to_csv(output_path, index=False)

    print("Cleaned shape:", df.shape)
    print("Extra columns added successfully.")
    print("Cancellation rows:", int(df["IsCancellation"].sum()))
    print("Missing Customer IDs:", int(df["Customer ID"].isna().sum()))
    print("Date range:", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())
    print("Saved to:", output_path.resolve())

