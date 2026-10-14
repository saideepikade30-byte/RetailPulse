
from pathlib import Path
import pandas as pd

print("=" * 55)
print("RETAILPULSE - DATA VALIDATION")
print("=" * 55)

file_path = Path(__file__).parent / "retail_cleaned.csv"

if not file_path.exists():
    print("ERROR: retail_cleaned.csv was not found.")
    raise SystemExit(1)

columns = [
    "Invoice",
    "Quantity",
    "Price",
    "Sales",
    "Customer ID",
    "InvoiceDate",
    "IsCancellation",
    "Description",
    "Country",
]


df = pd.read_csv(file_path, usecols=columns)

# Recalculate sales independently
expected_sales = df["Quantity"] * df["Price"]
difference = (df["Sales"] - expected_sales).abs()
mismatches = int((difference > 0.01).sum())

# Match the dashboard's sales criteria
is_cancellation = (
    df["IsCancellation"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

sales_rows = df[
    (~is_cancellation)
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
].copy()

print("\nDATA CHECK")
print("Total cleaned rows:", f"{len(df):,}")
print("Sales calculation mismatches:", f"{mismatches:,}")

print("\nDASHBOARD METRICS")
print("Gross Sales (£):", f"{sales_rows['Sales'].sum():,.2f}")
print("Total Orders:", f"{sales_rows['Invoice'].nunique():,}")
print("Total Customers:", f"{sales_rows['Customer ID'].nunique():,}")
print("Units Sold:", f"{sales_rows['Quantity'].sum():,.0f}")

print("\nDATE RANGE")
dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
print("First transaction:", dates.min())
print("Last transaction:", dates.max())

print("\nVALIDATION FINISHED")

# -----------------------------------
# Validate top products and countries
# -----------------------------------
print("\n" + "=" * 55)
print("TOP 10 PRODUCTS BY GROSS SALES")
print("=" * 55)

top_products = (
    sales_rows.groupby("Description", as_index=False)["Sales"]
    .sum()
    .sort_values("Sales", ascending=False)
    .head(10)
)

print(top_products.to_string(index=False, formatters={
    "Sales": lambda value: f"£{value:,.2f}"
}))

print("\n" + "=" * 55)
print("TOP 10 COUNTRIES BY GROSS SALES")
print("=" * 55)

top_countries = (
    sales_rows.groupby("Country", as_index=False)["Sales"]
    .sum()
    .sort_values("Sales", ascending=False)
    .head(10)
)

print(top_countries.to_string(index=False, formatters={
    "Sales": lambda value: f"£{value:,.2f}"
}))

# -----------------------------------
# Baseline forecast accuracy test
# -----------------------------------

# -----------------------------------
# Compare two baseline forecasts
# -----------------------------------
print("\n" + "=" * 60)
print("BASELINE FORECAST COMPARISON")
print("=" * 60)

sales_rows["InvoiceDate"] = pd.to_datetime(
    sales_rows["InvoiceDate"], errors="coerce"
)

monthly_sales = (
    sales_rows
    .dropna(subset=["InvoiceDate", "Sales"])
    .set_index("InvoiceDate")["Sales"]
    .resample("MS")
    .sum()
    .sort_index()
)

if len(monthly_sales) >= 15:
    # Exclude the latest month because it may be incomplete
    latest_month = monthly_sales.index.max()
    complete_months = monthly_sales[
        monthly_sales.index < latest_month
    ]

    test_months = complete_months.tail(3).index
    results = []

    for target_month in test_months:
        actual = float(complete_months.loc[target_month])

        # Method 1: average of the previous three months
        previous_three = complete_months[
            complete_months.index < target_month
        ].tail(3)

        moving_average = (
            float(previous_three.mean())
            if len(previous_three) == 3
            else float("nan")
        )

        # Method 2: sales from the same month one year earlier
        previous_year_month = (
            target_month - pd.DateOffset(years=1)
        )

        seasonal_forecast = (
            float(complete_months.loc[previous_year_month])
            if previous_year_month in complete_months.index
            else float("nan")
        )

        results.append({
            "Month": target_month.strftime("%Y-%m"),
            "Actual": actual,
            "Moving Average": moving_average,
            "Seasonal Forecast": seasonal_forecast,
        })

    result_df = pd.DataFrame(results)

    print("\nMonthly forecast comparison:")
    print(result_df.to_string(
        index=False,
        formatters={
            "Actual": lambda x: f"£{x:,.2f}",
            "Moving Average": lambda x: f"£{x:,.2f}",
            "Seasonal Forecast": lambda x: f"£{x:,.2f}",
        }
    ))

    print("\nACCURACY SUMMARY")

    for method in ["Moving Average", "Seasonal Forecast"]:
        valid = result_df.dropna(subset=[method, "Actual"]).copy()

        valid = valid[valid["Actual"] != 0]

        if not valid.empty:
            errors = valid[method] - valid["Actual"]
            mae = errors.abs().mean()
            mape = (
                errors.abs() / valid["Actual"].abs()
            ).mean() * 100

            print(f"\n{method}")
            print("MAE:", f"£{mae:,.2f}")
            print("MAPE:", f"{mape:.2f}%")

    print(
        "\nThese are baseline comparisons using only three test months. "
        "They do not establish AI model performance."
    )

else:
    print("Not enough monthly data for this comparison.")

# -----------------------------------
# 30-Day Forecast Backtest
# -----------------------------------
print("\n" + "=" * 55)
print("30-DAY FORECAST BACKTEST")
print("=" * 55)

# Use the same qualifying sale rows as the dashboard
daily_source = sales_rows.copy()
daily_source["SaleDate"] = (
    pd.to_datetime(daily_source["InvoiceDate"])
    .dt.normalize()
)

daily_sales = daily_source.groupby("SaleDate")["Sales"].sum()

# Exclude the potentially incomplete final day
last_observed = pd.to_datetime(df["InvoiceDate"]).max().normalize()
last_complete = last_observed - pd.Timedelta(days=1)

test_start = last_complete - pd.Timedelta(days=29)
train_end = test_start - pd.Timedelta(days=1)
train_start = train_end - pd.Timedelta(days=27)

# Include dates with no recorded sales as zero
all_dates = pd.date_range(
    start=pd.to_datetime(df["InvoiceDate"]).min().normalize(),
    end=last_complete,
    freq="D"
)
daily_sales = daily_sales.reindex(all_dates, fill_value=0)

training = daily_sales.loc[train_start:train_end]
actual = daily_sales.loc[test_start:last_complete]

print("Training period:", train_start.date(), "to", train_end.date())
print("Testing period:", test_start.date(), "to", last_complete.date())
print("Training days:", len(training))
print("Testing days:", len(actual))

if len(training) == 28 and len(actual) == 30:
    # Learn average sales for each weekday from training data only
    weekday_average = training.groupby(
        training.index.dayofweek
    ).mean()

    predictions = pd.Series(
        [
            float(weekday_average.get(day.dayofweek, 0.0))
            for day in actual.index
        ],
        index=actual.index
    )

    errors = predictions - actual
    absolute_errors = errors.abs()

    mae = absolute_errors.mean()
    total_actual = actual.sum()

    wape = (
        absolute_errors.sum() / total_actual * 100
        if total_actual != 0 else float("nan")
    )

    # MAPE is calculated only on days with positive actual sales,
    # because percentage error is undefined when actual sales are zero.
    nonzero_actual = actual > 0

    mape = (
        (absolute_errors[nonzero_actual] / actual[nonzero_actual]).mean() * 100
        if nonzero_actual.any() else float("nan")
    )

    print("\nActual sales total:", f"£{total_actual:,.2f}")
    print("Forecast sales total:", f"£{predictions.sum():,.2f}")
    print("Mean Absolute Error (MAE):", f"£{mae:,.2f}")
    print("Weighted Absolute Percentage Error (WAPE):", f"{wape:.2f}%")
    print("MAPE on nonzero-sales days:", f"{mape:.2f}%")
    print("Days with zero actual sales:", int((actual == 0).sum()))

    print("\nSample daily comparison:")
    comparison = pd.DataFrame({
        "Actual Sales": actual,
        "Forecast Sales": predictions
    })
    print(comparison.head(10).to_string(
        formatters={
            "Actual Sales": lambda x: f"£{x:,.2f}",
            "Forecast Sales": lambda x: f"£{x:,.2f}"
        }
    ))

    print("\nBacktest complete. This evaluates a baseline, not an AI model.")
else:
    print("Backtest could not run: unexpected number of training or test days.")

# -----------------------------------
# Compare 28, 56, and 84-day baselines
# -----------------------------------
print("\n" + "=" * 55)
print("WEEKDAY BASELINE COMPARISON")
print("=" * 55)

test_actual = daily_sales.loc[test_start:last_complete]

for window in [28, 56, 84]:
    train_end_compare = test_start - pd.Timedelta(days=1)
    train_start_compare = (
        test_start - pd.Timedelta(days=window)
    )

    window_training = daily_sales.loc[
        train_start_compare:train_end_compare
    ]

    if len(window_training) != window:
        print(f"Skipping {window}-day method: insufficient history.")
        continue

    weekday_means = window_training.groupby(
        window_training.index.dayofweek
    ).mean()

    window_predictions = pd.Series(
        [
            float(weekday_means.get(day.dayofweek, 0.0))
            for day in test_actual.index
        ],
        index=test_actual.index
    )

    abs_errors = (window_predictions - test_actual).abs()

    mae = abs_errors.mean()
    actual_total = test_actual.sum()

    wape = (
        abs_errors.sum() / actual_total * 100
        if actual_total != 0 else float("nan")
    )

    nonzero_days = test_actual > 0

    mape = (
        (
            abs_errors[nonzero_days]
            / test_actual[nonzero_days]
        ).mean() * 100
        if nonzero_days.any() else float("nan")
    )

    print(f"\nTraining window: {window} days")
    print("Forecast total:", f"£{window_predictions.sum():,.2f}")
    print("MAE:", f"£{mae:,.2f}")
    print("WAPE:", f"{wape:.2f}%")
    print("MAPE on nonzero-sales days:", f"{mape:.2f}%")

print("\nComparison finished. All methods use the same test dates.")


