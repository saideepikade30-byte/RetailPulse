
from pathlib import Path
import pandas as pd

print("=" * 60)
print("RETAILPULSE - FORECAST METHOD COMPARISON")
print("=" * 60)

file_path = Path(__file__).parent / "retail_cleaned.csv"

if not file_path.exists():
    raise FileNotFoundError("retail_cleaned.csv was not found.")

columns = [
    "InvoiceDate",
    "Quantity",
    "Price",
    "Sales",
    "IsCancellation",
]

df = pd.read_csv(file_path, usecols=columns)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

for col in ["Quantity", "Price", "Sales"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

is_cancellation = (
    df["IsCancellation"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

sales = df[
    (~is_cancellation)
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
].dropna(subset=["InvoiceDate", "Sales"]).copy()

sales["SaleDate"] = sales["InvoiceDate"].dt.normalize()

daily_sales = sales.groupby("SaleDate")["Sales"].sum()

last_observed = df["InvoiceDate"].max().normalize()
last_complete = last_observed - pd.Timedelta(days=1)
first_date = df["InvoiceDate"].min().normalize()

all_dates = pd.date_range(
    start=first_date,
    end=last_complete,
    freq="D"
)

daily_sales = daily_sales.reindex(all_dates, fill_value=0)

# Use common test periods with at least 84 training days available.
test_start = last_complete - pd.Timedelta(days=29)
test_starts = []

while test_start - pd.Timedelta(days=84) >= first_date:
    test_starts.append(test_start)
    test_start -= pd.Timedelta(days=30)

test_starts.sort()

methods = [
    "28-day weekday average",
    "56-day weekday average",
    "84-day weekday average",
    "Repeat last 7-day pattern",
]

records = []

for start in test_starts:
    end = start + pd.Timedelta(days=29)
    train_end = start - pd.Timedelta(days=1)

    actual = daily_sales.loc[start:end]

    if len(actual) != 30:
        continue

    predictions_by_method = {}

    # Three methods: average sales for each weekday.
    for window in [28, 56, 84]:
        train_start = start - pd.Timedelta(days=window)
        training = daily_sales.loc[train_start:train_end]

        if len(training) != window:
            continue

        weekday_means = training.groupby(
            training.index.dayofweek
        ).mean()

        predictions = pd.Series(
            [
                float(weekday_means.get(day.dayofweek, 0.0))
                for day in actual.index
            ],
            index=actual.index
        )

        predictions_by_method[
            f"{window}-day weekday average"
        ] = predictions

    # Repeat the last seven observed daily values cyclically.
    # No actual test-period values are used.
    last_seven = daily_sales.loc[
        train_end - pd.Timedelta(days=6):train_end
    ]

    if len(last_seven) == 7:
        repeated_values = [
            float(last_seven.iloc[i % 7])
            for i in range(30)
        ]

        predictions_by_method[
            "Repeat last 7-day pattern"
        ] = pd.Series(repeated_values, index=actual.index)

    for method, prediction in predictions_by_method.items():
        abs_error = (prediction - actual).abs()
        positive_actual = actual > 0

        records.append({
            "Test Start": start.strftime("%Y-%m-%d"),
            "Method": method,
            "Actual Total": actual.sum(),
            "Forecast Total": prediction.sum(),
            "Absolute Error Sum": abs_error.sum(),
            "Daily MAE": abs_error.mean(),
            "WAPE (%)": (
                abs_error.sum() / actual.sum() * 100
                if actual.sum() != 0 else float("nan")
            ),
            "MAPE (%)": (
                (
                    abs_error[positive_actual]
                    / actual[positive_actual]
                ).mean() * 100
                if positive_actual.any() else float("nan")
            ),
        })

if not records:
    print("No valid test periods were available.")
else:
    results = pd.DataFrame(records)

    summaries = []

    for method, group in results.groupby("Method", sort=False):
        total_actual = group["Actual Total"].sum()
        total_abs_error = group["Absolute Error Sum"].sum()

        summaries.append({
            "Method": method,
            "Test Periods": group["Test Start"].nunique(),
            "Test Days": group["Test Start"].nunique() * 30,
            "Overall WAPE (%)": (
                total_abs_error / total_actual * 100
                if total_actual else float("nan")
            ),
            "Average Period MAPE (%)": group["MAPE (%)"].mean(),
            "Average Daily MAE (£)": group["Daily MAE"].mean(),
        })

    summary = pd.DataFrame(summaries).sort_values(
        "Overall WAPE (%)"
    )

    print("\nOVERALL COMPARISON — LOWER ERRORS ARE BETTER")
    print(summary.to_string(
        index=False,
        formatters={
            "Overall WAPE (%)": lambda x: f"{x:.2f}%",
            "Average Period MAPE (%)": lambda x: f"{x:.2f}%",
            "Average Daily MAE (£)": lambda x: f"£{x:,.2f}",
        }
    ))

    output_path = (
        Path(__file__).parent / "forecast_method_comparison.csv"
    )
    summary.to_csv(output_path, index=False)

    details_path = (
        Path(__file__).parent / "forecast_method_details.csv"
    )
    results.to_csv(details_path, index=False)

    print("\nComparison saved to:", output_path.name)
    print("Period details saved to:", details_path.name)
    print("\nAll methods use the same test periods.")
    print("These are baselines, not trained AI models.")
