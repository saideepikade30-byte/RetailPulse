
from pathlib import Path
import logging

import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

print("=" * 60)
print("RETAILPULSE - PROPHET MULTI-PERIOD BACKTEST")
print("=" * 60)

# Load the cleaned dataset
file_path = Path(__file__).parent / "retail_cleaned.csv"

df = pd.read_csv(
    file_path,
    usecols=[
        "InvoiceDate",
        "Quantity",
        "Price",
        "Sales",
        "IsCancellation",
    ],
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

for column in ["Quantity", "Price", "Sales"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

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
    freq="D",
)

daily_sales = daily_sales.reindex(all_dates, fill_value=0)

# Create non-overlapping 30-day test periods.
# Each method is evaluated on the same dates.
test_start = last_complete - pd.Timedelta(days=29)
test_starts = []

while test_start - pd.Timedelta(days=365) >= first_date:
    test_starts.append(test_start)
    test_start -= pd.Timedelta(days=30)

test_starts.sort()

all_actual = []
prophet_predictions = []
baseline_predictions = []
details = []

for number, start in enumerate(test_starts, start=1):
    end = start + pd.Timedelta(days=29)
    train_end = start - pd.Timedelta(days=1)

    training = daily_sales.loc[:train_end]
    actual = daily_sales.loc[start:end]

    if len(training) < 60 or len(actual) != 30:
        continue

    print(
        f"\nTest period {number}/{len(test_starts)}:",
        start.date(), "to", end.date(),
        "| Training days:", len(training),
    )

    # Method 1: weekday average using the previous 56 days
    recent_history = training.tail(56)
    weekday_average = recent_history.groupby(
        recent_history.index.dayofweek
    ).mean()

    baseline = pd.Series(
        [
            float(weekday_average.get(day.dayofweek, 0.0))
            for day in actual.index
        ],
        index=actual.index,
    )

    # Method 2: train Prophet using training data only
    training_df = pd.DataFrame({
        "ds": training.index,
        "y": training.to_numpy(dtype=float),
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )

    model.fit(training_df)

    forecast = model.predict(
        pd.DataFrame({"ds": actual.index})
    )

    prediction = pd.Series(
        forecast["yhat"].clip(lower=0).to_numpy(),
        index=actual.index,
    )

    all_actual.extend(actual.tolist())
    baseline_predictions.extend(baseline.tolist())
    prophet_predictions.extend(prediction.tolist())

    for day in actual.index:
        details.append({
            "Date": day,
            "Actual Sales": float(actual.loc[day]),
            "56-Day Baseline": float(baseline.loc[day]),
            "Prophet Forecast": float(prediction.loc[day]),
        })

# Calculate metrics on the combined, non-overlapping test days.
if not all_actual:
    print("\nNo valid test periods were available.")
else:
    actual = pd.Series(all_actual, dtype=float)
    baseline = pd.Series(baseline_predictions, dtype=float)
    prophet_pred = pd.Series(prophet_predictions, dtype=float)

    nonzero_actual = actual > 0

    def calculate_metrics(prediction):
        absolute_error = (prediction - actual).abs()

        mae = absolute_error.mean()
        wape = absolute_error.sum() / actual.sum() * 100

        mape = (
            (
                absolute_error[nonzero_actual]
                / actual[nonzero_actual]
            ).mean() * 100
            if nonzero_actual.any()
            else float("nan")
        )

        return mae, wape, mape

    baseline_mae, baseline_wape, baseline_mape = (
        calculate_metrics(baseline)
    )
    prophet_mae, prophet_wape, prophet_mape = (
        calculate_metrics(prophet_pred)
    )

    print("\n" + "=" * 60)
    print("OVERALL COMPARISON")
    print("=" * 60)
    print("Test periods:", len(all_actual) // 30)
    print("Test days:", len(actual))

    print("\n56-Day Weekday Baseline")
    print("MAE:", f"£{baseline_mae:,.2f}")
    print("WAPE:", f"{baseline_wape:.2f}%")
    print("MAPE on nonzero-sales days:", f"{baseline_mape:.2f}%")

    print("\nProphet")
    print("MAE:", f"£{prophet_mae:,.2f}")
    print("WAPE:", f"{prophet_wape:.2f}%")
    print("MAPE on nonzero-sales days:", f"{prophet_mape:.2f}%")

    output_path = (
        Path(__file__).parent / "prophet_multi_period_results.csv"
    )
    pd.DataFrame(details).to_csv(output_path, index=False)

    print("\nDaily results saved to:", output_path.name)
    print("Lower errors are better.")
    print("These are historical backtest results, not future guarantees.")
