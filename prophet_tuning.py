
from pathlib import Path
import logging
import warnings

import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Yearly seasonality is enabled")

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "retail_cleaned.csv.gz"
OUTPUT_FILE = BASE_DIR / "prophet_tuning_results.csv"

print("=" * 65)
print("RETAILPULSE - PROPHET CONFIGURATION TUNING")
print("=" * 65)

df = pd.read_csv(
    DATA_FILE,
    usecols=[
        "InvoiceDate", "Quantity", "Price",
        "Sales", "IsCancellation"
    ],
     compression="gzip",
    low_memory=False,
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

for col in ["Quantity", "Price", "Sales"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

cancelled = (
    df["IsCancellation"].astype(str).str.strip().str.lower()
    .eq("true")
)

sales = df[
    (~cancelled) & (df["Quantity"] > 0) & (df["Price"] > 0)
].dropna(subset=["InvoiceDate", "Sales"]).copy()

sales["SaleDate"] = sales["InvoiceDate"].dt.normalize()
daily = sales.groupby("SaleDate")["Sales"].sum()

first_date = df["InvoiceDate"].min().normalize()
last_complete = df["InvoiceDate"].max().normalize() - pd.Timedelta(days=1)

dates = pd.date_range(first_date, last_complete, freq="D")
daily = daily.reindex(dates, fill_value=0)

# Use exactly the same 12 test periods as prophet_backtest.py.
test_start = last_complete - pd.Timedelta(days=29)
test_starts = []

while test_start - pd.Timedelta(days=365) >= first_date:
    test_starts.append(test_start)
    test_start -= pd.Timedelta(days=30)

test_starts.sort()

# Compare the current configuration with alternatives.
configs = [
    {
        "name": "Current Prophet",
        "yearly": True,
        "mode": "additive",
        "changepoint": 0.05,
        "seasonality_prior": 10.0,
    },
    {
        "name": "No Yearly Seasonality",
        "yearly": False,
        "mode": "additive",
        "changepoint": 0.05,
        "seasonality_prior": 10.0,
    },
    {
        "name": "Multiplicative Seasonality",
        "yearly": True,
        "mode": "multiplicative",
        "changepoint": 0.05,
        "seasonality_prior": 10.0,
    },
    {
        "name": "Lower Changepoint Flexibility",
        "yearly": True,
        "mode": "additive",
        "changepoint": 0.01,
        "seasonality_prior": 10.0,
    },
    {
        "name": "Higher Changepoint Flexibility",
        "yearly": True,
        "mode": "additive",
        "changepoint": 0.20,
        "seasonality_prior": 10.0,
    },
    {
        "name": "Stronger Seasonality Regularization",
        "yearly": True,
        "mode": "additive",
        "changepoint": 0.05,
        "seasonality_prior": 1.0,
    },
]

results = []

for config in configs:
    actual_all = []
    pred_all = []
    print(f"\nTesting: {config['name']}")

    for start in test_starts:
        end = start + pd.Timedelta(days=29)
        train_end = start - pd.Timedelta(days=1)

        train = daily.loc[:train_end]
        actual = daily.loc[start:end]

        if len(train) < 60 or len(actual) != 30:
            continue

        train_df = pd.DataFrame({
            "ds": train.index,
            "y": train.to_numpy(dtype=float),
        })

        model = Prophet(
            yearly_seasonality=config["yearly"],
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=config["mode"],
            changepoint_prior_scale=config["changepoint"],
            seasonality_prior_scale=config["seasonality_prior"],
        )

        model.fit(train_df)
        forecast = model.predict(pd.DataFrame({"ds": actual.index}))
        prediction = forecast["yhat"].clip(lower=0).to_numpy()

        actual_all.extend(actual.to_numpy(dtype=float))
        pred_all.extend(prediction)

    actual_s = pd.Series(actual_all, dtype=float)
    pred_s = pd.Series(pred_all, dtype=float)

    abs_error = (pred_s - actual_s).abs()
    wape = abs_error.sum() / actual_s.sum() * 100
    mae = abs_error.mean()

    nonzero = actual_s > 0
    mape = (
        (abs_error[nonzero] / actual_s[nonzero]).mean() * 100
        if nonzero.any() else float("nan")
    )

    results.append({
        "Method": config["name"],
        "Test Days": len(actual_s),
        "MAE (£)": round(mae, 2),
        "WAPE (%)": round(wape, 2),
        "MAPE (%)": round(mape, 2),
    })

    print(
        f"  WAPE: {wape:.2f}% | "
        f"MAE: £{mae:,.2f} | MAPE: {mape:.2f}%"
    )

result_df = pd.DataFrame(results).sort_values("WAPE (%)")
result_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("FINAL RANKING — LOWER ERRORS ARE BETTER")
print("=" * 65)
print(result_df.to_string(index=False))
print(f"\nSaved results to: {OUTPUT_FILE.name}")
