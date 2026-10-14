
from pathlib import Path
import pandas as pd

file_path = Path(__file__).parent / "prophet_multi_period_results.csv"

if not file_path.exists():
    raise FileNotFoundError(
        "prophet_multi_period_results.csv was not found. "
        "Run prophet_backtest.py first."
    )

df = pd.read_csv(file_path, parse_dates=["Date"])
df = df.sort_values("Date").tail(30).copy()

actual = df["Actual Sales"]

print("=" * 55)
print("MOST RECENT 30-DAY FORECAST COMPARISON")
print("=" * 55)
print("Period:", df["Date"].min().date(), "to", df["Date"].max().date())

for method in ["56-Day Baseline", "Prophet Forecast"]:
    forecast = df[method]
    errors = (forecast - actual).abs()

    mae = errors.mean()
    wape = errors.sum() / actual.sum() * 100

    nonzero = actual > 0
    mape = (
        (errors[nonzero] / actual[nonzero]).mean() * 100
        if nonzero.any() else float("nan")
    )

    print(f"\n{method}")
    print("Forecast total:", f"£{forecast.sum():,.2f}")
    print("MAE:", f"£{mae:,.2f}")
    print("WAPE:", f"{wape:.2f}%")
    print("MAPE on nonzero-sales days:", f"{mape:.2f}%")
