
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------
# 1. Dashboard settings
# -----------------------------------
st.set_page_config(
    page_title="RetailPulse Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("RetailPulse")
st.subheader("AI-Powered Customer Analytics & Demand Forecasting")
st.caption("Explore sales trends, products, customers, and order patterns.")

DATA_FILE = Path(__file__).parent / "retail_cleaned.csv.gz"

REQUIRED_COLUMNS = [
    "Invoice",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "Sales",
    "MonthYear",
    "DayName",
    "Hour",
    "IsCancellation",
]


# -----------------------------------
# 2. Load the cleaned dataset
# -----------------------------------
@st.cache_data(show_spinner="Loading the retail dataset...")
def load_data(file_path, modified_time):
    data = pd.read_csv(
        file_path,
        usecols=REQUIRED_COLUMNS,
        low_memory=False,
        compression="gzip"
    )

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )
    data["Quantity"] = pd.to_numeric(
        data["Quantity"], errors="coerce"
    )
    data["Price"] = pd.to_numeric(
        data["Price"], errors="coerce"
    )
    data["Sales"] = pd.to_numeric(
        data["Sales"], errors="coerce"
    )
    data["Hour"] = pd.to_numeric(
        data["Hour"], errors="coerce"
    )
    data["Customer ID"] = pd.to_numeric(
        data["Customer ID"], errors="coerce"
    )

    data["Country"] = data["Country"].fillna("Unknown").astype(str)
    data["Description"] = (
        data["Description"].fillna("Unknown product").astype(str)
    )
    data["MonthYear"] = data["MonthYear"].fillna("Unknown").astype(str)
    data["DayName"] = data["DayName"].fillna("Unknown").astype(str)

    data["IsCancellation"] = (
        data["IsCancellation"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    data = data.dropna(subset=["InvoiceDate", "Quantity", "Price", "Sales"])
    return data


# -----------------------------------
# 3. Check the data file
# -----------------------------------
if not DATA_FILE.exists():
    st.error("retail_cleaned.csv was not found.")
    st.write("Please keep app.py and retail_cleaned.csv in the same folder.")
    st.stop()

try:
    df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as error:
    st.error("The dataset could not be loaded.")
    st.code(str(error))
    st.stop()

if df.empty:
    st.warning("The dataset has no usable rows.")
    st.stop()


# -----------------------------------
# 4. Sidebar filters
# -----------------------------------
st.sidebar.header("Dashboard Filters")

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

selected_dates = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

country_options = ["All countries"] + sorted(df["Country"].unique().tolist())

selected_country = st.sidebar.selectbox(
    "Select country",
    country_options
)

if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
    st.info("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = selected_dates

if start_date > end_date:
    st.warning("The start date must be before the end date.")
    st.stop()

filtered = df[
    (df["InvoiceDate"] >= pd.Timestamp(start_date))
    & (df["InvoiceDate"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
].copy()

if selected_country != "All countries":
    filtered = filtered[filtered["Country"] == selected_country].copy()

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()


# -----------------------------------
# 5. Define sale and cancellation rows
# -----------------------------------
# These KPIs include positive-quantity, positive-price sale lines.
# Returns and cancellation invoices are excluded from gross sales.
sales_rows = filtered[
    (~filtered["IsCancellation"])
    & (filtered["Quantity"] > 0)
    & (filtered["Price"] > 0)
].copy()

total_sales = sales_rows["Sales"].sum()
total_orders = sales_rows["Invoice"].nunique()
total_customers = sales_rows["Customer ID"].nunique()
total_units = sales_rows["Quantity"].sum()
cancellation_lines = int(filtered["IsCancellation"].sum())


# -----------------------------------
# 6. Key performance indicators
# -----------------------------------
st.markdown("### Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Gross Sales", f"£{total_sales:,.2f}")
col2.metric("Orders", f"{total_orders:,}")
col3.metric("Unique Customers", f"{total_customers:,}")
col4.metric("Units Sold", f"{total_units:,.0f}")

st.caption(
    "Gross Sales excludes cancellation invoices, negative quantities, "
    "and non-positive prices. Returns are not deducted from this figure. "
    "Currency is shown in GBP (£), as the dataset's Price field is in pounds."
)

st.metric("Cancellation Lines", f"{cancellation_lines:,}")

st.divider()


# -----------------------------------
# 7. Monthly sales trend
# -----------------------------------
st.markdown("### Monthly Sales Trend")

if not sales_rows.empty:
    monthly_sales = (
        sales_rows.groupby("MonthYear", as_index=False)["Sales"]
        .sum()
        .sort_values("MonthYear")
    )

    fig_monthly = px.line(
        monthly_sales,
        x="MonthYear",
        y="Sales",
        markers=True,
        title="Gross Sales by Month",
        labels={
            "MonthYear": "Month",
            "Sales": "Gross Sales (£)"
        }
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
else:
    st.info("No qualifying sale lines are available for the selected filters.")


# -----------------------------------
# 8. Top products
# -----------------------------------
st.markdown("### Top 10 Products by Gross Sales")

if not sales_rows.empty:
    top_products = (
        sales_rows.groupby("Description", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
        .sort_values("Sales")
    )

    fig_products = px.bar(
        top_products,
        x="Sales",
        y="Description",
        orientation="h",
        title="Top Products",
        labels={
            "Sales": "Gross Sales (£)",
            "Description": "Product"
        }
    )
    st.plotly_chart(fig_products, use_container_width=True)
else:
    st.info("No product sales are available for the selected filters.")


# -----------------------------------
# 9. Sales by country
# -----------------------------------
st.markdown("### Country-wise Sales")

if not sales_rows.empty:
    country_sales = (
        sales_rows.groupby("Country", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
        .sort_values("Sales")
    )

    fig_country = px.bar(
        country_sales,
        x="Sales",
        y="Country",
        orientation="h",
        title="Top 10 Countries by Gross Sales",
        labels={
            "Sales": "Gross Sales (£)",
            "Country": "Country"
        }
    )
    st.plotly_chart(fig_country, use_container_width=True)
else:
    st.info("No country sales are available for the selected filters.")


# -----------------------------------
# 10. Shopping patterns
# -----------------------------------
st.markdown("### Shopping Patterns")

left_col, right_col = st.columns(2)

with left_col:
    hourly_data = sales_rows.dropna(subset=["Hour"])

    if not hourly_data.empty:
        hourly_sales = (
            hourly_data.groupby("Hour", as_index=False)["Sales"]
            .sum()
            .sort_values("Hour")
        )

        fig_hour = px.bar(
            hourly_sales,
            x="Hour",
            y="Sales",
            title="Gross Sales by Hour",
            labels={"Hour": "Hour of Day", "Sales": "Gross Sales (£)"}
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("No hourly sales data is available.")

with right_col:
    if not sales_rows.empty:
        weekday_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]

        weekday_sales = (
            sales_rows.groupby("DayName", as_index=False)["Sales"]
            .sum()
        )
        weekday_sales["DayName"] = pd.Categorical(
            weekday_sales["DayName"],
            categories=weekday_order,
            ordered=True
        )
        weekday_sales = weekday_sales.sort_values("DayName")

        fig_weekday = px.bar(
            weekday_sales,
            x="DayName",
            y="Sales",
            title="Gross Sales by Weekday",
            labels={"DayName": "Weekday", "Sales": "Gross Sales (£)"}
        )
        st.plotly_chart(fig_weekday, use_container_width=True)
    else:
        st.info("No weekday sales data is available.")


# -----------------------------------
# 11. Preview the filtered data
# -----------------------------------
with st.expander("View filtered transaction data"):
    st.write(f"Rows matching filters: {len(filtered):,}")
    st.dataframe(filtered.head(100), use_container_width=True)
    # -----------------------------------
# 12. Demand Forecasting
# ---------------------------------------

# -----------------------------------
# 12. Demand Forecasting
# -----------------------------------
st.divider()
st.markdown("### Demand Forecasting")

st.write(
    "This seasonal baseline forecasts each future month's sales "
    "using the sales from the same month one year earlier."
)

st.caption(
    "Forecasts use all available historical data for the selected "
    "country, regardless of the date range selected for other charts."
)

# Start with the full cleaned dataset and apply the sales criteria
forecast_source = df[
    (~df["IsCancellation"])
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
].copy()

# Keep the selected country, if one is selected
if selected_country != "All countries":
    forecast_source = forecast_source[
        forecast_source["Country"] == selected_country
    ].copy()

if forecast_source.empty:
    st.info("No qualifying historical sales are available to forecast.")

else:
    # The dataset's last observed month is incomplete:
    # its final transaction is on 9 December 2011.
    last_transaction = df["InvoiceDate"].max()
    last_complete_period = last_transaction.to_period("M") - 1

    history_start = (
        df["InvoiceDate"].min().to_period("M").to_timestamp()
    )
    last_complete_month = last_complete_period.to_timestamp()

    all_months = pd.date_range(
        start=history_start,
        end=last_complete_month,
        freq="MS"
    )

    # Aggregate qualifying sales by month
    monthly_history = (
        forecast_source
        .set_index("InvoiceDate")["Sales"]
        .resample("MS")
        .sum()
        .reindex(all_months, fill_value=0)
    )

    if len(monthly_history) < 12:
        st.info("At least 12 months of historical data are needed.")
    else:
        # Forecast the next 12 months using the same month last year
        forecast_dates = pd.date_range(
            start=(last_complete_period + 1).to_timestamp(),
            periods=12,
            freq="MS"
        )

        forecast_values = []

        for target_month in forecast_dates:
            previous_year_month = (
                target_month - pd.DateOffset(years=1)
            )

            if previous_year_month in monthly_history.index:
                forecast_values.append(
                    float(monthly_history.loc[previous_year_month])
                )
            else:
                forecast_values.append(0.0)

        historical_plot = (
            monthly_history.rename("Sales")
            .rename_axis("Month")
            .reset_index()
        )

        forecast_plot = pd.DataFrame({
            "Month": forecast_dates,
            "Forecast Sales": forecast_values
        })

        # Display historical sales
        fig_forecast = px.line(
            historical_plot,
            x="Month",
            y="Sales",
            markers=True,
            title="Historical Sales and 12-Month Seasonal Forecast",
            labels={
                "Month": "Month",
                "Sales": "Historical Gross Sales (£)"
            }
        )

        # Add the future forecast
        fig_forecast.add_scatter(
            x=forecast_plot["Month"],
            y=forecast_plot["Forecast Sales"],
            mode="lines+markers",
            name="Seasonal Baseline Forecast",
            line=dict(dash="dash")
        )

        st.plotly_chart(fig_forecast, width="stretch")

        st.metric(
            "Next-Month Baseline Forecast",
            f"£{forecast_values[0]:,.2f}"
        )
# -----------------------------------
# 13. Daily Demand Forecast - Next 30 Days
# -----------------------------------
st.divider()
st.markdown("### 30-Day Demand Forecast")

st.write(
    "This baseline estimates daily gross sales for the next 30 days "
    "using average sales for each weekday over the previous 56 days."
)

if forecast_source.empty:
    st.info("No qualifying sales are available for forecasting.")

else:
    # The last observed date is 9 December 2011.
    # Exclude this potentially incomplete day from training.
    last_observed_date = df["InvoiceDate"].max().normalize()
    last_complete_date = (
        last_observed_date - pd.Timedelta(days=1)
    )

    first_date = df["InvoiceDate"].min().normalize()

    # Build daily sales history through the last complete day
    daily_index = pd.date_range(
        start=first_date,
        end=last_complete_date,
        freq="D"
    )

    daily_source = forecast_source.copy()
    daily_source["SaleDate"] = (
        daily_source["InvoiceDate"].dt.normalize()
    )

    daily_sales = (
        daily_source.groupby("SaleDate")["Sales"]
        .sum()
        .reindex(daily_index, fill_value=0)
    )

    if len(daily_sales) < 28:
        st.info("At least 28 days of history are required.")
    else:
        # Use the previous 56 complete days to learn
        # the average sales for each day of the week.
        training_data = daily_sales.tail(56)

        weekday_average = training_data.groupby(
            training_data.index.dayofweek
        ).mean()

        # Forecast 30 days after the latest observed date.
        forecast_dates = pd.date_range(
            start=last_observed_date + pd.Timedelta(days=1),
            periods=30,
            freq="D"
        )

        forecast_values = [
            float(weekday_average.get(day.dayofweek, 0.0))
            for day in forecast_dates
        ]

        # Show the last 90 days of history
        historical_daily = (
            daily_sales.tail(90)
            .rename("Actual Gross Sales")
            .rename_axis("Date")
            .reset_index()
        )

        fig_daily = px.line(
            historical_daily,
            x="Date",
            y="Actual Gross Sales",
            title="Daily Historical Sales and 30-Day Forecast",
            labels={
                "Date": "Date",
                "Actual Gross Sales": "Gross Sales (£)"
            }
        )

        fig_daily.add_scatter(
            x=forecast_dates,
            y=forecast_values,
            mode="lines+markers",
            name="30-Day Baseline Forecast",
            line=dict(dash="dash")
        )

        st.plotly_chart(fig_daily, width="stretch")

        col1, col2 = st.columns(2)

        col1.metric(
            "Forecast Sales — Next 30 Days",
            f"£{sum(forecast_values):,.2f}"
        )

        col2.metric(
            "Average Forecast per Day",
            f"£{sum(forecast_values) / 30:,.2f}"
        )

        forecast_table = pd.DataFrame({
            "Date": forecast_dates,
            "Forecast Sales (£)": forecast_values
        })

        with st.expander("View daily forecast values"):
            st.dataframe(
                forecast_table,
                hide_index=True,
                width="stretch"
            )
            
# -----------------------------------
# 14. Trained Prophet 30-Day Forecast
# -----------------------------------
st.divider()
st.markdown("### Prophet 30-Day Forecast")

st.write(
    "This model learns historical daily sales patterns and forecasts "
    "gross sales for the next 30 days."
)

st.caption(
    "The model is trained on historical data for the selected country. "
    "Its accuracy has not met the project's 12% MAPE target yet."
)

if forecast_source.empty:
    st.info("No qualifying sales are available for this forecast.")

else:
    from prophet import Prophet

    prophet_source = forecast_source.copy()

    prophet_source["SaleDate"] = (
        prophet_source["InvoiceDate"].dt.normalize()
    )

    # Exclude the latest, potentially incomplete day
    last_country_observed = prophet_source["SaleDate"].max()
    prophet_last_complete = (
        last_country_observed - pd.Timedelta(days=1)
    )

    prophet_first_date = prophet_source["SaleDate"].min()

    prophet_dates = pd.date_range(
        start=prophet_first_date,
        end=prophet_last_complete,
        freq="D"
    )

    prophet_daily_sales = (
        prophet_source.groupby("SaleDate")["Sales"]
        .sum()
        .reindex(prophet_dates, fill_value=0)
    )

    prophet_train = pd.DataFrame({
        "ds": prophet_daily_sales.index,
        "y": prophet_daily_sales.values
    })

    if len(prophet_train) < 60:
        st.info("At least 60 days of historical data are needed.")

    else:
        # Use yearly seasonality only when sufficient history exists.
        use_yearly_seasonality = len(prophet_train) >= 730

        if not use_yearly_seasonality:
            st.caption(
                "Yearly seasonality is disabled because this selection "
                "does not contain at least 730 complete daily observations."
            )

        with st.spinner("Training Prophet and generating forecasts..."):
            prophet_model = Prophet(
                yearly_seasonality=use_yearly_seasonality,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="additive",
                changepoint_prior_scale=0.05,
            )

            prophet_model.fit(prophet_train)

            prophet_future_dates = prophet_model.make_future_dataframe(
                periods=30,
                freq="D",
                include_history=True
            )

            prophet_prediction = prophet_model.predict(
                prophet_future_dates
            )

        # Future values only, starting after the last complete day
        future_only = prophet_prediction[
            prophet_prediction["ds"] > prophet_last_complete
        ].head(30).copy()

        future_only["Forecast Sales (£)"] = (
            future_only["yhat"].clip(lower=0)
        )

        future_only = future_only.rename(
            columns={"ds": "Date"}
        )

        prophet_total = future_only["Forecast Sales (£)"].sum()

        col1, col2 = st.columns(2)

        col1.metric(
            "Prophet Forecast — Next 30 Days",
            f"£{prophet_total:,.2f}"
        )

        col2.metric(
            "Average Forecast per Day",
            f"£{prophet_total / 30:,.2f}"
        )

        # Compare recent actual sales with the model prediction
        history_view = prophet_daily_sales.tail(90).rename(
            "Actual Gross Sales"
        ).rename_axis("Date").reset_index()

        model_view = prophet_prediction[
            prophet_prediction["ds"] >= (
                prophet_last_complete - pd.Timedelta(days=14)
            )
        ].copy()

        model_view["yhat"] = model_view["yhat"].clip(lower=0)

        fig_prophet = px.line(
            history_view,
            x="Date",
            y="Actual Gross Sales",
            title="Historical Sales and Prophet Forecast",
            labels={
                "Date": "Date",
                "Actual Gross Sales": "Gross Sales (£)"
            }
        )

        fig_prophet.add_scatter(
            x=model_view["ds"],
            y=model_view["yhat"],
            mode="lines",
            name="Prophet model estimate",
            line=dict(dash="dash")
        )

        st.plotly_chart(fig_prophet, width="stretch")

        with st.expander("View the 30 daily forecast values"):
            st.dataframe(
                future_only[["Date", "Forecast Sales (£)"]],
                hide_index=True,
                width="stretch"
            )

        st.download_button(
            "Download Prophet forecast CSV",
            data=future_only[
                ["Date", "Forecast Sales (£)"]
            ].to_csv(index=False),
            file_name="retailpulse_prophet_30_day_forecast.csv",
            mime="text/csv"
        )

        st.caption(
            "Prophet is a trained time-series model. Forecasts are estimates, "
            "not guaranteed future sales. The backtest did not meet the "
            "12% MAPE target, so treat these predictions with caution. "
            "Returns are not deducted from gross sales."
        )


        st.caption(
            "This is a weekday-average baseline, not a trained AI model. "
            "The potentially incomplete final day is excluded from training. "
            "Forecast accuracy must be measured separately on held-out data. "
            "Returns are not deducted from gross sales."
        )


# -----------------------------------
# 15. Customer Segmentation with K-Means
# -----------------------------------
st.divider()
st.markdown("## Customer Segmentation")

st.write(
    "Explore customer groups identified using K-Means clustering "
    "on Recency, Frequency, and Monetary (RFM) features."
)

SEGMENT_FILE = Path(__file__).parent / "rfm_kmeans_customers.csv"

if not SEGMENT_FILE.exists():
    st.warning(
        "K-Means results were not found. "
        "Please run rfm_kmeans.py first."
    )
else:
    customers = pd.read_csv(SEGMENT_FILE)

    cluster_summary = customers.groupby(
        "KMeans_Cluster", as_index=False
    ).agg(
        Customer_Count=("Customer ID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
    ).round(2)

    col1, col2 = st.columns(2)

    col1.metric(
        "Customers Analyzed",
        f"{len(customers):,}"
    )

    col2.metric(
        "Number of Clusters",
        f"{customers['KMeans_Cluster'].nunique()}"
    )

    fig_clusters = px.bar(
        cluster_summary.sort_values("Customer_Count"),
        x="Customer_Count",
        y="KMeans_Cluster",
        orientation="h",
        text="Customer_Count",
        title="Customer Count by K-Means Cluster",
        labels={
            "Customer_Count": "Number of Customers",
            "KMeans_Cluster": "Cluster",
        },
    )

    st.plotly_chart(fig_clusters, width="stretch")

    st.markdown("### Cluster Characteristics")

    st.dataframe(
        cluster_summary,
        hide_index=True,
        width="stretch"
    )
    
    st.markdown("### Business Recommendations")

    recommendations = {
        1: ("Occasional, Low-Spend", "Offer relevant product promotions."),
        2: ("Active Repeat Customers", "Encourage repeat purchases."),
        3: ("High-Value Champions", "Reward loyalty and offer premium products."),
        4: ("Regular Customers", "Encourage more frequent purchases."),
        5: ("At-Risk Customers", "Send re-engagement offers."),
        6: ("Inactive, Low-Spend", "Try a win-back campaign."),
    }

    cluster_summary["Segment_Name"] = (
        cluster_summary["KMeans_Cluster"].map(
            lambda cluster: recommendations.get(
                int(cluster), ("Other", "Review customer behaviour.")
            )[0]
        )
    )

    cluster_summary["Recommended_Action"] = (
        cluster_summary["KMeans_Cluster"].map(
            lambda cluster: recommendations.get(
                int(cluster), ("Other", "Review customer behaviour.")
            )[1]
        )
    )

    st.dataframe(
        cluster_summary[
            [
                "KMeans_Cluster",
                "Segment_Name",
                "Customer_Count",
                "Avg_Recency",
                "Avg_Frequency",
                "Avg_Monetary",
                "Recommended_Action",
            ]
        ],
        hide_index=True,
        width="stretch",
    )


    st.download_button(
        "Download K-Means Customer Data",
        data=customers.to_csv(index=False),
        file_name="rfm_kmeans_customers.csv",
        mime="text/csv",
    )

    st.caption(
        "Cluster numbers are identifiers, not rankings. "
        "Monetary values are based on the historical dataset "
        "and are not guaranteed future spending."
    )
    
# 15. Forecast Model Evaluation
st.markdown("---")
st.subheader("Forecast Model Evaluation")

st.write(
    "Historical backtesting compares Prophet with a "
    "56-day weekday-average baseline over the same "
    "12 test periods (360 days)."
)

evaluation = pd.DataFrame({
    "Forecast Method": [
        "56-Day Weekday Baseline",
        "Prophet"
    ],
    "MAE (£)": [
        10938.56,
        8907.76
    ],
    "WAPE (%)": [
        39.82,
        32.43
    ],
    "MAPE (%) — Nonzero Sales Days": [
        40.80,
        33.68
    ]
})

st.dataframe(
    evaluation,
    hide_index=True,
    width="stretch"
)

baseline_wape = 39.82
prophet_wape = 32.43
improvement = (
    (baseline_wape - prophet_wape)
    / baseline_wape
) * 100

col1, col2 = st.columns(2)

col1.metric(
    "Prophet WAPE",
    f"{prophet_wape:.2f}%"
)

col2.metric(
    "Relative Error Reduction",
    f"{improvement:.1f}%"
)

st.caption(
    "Evaluation: 12 historical test periods covering "
    "360 days. Lower errors are better. These results "
    "are historical measurements, not guarantees of "
    "future forecast accuracy. The original target "
    "of MAPE <= 12% has not been achieved."
)





