# RetailPulse – AI-Powered Customer Analytics & Demand Forecasting

RetailPulse is a Python-based retail analytics platform built with Streamlit. It helps explore retail sales, customer purchasing behavior, customer segmentation, churn risk, and demand forecasting.

## Features

- Interactive sales dashboard with date and country filters
- Key performance indicators: sales, orders, customers, and units sold
- RFM (Recency, Frequency, Monetary) customer analysis
- Customer segmentation using K-Means clustering
- Customer churn analysis using a Random Forest model
- Sales forecasting and backtesting with Prophet
- Data validation and forecasting-method comparisons

## Technology Stack

- Python
- Pandas
- Streamlit
- Plotly
- Scikit-learn
- Prophet
- Matplotlib

## Dataset

The project uses the Online Retail II dataset from the UCI Machine Learning Repository.

The raw dataset and generated data files are excluded from this repository because of their size. You must obtain the dataset separately and configure the required data paths before running the project.

## Run Locally

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the required dataset and generated files in the expected locations.
5. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL shown in the terminal.

## Important Notes

- Forecasts are estimates, not guaranteed future sales.
- Model performance should be evaluated on held-out data before business use.
- Churn predictions depend on the target definition and evaluation period.
- The dashboard requires its expected data files to be available.

## Project Status

Development and evaluation are in progress. Public deployment and production readiness have not yet been confirmed.
