
import pandas as pd

# Load the scored customer data
rfm = pd.read_csv("rfm_scored_customers.csv")

def assign_segment(row):
    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 4:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "Recent Customers"
    elif r <= 2 and f >= 3 and m >= 3:
        return "At Risk - Valuable"
    elif r <= 2 and f <= 2:
        return "Need Attention"
    else:
        return "Regular Customers"

rfm["CustomerSegment"] = rfm.apply(assign_segment, axis=1)

# Save customer segments
rfm.to_csv("rfm_customer_segments.csv", index=False)

print("CUSTOMER SEGMENTATION COMPLETED")
print("\nCustomers in each segment:")
print(rfm["CustomerSegment"].value_counts().to_string())

print("\nAverage metrics by segment:")
print(
    rfm.groupby("CustomerSegment")[
        ["Recency", "Frequency", "Monetary"]
    ].mean().round(2).to_string()
)

print("\nSaved to: rfm_customer_segments.csv")
