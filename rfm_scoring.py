
import pandas as pd

# Load the RFM table
rfm = pd.read_csv("rfm_customers.csv")

# Score customers from 1 to 5
# Lower Recency is better, so reverse its score
rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Combine the three scores
rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

# Save scored customers
rfm.to_csv("rfm_scored_customers.csv", index=False)

print("RFM SCORING COMPLETED")
print("\nScore distribution:")
print(rfm[["R_Score", "F_Score", "M_Score"]].describe().round(2))

print("\nSample scored customers:")
print(rfm.head(10).to_string(index=False))

print("\nSaved to: rfm_scored_customers.csv")
