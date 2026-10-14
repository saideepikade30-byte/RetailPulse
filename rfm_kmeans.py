
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load customer RFM data
df = pd.read_csv("rfm_customers.csv")

# 2. Select features for clustering
features = ["Recency", "Frequency", "Monetary"]
X = df[features].copy()

# 3. Handle extreme values and skew before scaling
# Log transform Frequency and Monetary to reduce skew
X["Frequency"] = X["Frequency"].clip(lower=0)
X["Monetary"] = X["Monetary"].clip(lower=0)
X["Frequency"] = __import__("numpy").log1p(X["Frequency"])
X["Monetary"] = __import__("numpy").log1p(X["Monetary"])

# 4. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply K-Means with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled) + 1

# 6. Save customer cluster assignments
df.to_csv("rfm_kmeans_customers.csv", index=False)

# 7. Summarize clusters
summary = df.groupby("KMeans_Cluster").agg(
    Customer_Count=("Customer ID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean")
).round(2)

print("\nK-MEANS CLUSTER SUMMARY")
print(summary.to_string())

summary.to_csv("rfm_kmeans_summary.csv")
print("Saved: rfm_kmeans_summary.csv")

print("\nTotal customers:", len(df))
print("Saved: rfm_kmeans_customers.csv")

# 8. Plot cluster sizes
summary["Customer_Count"].sort_values().plot(
    kind="barh", figsize=(9, 5)
)
plt.title("RetailPulse K-Means Customer Clusters")
plt.xlabel("Number of Customers")
plt.ylabel("K-Means Cluster")
plt.tight_layout()
plt.savefig("rfm_kmeans_clusters.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: rfm_kmeans_clusters.png")
