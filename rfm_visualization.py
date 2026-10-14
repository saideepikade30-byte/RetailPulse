
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("rfm_customer_segments.csv")

segment_counts = (
    df["CustomerSegment"]
    .value_counts()
    .sort_values(ascending=True)
)

plt.figure(figsize=(10, 6))
bars = plt.barh(segment_counts.index, segment_counts.values)

plt.bar_label(bars, padding=5, fmt="%.0f")
plt.title("RetailPulse Customer Segmentation")
plt.xlabel("Number of Customers")
plt.ylabel("Customer Segment")
plt.tight_layout()

plt.savefig("rfm_customer_segments.png", dpi=300, bbox_inches="tight")
plt.show()

print("Chart saved successfully: rfm_customer_segments.png")
