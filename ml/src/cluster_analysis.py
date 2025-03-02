import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 📌 Load Clustered Data
data_path = "../data/processed/clustered_data.csv"
df = pd.read_csv(data_path)

print(f"✅ Loaded dataset with shape: {df.shape}")

# 📌 Check cluster distributions
print("\n📌 Cluster Distribution:")
print(df["gmm_cluster"].value_counts())

# 📌 Visualize Cluster Sizes
plt.figure(figsize=(8, 4))
sns.countplot(x=df["gmm_cluster"], palette="tab10")
plt.title("Cluster Distribution (GMM)")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# 📌 Calculate Silhouette Scores
silhouette_gmm = silhouette_score(df.iloc[:, :-1], df["gmm_cluster"])
print(f"✅ Silhouette Score (GMM): {silhouette_gmm:.3f}")

# 📌 Calculate Davies-Bouldin Score
dbi_gmm = davies_bouldin_score(df.iloc[:, :-1], df["gmm_cluster"])
print(f"✅ Davies-Bouldin Index (GMM): {dbi_gmm:.3f}")

# 📌 Save Evaluation Results
results = {
    "Silhouette Score (GMM)": silhouette_gmm,
    "Davies-Bouldin Index (GMM)": dbi_gmm
}
pd.DataFrame([results]).to_csv("../data/processed/cluster_evaluation.csv", index=False)

print("\n✅ Cluster Analysis & Evaluation Completed! Results saved.")
