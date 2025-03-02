import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 📌 Load the Selected Features Dataset
data_path = "../data/processed/selected_features.csv"
df = pd.read_csv(data_path)

print(f"✅ Loaded dataset with shape: {df.shape}")

# 📌 Standardize Data (For Better Clustering)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 📌 Apply PCA for Dimensionality Reduction (Retain 98% Variance for Better Clustering)
pca = PCA(n_components=0.98)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled))
print(f"✅ PCA applied, reduced to {df_pca.shape[1]} dimensions.")

# --- FIND OPTIMAL CLUSTERS FOR GMM ---
print("\n⚡ Finding optimal clusters using Bayesian Information Criterion (BIC)...")

bic_scores = []
silhouette_scores = []
best_k = 2
best_silhouette = -1

for k in range(2, 11):  # Trying clusters from 2 to 10
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm_labels = gmm.fit_predict(df_pca)
    
    # Compute BIC (Lower is better)
    bic_scores.append(gmm.bic(df_pca))
    
    # Compute Silhouette Score (Higher is better)
    silhouette = silhouette_score(df_pca, gmm_labels)
    silhouette_scores.append(silhouette)

    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_k = k

# 📌 Plot BIC Scores for different cluster numbers
plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), bic_scores, marker='o', linestyle='--', label="BIC Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Bayesian Information Criterion (BIC)")
plt.title("Optimal Cluster Selection using BIC")
plt.legend()
plt.show()

print(f"✅ Optimal Number of Clusters for GMM: {best_k}")

# --- APPLY GAUSSIAN MIXTURE MODEL (GMM) ---
print("\n⚡ Running Optimized Gaussian Mixture Model...")
gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
df_pca["gmm_cluster"] = gmm.fit_predict(df_pca)

# 📌 Evaluate GMM Clustering
gmm_silhouette = silhouette_score(df_pca.iloc[:, :-1], df_pca["gmm_cluster"])
gmm_davies_bouldin = davies_bouldin_score(df_pca.iloc[:, :-1], df_pca["gmm_cluster"])

print(f"✅ GMM Silhouette Score: {gmm_silhouette:.4f}")
print(f"✅ GMM Davies-Bouldin Index: {gmm_davies_bouldin:.4f}")

# --- APPLY HIERARCHICAL CLUSTERING ---
print("\n⚡ Running Optimized Hierarchical Clustering...")

# 📌 Dynamically adjust sample size for hierarchical clustering
sample_size = min(len(df_pca), 5000)  # Reduce data if too large
df_sampled = df_pca.sample(n=sample_size, random_state=42)

# 📌 Generate Linkage Matrix (Using Complete Linkage for Better Separation)
linkage_matrix = linkage(df_sampled.iloc[:, :-1], method="complete")

# 📌 Plot Dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram (Optimized)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# 📌 Assign Clusters from Dendrogram
df_sampled["hierarchical_cluster"] = fcluster(linkage_matrix, best_k, criterion="maxclust")

print(f"✅ Hierarchical Clustering Applied with {best_k} clusters (on {sample_size} samples).")

# --- VISUALIZATION ---
# 📌 Scatter Plot of GMM Clusters
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df_pca.iloc[:, 0], y=df_pca.iloc[:, 1], hue=df_pca["gmm_cluster"], palette="tab10")
plt.title("Optimized Gaussian Mixture Model (GMM) Clustering Results")
plt.show()

# 📌 Scatter Plot of Hierarchical Clusters
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df_sampled.iloc[:, 0], y=df_sampled.iloc[:, 1], hue=df_sampled["hierarchical_cluster"], palette="tab10")
plt.title("Optimized Hierarchical Clustering Results (Sampled Data)")
plt.show()

# 📌 Save Clustered Data
df_pca.to_csv("../data/processed/optimized_clustered_data.csv", index=False)
print("\n✅ Optimized Clustering Completed! Clustered dataset saved.")
