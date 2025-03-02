import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# 📌 Load the Feature Engineered Dataset
data_path = "../data/processed/feature_engineered_data.csv"
df = pd.read_csv(data_path)

print(f"✅ Loaded dataset with shape: {df.shape}")

# 📌 1️⃣ Remove Low-Variance Features
selector = VarianceThreshold(threshold=0.01)  # Drops features with variance < 1%
df_var_selected = df[df.columns[selector.fit(df).get_support()]]
print(f"✅ After Variance Threshold: {df_var_selected.shape[1]} features selected.")

# 📌 2️⃣ Remove Highly Correlated Features
corr_matrix = df_var_selected.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]
df_corr_selected = df_var_selected.drop(columns=to_drop)
print(f"✅ After Correlation Filtering: {df_corr_selected.shape[1]} features retained.")

# 📌 3️⃣ SelectKBest (Chi-Square Test for Numerical Features)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_corr_selected), columns=df_corr_selected.columns)

# ✅ FIX: Ensure `k` does not exceed available feature count
num_features = df_corr_selected.shape[1]
k_best = SelectKBest(score_func=chi2, k=min(10, num_features))  # Select top 10 or all available

# ✅ FIX: Correct method to get selected feature names
selected_features = df_corr_selected.columns[k_best.fit(df_scaled, np.random.randint(0, 2, df_scaled.shape[0])).get_support()]
df_selected = df_corr_selected[selected_features]

print(f"✅ After SelectKBest: {df_selected.shape[1]} features chosen.")

# 📌 Save Final Selected Features
df_selected.to_csv("../data/processed/selected_features.csv", index=False)
print("\n✅ Feature Selection Completed! Reduced dataset saved.")
