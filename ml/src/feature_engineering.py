import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import skew

# 📌 Load the Cleaned Dataset
data_path = "../data/processed/cleaned_data.csv"
df = pd.read_csv(data_path)

print(f"✅ Loaded dataset with shape: {df.shape}")

# 📌 Standardize Column Names (Lowercase, Remove Spaces)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 📌 Use "age_category" instead of "age"
if "age_category" in df.columns:
    df["age_group"] = df["age_category"]
else:
    print("⚠️ ERROR: 'age_category' column not found!")
    exit(1)  # Stop execution if age_category is missing

# 📌 Handle Missing Values Again (Ensure No NaNs Before PCA)
df.fillna(df.median(numeric_only=True), inplace=True)  # Numeric: Replace NaN with median
df.fillna(df.mode().iloc[0], inplace=True)  # Categorical: Replace NaN with most frequent value

# 📌 Feature Selection: Remove Low Variance Features
selector = VarianceThreshold(threshold=0.01)  # Remove features with variance < 1%
df = df[df.columns[selector.fit(df).get_support()]]

print(f"✅ Selected {df.shape[1]} high-variance features.")

# 📌 Feature Creation
if "weight_(kg)" in df.columns and "height_(cm)" in df.columns:
    df["bmi_calculated"] = df["weight_(kg)"] / ((df["height_(cm)"] / 100) ** 2)  # Create BMI if missing

if "exercise" in df.columns and "heart_disease" in df.columns:
    df["exercise_heart"] = df["exercise"] * df["heart_disease"]  # Interaction term
else:
    df["exercise_heart"] = 0  # Default if columns are missing

# 📌 Feature Encoding (One-Hot Encoding)
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_features]))
encoded_features.columns = encoder.get_feature_names_out(categorical_features)

df.drop(columns=categorical_features, inplace=True)  # Drop original categorical columns
df = pd.concat([df, encoded_features], axis=1)  # Add encoded features

# 📌 Feature Scaling (Z-score Normalization)
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 📌 Fix: Replace Negative Values Before Log Transformation
skewed_cols = df[numeric_cols].apply(lambda x: skew(x)).abs()
high_skew = skewed_cols[skewed_cols > 0.75].index.tolist()

# Replace negative values with 0 before log transformation
df[high_skew] = df[high_skew].clip(lower=0)
df[high_skew] = np.log1p(df[high_skew])  # Apply log transformation

# 📌 Ensure No NaNs Before PCA
df.dropna(inplace=True)  # Remove any remaining NaNs

# 📌 Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
df_pca = pca.fit_transform(df)
df_pca = pd.DataFrame(df_pca)
print(f"✅ Reduced dimensions to {df_pca.shape[1]} using PCA.")

# 📌 Save Final Processed Data
df_pca.to_csv("../data/processed/feature_engineered_data.csv", index=False)
print("\n✅ Feature Engineering Completed! Processed dataset saved.")
