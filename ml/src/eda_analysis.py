import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

# 📌 Load All Cleaned Datasets
datasets = {
    "cooking_fuel": pd.read_csv("../data/cleaned/cooking_fuel_cleaned.csv"),
    "married_surveys": pd.read_csv("../data/cleaned/married_surveys_cleaned.csv"),
    "population": pd.read_csv("../data/cleaned/population_cleaned.csv"),
    "cardiovascular": pd.read_csv("../data/cleaned/cardiovascular_cleaned.csv"),
    "diabetes_data": pd.read_csv("../data/cleaned/diabetes_data_cleaned.csv"),
    "diabetes": pd.read_csv("../data/cleaned/diabetes_cleaned.csv"),
    "fed_cycle": pd.read_csv("../data/cleaned/fed_cycle_cleaned.csv"),
    "food_group1": pd.read_csv("../data/cleaned/food_group1_cleaned.csv"),
    "food_group2": pd.read_csv("../data/cleaned/food_group2_cleaned.csv"),
    "food_group3": pd.read_csv("../data/cleaned/food_group3_cleaned.csv"),
    "food_group4": pd.read_csv("../data/cleaned/food_group4_cleaned.csv"),
    "food_group5": pd.read_csv("../data/cleaned/food_group5_cleaned.csv"),
    "inquirer_basic": pd.read_csv("../data/cleaned/inquirer_basic_cleaned.csv"),
    "kidney_disease": pd.read_csv("../data/cleaned/kidney_disease_cleaned.csv"),
    "maternal_health": pd.read_csv("../data/cleaned/maternal_health_cleaned.csv"),
    "rows": pd.read_csv("../data/cleaned/rows_cleaned.csv"),
}

# 📌 Standardize Column Names (Lowercase + Underscore)
for name, df in datasets.items():
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    datasets[name] = df

# 📌 Ensure "age" column exists and is numeric
def check_and_clean_age(df):
    """Ensure 'age' column exists, convert to numeric, and handle missing values."""
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")  # Convert non-numeric values to NaN
        df.dropna(subset=["age"], inplace=True)  # Drop rows where age is NaN
        df["age"] = df["age"].astype(int)  # Convert to integer
    return df

for dataset in ["cardiovascular", "diabetes", "kidney_disease", "maternal_health"]:
    if dataset in datasets:
        datasets[dataset] = check_and_clean_age(datasets[dataset])

# 📌 Debug: Print columns of each dataset before merging
for name, df in datasets.items():
    print(f"🔍 {name} Columns: {df.columns.tolist()}")

# 📌 Merge Relevant Datasets on "Age" (Only if 'age' exists in both)
merged_df = datasets["cardiovascular"]

for dataset in ["diabetes", "kidney_disease", "maternal_health"]:
    if "age" in datasets[dataset].columns and "age" in merged_df.columns:
        print(f"✅ Merging {dataset} on 'age'")
        merged_df = merged_df.merge(datasets[dataset], on="age", how="left")
    else:
        print(f"⚠️ Skipping merge for {dataset}, 'age' column missing!")

# 📌 Ensure "clientid" exists before merging `fed_cycle`
if "clientid" in datasets["fed_cycle"].columns and "clientid" in merged_df.columns:
    print("✅ Merging fed_cycle on 'clientid'")
    merged_df = merged_df.merge(datasets["fed_cycle"], on="clientid", how="left")
else:
    print("⚠️ Skipping merge for fed_cycle, 'clientid' column missing!")

# 📌 Drop Duplicate Columns
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# 📌 Encode Categorical Features
def encode_categorical(df):
    """Convert categorical columns to numerical values."""
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df

merged_df = encode_categorical(merged_df)

# 📌 Feature Scaling
scaler = StandardScaler()
numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols])

# 📌 Save Processed Dataset
merged_df.to_csv("../data/processed/cleaned_data.csv", index=False)

# 📌 Exploratory Data Analysis (EDA)

## **1️⃣ Feature Distributions**
def plot_feature_distributions(df):
    """Plot histograms for numerical features."""
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(df.select_dtypes(include=['float64', 'int64']).columns[:6]):
        plt.subplot(2, 3, i + 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

plot_feature_distributions(merged_df)

## **2️⃣ Correlation Heatmap**
plt.figure(figsize=(12, 8))
sns.heatmap(merged_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

## **3️⃣ Outlier Detection using Boxplots**
plt.figure(figsize=(15, 8))
for i, col in enumerate(merged_df.select_dtypes(include=['float64', 'int64']).columns[:6]):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x=merged_df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

## **4️⃣ Feature Selection (Removing Low Variance Features)**
def feature_selection(df):
    """Select high variance features for better clustering."""
    selector = VarianceThreshold(threshold=0.01)  # Remove features with low variance
    df_selected = df[df.columns[selector.fit(df).get_support()]]
    return df_selected

selected_features_df = feature_selection(merged_df)
selected_features_df.to_csv("../data/processed/selected_features.csv", index=False)

print("\n✅ EDA Completed! Cleaned dataset and selected features saved.")
