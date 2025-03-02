import pandas as pd
import numpy as np
from data_loader import load_dataset

# 📌 Load all datasets
datasets = {
    "cooking_fuel": load_dataset("cooking_fuel"),
    "married_surveys": load_dataset("married_surveys"),
    "population": load_dataset("population"),
    "cardiovascular": load_dataset("cardiovascular"),
    "diabetes_data": load_dataset("diabetes_data"),
    "diabetes": load_dataset("diabetes"),
    "fed_cycle": load_dataset("fed_cycle"),
    "food_group1": load_dataset("food_group1"),
    "food_group2": load_dataset("food_group2"),
    "food_group3": load_dataset("food_group3"),
    "food_group4": load_dataset("food_group4"),
    "food_group5": load_dataset("food_group5"),
    "inquirer_basic": load_dataset("inquirer_basic"),
    "kidney_disease": load_dataset("kidney_disease"),
    "maternal_health": load_dataset("maternal_health"),
    "rows": load_dataset("rows"),
}

# 📌 Standardize Column Names
for name, df in datasets.items():
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    datasets[name] = df

# 📌 Remove Duplicate Rows
for name, df in datasets.items():
    datasets[name] = df.drop_duplicates()

# 📌 Handle Missing Values
def clean_missing_values(df):
    """Fill or drop missing values based on the feature type"""
    df = df.dropna(thresh=len(df) * 0.7, axis=1)  # Drop columns with >30% missing
    for col in df.columns:
        if df[col].dtype == "object":  # Categorical data
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:  # Numerical data
            df[col].fillna(df[col].median(), inplace=True)
    return df

for name, df in datasets.items():
    datasets[name] = clean_missing_values(df)

# 📌 Convert Data Types
def convert_data_types(df):
    """Convert numeric columns and handle categorical ones"""
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
    return df

for name, df in datasets.items():
    datasets[name] = convert_data_types(df)

# 📌 Ensure "Age" is Numeric in Relevant Datasets
for name in ["cardiovascular", "diabetes", "kidney_disease", "maternal_health"]:
    if "age" in datasets[name].columns:
        datasets[name]["age"] = pd.to_numeric(datasets[name]["age"], errors="coerce").astype("Int64")

# 📌 Handle Outliers (Capping extreme values)
def remove_outliers(df):
    """Clip values in numerical columns to remove extreme outliers"""
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

for name, df in datasets.items():
    datasets[name] = remove_outliers(df)

# 📌 Save Cleaned Datasets
for name, df in datasets.items():
    df.to_csv(f"../data/cleaned/{name}_cleaned.csv", index=False)

print("\n✅ Data Cleaning Completed! Cleaned datasets saved.")
