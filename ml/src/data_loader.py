import pandas as pd

# Define paths for different datasets
DATA_PATHS = {
    "cooking_fuel": "../data/raw/LivWell/all_labels_cooking_fuel_completed_coal_as_traditional.csv",
    "married_surveys": "../data/raw/LivWell/ever_married_surveys.csv",
    "population": "../data/raw/LivWell/populationWB.csv",
    "cardiovascular": "../data/raw/CVD_cleaned.csv",
    "diabetes_data": "../data/raw/diabetes_data.csv",
    "diabetes": "../data/raw/Diabetes.csv",
    "fed_cycle": "../data/raw/FedCycleData071012 (2).csv",
    "food_group1": "../data/raw/FOOD-DATA-GROUP1.csv",
    "food_group2": "../data/raw/FOOD-DATA-GROUP2.csv",
    "food_group3": "../data/raw/FOOD-DATA-GROUP3.csv",
    "food_group4": "../data/raw/FOOD-DATA-GROUP4.csv",
    "food_group5": "../data/raw/FOOD-DATA-GROUP5.csv",
    "inquirer_basic": "../data/raw/inquirerbasic.csv",
    "kidney_disease": "../data/raw/kidney_disease.csv",
    "maternal_health": "../data/raw/Maternal_Health_Risk_Data_Set.csv",
    "rows": "../data/raw/rows.csv",
}


features = {}

for name, path in DATA_PATHS.items():
    try:
        df = pd.read_csv(path, nrows=1)  # Read only the first row to get columns
        features[name] = list(df.columns)
    except Exception as e:
        features[name] = f"Error reading file: {e}"

# Print the extracted features
for dataset, columns in features.items():
    print(f"{dataset}: {', '.join(columns)}")

def load_dataset(dataset_name):
    """Loads a dataset by name and returns a pandas DataFrame."""
    if dataset_name in DATA_PATHS:
        try:
            return pd.read_csv(DATA_PATHS[dataset_name])
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
    else:
        print(f"Dataset '{dataset_name}' not found.")
        return None

