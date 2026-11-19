
import pandas as pd 
from sklearn.model_selection import train_test_split


def load_data(path="data/marketing_selected_features.csv"):
    
    data = pd.read_csv(path)

    print("Data loaded successfully.")
    print("=" * 60)
    print(data.head())
    print("=" * 60)
    print(f"Data shape: {data.shape}")
    print("=" * 60)
    print(f"Data columns: {data.columns.tolist()}")
    print("=" * 60)
    print(f"Data statistics:\n{data.describe()}")

    return data



def select_features(data):
    
    features = ['timestamp', 'campaign', 'cost', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']
    target = "conversion"

    required = features + [target]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    selected_data = data[required].copy()

    print("\nCorrelation with conversion:")
    print(selected_data.corr(numeric_only=True)[target])

    X = selected_data[features]
    y = selected_data[target]

    return X, y


def splitData(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
