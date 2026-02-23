# src/feature_engineering.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_features_target(df, target_col='co2_emission'):
    """Split dataframe into features X and target y"""
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataframe columns")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Split features and target into training and testing sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler