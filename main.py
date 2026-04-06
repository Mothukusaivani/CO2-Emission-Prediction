from src.data_processing import load_data, clean_data
from src.feature_engineering import (
    split_features_target,
    train_test_split_data,
    scale_features
)
from src.models import get_models, train_model
from src.evaluation import compare_models
import joblib
import os
import json
import pandas as pd
# --------------------------------------------------
# 1. Load and clean data
# --------------------------------------------------
df = load_data('data/co2_emissions.csv')
df = clean_data(df)

# IMPORTANT: keep full copy for dashboard metadata
df_full = df.copy().reset_index(drop=True)

# --------------------------------------------------
# 2. Split features and target
# --------------------------------------------------
X, y = split_features_target(df_full, target_col='co2')

# Reset index to maintain alignment
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# --------------------------------------------------
# 3. Train-test split (ONLY ONCE)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

# Capture test indices BEFORE scaling
test_indices = X_test.index

# --------------------------------------------------
# 4. Scale features
# --------------------------------------------------
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# --------------------------------------------------
# 5. Train all models
# --------------------------------------------------
models = get_models()
trained_models = {}

for name, model in models.items():
    if name in ['Linear Regression', 'MLP Regressor']:
        trained_models[name] = train_model(model, X_train_scaled, y_train)
    else:
        trained_models[name] = train_model(model, X_train, y_train)

# --------------------------------------------------
# 6. Evaluate models
# --------------------------------------------------
results_df = compare_models(trained_models, X_test_scaled, X_test, y_test)

print("\nModel Comparison:")
print(results_df)

# --------------------------------------------------
# 7. Save models and metadata
# --------------------------------------------------
os.makedirs('saved_models', exist_ok=True)

for name, model in trained_models.items():
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"saved_models/{filename}")

joblib.dump(scaler, 'saved_models/scaler.pkl')
joblib.dump(list(X_train.columns), 'saved_models/feature_columns.pkl')

results_df.to_csv("saved_models/model_comparison_results.csv", index=False)

results_dict = results_df.to_dict(orient="records")
with open("saved_models/model_metrics.json", "w") as f:
    json.dump(results_dict, f, indent=4)

print("\nAll models and metrics saved successfully.")

# --------------------------------------------------
# 8. Save test predictions WITH country metadata
# --------------------------------------------------

# Metadata columns we want to keep for dashboard
metadata_cols = [
    "country",
    "year",
    "population",
    "gdp",
    "cement_co2",
    "coal_co2",
    "oil_co2",
    "gas_co2"
]

# Keep only columns that exist
metadata_cols = [c for c in metadata_cols if c in df_full.columns]

# Reset index to keep alignment with X
metadata_df = df_full[metadata_cols].reset_index(drop=True)

# Select metadata rows corresponding to test set
metadata_test = metadata_df.loc[X_test.index].reset_index(drop=True)

# Reset y_test for alignment
y_test_reset = y_test.reset_index(drop=True)

test_predictions = []

for name, model in trained_models.items():

    if name in ['Linear Regression', 'MLP Regressor']:
        preds = model.predict(X_test_scaled)
    else:
        preds = model.predict(X_test)

    preds = pd.Series(preds).reset_index(drop=True)

    temp_df = metadata_test.copy()
    temp_df["model"] = name
    temp_df["actual"] = y_test_reset
    temp_df["predicted"] = preds

    test_predictions.append(temp_df)

test_predictions_df = pd.concat(test_predictions, ignore_index=True)

test_predictions_df.to_csv(
    "saved_models/test_predictions.csv",
    index=False
)

print("\nTest predictions saved successfully WITH country column.")
