from src.data_processing import load_data, clean_data
from src.feature_engineering import split_features_target, train_test_split_data, scale_features
from src.models import get_models, train_model
from src.evaluation import compare_models
import joblib
import os

# 1. Load and clean data
df = load_data('data/co2_emissions.csv')
df = clean_data(df)

# 2. Split features and target
X, y = split_features_target(df, target_col='co2')

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

# 4. Scale features for Linear Regression & MLP
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# 5. Train all models
models = get_models()
trained_models = {}
for name, model in models.items():
    if name in ['Linear Regression', 'MLP Regressor']:
        trained_models[name] = train_model(model, X_train_scaled, y_train)
    else:
        trained_models[name] = train_model(model, X_train, y_train)

# 6. Evaluate models
results_df = compare_models(trained_models, X_test_scaled, X_test, y_test)
print("Model Comparison:\n", results_df)

# 7. Save the best model, scaler, and feature columns
os.makedirs('saved_models', exist_ok=True)
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

joblib.dump(best_model, 'saved_models/best_co2_model.pkl')
joblib.dump(scaler, 'saved_models/scaler.pkl')              # Save scaler
joblib.dump(list(X_train.columns), 'saved_models/feature_columns.pkl')  # Save feature columns

print(f"Best model saved: {best_model_name}")