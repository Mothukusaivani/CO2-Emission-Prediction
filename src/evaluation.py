from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np


def evaluate_model(model, X_test, y_test):
    """Return RMSE and R2 score for a model"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)  # always works
    rmse = np.sqrt(mse)  # compute RMSE manually
    r2 = r2_score(y_test, y_pred)
    return rmse, r2


def compare_models(models, X_test_scaled, X_test, y_test):
    """Compare all trained models and return a sorted dataframe"""
    results = []
    for name, model in models.items():
        # Scale-aware evaluation
        if name in ['Linear Regression', 'MLP Regressor']:
            rmse, r2 = evaluate_model(model, X_test_scaled, y_test)
        else:
            rmse, r2 = evaluate_model(model, X_test, y_test)
        results.append({'Model': name, 'RMSE': rmse, 'R2': r2})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='RMSE').reset_index(drop=True)
    return results_df