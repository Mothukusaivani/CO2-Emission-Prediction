from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def get_models():
    models = {
        'Linear Regression': LinearRegression(),
        'MLP Regressor': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        'CatBoost': CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=42)
    }
    return models

def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)