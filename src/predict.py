import joblib

def load_model(path='saved_models/best_co2_model.pkl'):
    return joblib.load(path)

def make_predictions(model, X_sample, scaler=None):
    if scaler:
        X_sample = scaler.transform(X_sample)
    return model.predict(X_sample)