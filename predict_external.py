import pandas as pd
import joblib
import numpy as np

# ============================
# Load saved model assets
# ============================
best_model = joblib.load('saved_models/best_co2_model.pkl')
scaler = joblib.load('saved_models/scaler.pkl')
feature_columns = joblib.load('saved_models/feature_columns.pkl')

print("\nEnter details for CO2 prediction:\n")

# ============================
# Take user input
# ============================
country = input("Country: ")
year = int(input("Year: "))
population = float(input("Population: "))
gdp = float(input("GDP: "))
coal = float(input("Coal Consumption: "))
oil = float(input("Oil Consumption: "))
gas = float(input("Gas Consumption: "))
cement = float(input("Cement CO2: "))
flaring = float(input("Flaring CO2: "))
other = float(input("Other Energy CO2: "))

# ============================
# Create input dictionary
# ============================
input_dict = {
    'country': country,
    'year': year,
    'population': population,
    'gdp': gdp,
    'coal_consumption': coal,
    'oil_consumption': oil,
    'gas_consumption': gas,
    'cement_co2': cement,
    'flaring_co2': flaring,
    'other_energy_co2': other
}

# ============================
# Convert to DataFrame
# ============================
input_df = pd.DataFrame([input_dict])

# ============================
# One-hot encode
# ============================
input_df = pd.get_dummies(input_df)

# ============================
# Align columns with training
# ============================
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ============================
# Decide whether scaling is required
# ============================
model_name = type(best_model).__name__

if model_name in ["LinearRegression", "MLPRegressor"]:
    input_final = scaler.transform(input_df)
else:
    input_final = input_df.values

# ============================
# Predict
# ============================
prediction = best_model.predict(input_final)

print("\n========== RESULT ==========")
print(f"Predicted CO2 Emission: {prediction[0]:.2f}")
print("============================")