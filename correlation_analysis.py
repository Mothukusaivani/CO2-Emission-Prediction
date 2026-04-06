import pandas as pd

# Load dataset
df = pd.read_csv("data/co2_emissions.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Select numeric columns
numeric_df = df.select_dtypes(include=["number"])

# Correlation with CO2
corr = numeric_df.corr()["co2"].sort_values(ascending=False)

print("\nTop 20 features correlated with CO2:\n")
print(corr.head(20))
