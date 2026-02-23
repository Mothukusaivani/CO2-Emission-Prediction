import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    for col in df.select_dtypes(include='number'):
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna(df[col].mode()[0])
    df = pd.get_dummies(df, drop_first=True)
    return df