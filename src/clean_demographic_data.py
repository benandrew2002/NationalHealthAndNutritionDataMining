# src/clean_data.py
import pandas as pd
import os

RAW_DATA_PATH = 'data/raw/demographic.csv'
CLEANED_DATA_PATH = 'data/cleaned/demographic_cleaned.csv'

def clean_data(df):
    # Example cleaning steps
    df.dropna(subset=['RIDAGEYR', 'RIAGENDR', 'WTINT2YR', 'WTMEC2YR'], inplace=True)
    df['WTINT2YR'] = pd.to_numeric(df['WTINT2YR'], errors='coerce')
    df['WTINT2YR'] = df['WTINT2YR'].round(2)
    df['WTMEC2YR'] = pd.to_numeric(df['WTMEC2YR'], errors='coerce')
    df['WTMEC2YR'] = df['WTMEC2YR'].round(2)
    return df

def main():
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)
    # Clean data
    cleaned_df = clean_data(df)
    # Save cleaned data
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print("Data cleaned and saved to", CLEANED_DATA_PATH)

if __name__ == "__main__":
    main()

