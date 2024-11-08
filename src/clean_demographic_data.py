# src/clean_data.py
import pandas as pd
import os

RAW_DATA_PATH = 'data/raw/ndemographic.csv'
CLEANED_DATA_PATH = 'data/cleaned/demographic_cleaned.csv'

def clean_data(df):
    # Example cleaning steps (more can be added as needed)
    df.dropna(subset=['Age', 'Gender', 'BMI', 'Income'], inplace=True)
    df['Gender'] = df['Gender'].str.capitalize()
    df = df[(df['BMI'] > 10) & (df['BMI'] < 60)]  # BMI range filtering as an example
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

