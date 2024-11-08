import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('demographic.csv')

# Remove or impute null values in crucial columns
df.dropna(subset=['Age', 'Gender', 'BMI', 'Income'], inplace=True)

# Remove outliers outside the 1st and 99th percentile
q_low, q_high = df['BMI'].quantile(0.01), df['BMI'].quantile(0.99)
df = df[(df['BMI'] >= q_low) & (df['BMI'] <= q_high)]

# Convert categorical columns to consistent casing
df['Gender'] = df['Gender'].str.capitalize()

# Handle invalid values (e.g., negative or unrealistic values for height/weight)
df = df[df['Age'] > 0]
df = df[(df['Height'] > 0) & (df['Weight'] > 0)]

# Display cleaned data
print(df.head())
