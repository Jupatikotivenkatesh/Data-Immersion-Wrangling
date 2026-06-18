"""Data Immersion & Wrangling — ApexPlanet DataAnalytics Dataset
Author: Data Analytics Intern
Purpose: Load, clean, transform, and engineer features from raw sales data"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────
# STEP 1: LOAD THE DATASET
# ─────────────────────────────────────────────────────────
df = pd.read_excel("../data/raw_data.xlsx")
print("=" * 60)
print("STEP 1 — DATASET LOADED")
print("=" * 60)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nFirst 5 Rows:")
print(df.head())

# ─────────────────────────────────────────────────────────
# STEP 2: DATA QUALITY ASSESSMENT
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — DATA QUALITY ASSESSMENT")
print("=" * 60)

# Check missing values
missing_count = df.isnull().sum()
missing_pct   = (missing_count / len(df) * 100).round(2)
print("\nMissing Values:")
print(pd.DataFrame({
    "Missing Count"  : missing_count,
    "Missing Percent": missing_pct
}))

# Check duplicates
print(f"\nDuplicate rows        : {df.duplicated().sum()}")
print(f"Duplicate Order_IDs   : {df['Order_ID'].duplicated().sum()}")
print(f"Duplicate Customer_IDs: {df['Customer_ID'].duplicated().sum()}")

# Check unique values in categorical columns
print("\nUnique Categories:")
for col in ['Gender', 'City', 'Category', 'Product']:
    print(f"  {col}: {df[col].unique().tolist()}")

# IQR Outlier Detection
print("\nOutlier Detection (IQR Method):")
for col in ['Age', 'Quantity', 'Unit_Price', 'Total_Sales']:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    print(f"  {col:<15}: Lower={lower:.1f}, Upper={upper:.1f}, Outliers={n_out}")

# Validate Total_Sales = Quantity x Unit_Price
df['Calc_Sales'] = df['Quantity'] * df['Unit_Price']
mismatch = df[abs(df['Total_Sales'] - df['Calc_Sales']) > 1]
print(f"\nTotal_Sales validation mismatches: {len(mismatch)}")
df.drop(columns=['Calc_Sales'], inplace=True)

# ─────────────────────────────────────────────────────────
# STEP 3: DATA CLEANING
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — DATA CLEANING")
print("=" * 60)

# Always work on a copy — never modify the original
df_clean = df.copy()

# 3a. Convert Order_Date from string to datetime
df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'], errors='coerce')
print(f"Order_Date converted to: {df_clean['Order_Date'].dtype}")

# 3b. Fill missing Age with median
age_median = df_clean['Age'].median()
df_clean['Age'] = df_clean['Age'].fillna(age_median).astype(int)
print(f"Age: missing values filled with median ({age_median})")

# 3c. Fill missing City with 'Unknown'
df_clean['City'] = df_clean['City'].fillna('Unknown')
print(f"City: missing values filled with 'Unknown'")

# 3d. Strip whitespace from all text columns
text_cols = ['Customer_Name', 'City', 'Gender', 'Product', 'Category']
for col in text_cols:
    df_clean[col] = df_clean[col].str.strip()
print("Whitespace stripped from all text columns")

# 3e. Remove duplicate Order_IDs — keep first occurrence
rows_before = len(df_clean)
df_clean    = df_clean.drop_duplicates(subset=['Order_ID'], keep='first')
rows_after  = len(df_clean)
print(f"Duplicate Order_IDs removed: {rows_before - rows_after} rows dropped")
print(f"Dataset shape after cleaning: {df_clean.shape}")

# 3f. Final verification
print("\nVerification after cleaning:")
print(f"  Missing values  : {df_clean.isnull().sum().sum()}")
print(f"  Duplicate IDs   : {df_clean['Order_ID'].duplicated().sum()}")
print(f"  Final shape     : {df_clean.shape}")

# ─────────────────────────────────────────────────────────
# STEP 4: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — FEATURE ENGINEERING")
print("=" * 60)

# 4a. Time features from Order_Date
df_clean['Order_Year']    = df_clean['Order_Date'].dt.year
df_clean['Order_Month']   = df_clean['Order_Date'].dt.month
df_clean['Order_Quarter'] = df_clean['Order_Date'].dt.quarter
df_clean['Day_of_Week']   = df_clean['Order_Date'].dt.day_name()
print("Time features created: Year, Month, Quarter, Day_of_Week")

# 4b. Age Group segmentation
bins   = [17, 25, 35, 50, 100]
labels = ['Young Adult', 'Adult', 'Middle Age', 'Senior']
df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=bins, labels=labels)
print("Age_Group created: Young Adult / Adult / Middle Age / Senior")

# 4c. Sales Category based on quartiles
q1  = df_clean['Total_Sales'].quantile(0.25)
med = df_clean['Total_Sales'].median()
q3  = df_clean['Total_Sales'].quantile(0.75)

def sales_category(val):
    if val <= q1:    return 'Low'
    elif val <= med: return 'Medium-Low'
    elif val <= q3:  return 'Medium-High'
    else:            return 'High'

df_clean['Sales_Category'] = df_clean['Total_Sales'].apply(sales_category)
print("Sales_Category created: Low / Medium-Low / Medium-High / High")

# 4d. Revenue per unit
df_clean['Revenue_per_Unit'] = (df_clean['Total_Sales'] / df_clean['Quantity']).round(2)
print("Revenue_per_Unit created")

# 4e. Weekend flag
df_clean['Is_Weekend'] = df_clean['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)
print("Is_Weekend flag created (1=Weekend, 0=Weekday)")

# ─────────────────────────────────────────────────────────
# STEP 5: SAVE FINAL DATASET
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — SAVING FINAL DATASET")
print("=" * 60)

output_path = "../data/cleaned_data.csv"
df_clean.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")
print(f"Final shape             : {df_clean.shape}")
print(f"Total new features added: 8")

print("\nFinal columns:")
for i, col in enumerate(df_clean.columns, 1):
    print(f"  {i:02d}. {col}")

print("\n✅ Pipeline complete. Dataset is analysis-ready.")