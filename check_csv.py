import pandas as pd
import numpy as np

features_path = 'c:/Claude/science_article/data/features/full_dataset.csv'
df = pd.read_csv(features_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]} ...")

if 's2_NDVI_summer' in df.columns:
    print(f"s2_NDVI_summer non-null count: {df['s2_NDVI_summer'].notna().sum()}")
else:
    print("s2_NDVI_summer not found in columns")

print(f"k non-null count: {df['k'].notna().sum() if 'k' in df.columns else 'N/A'}")
print(f"s2_BSI_spring non-null count: {df['s2_BSI_spring'].notna().sum() if 's2_BSI_spring' in df.columns else 'N/A'}")

mask = df[["k", "s2_BSI_spring"]].notna().all(axis=1) if "k" in df.columns and "s2_BSI_spring" in df.columns else []
print(f"Rows with both k and s2_BSI_spring: {sum(mask)}")
