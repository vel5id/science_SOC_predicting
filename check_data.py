import pandas as pd
import numpy as np
import os

df = pd.read_csv("c:/Claude/science_article/data/features/full_dataset.csv")
print(f"Total rows: {len(df)}")
# SoilGrids issue
sg_cols = [c for c in df.columns if "sg_" in c]
print(f"SoilGrids columns in dataset: {sg_cols}")
print(f"Trace elements == 0: \n{(df[['mg', 'fe', 'mo', 'zn', 'cu', 'mn']] == 0).sum()}")
if 'k' in df.columns:
    print(f"k == 999: {(df['k'] == 999).sum()}")

# To fix this, we should delete the corrupt soilgrids file
sg_file = "c:/Claude/science_article/data/soil_maps/soilgrids_features.csv"
if os.path.exists(sg_file):
    print(f"Testing reading {sg_file}")
    sdf = pd.read_csv(sg_file)
    print(sdf.columns.tolist())
