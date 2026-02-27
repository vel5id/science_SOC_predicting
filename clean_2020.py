import sqlite3
import pandas as pd
import glob
import os

print("Starting to clean 2020 data...")

# 1. Clean SQLite database
db_path = "data/soil_analysis.db"
if os.path.exists(db_path):
    print(f"Cleaning database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables containing 'year' column
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [c[1] for c in cursor.fetchall()]
        if 'year' in columns:
            cursor.execute(f"DELETE FROM {table} WHERE year = 2020;")
            print(f"  Deleted 2020 records from table: {table}")
    
    conn.commit()
    conn.execute("VACUUM;")
    conn.close()
    print("Database cleaned.")
else:
    print(f"Database {db_path} not found.")

# 2. Clean CSV and XLSX files in data/features/ and data/
directories_to_clean = ["data", "data/features", "data/climate", "data/glcm", "data/landsat8", "data/sentinel1", "data/sentinel2", "data/spectral_eng", "data/temperature", "data/semivariograms"]

for folder in directories_to_clean:
    if os.path.exists(folder):
        print(f"\nCleaning datasets in {folder}...")
        for file in glob.glob(f"{folder}/*.*"):
            if not (file.endswith(".csv") or file.endswith(".xlsx")):
                continue
                
            try:
                if file.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                    
                if 'year' in df.columns:
                    n_before = len(df)
                    df = df[df['year'] != 2020]
                    n_after = len(df)
                    
                    if n_before != n_after:
                        print(f"    Removed {n_before - n_after} rows for 2020 in {file}.")
                        if file.endswith(".csv"):
                            df.to_csv(file, index=False)
                        else:
                            df.to_excel(file, index=False)
            except Exception as e:
                # Some files might not be readable as simple df
                pass

print("Done.")
