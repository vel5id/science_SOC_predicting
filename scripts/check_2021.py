import pandas as pd
import sys

try:
    print("Loading data...")
    df = pd.read_csv('data/features/master_dataset.csv')
    print("Data loaded. Shape:", df.shape)
    
    years = sorted(df['year'].unique())
    print('\nTotal rows per year:')
    print(df['year'].value_counts().sort_index())
    
    groups = {
        'S2': [c for c in df.columns if c.startswith('s2_')],
        'L8': [c for c in df.columns if c.startswith('l8_')],
        'S1': [c for c in df.columns if c.startswith('s1_')],
        'GLCM': [c for c in df.columns if c.startswith('glcm_')],
        'Composite': [c for c in df.columns if c.startswith('cs_')],
        'Delta': [c for c in df.columns if c.startswith('delta_')]
    }
    
    print('\nAnalyzing missing data (NaNs) by year for RS groups...')
    for g_name, cols in groups.items():
        if not cols: 
            continue
        print(f'\n--- {g_name} ({len(cols)} features) ---')
        for y in years:
            sub = df[df['year'] == y][cols]
            total_cells = sub.size
            if total_cells == 0:
                continue
            missing_cells = sub.isna().sum().sum()
            pct = (missing_cells / total_cells * 100)
            
            rows_with_any_nan = sub.isna().any(axis=1).sum()
            row_pct = (rows_with_any_nan / len(sub) * 100)
            
            print(f'  {y}: {pct:5.1f}% cells missing | {row_pct:5.1f}% rows have >=1 missing feature')
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
