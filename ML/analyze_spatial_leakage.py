"""
Analyze spatial distances between fields/farms to assess
whether Field-LOFO has spatial leakage.
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

df = pd.read_csv("data/features/master_dataset.csv", low_memory=False)

# 1. Find coordinate columns
print("=== COORDINATE COLUMNS ===")
coord_candidates = [c for c in df.columns if any(x in c.lower() 
    for x in ['lat','lon','x','y','centr','coord','easting','northing'])]
print(f"Candidates: {coord_candidates[:30]}")

geo_cols = [c for c in df.columns if 'geo' in c.lower()]
grid_cols = [c for c in df.columns if 'grid' in c.lower()]
print(f"Geo cols: {geo_cols[:10]}")
print(f"Grid cols: {grid_cols[:10]}")

# Try to find actual lat/lon
for pattern in [('centroid_lat','centroid_lon'), ('latitude','longitude'),
                ('lat','lon'), ('y','x'), ('grid_lat','grid_lon')]:
    if pattern[0] in df.columns and pattern[1] in df.columns:
        lat_col, lon_col = pattern
        print(f"\nUsing: {lat_col}, {lon_col}")
        break
else:
    # Check all numeric columns for lat/lon range
    for c in df.select_dtypes(include='number').columns:
        vals = df[c].dropna()
        if len(vals) > 0 and 40 < vals.mean() < 60 and vals.std() < 5:
            print(f"  Possible lat: {c} (mean={vals.mean():.2f}, std={vals.std():.4f})")
        if len(vals) > 0 and 50 < vals.mean() < 90 and vals.std() < 10:
            print(f"  Possible lon: {c} (mean={vals.mean():.2f}, std={vals.std():.4f})")
    lat_col, lon_col = None, None

if lat_col and lon_col:
    df_coords = df[['field_name', 'farm', lat_col, lon_col]].copy()
    df_coords = df_coords.dropna(subset=[lat_col, lon_col])
    
    print(f"\n=== DATASET OVERVIEW ===")
    print(f"Total samples: {len(df_coords)}")
    print(f"Fields: {df_coords['field_name'].nunique()}")
    print(f"Farms: {df_coords['farm'].nunique()}")
    print(f"Lat range: {df_coords[lat_col].min():.4f} - {df_coords[lat_col].max():.4f}")
    print(f"Lon range: {df_coords[lon_col].min():.4f} - {df_coords[lon_col].max():.4f}")
    
    # 2. Compute field centroids
    field_centroids = df_coords.groupby('field_name').agg({
        lat_col: 'mean', lon_col: 'mean', 'farm': 'first'
    }).reset_index()
    field_centroids.columns = ['field', 'lat', 'lon', 'farm']
    
    # 3. Compute pairwise distances between field centroids (in km, approximate)
    lat_rad = np.radians(field_centroids['lat'].values)
    lon_rad = np.radians(field_centroids['lon'].values)
    
    # Haversine distance matrix
    n = len(field_centroids)
    dist_km = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dlat = lat_rad[j] - lat_rad[i]
            dlon = lon_rad[j] - lon_rad[i]
            a = np.sin(dlat/2)**2 + np.cos(lat_rad[i])*np.cos(lat_rad[j])*np.sin(dlon/2)**2
            dist_km[i,j] = dist_km[j,i] = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    farms = field_centroids['farm'].values
    
    # 4. Intra-farm distances (between fields of the SAME farm)
    print(f"\n=== INTRA-FARM DISTANCES (same farm, different fields) ===")
    intra_dists = []
    for farm in np.unique(farms):
        idx = np.where(farms == farm)[0]
        if len(idx) > 1:
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    intra_dists.append(dist_km[idx[i], idx[j]])
    
    intra_dists = np.array(intra_dists)
    print(f"N pairs: {len(intra_dists)}")
    print(f"Min:    {intra_dists.min():.2f} km")
    print(f"Median: {np.median(intra_dists):.2f} km")
    print(f"Mean:   {intra_dists.mean():.2f} km")
    print(f"Max:    {intra_dists.max():.2f} km")
    print(f"P25:    {np.percentile(intra_dists, 25):.2f} km")
    print(f"P75:    {np.percentile(intra_dists, 75):.2f} km")
    print(f"<1km:   {(intra_dists < 1).sum()} ({100*(intra_dists < 1).mean():.1f}%)")
    print(f"<5km:   {(intra_dists < 5).sum()} ({100*(intra_dists < 5).mean():.1f}%)")
    print(f"<10km:  {(intra_dists < 10).sum()} ({100*(intra_dists < 10).mean():.1f}%)")
    
    # 5. Inter-farm distances (between fields of DIFFERENT farms)
    print(f"\n=== INTER-FARM DISTANCES (different farms) ===")
    inter_dists = []
    for i in range(n):
        for j in range(i+1, n):
            if farms[i] != farms[j]:
                inter_dists.append(dist_km[i, j])
    
    inter_dists = np.array(inter_dists)
    print(f"N pairs: {len(inter_dists)}")
    print(f"Min:    {inter_dists.min():.2f} km")
    print(f"Median: {np.median(inter_dists):.2f} km")
    print(f"Mean:   {inter_dists.mean():.2f} km")
    print(f"Max:    {inter_dists.max():.2f} km")
    
    # 6. For each field: nearest neighbor distance (to any other field)
    print(f"\n=== NEAREST NEIGHBOR (field → closest field) ===")
    nn_same_farm = []
    nn_diff_farm = []
    for i in range(n):
        # Nearest within same farm
        same_idx = [j for j in range(n) if j != i and farms[j] == farms[i]]
        if same_idx:
            nn_same_farm.append(min(dist_km[i, j] for j in same_idx))
        
        # Nearest in different farm
        diff_idx = [j for j in range(n) if farms[j] != farms[i]]
        if diff_idx:
            nn_diff_farm.append(min(dist_km[i, j] for j in diff_idx))
    
    nn_same_farm = np.array(nn_same_farm)
    nn_diff_farm = np.array(nn_diff_farm)
    
    print(f"Nearest SAME-farm field:  median={np.median(nn_same_farm):.2f} km, "
          f"mean={nn_same_farm.mean():.2f} km, max={nn_same_farm.max():.2f} km")
    print(f"Nearest DIFF-farm field:  median={np.median(nn_diff_farm):.2f} km, "
          f"mean={nn_diff_farm.mean():.2f} km, min={nn_diff_farm.min():.2f} km")
    
    # Only compute ratio for fields that have same-farm neighbors
    # Some fields may be the only field in their farm
    nn_paired = []
    for i in range(n):
        same_idx = [j for j in range(n) if j != i and farms[j] == farms[i]]
        diff_idx = [j for j in range(n) if farms[j] != farms[i]]
        if same_idx and diff_idx:
            d_same = min(dist_km[i, j] for j in same_idx)
            d_diff = min(dist_km[i, j] for j in diff_idx)
            nn_paired.append((d_same, d_diff))
    
    nn_paired = np.array(nn_paired)
    ratio = nn_paired[:, 1] / (nn_paired[:, 0] + 1e-6)
    print(f"\nRatio (nearest-diff / nearest-same): "
          f"median={np.median(ratio):.1f}x, mean={ratio.mean():.1f}x")
    print(f"Fields where nearest-diff < nearest-same: "
          f"{(nn_paired[:,1] < nn_paired[:,0]).sum()} / {len(nn_paired)}")
    
    # 7. Per-farm breakdown
    print(f"\n=== PER-FARM BREAKDOWN ===")
    print(f"{'Farm':<25} {'Fields':>6} {'Samples':>8} {'IntraD_med':>10} {'Nearest_other':>14}")
    for farm in sorted(np.unique(farms)):
        farm_idx = np.where(farms == farm)[0]
        n_fields = len(farm_idx)
        n_samples = (df_coords['farm'] == farm).sum()
        
        if n_fields > 1:
            intra = []
            for i in range(len(farm_idx)):
                for j in range(i+1, len(farm_idx)):
                    intra.append(dist_km[farm_idx[i], farm_idx[j]])
            med_intra = np.median(intra)
        else:
            med_intra = 0
        
        # Nearest field from a different farm
        nearest_other = min(
            dist_km[fi, fj] 
            for fi in farm_idx 
            for fj in range(n) 
            if farms[fj] != farm
        )
        
        print(f"{str(farm):<25} {n_fields:>6} {n_samples:>8} {med_intra:>10.2f} {nearest_other:>14.2f}")
    
    # 8. Key question: in Field-LOFO, when we hold out one field,
    # how close is the nearest training point?
    print(f"\n=== FIELD-LOFO SPATIAL BUFFER ===")
    print("When a field is held out, nearest training sample comes from:")
    print("  Same farm (adjacent field) — this is SPATIAL LEAKAGE if distance is small")
    print()
    
    # For each field, compute: (a) distance to nearest same-farm field,
    # (b) number of same-farm samples still in training
    for i in range(n):
        field = field_centroids.iloc[i]['field']
        farm = farms[i]
        same_farm_idx = [j for j in range(n) if j != i and farms[j] == farm]
        n_same_farm_fields = len(same_farm_idx)
        if same_farm_idx:
            nearest_same = min(dist_km[i, j] for j in same_farm_idx)
        else:
            nearest_same = float('inf')
        
        n_same_farm_samples = df_coords[
            (df_coords['farm'] == farm) & (df_coords['field_name'] != field)
        ].shape[0]
    
    # Summary statistics
    buffer_distances = []
    for i in range(n):
        same_farm_idx = [j for j in range(n) if j != i and farms[j] == farm]
        if same_farm_idx:
            buffer_distances.append(min(dist_km[i, j] for j in same_farm_idx))
    
    # Simpler: just report the fraction of fields where nearest training data is < X km
    print("Fraction of fields where nearest TRAINING sample (Field-LOFO) is:")
    for thresh in [0.5, 1, 2, 5, 10, 20]:
        frac = (nn_same_farm < thresh).mean()
        print(f"  < {thresh:>4} km: {100*frac:.1f}% of fields")
    
    print(f"\n=== CONCLUSION ===")
    med_intra = np.median(nn_same_farm)
    med_inter = np.median(nn_diff_farm)
    if med_intra < 5:
        print(f"⚠ SPATIAL LEAKAGE LIKELY in Field-LOFO:")
        print(f"  Median nearest same-farm field: {med_intra:.2f} km")
        print(f"  Median nearest diff-farm field: {med_inter:.2f} km")
        print(f"  Fields within the same farm are {med_inter/med_intra:.1f}x closer")
        print(f"  → Farm-LOFO is the correct strict strategy")
    else:
        print(f"  Intra-farm distances ({med_intra:.1f} km) are large enough")
        print(f"  → Field-LOFO may be acceptable")
else:
    print("\nCould not find lat/lon columns. Listing all columns:")
    for c in df.columns:
        print(f"  {c}")
