"""Quick DB schema + sample inspection."""
import sqlite3

conn = sqlite3.connect("data/soil_analysis.db")
cur = conn.cursor()

tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)

for t in tables:
    print(f"\n=== {t} ===")
    cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
    for c in cols:
        print(f"  {c[1]:30s} {c[2]}")
    
    count = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  Rows: {count}")

    # Sample data
    rows = cur.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()
    col_names = [c[1] for c in cols]
    for r in rows:
        print("  ---")
        for cn, v in zip(col_names, r):
            print(f"    {cn}: {v}")

# Specific queries for planning
print("\n=== SAMPLING DATES ===")
for r in cur.execute("SELECT DISTINCT sampling_date FROM soil_analysis ORDER BY sampling_date LIMIT 20").fetchall():
    print(f"  {r[0]}")

print("\n=== YEARS ===")
for r in cur.execute("SELECT DISTINCT year FROM soil_analysis ORDER BY year").fetchall():
    print(f"  {r[0]}")

print("\n=== FARMS ===")
for r in cur.execute("SELECT DISTINCT farm FROM soil_analysis ORDER BY farm").fetchall():
    print(f"  {r[0]}")

print("\n=== COORDINATE RANGE ===")
cur.execute("SELECT MIN(centroid_lon), MAX(centroid_lon), MIN(centroid_lat), MAX(centroid_lat) FROM soil_analysis")
r = cur.fetchone()
print(f"  Lon: {r[0]} .. {r[1]}")
print(f"  Lat: {r[2]} .. {r[3]}")

print("\n=== GEOMETRY SAMPLE ===")
for r in cur.execute("SELECT geometry_wkt FROM soil_analysis WHERE geometry_wkt IS NOT NULL LIMIT 2").fetchall():
    print(f"  {str(r[0])[:200]}")

conn.close()
