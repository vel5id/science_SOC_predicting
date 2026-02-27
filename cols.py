import pandas as pd
df = pd.read_csv("c:/Claude/science_article/data/features/full_dataset.csv")

with open("c:/Claude/science_article/cols.txt", "w") as f:
    for c in df.columns:
        if 'sg' in c:
            f.write(f"{c}\n")

with open("c:/Claude/science_article/info.txt", "w") as f:
    f.write(f"999 cols: {[c for c in df.columns if (df[c] == 999).any()]}\n")
    trace = df[['mg','fe','mo','zn','cu','mn']]
    f.write(f"0 counts trace: {(trace == 0).sum().to_dict()}\n")
