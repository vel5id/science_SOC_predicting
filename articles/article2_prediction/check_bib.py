import re, glob, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
keys = set()
for f in glob.glob("sections/*.tex"):
    with open(f) as fh:
        for m in re.finditer(r"\\cite[tp]*\{([^}]+)\}", fh.read()):
            for k in m.group(1).split(","):
                keys.add(k.strip())

with open("sections/references.bib") as bib:
    bib_text = bib.read()
bib_keys = set(re.findall(r"@\w+\{(\w+),", bib_text))

missing = sorted(keys - bib_keys)
print(f"Cited: {len(keys)}, In bib: {len(bib_keys)}")
if missing:
    print("MISSING:", missing)
else:
    print("All keys found!")
