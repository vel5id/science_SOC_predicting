import re, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

missing = ['Chen2024', 'Hengl2018', 'Khaledian2020', 'Meyer2019', 'Minasny2016',
           'Nussbaum2018', 'Padarian2019b', 'Poggio2021', 'Riihimaki2021',
           'Sorenson2021', 'Steffens2013', 'TaghizadehMehrjardi2020',
           'Tsakiridis2020', 'Vaysse2015', 'Wang2021', 'Were2015', 'Xu2020',
           'Zeraatpisheh2019', 'Zhang2021b']

root_bib = "../../references.bib"
with open(root_bib) as f:
    root_text = f.read()

# Parse entries from root bib
entries = {}
for m in re.finditer(r'(@\w+\{(\w+),.*?)(?=\n@|\Z)', root_text, re.DOTALL):
    entries[m.group(2)] = m.group(1).strip()

found = []
not_found = []
for key in missing:
    if key in entries:
        found.append(key)
    else:
        not_found.append(key)

print(f"Found {len(found)} of {len(missing)} missing entries in root bib")
if not_found:
    print("NOT FOUND in root bib:", not_found)

if found:
    with open("sections/references.bib", "a") as f:
        f.write("\n\n% === Entries copied from root references.bib ===\n\n")
        for key in found:
            f.write(entries[key] + "\n\n")
    print(f"Appended {len(found)} entries to sections/references.bib")
