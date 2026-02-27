import re

with open('references.bib', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Zizala
content = re.sub(
    r'@article\{Zizala2022,[\s\S]*?\}',
    '''@article{Zizala2019,
  author  = {Žížala, Daniel and Minařík, Robert and Zádorová, Tereza},
  title   = {Soil Organic Carbon Mapping Using Multispectral Remote Sensing Data: Prediction Ability of Data with Different Spatial and Spectral Resolutions},
  journal = {Remote Sensing},
  volume  = {11},
  number  = {24},
  pages   = {2947},
  year    = {2019},
  doi     = {10.3390/rs11242947},
}''',
    content
)

with open('new_refs.bib', 'r', encoding='utf-8') as f:
    new_refs = f.read()

# Extract keys from references.bib
existing_keys = set(re.findall(r'@\w+\{([^,]+),', content))

# Parse new_refs.bib
new_entries = re.findall(r'(@\w+\{([^,]+),[\s\S]*?\n\})', new_refs)

added_count = 0
for entry, key in new_entries:
    if key not in existing_keys and added_count < 20:
        content += '\n' + entry
        existing_keys.add(key)
        added_count += 1

with open('references.bib', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Added {added_count} new references.")
