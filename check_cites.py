#!/usr/bin/env python3
"""Check which citation keys in article2 sections are missing from references.bib."""
import re, glob

# Get all citation keys from tex files
keys = set()
for f in sorted(glob.glob('articles/article2_prediction/sections/*.tex')):
    with open(f) as fh:
        text = fh.read()
    for m in re.finditer(r'\\cite[tp]\{([^}]+)\}', text):
        for k in m.group(1).split(','):
            keys.add(k.strip())

# Get all bib entry keys from references.bib
bib_keys = set()
with open('references.bib') as fh:
    for line in fh:
        m = re.match(r'@\w+\{(\S+?),', line)
        if m:
            bib_keys.add(m.group(1))

keys = sorted(keys)
missing = sorted(k for k in keys if k not in bib_keys)

print(f'=== ALL UNIQUE CITATION KEYS ({len(keys)}) ===')
for k in keys:
    status = '  [MISSING]' if k not in bib_keys else ''
    print(f'  {k}{status}')

print(f'\n=== MISSING FROM references.bib ({len(missing)}) ===')
for k in missing:
    print(f'  {k}')

print(f'\n=== PRESENT ({len(keys) - len(missing)}) ===')
for k in keys:
    if k in bib_keys:
        print(f'  {k}')
