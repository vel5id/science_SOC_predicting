import urllib.request, json, urllib.parse, re, time

EXISTING_DOIS = set()
EXISTING_KEYS = {
    'Attarzadeh2018','Bartholomeus2008','Bauer2019','Behrens2018','BenDor2009',
    'Breiman2001','Castaldi2019','Chen2016','Dahhani2024','Dorogush2018',
    'Drusch2012','Dvorakova2021','Farr2007','Fick2017','Friedman2001','Gao2017',
    'Garg2020','Gebbers2010','Geurts2006','Gorelick2017','Grinsztajn2022',
    'Hajj2018','Haralick1973','Hastie2009','He2016','Hengl2017','Hu2018',
    'Hughes1968','Kakhani2024','Kraemer2015','Li2022','Li2023','Liu2022',
    'Lu2012','McBratney2003','Meyer2021','Montanarella2016','Morellos2016',
    'MunozSabater2021','Möller2022','Nath2021','Nawar2020','Padarian2019',
    'Paszke2019','Pedregosa2011','Pribyl2010','Roberts2017','Rohmer2024',
    'Roy2014','Scherer2001','Shirani2024','Sims1996','Smola2004','Swinnen2017',
    'Torres2012','Tucker1979','Vaudour2019','ViscarraRossel2006','ViscarraRossel2010',
    'Wadoux2019','Wadoux2020','Wadoux2021','Wang2023SSL4EO','Xu2023','Yang2023',
    'Zeng2022','Zhang2021','Zhong2024','Zizala2019','Žížala2019',
}

QUERIES = [
    "soil pH prediction remote sensing machine learning",
    "spatial cross-validation soil mapping leave-one-out",
    "transfer learning remote sensing soil properties",
    "SHAP feature importance soil digital mapping",
    "soil nitrogen prediction satellite imagery",
    "potassium soil mapping multispectral",
    "phosphorus soil remote sensing cropland",
    "Central Asia Kazakhstan soil agriculture degradation",
    "SAR backscatter soil properties prediction",
    "SoilGrids global soil mapping validation",
    "bare soil composite satellite soil organic carbon",
    "hyperspectral soil organic carbon regression",
    "confounding variable soil spectral remote sensing",
    "precision agriculture soil spatial variability",
    "XGBoost gradient boosting soil prediction",
    "convolutional neural network remote sensing regression",
    "chernozem soil carbon organic matter northern",
    "soil texture clay content mapping Sentinel",
    "area of applicability prediction model extrapolation",
    "stacking ensemble model soil geochemical prediction",
]

def fetch_crossref(query, n=5):
    q = urllib.parse.quote(query)
    url = (f'https://api.crossref.org/works?query.bibliographic={q}'
           f'&select=DOI,title,author,container-title,published,volume,issue,page,type'
           f'&rows={n}&filter=type:journal-article')
    req = urllib.request.Request(url, headers={'User-Agent': 'mailto:research@example.com'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())['message']['items']
    except Exception as e:
        print(f"  ERR: {e}")
        return []

def verify_doi(doi):
    url = f'https://api.crossref.org/works/{urllib.parse.quote(doi)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'mailto:research@example.com'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            msg = data['message']
            return msg.get('title', [''])[0], True
    except:
        return '', False

def make_key(authors, year):
    if not authors:
        return f'Unknown{year}'
    fam = authors[0].get('family', 'Unknown').replace(' ', '').replace('-', '')
    # strip non-ascii
    fam = ''.join(c for c in fam if c.isalpha() or c.isdigit())
    return f'{fam}{year}'

def make_bibtex(item):
    doi = item.get('DOI', '').strip()
    if not doi or doi.lower() in EXISTING_DOIS:
        return None, None

    title = item.get('title', [''])[0].strip()
    authors = item.get('author', [])
    journal = item.get('container-title', [''])[0].strip()
    year_parts = item.get('published', {}).get('date-parts', [[None]])[0]
    year = year_parts[0] if year_parts else None
    volume = item.get('volume', '')
    issue = item.get('issue', '')
    page = item.get('page', '')

    if not (title and authors and journal and year):
        return None, None

    author_str = ' and '.join(
        f"{a.get('family', '')}, {a.get('given', '')}"
        for a in authors if 'family' in a
    )
    if not author_str:
        return None, None

    key = make_key(authors, year)
    if key in EXISTING_KEYS:
        key = key + 'b'
    if key in EXISTING_KEYS:
        return None, None

    bib = f'\n@article{{{key},\n'
    bib += f'  author  = {{{author_str}}},\n'
    bib += f'  title   = {{{{{title}}}}},\n'
    bib += f'  journal = {{{journal}}},\n'
    if volume: bib += f'  volume  = {{{volume}}},\n'
    if issue:  bib += f'  number  = {{{issue}}},\n'
    if page:   bib += f'  pages   = {{{page}}},\n'
    bib += f'  year    = {{{year}}},\n'
    bib += f'  doi     = {{{doi}}},\n'
    bib += '}\n'
    return key, bib

collected = []
seen_dois = set()

for query in QUERIES:
    if len(collected) >= 20:
        break
    print(f"  Query: {query[:60]}")
    items = fetch_crossref(query, n=8)
    for item in items:
        if len(collected) >= 20:
            break
        doi = item.get('DOI', '').strip().lower()
        if not doi or doi in seen_dois or doi in EXISTING_DOIS:
            continue
        key, bib = make_bibtex(item)
        if key and bib:
            # quick verify
            _, ok = verify_doi(item['DOI'])
            if ok:
                collected.append((key, bib, item.get('title',[''])[0]))
                seen_dois.add(doi)
                EXISTING_KEYS.add(key)
                print(f"    + [{key}] {item.get('title',[''])[0][:70]}")
        time.sleep(0.3)
    time.sleep(0.5)

print(f"\nTotal collected: {len(collected)}")

# Write to new_20_refs.bib
with open('new_20_refs.bib', 'w', encoding='utf-8') as f:
    f.write('% === 20 new references ===\n')
    for key, bib, title in collected:
        f.write(bib)

print("Saved to new_20_refs.bib")
