import urllib.request, json, urllib.parse, time

EXISTING_KEYS = {
    "Attarzadeh2018","Bartholomeus2008","Bauer2019","Behrens2018","BenDor2009",
    "Breiman2001","Castaldi2019","Chen2016","Dahhani2024","Dorogush2018",
    "Drusch2012","Dvorakova2021","Farr2007","Fick2017","Friedman2001","Gao2017",
    "Garg2020","Gebbers2010","Geurts2006","Gorelick2017","Grinsztajn2022",
    "Hajj2018","Haralick1973","Hastie2009","He2016","Hengl2017","Hu2018",
    "Hughes1968","Kakhani2024","Kraemer2015","Li2022","Li2023","Liu2022",
    "Lu2012","McBratney2003","Meyer2021","Montanarella2016","Morellos2016",
    "MunozSabater2021","Moller2022","Nath2021","Nawar2020","Padarian2019",
    "Paszke2019","Pedregosa2011","Pribyl2010","Roberts2017","Rohmer2024",
    "Roy2014","Scherer2001","Shirani2024","Sims1996","Smola2004","Swinnen2017",
    "Torres2012","Tucker1979","Vaudour2019","ViscarraRossel2006","ViscarraRossel2010",
    "Wadoux2019","Wadoux2020","Wadoux2021","Wang2023SSL4EO","Xu2023","Yang2023",
    "Zeng2022","Zhang2021","Zhong2024","Zizala2019","Zizala2019b",
}

# 20 verified DOIs with exact paper titles confirmed by search
FINAL_DOIS = [
    "10.5194/soil-7-217-2021",         # Poggio 2021 - SoilGrids 2.0
    "10.7717/peerj.5518",              # Hengl 2018 - RF generic framework for spatial
    "10.1016/j.ecolmodel.2019.108815", # Meyer 2019 - spatial predictor selection ML
    "10.1016/j.geoderma.2018.09.006",  # Zeraatpisheh 2019 - DSM multiple ML Iran
    "10.1016/j.geoderma.2021.115316",  # Sorenson 2021 - bare soil composite legacy
    "10.1016/j.geoderma.2012.11.011",  # Steffens 2013 - imaging spectroscopy Luvisol
    "10.1016/j.geoderma.2020.114358",  # Xu 2020 - hyperspectral soil carbon fractions
    "10.1016/j.ecolind.2014.12.028",   # Were 2015 - SVR/ANN/RF for SOC mapping
    "10.5194/soil-4-1-2018",           # Nussbaum 2018 - eval DSM approaches
    "10.1016/j.geoderma.2015.07.017",  # Minasny 2016 - brief history DSM
    "10.1016/j.apm.2019.12.016",       # Khaledian 2020 - selecting ML for DSM
    "10.5194/soil-5-79-2019",          # Padarian 2019 - deep learning for DSM
    "10.1016/j.geoderma.2020.114552",  # Taghizadeh 2020 - multi-task CNN vs RF
    "10.1016/j.geoderma.2020.114809",  # Zhang 2020 - self-training semi-supervised
    "10.1016/j.geoderma.2020.114208",  # Tsakiridis 2020 - simultaneous prediction VNIR
    "10.1016/j.geoderma.2021.115366",  # Zhong 2021 - soil properties feature extraction
    "10.1029/2021wr029871",            # Riihimaki 2021 - TWI proxy for soil moisture
    "10.7717/peerj.17836",             # Chen 2024 - SOC estimation RS data-driven
    "10.1109/access.2021.3080689",     # Wang 2021 - SOC Sentinel-2
    "10.1016/j.geodrs.2014.11.003",    # Vaysse 2015 - evaluating DSM GlobalSoilMap
]

def fetch_doi(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:researcher@example.com"})
    try:
        with urllib.request.urlopen(req, timeout=14) as resp:
            return json.loads(resp.read().decode())["message"]
    except Exception as e:
        print(f"    ERR: {e}")
        return None

def make_key(authors, year):
    fam = (authors[0].get("family", "Unknown") if authors else "Unknown")
    fam = "".join(c for c in fam if c.isalpha() or c.isdigit())
    return f"{fam}{year}"

collected = []
seen = set(EXISTING_KEYS)

for doi in FINAL_DOIS:
    print(f"Fetching {doi[:45]}...")
    msg = fetch_doi(doi)
    if not msg:
        print(f"  FAIL: {doi}")
        time.sleep(0.6)
        continue
    title   = msg.get("title", [""])[0].strip()
    authors = msg.get("author", [])
    journal = (msg.get("container-title") or [""])[0].strip()
    y_parts = msg.get("published", {}).get("date-parts", [[None]])[0]
    year    = y_parts[0] if y_parts else None
    volume  = msg.get("volume", "")
    issue   = msg.get("issue", "")
    pages   = msg.get("page", "")
    adoi    = msg.get("DOI", doi)
    first_a = (authors[0].get("family", "?") if authors else "?")
    print(f"  [{first_a} {year}] {title[:65]}")
    if not (title and authors and journal and year):
        print("  SKIP (incomplete)")
        time.sleep(0.6)
        continue
    key = make_key(authors, year)
    sfx = ""
    while key + sfx in seen:
        sfx = chr(ord(sfx)+1) if sfx else "b"
    key += sfx
    seen.add(key)
    author_str = " and ".join(
        f"{a.get('family','')}, {a.get('given','')}"
        for a in authors if "family" in a
    )
    bib  = f"\n@article{{{key},\n"
    bib += f"  author  = {{{author_str}}},\n"
    bib += f"  title   = {{{{{title}}}}},\n"
    bib += f"  journal = {{{journal}}},\n"
    if volume: bib += f"  volume  = {{{volume}}},\n"
    if issue:  bib += f"  number  = {{{issue}}},\n"
    if pages:  bib += f"  pages   = {{{pages}}},\n"
    bib += f"  year    = {{{year}}},\n"
    bib += f"  doi     = {{{adoi}}},\n"
    bib += "}\n"
    collected.append((key, bib, doi, title))
    time.sleep(0.6)

print(f"\n=== COLLECTED {len(collected)} ===")
for k, _, d, t in collected:
    print(f"  [{k}] {t[:65]}")

with open("new_final_20.bib", "w", encoding="utf-8") as f:
    f.write("% === 20 additional references (verified) ===\n")
    for _, bib, _, _ in collected:
        f.write(bib)
print(f"\nSaved to new_final_20.bib")
