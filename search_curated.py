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

TARGET_DOIS = [
    "10.5194/soil-7-217-2021",
    "10.7717/peerj.5518",
    "10.1016/j.geoderma.2016.01.003",
    "10.1016/j.geoderma.2015.04.021",
    "10.1016/j.still.2019.104407",
    "10.1016/j.ecolmodel.2019.108815",
    "10.1016/j.geoderma.2018.09.006",
    "10.1016/j.geoderma.2017.01.020",
    "10.1016/j.catena.2019.104418",
    "10.1016/j.catena.2020.104729",
    "10.1016/j.geoderma.2021.115170",
    "10.1016/j.geoderma.2022.116078",
    "10.1016/j.geoderma.2021.115316",
    "10.1016/j.rse.2021.112786",
    "10.1016/j.geoderma.2019.114070",
    "10.1016/j.geoderma.2012.11.011",
    "10.1016/j.geoderma.2020.114358",
    "10.1016/j.geoderma.2019.04.027",
    "10.1016/j.geoderma.2020.114455",
    "10.1016/j.geoderma.2020.114112",
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

for doi in TARGET_DOIS:
    print(f"Fetching {doi} ...")
    msg = fetch_doi(doi)
    if not msg:
        print(f"  FAIL: {doi}")
        time.sleep(0.5)
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
    print(f"  -> [{year}] {title[:70]}")
    if not (title and authors and journal and year):
        print("  SKIP (incomplete)")
        time.sleep(0.5)
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
    bib = f"\n@article{{{key},\n"
    bib += f"  author  = {{{author_str}}},\n"
    bib += f"  title   = {{{{{title}}}}},\n"
    bib += f"  journal = {{{journal}}},\n"
    if volume: bib += f"  volume  = {{{volume}}},\n"
    if issue:  bib += f"  number  = {{{issue}}},\n"
    if pages:  bib += f"  pages   = {{{pages}}},\n"
    bib += f"  year    = {{{year}}},\n"
    bib += f"  doi     = {{{adoi}}},\n"
    bib += "}\n"
    collected.append((key, bib, title))
    time.sleep(0.5)

print(f"\nTotal collected: {len(collected)}")
with open("new_20_curated.bib", "w", encoding="utf-8") as f:
    f.write("% === curated new references ===\n")
    for _k, bib, _t in collected:
        f.write(bib)
print("Saved to new_20_curated.bib")
print()
print("=== SUMMARY ===")
for k, _, t in collected:
    print(f"  [{k}] {t[:70]}")
