import urllib.request, json, urllib.parse, time, sys

# Papers to find by title+author (all from soil/DSM/ML domain)
TARGETS = [
    ("A comparative assessment of support vector regression, artificial neural networks, and random forests for predicting and mapping soil organic carbon stocks", "Were"),
    ("Evaluation of digital soil mapping approaches with large sets of environmental covariates", "Nussbaum"),
    ("Synergetic use of Sentinel-1 Sentinel-2 SRTM and soil texture for digital soil mapping", "Taghizadeh-Mehrjardi"),
    ("Digital soil mapping: A brief history and some lessons", "Minasny"),
    ("Selecting appropriate machine learning methods for digital soil mapping", "Khaledian"),
    ("Improvements to the representativeness of the multivariate environmental space used in digital soil mapping", "Stumpf"),
    ("Predictors of presence and abundance for an obligate grassland bird", "skip"),
    ("Using deep learning for digital soil mapping", "Padarian"),
    ("Uncertainty quantification at the point-of-care: Deep learning-based prediction intervals for spatial and temporal predictions", "skip"),
    ("Random forests in categorical abundance models for digital soil mapping", "Vaysse"),
    ("Modelling and mapping soil properties with limited training data using semi-supervised machine learning", "Yu"),
    ("Spatial uncertainty of grassland soil properties estimated by hyperspectral imaging using machine learning", "skip"),
    ("Quantile Regression Forests", "Meinshausen"),
    ("Predicting soil organic carbon content using satellite remote sensing data with machine learning", "Song"),
    ("Multi-task convolutional neural networks outperformed random forest for mapping soil particle size fractions in cluttered areas", "Kakhani"),
    ("Deep learning convolutional neural networks for the prediction of soil properties from VNIR spectra", "Ng"),
    ("Detection of soil organic matter content using multispectral imagery", "Gomez"),
    ("Topographic wetness index as a proxy for soil moisture: The importance of scale", "Grabs"),
    ("Digital soil mapping using multiple logistic regression on terrain parameters in semi-arid regions", "Nabiollahi"),
    ("Soil nutrient contents and stoichiometry within aggregate fractions responded to cropping history and N fertilization", "skip"),
]

def search_title(title, author):
    if title == "skip":
        return None
    q = f"{title} {author}"
    url = "https://api.crossref.org/works?query.bibliographic=" + urllib.parse.quote(q) + "&rows=3&select=DOI,title,author,container-title,published,volume,issue,page"
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:researcher@example.com"})
    try:
        with urllib.request.urlopen(req, timeout=14) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("message", {}).get("items", [])
        return items
    except Exception as e:
        print(f"  ERR: {e}")
        return []

def make_key(authors, year):
    fam = (authors[0].get("family", "Unknown") if authors else "Unknown")
    fam = "".join(c for c in fam if c.isalpha() or c.isdigit())
    return f"{fam}{year}"

collected = []
for want_title, want_author in TARGETS:
    if want_title == "skip" or want_author == "skip":
        continue
    print(f"\nSearching: {want_title[:60]}...")
    items = search_title(want_title, want_author)
    if not items:
        print("  no results")
        time.sleep(0.5)
        continue
    for item in items:
        t = (item.get("title") or [""])[0]
        a = (item.get("author") or [{}])[0].get("family", "?")
        d = item.get("DOI", "")
        print(f"  [{a}] {t[:65]} | {d}")
    time.sleep(0.5)
print("\nDone")
