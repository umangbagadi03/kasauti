import sys
sys.path.append("..")
from agents.knowledge_agent import ingest_knowledge_base
import requests, json

def fetch_off_ingredients():
    """Pull ingredient taxonomy from Open Food Facts."""
    url = "https://world.openfoodfacts.org/data/taxonomies/ingredients.json"
    print("Fetching Open Food Facts taxonomy...")
    data = requests.get(url, timeout=30).json()
    entries = []
    for key, val in list(data.items())[:5000]:  # top 5k entries
        name = key.replace("en:", "").replace("-", " ")
        desc = val.get("wikidata", {})
        text = f"{name}: food ingredient. {json.dumps(desc)[:200]}"
        entries.append({"id": f"off_{key}", "text": text, "source": "Open Food Facts"})
    return entries

def load_manual_entries():
    """Your curated high-priority entries — E-numbers, FSSAI list etc."""
    return [
        {"id": "e621", "source": "EFSA",
         "text": "E621 (MSG / Monosodium Glutamate): flavor enhancer. ADI: not specified (EFSA 2017). Generally recognized as safe but linked to headaches in sensitive individuals. Permitted in India under FSSAI."},
        {"id": "e471", "source": "EFSA",
         "text": "E471 (Mono and diglycerides of fatty acids): emulsifier. Derived from palm/soy oil. EFSA considers safe at current levels. May contain trans fats depending on source."},
        {"id": "e110", "source": "EFSA",
         "text": "E110 (Sunset Yellow FCF): artificial color. ADI: 4 mg/kg body weight (EFSA). Linked to hyperactivity in children. Banned in Norway and Finland."},
        {"id": "e250", "source": "WHO",
         "text": "E250 (Sodium Nitrite): preservative in processed meats. WHO classifies processed meat as Group 1 carcinogen. Linked to colorectal cancer at high consumption."},
        {"id": "e951", "source": "FDA",
         "text": "E951 (Aspartame): artificial sweetener. ADI: 40 mg/kg (EFSA), 50 mg/kg (FDA). IARC classified as possibly carcinogenic (Group 2B) in 2023. Avoid in phenylketonuria."},
        {"id": "hfcs", "source": "Research",
         "text": "High Fructose Corn Syrup (HFCS): sweetener linked to obesity, insulin resistance, and non-alcoholic fatty liver disease at high consumption. Not separately listed on Indian labels."},
    ]

if __name__ == "__main__":
    entries = load_manual_entries()

    # Optionally add OFF taxonomy (large, takes ~2 min)
    # entries += fetch_off_ingredients()

    ingest_knowledge_base(entries)
    print("Done. ChromaDB ready at ./chroma_db")