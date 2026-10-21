import os, json, requests
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from state import ACHISState

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

# ── ChromaDB setup (runs locally, no server needed)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast

# =========================
# 🔥 ADD THESE GLOBALS
# =========================
 
# how much trust or importance is given to the information retrieved from each source
SOURCE_WEIGHTS = {
    "Open Food Facts": 0.6,
    "Knowledge base": 0.8,
    "Live regulatory search": 0.9,
}

CACHE = {}  # MCP cache


# =========================
# 🔥 MAIN AGENT (UPDATED)
# =========================

def knowledge_agent(state: ACHISState) -> ACHISState:
    ingredients = state.get("decoded_ingredients", [])
    names = []
    for i in ingredients:
        if i.get("name"):   # from ingredient dict, get the "name" key. If it exists, add to names list. This is for querying external sources.
            names.append(i["name"])

    evidence = []  # this will hold all the findings from external sources, which will be fed into reasoning later. Each finding is a dict with keys like "source", "ingredient", "finding", "confidence", etc.

    for name in names[:6]:
        # Open Food Facts
        off = query_open_food_facts(name)
        if off:
            off["confidence"] = SOURCE_WEIGHTS["Open Food Facts"]
            off["type"] = "product"
            evidence.append(off)

        # 2️⃣ Vector DB
        rag_hits = query_vector_db(name)
        for hit in rag_hits:
            hit["confidence"] = SOURCE_WEIGHTS["Knowledge base"]
            hit["type"] = "research"
        evidence.extend(rag_hits)

        # 3️⃣ MCP / Live Search (with caching)
        live_hits = cached_mcp_search(name)
        for hit in live_hits:
            hit["confidence"] = SOURCE_WEIGHTS["Live regulatory search"]
            hit["type"] = "regulatory"
        evidence.extend(live_hits)

    # 🔥 CLEAN PIPELINE
    evidence = deduplicate(evidence)  # This function removes duplicate entries from the evidence list
    evidence = rank_evidence(evidence)[:10] # keep top 10 most relevant pieces of evidence

    banned_status = check_banned_status(names)

    return {
        **state,
        "external_evidence": evidence,
        "banned_status": banned_status,
    }


# =========================
# 🔥 MCP CACHE
# =========================

def cached_mcp_search(name: str):
    if name in CACHE:
        return CACHE[name]

    result = query_mcp_search(name)
    CACHE[name] = result
    return result


# =========================
# 🔥 DEDUPLICATION
# =========================

def deduplicate(evidence):
    seen = set()
    unique = []

    for e in evidence:
        key = e["finding"][:120]  # partial hash
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# =========================
# 🔥 RANKING
# =========================

def rank_evidence(evidence):
    return sorted(
        evidence,
        key=lambda x: (
            x.get("confidence", 0.5),
            x.get("relevance_score", 0.5)
        ),
        reverse=True
    )


# SOURCE 1: OPEN FOOD FACTS API
def query_open_food_facts(ingredient_name: str) -> dict | None:
    """
    Searches Open Food Facts ingredient taxonomy.
    3M+ products, free, no key needed.
    """
    try:
        url = (
            f"https://world.openfoodfacts.org/cgi/search.pl"
            f"?search_terms={requests.utils.quote(ingredient_name)}"
            f"&search_simple=1&action=process&json=1&page_size=1"
        )
        resp = requests.get(url, timeout=5)
        data = resp.json()
        products = data.get("products", [])
        if not products:
            return None

        p = products[0]
        return {  # this dict will be added to evidence list, which will be fed into reasoning later simailr to json is dictionary in python
            "source": "Open Food Facts",
            "ingredient": ingredient_name,
            "finding": (
                f"Found in product: {p.get('product_name', 'Unknown')}. "
                f"Nutriscore: {p.get('nutriscore_grade', 'N/A').upper()}. "
                f"Nova group (processing level): {p.get('nova_group', 'N/A')}. "
                f"Additives: {', '.join(p.get('additives_tags', [])) or 'none listed'}."
            ),
            "nova_group": p.get("nova_group"),
            "nutriscore":  p.get("nutriscore_grade"),
        }
    except Exception:
        return None


# SOURCE 2: CHROMADB VECTOR SEARCH (RAG)
def query_vector_db(ingredient_name: str, top_k: int = 2) -> list[dict]:
    """
    Semantic search over locally embedded knowledge base.
    Populate with ingest_knowledge_base() below.
    """
    try:
        collection = get_or_create_collection()
        if collection.count() == 0:
            return []  # empty DB — skip silently

        query_embedding = embedder.encode(ingredient_name).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist < 0.5:  # cosine distance threshold — skip weak matches
                hits.append({
                    "source": meta.get("source", "Knowledge base"),
                    "ingredient": ingredient_name,
                    "finding": doc,
                    "relevance_score": round(1 - dist, 2),
                })
        return hits
    except Exception:
        return []
  

# SOURCE 3: MCP / LIVE WEB SEARCH

def query_mcp_search(ingredient_name: str) -> list[dict]:
    """
    Uses Gemini with web search grounding OR Exa MCP to fetch
    live regulatory + research info.
    Falls back gracefully if no MCP server is connected.
    """
    try:
        # Option A: Gemini grounding (simplest, no extra infra)
        prompt = f"""
Search for health and safety information about the food ingredient: "{ingredient_name}"

Focus on:
1. Any WHO, EFSA, FSSAI, or FDA classification
2. Known health risks or safe daily limits (ADI)
3. Whether it is banned or restricted in any country

Respond in 2-3 sentences. Cite your source (WHO / EFSA / FDA / FSSAI).
If no reliable information found, say: "No regulatory data found."
"""
        response = llm.invoke(prompt)
        text = response.content.strip()

        if "no regulatory data" in text.lower():
            return []

        return [{
            "source": "Live regulatory search",
            "ingredient": ingredient_name,
            "finding": text,
        }]

    except Exception:
        return []
def check_banned_status(ingredient_names: list[str]) -> str | None:
    """
    Ask Gemini to check if any ingredient is banned in India / EU / US.
    Returns a status string or None.
    """
    if not ingredient_names:
        return None
    try:
        prompt = f"""
Check if any of these food ingredients are banned or restricted in India (FSSAI), EU (EFSA), or US (FDA):
{', '.join(ingredient_names)}

If any are banned, respond: "BANNED: <ingredient> — <reason> (<country/region>)"
If none are banned, respond: "None banned"
Keep it to one line per banned ingredient.
"""
        response = llm.invoke(prompt)
        text = response.content.strip()
        if text.lower().startswith("none"):
            return None
        return text
    except Exception:
        return None
