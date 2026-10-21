"""Curated knowledge: additive/ingredient lookup, label splitting, allergen detection."""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..models import MatchedIngredient

_DIR = Path(__file__).parent

# Additive categories NOVA treats as markers of ultra-processing ("cosmetic" additives).
UPF_CATEGORIES = {
    "color_synthetic", "color_natural", "flavor_enhancer", "flavouring", "flavouring_artificial",
    "emulsifier", "sweetener", "sweetener_polyol", "thickener", "modified_starch", "glazing_agent",
}

# Allergen keywords. Negative lookbehinds stop "cocoa butter" or "coconut milk" counting as dairy.
_NOT_DAIRY = r"(?<!cocoa )(?<!peanut )(?<!shea )(?<!coconut )(?<!almond )(?<!soy )(?<!soya )(?<!oat )(?<!rice )"
ALLERGENS: dict[str, list[str]] = {
    "milk": [_NOT_DAIRY + r"\bmilk", _NOT_DAIRY + r"\bbutter\b", r"\bwhey\b", r"\bcasein", r"\blactose\b",
             r"\bghee\b", r"\bcream\b(?!\s*(?:filling|biscuit))", r"\bcheese\b", r"\bpaneer\b", r"\bcurd\b", r"\byogh?urt\b", r"\bkhoa\b"],
    "egg": [r"\beggs?\b", r"\balbumin\b", r"\begg (?:white|yolk|powder)"],
    "soy": [r"\bsoy", r"\bsoya\b", r"\bsoybeans?\b", r"\btofu\b", r"\bedamame\b"],
    "gluten": [r"\bwheat\b", r"\bmaida\b", r"\batta\b", r"\bsemolina\b", r"\bsooji\b", r"\bsuji\b", r"\brava\b",
               r"\bbarley\b", r"\brye\b", r"\bmalt(?:ed)?\b", r"\bgluten\b", r"\bspelt\b", r"\boats?\b", r"\bseitan\b"],
    "peanut": [r"\bpeanuts?\b", r"\bgroundnuts?\b"],
    "tree_nuts": [r"\b(?:tree )?nuts\b", r"\balmonds?\b", r"\bcashews?\b", r"\bwalnuts?\b", r"\bpistachios?\b", r"\bhazelnuts?\b",
                  r"\bpecans?\b", r"\bmacadamia", r"\bbrazil nuts?\b"],
    "fish": [r"\bfish\b", r"\banchov", r"\btuna\b", r"\bsardines?\b", r"\bsalmon\b"],
    "crustacean": [r"\bshrimps?\b", r"\bprawns?\b", r"\bcrabs?\b", r"\blobsters?\b"],
    "sesame": [r"\bsesame\b", r"\btil\b", r"\bgingelly\b"],
    "mustard": [r"\bmustard\b"],
    "sulphites": [r"\bsulph?ites?\b", r"\bsulfites?\b", r"\bmetabisulph?ite", r"\bsulph?ur dioxide\b", r"\bsulfur dioxide\b"],
    "celery": [r"\bcelery\b"],
}
_ALLERGEN_RE = {k: [re.compile(p, re.I) for p in v] for k, v in ALLERGENS.items()}

# E/INS codes: "E471", "INS 150d", "e-322", or bare "(471)"; never "100%" / "500mg".
_CODE_RE = re.compile(
    r"(?<![\w.])(?:e|ins)?\s?-?\s?(\d{3,4})(?:\s?\(?([a-f])\)?(?![a-z]))?(?:\s?\((?:i|ii|iii|iv|v)\))?"
    r"(?!\s?(?:%|\d|mg\b|g\b|ml\b|kcal\b|kj\b|gm\b))",
    re.I,
)


@dataclass
class _Entry:
    key: str
    name: str
    category: str
    risk: str
    note: str
    code: str | None = None
    sources: list[str] = field(default_factory=list)
    not_permitted_in: list[str] = field(default_factory=list)
    conditions: dict[str, str] = field(default_factory=dict)
    upf: bool = False
    generic: bool = False
    allergen: str | None = None


@dataclass
class KnowledgeBase:
    by_code: dict[str, _Entry]
    phrases: list[tuple[re.Pattern, _Entry]]  # sorted longest phrase first
    whole_foods: list[re.Pattern]

    def entry_for(self, key: str) -> _Entry | None:
        return self.by_code.get(key) or next((e for _, e in self.phrases if e.key == key), None)


def _phrase_re(phrase: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + re.escape(phrase.lower()) + r"(?![a-z])")


@lru_cache(maxsize=1)
def load() -> KnowledgeBase:
    additives = json.loads((_DIR / "additives.json").read_text())
    ingredients = json.loads((_DIR / "ingredients.json").read_text())

    by_code: dict[str, _Entry] = {}
    phrases: list[tuple[str, _Entry]] = []
    for code, a in additives.items():
        if code.startswith("_"):
            continue
        e = _Entry(
            key=f"E{code.upper()}", name=a["name"], category=a["category"], risk=a["risk"], note=a["note"],
            code=f"E{code}", sources=a.get("sources", []), not_permitted_in=a.get("not_permitted_in", []),
            conditions=a.get("conditions", {}), upf=a["category"] in UPF_CATEGORIES, allergen=a.get("allergen"),
        )
        by_code[code.lower()] = e
        for p in [a["name"], *a.get("aliases", [])]:
            phrases.append((p.lower(), e))

    for i in ingredients["entries"]:
        e = _Entry(
            key=i["id"], name=i["name"], category=i["category"], risk=i["risk"], note=i["note"],
            sources=i.get("sources", []), not_permitted_in=i.get("not_permitted_in", []),
            conditions=i.get("conditions", {}), upf=i.get("upf", False) or i["category"] in UPF_CATEGORIES,
            generic=i.get("generic", False),
        )
        for p in i["patterns"]:
            phrases.append((p.lower(), e))

    phrases.sort(key=lambda t: len(t[0]), reverse=True)
    return KnowledgeBase(
        by_code=by_code,
        phrases=[(_phrase_re(p), e) for p, e in phrases],
        whole_foods=sorted((_phrase_re(w) for w in ingredients["whole_foods"]), key=lambda r: -len(r.pattern)),
    )


# ---------------------------------------------------------------- label parsing

def split_ingredients(text: str) -> list[str]:
    """Split an ingredient list on commas/semicolons that are not inside brackets."""
    text = re.sub(r"^\s*ingredients?\s*[:\-]\s*", "", text.strip(), flags=re.I)
    parts, depth, buf = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip(" .\n\t") for p in parts if p.strip(" .\n\t")]


def _leaves(item: str) -> tuple[str, list[str]]:
    """'Emulsifiers (322, 471)' -> ('Emulsifiers', ['322', '471'])."""
    m = re.match(r"^(.*?)[\(\[](.*)[\)\]]\s*(.*)$", item, re.S)
    if not m:
        return item, []
    outer = (m.group(1) + " " + m.group(3)).strip()
    inner = m.group(2)
    # "Caramel colour (150d)" / "Sugar (12%)" — the bracket holds a code or a percentage
    children = split_ingredients(inner) if inner.strip() else []
    children = [c for c in children if not re.fullmatch(r"[\d.]+\s?%", c)]
    return outer, children


@dataclass
class MatchOutcome:
    matched: list[MatchedIngredient]
    unknown: list[str]
    positions: dict[str, int]  # entry key -> first top-level position (0 = first ingredient)
    entries: dict[str, _Entry]


def match_ingredients(items: list[str]) -> MatchOutcome:
    kb = load()
    matched: dict[str, MatchedIngredient] = {}
    positions: dict[str, int] = {}
    entries: dict[str, _Entry] = {}
    unknown: list[str] = []

    def add(e: _Entry, original: str, pos: int):
        if e.key not in matched:
            matched[e.key] = MatchedIngredient(
                original=original, name=e.name, code=e.code, category=e.category, risk=e.risk,
                note=e.note, sources=e.sources, not_permitted_in=e.not_permitted_in,
            )
            positions[e.key] = pos
            entries[e.key] = e

    def match_leaf(text: str, pos: int, has_children: bool) -> bool:
        low = text.lower().strip()
        if not low:
            return True
        found = False
        for m in _CODE_RE.finditer(low):
            num, letter = m.group(1), (m.group(2) or "").lower()
            e = kb.by_code.get(num + letter) or kb.by_code.get(num)
            if e:
                add(e, text, pos)
                found = True
        if found:
            return True
        for rx, e in kb.phrases:
            if rx.search(low):
                if e.generic and has_children:
                    return True  # "Emulsifiers (...)": the children say which ones
                add(e, text, pos)
                return True
        if any(rx.search(low) for rx in kb.whole_foods):
            return True
        return has_children  # a bare group label like "Cream filling (...)" isn't "unknown"

    for pos, item in enumerate(items):
        outer, children = _leaves(item)
        outer_ok = match_leaf(outer, pos, bool(children))
        child_ok = [match_leaf(c, pos, False) for c in children]
        if not outer_ok:
            unknown.append(outer)
        unknown.extend(c for c, ok in zip(children, child_ok) if not ok)

    return MatchOutcome(list(matched.values()), list(dict.fromkeys(unknown)), positions, entries)


def detect_allergens(items: list[str], statements: list[str]) -> dict[str, str]:
    """Return {allergen: "contains" | "may_contain"}."""
    found: dict[str, str] = {}

    def scan(text: str, level: str):
        for allergen, patterns in _ALLERGEN_RE.items():
            if any(p.search(text) for p in patterns) and found.get(allergen) != "contains":
                found[allergen] = level

    for item in items:
        scan(item, "contains")
    trace = r"may contain|traces? of|processed in a facility|same (?:facility|line)"
    for s in statements:
        # "Contains: wheat, soy. May contain nuts." -> only the part after the trace marker is "may contain"
        head, *tail = re.split(f"(?={trace})", s, maxsplit=1, flags=re.I)
        scan(head, "contains")
        if tail:
            scan(tail[0], "may_contain")
    return found
