"""CLI.

  python -m kasauti analyze --image label.jpg --condition diabetes --provider anthropic
  python -m kasauti analyze --text-file samples/biscuit.txt --allergy milk --no-llm
  python -m kasauti analyze --barcode <EAN-13 barcode>
  python -m kasauti providers
  python -m kasauti serve
"""

import argparse
import json
import sys
from pathlib import Path

from .config import PROVIDERS, settings
from .llm import LLMError, available_providers

ICON = {"avoid": "⛔", "high": "🔴", "moderate": "🟠", "low": "🟡"}
VERDICT = {"Safe": "🟢 SAFE", "Moderate": "🟠 MODERATE", "Risky": "🔴 RISKY", "Unknown": "⚪ UNKNOWN"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kasauti", description="Analyse packaged-food labels.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse a label")
    src = a.add_argument_group("input (at least one)")
    src.add_argument("--image", type=Path)
    src.add_argument("--text")
    src.add_argument("--text-file", type=Path)
    src.add_argument("--barcode")
    a.add_argument("--condition", action="append", default=[],
                   help="diabetes, hypertension, kidney_disease, pku, celiac, pregnancy, gout, weight_loss")
    a.add_argument("--allergy", action="append", default=[], help="e.g. milk, peanut, tree_nuts, gluten, soy")
    a.add_argument("--region", default=settings.default_region, help="IN, EU or US")
    a.add_argument("--language", default="English")
    a.add_argument("--provider", choices=PROVIDERS)
    a.add_argument("--model")
    a.add_argument("--no-llm", action="store_true", help="offline: rules only (text/barcode input)")
    a.add_argument("--json", action="store_true", help="print the full JSON result")

    sub.add_parser("providers", help="show which LLM providers are configured")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)

    if args.cmd == "providers":
        for name, info in available_providers().items():
            mark = "✓" if info["configured"] else "✗ (no API key)"
            star = " ← default" if name == settings.provider else ""
            print(f"{name:10} {mark:15} model: {info['default_model']}{star}")
        return 0

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("kasauti.api:app", host=args.host, port=args.port)
        return 0

    from .models import UserProfile
    from .pipeline import analyze

    text = args.text or (args.text_file.read_text() if args.text_file else None)
    image = args.image.read_bytes() if args.image else None
    if not (image or text or args.barcode):
        p.error("give --image, --text, --text-file or --barcode")

    try:
        profile = UserProfile(conditions=args.condition, allergies=args.allergy, region=args.region.upper(), language=args.language)
        r = analyze(image=image, text=text, barcode=args.barcode, profile=profile,
                    provider=args.provider, model=args.model, use_llm=not args.no_llm)
    except (LLMError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(r.model_dump(), indent=2, ensure_ascii=False))
        return 0

    name = " — ".join(x for x in (r.label.brand, r.label.product_name) if x) or "Product"
    print(f"\n{name}\n{VERDICT[r.verdict]}   score {r.score}/10   confidence {r.confidence:.0%}"
          + (f"   via {r.provider}/{r.model}" if r.provider else "   (offline)"))
    print(f"\n{r.explanation}\n")
    order = {"avoid": 0, "high": 1, "moderate": 2, "low": 3}
    for s in sorted(r.signals, key=lambda s: order[s.level]):
        print(f"  {ICON[s.level]} {s.title}\n      {s.detail}")
    for pos in r.positives:
        print(f"  ✅ {pos}")
    if r.tips:
        print("\nTips:\n" + "\n".join(f"  • {t}" for t in r.tips))
    if r.unknown_ingredients:
        print(f"\nNot recognised: {', '.join(r.unknown_ingredients)}")
    for w in r.warnings:
        print(f"⚠️  {w}")
    print("\nInformational only — not medical advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
