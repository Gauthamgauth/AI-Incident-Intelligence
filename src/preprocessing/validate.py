"""
Validates the cleaned incident dataset for missing/bad timestamps and
other data-quality issues before loading into storage.
"""

import json
import logging
from pathlib import Path
from collections import Counter

PROCESSED_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "incidents_clean.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_clean() -> list[dict]:
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(incidents: list[dict]) -> None:
    total = len(incidents)

    missing_created = [i for i in incidents if not i.get("created_at")]
    missing_resolved = [i for i in incidents if not i.get("resolved_at")]
    missing_title = [i for i in incidents if not i.get("title")]
    missing_updates = [i for i in incidents if not i.get("updates_text")]

    # resolved_at missing but status says resolved -> actual data problem
    inconsistent_resolved = [
        i for i in incidents
        if i.get("status") == "resolved" and not i.get("resolved_at")
    ]

    status_counts = Counter(i.get("status") for i in incidents)
    source_counts = Counter(i.get("source") for i in incidents)

    print(f"Total incidents: {total}\n")

    print(f"Missing created_at : {len(missing_created)}  "
          f"({len(missing_created)/total:.1%})  <- should be ~0, this is a parse failure")
    print(f"Missing resolved_at: {len(missing_resolved)}  "
          f"({len(missing_resolved)/total:.1%})  <- expected for ongoing/unresolved incidents")
    print(f"Missing title       : {len(missing_title)}")
    print(f"Missing updates_text: {len(missing_updates)}")
    print(f"Inconsistent (status=resolved but resolved_at missing): {len(inconsistent_resolved)}")

    print(f"\nStatus breakdown: {dict(status_counts)}")
    print(f"Source breakdown: {dict(source_counts)}")

    if missing_created:
        print("\n--- Incidents missing created_at (first 5) ---")
        for inc in missing_created[:5]:
            print(f"  [{inc['source']}] id={inc['id']}  title={inc['title']!r}")

    if inconsistent_resolved:
        print("\n--- Inconsistent: resolved but no resolved_at (first 5) ---")
        for inc in inconsistent_resolved[:5]:
            print(f"  [{inc['source']}] id={inc['id']}  title={inc['title']!r}")


if __name__ == "__main__":
    data = load_clean()
    validate(data)