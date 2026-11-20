"""
Preprocessing module: loads raw incident JSON from data/raw/, normalizes
every source into one consistent schema, cleans text fields, deduplicates,
and saves the result to data/processed/incidents_clean.json.
"""

import json
import logging
import re
from pathlib import Path
from datetime import datetime

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "incidents_clean.json"

SOURCE_FILES = ["github", "cloudflare", "openai", "anthropic"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading raw files
# ---------------------------------------------------------------------------

def load_raw(source_name: str) -> list[dict]:
    """Loads a raw JSON file from data/raw/<source_name>.json."""
    path = RAW_DATA_DIR / f"{source_name}.json"
    if not path.exists():
        logger.warning(f"No raw file found for {source_name}: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} raw incidents from {source_name}")
    return data


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def clean_text(text: str | None) -> str:
    """Strips HTML tags, collapses whitespace, trims."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)       # strip HTML tags
    text = re.sub(r"\s+", " ", text)           # collapse whitespace/newlines
    return text.strip()


def parse_datetime(value: str | None) -> str | None:
    """Normalizes ISO timestamps to a consistent format, or returns None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.isoformat()
    except ValueError:
        logger.warning(f"Could not parse datetime: {value}")
        return None


# ---------------------------------------------------------------------------
# Normalization (all 4 sources share the Atlassian Statuspage schema)
# ---------------------------------------------------------------------------

def normalize_incident(raw: dict, source_name: str) -> dict:
    """
    Converts one raw Statuspage-style incident into the unified schema:
    id, source, title, impact, status, created_at, resolved_at,
    components, updates_text (merged narrative of all updates).
    """
    updates = raw.get("incident_updates", []) or []
    # Oldest first, so the narrative reads chronologically
    updates_sorted = sorted(updates, key=lambda u: u.get("created_at") or "")

    updates_text = " ".join(
        clean_text(u.get("body")) for u in updates_sorted if u.get("body")
    )

    components = [
        c.get("name") for c in (raw.get("components") or []) if c.get("name")
    ]

    return {
        "id": raw.get("id"),
        "source": source_name,
        "title": clean_text(raw.get("name")),
        "impact": raw.get("impact"),              # none/minor/major/critical
        "status": raw.get("status"),               # resolved/investigating/etc.
        "created_at": parse_datetime(raw.get("created_at")),
        "resolved_at": parse_datetime(raw.get("resolved_at")),
        "shortlink": raw.get("shortlink"),
        "components": components,
        "updates_text": updates_text,
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(incidents: list[dict]) -> list[dict]:
    """Removes duplicates by (source, id) pair — safe since ids are per-source."""
    seen = set()
    unique = []
    for inc in incidents:
        key = (inc["source"], inc["id"])
        if key not in seen:
            seen.add(key)
            unique.append(inc)
    removed = len(incidents) - len(unique)
    if removed:
        logger.info(f"Removed {removed} duplicate incidents")
    return unique


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> list[dict]:
    all_incidents = []

    for source_name in SOURCE_FILES:
        raw_incidents = load_raw(source_name)
        normalized = [normalize_incident(r, source_name) for r in raw_incidents]
        all_incidents.extend(normalized)

    all_incidents = deduplicate(all_incidents)

    # Drop incidents with no usable title (malformed records)
    before = len(all_incidents)
    all_incidents = [inc for inc in all_incidents if inc["title"]]
    dropped = before - len(all_incidents)
    if dropped:
        logger.info(f"Dropped {dropped} incidents with empty titles")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_incidents, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(all_incidents)} cleaned incidents -> {OUTPUT_FILE}")
    return all_incidents


if __name__ == "__main__":
    result = run()
    print(f"\nTotal cleaned incidents: {len(result)}")

    # Print one sample so you can eyeball the schema
    if result:
        print("\n--- Sample cleaned incident ---")
        print(json.dumps(result[0], indent=2, ensure_ascii=False)[:800])