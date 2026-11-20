"""
Ingestion module: fetches incident data from status page APIs for
GitHub, Cloudflare, OpenAI, and Anthropic.
Saves raw JSON dumps to data/raw/ for the preprocessing stage.
"""

import time
import json
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# Config


SOURCES = [
    {
        "name": "github",
        "type": "statuspage_api",
        "base_url": "https://www.githubstatus.com/api/v2/incidents.json",
    },
    {
        "name": "cloudflare",
        "type": "statuspage_api",
        "base_url": "https://www.cloudflarestatus.com/api/v2/incidents.json",
    },
    {
        "name": "openai",
        "type": "statuspage_api",
        "base_url": "https://status.openai.com/api/v2/incidents.json",
    },
    {
        "name": "anthropic",
        "type": "statuspage_api",
        "base_url": "https://status.anthropic.com/api/v2/incidents.json",
    },
]

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
REQUEST_TIMEOUT = 15
RETRY_COUNT = 3
RETRY_BACKOFF = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)



# Core fetch logic


def _request_with_retry(url: str):
    """GET a URL with retries + exponential backoff. Returns parsed JSON or None."""
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt}/{RETRY_COUNT} failed for {url}: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF * attempt)
    logger.error(f"All retries failed for {url}")
    return None


def fetch_statuspage_incidents(source: dict) -> list[dict]:
    """Fetches incidents from an Atlassian Statuspage-style JSON API."""
    name = source["name"]
    url = source["base_url"]
    logger.info(f"Fetching incidents for: {name}")

    data = _request_with_retry(url)
    if data is None:
        return []

    incidents = data.get("incidents", [])
    logger.info(f"{name}: fetched {len(incidents)} incidents")
    return incidents


def fetch_html_incidents(source: dict) -> list[dict]:
    """
    Fallback scraper for providers with no public API.
    Not used by the current SOURCES list — kept here in case you add a
    provider later that has no JSON API and needs HTML scraping instead.
    """
    name = source["name"]
    url = source["base_url"]
    logger.info(f"Scraping HTML incidents for: {name}")

    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (incident-intel-bot)"},
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    incidents = []

    for day_block in soup.select(".status-day"):
        date_el = day_block.select_one(".date")
        date_text = date_el.get_text(strip=True) if date_el else None

        for incident_block in day_block.select(".incident-title a"):
            incidents.append({
                "title": incident_block.get_text(strip=True),
                "url": incident_block.get("href"),
                "date": date_text,
                "source": name,
            })

    logger.info(f"{name}: scraped {len(incidents)} incidents from HTML")
    return incidents


def save_raw(source_name: str, incidents: list[dict]) -> None:
    """Writes raw incidents to data/raw/<source_name>.json."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"{source_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(incidents)} incidents -> {out_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> dict:
    """Fetches all configured sources and saves raw dumps. Returns a count summary."""
    counts = {}

    for source in SOURCES:
        source_type = source["type"]

        if source_type == "statuspage_api":
            incidents = fetch_statuspage_incidents(source)
        elif source_type == "html":
            incidents = fetch_html_incidents(source)
        else:
            logger.warning(f"Unknown source type for {source['name']}: {source_type}")
            continue

        save_raw(source["name"], incidents)
        counts[source["name"]] = len(incidents)

    return counts


if __name__ == "__main__":
    summary = run()
    print("\n--- Fetch summary ---")
    for name, count in summary.items():
        print(f"{name}: {count} incidents")
    print(f"Total: {sum(summary.values())}")
# --------------------------------------------------------------------------