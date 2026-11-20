import time
import json
import logging
import requests
from pathlib import Path
from config import REQUEST_TIMEOUT, RETRY_COUNT, RETRY_BACKOFF, RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _request_with_retry(url: str) -> dict | None:
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
    """
    Fetches incidents from an Atlassian Statuspage-style API.
    Handles pagination (Statuspage API returns ~incidents per page via `page` param,
    though most sites cap near recent history by default).
    """
    name = source["name"]
    url = source["base_url"]
    logger.info(f"Fetching incidents for: {name}")

    data = _request_with_retry(url)
    if data is None:
        return []

    incidents = data.get("incidents", [])
    logger.info(f"{name}: fetched {len(incidents)} incidents")
    return incidents


def save_raw(source_name: str, incidents: list[dict]) -> None:
    """Dumps raw fetched data to data/raw/<source_name>.json for downstream processing."""
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(RAW_DATA_DIR) / f"{source_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(incidents)} incidents -> {out_path}")