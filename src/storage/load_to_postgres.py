"""
Loads data/processed/incidents_clean.json into PostgreSQL.
Creates the incidents table if it doesn't exist, then upserts all records.
"""

import os
import json
import logging
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_FILE = PROJECT_ROOT / "data" / "processed" / "incidents_clean.json"
SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"

load_dotenv(PROJECT_ROOT / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def apply_schema(conn) -> None:
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()
    logger.info("Schema applied (table created if not existing)")


def load_incidents(conn, incidents: list[dict]) -> None:
    rows = [
        (
            inc["id"],
            inc["source"],
            inc["title"],
            inc.get("impact"),
            inc.get("status"),
            inc.get("created_at"),
            inc.get("resolved_at"),
            inc.get("shortlink"),
            inc.get("components") or [],
            inc.get("updates_text"),
        )
        for inc in incidents
    ]

    insert_query = """
        INSERT INTO incidents
            (id, source, title, impact, status, created_at, resolved_at,
             shortlink, components, updates_text)
        VALUES %s
        ON CONFLICT (source, id) DO UPDATE SET
            title = EXCLUDED.title,
            impact = EXCLUDED.impact,
            status = EXCLUDED.status,
            created_at = EXCLUDED.created_at,
            resolved_at = EXCLUDED.resolved_at,
            shortlink = EXCLUDED.shortlink,
            components = EXCLUDED.components,
            updates_text = EXCLUDED.updates_text;
    """

    with conn.cursor() as cur:
        execute_values(cur, insert_query, rows)
    conn.commit()
    logger.info(f"Upserted {len(rows)} incidents into Postgres")


def run() -> None:
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        incidents = json.load(f)
    logger.info(f"Loaded {len(incidents)} incidents from {CLEAN_FILE}")

    conn = get_connection()
    try:
        apply_schema(conn)
        load_incidents(conn, incidents)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM incidents;")
            total = cur.fetchone()[0]
        logger.info(f"Total rows now in incidents table: {total}")
    finally:
        conn.close()


if __name__ == "__main__":
    run()