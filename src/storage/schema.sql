CREATE TABLE IF NOT EXISTS incidents (
    pk              SERIAL PRIMARY KEY,
    id              TEXT NOT NULL,
    source          TEXT NOT NULL,
    title           TEXT NOT NULL,
    impact          TEXT,
    status          TEXT,
    created_at      TIMESTAMPTZ,
    resolved_at     TIMESTAMPTZ,
    shortlink       TEXT,
    components      TEXT[],
    updates_text    TEXT,
    UNIQUE (source, id)
);

CREATE INDEX IF NOT EXISTS idx_incidents_source ON incidents (source);
CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents (created_at);
CREATE INDEX IF NOT EXISTS idx_incidents_impact ON incidents (impact);