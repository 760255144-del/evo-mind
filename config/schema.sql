-- ============================================================
-- Schema Version Tracking (migration system)
-- ============================================================
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT NOT NULL
);

-- ============================================================
-- Sessions
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    summary         TEXT,
    memory_count    INTEGER NOT NULL DEFAULT 0,
    metadata_json   TEXT DEFAULT '{}'
);

-- ============================================================
-- Core Memory Store
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (
    id                      TEXT PRIMARY KEY,
    memory_type             TEXT NOT NULL CHECK (
                                memory_type IN (
                                    'episodic',
                                    'semantic',
                                    'procedural',
                                    'feedback'
                                )
                            ),
    content_json            TEXT NOT NULL,
    content_hash            TEXT NOT NULL UNIQUE,
    content_plain           TEXT,
    embedding_id            TEXT,
    importance              REAL NOT NULL DEFAULT 0.5
                                CHECK (importance >= 0.0 AND importance <= 1.0),
    access_count            INTEGER NOT NULL DEFAULT 0,
    last_accessed_at        TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    session_id              TEXT REFERENCES sessions(id),
    source                  TEXT NOT NULL DEFAULT 'direct'
                                CHECK (source IN ('direct', 'consolidation', 'deduction', 'plugin', 'import')),
    status                  TEXT NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'consolidating', 'consolidated', 'archived', 'pruned')),
    consolidation_version   INTEGER NOT NULL DEFAULT 0,
    consolidation_run_id    TEXT,
    metadata_json           TEXT DEFAULT '{}',
    deleted_at              TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_mem_type_status   ON memories(memory_type, status);
CREATE INDEX IF NOT EXISTS idx_mem_session       ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_mem_importance    ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_mem_created       ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_mem_accessed      ON memories(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_mem_hash          ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_mem_consolidation ON memories(status, consolidation_version)
    WHERE status IN ('active', 'consolidating');

-- ============================================================
-- FTS5 Full-Text Search Index
-- ============================================================
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content_plain,
    content='memories',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS mem_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content_plain) VALUES (new.rowid, new.content_plain);
END;

CREATE TRIGGER IF NOT EXISTS mem_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content_plain)
        VALUES ('delete', old.rowid, old.content_plain);
END;

CREATE TRIGGER IF NOT EXISTS mem_fts_update AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content_plain)
        VALUES ('delete', old.rowid, old.content_plain);
    INSERT INTO memories_fts(rowid, content_plain) VALUES (new.rowid, new.content_plain);
END;

-- ============================================================
-- Memory Relationships (directed, weighted graph)
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_relationships (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id       TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation_type   TEXT NOT NULL CHECK (
                        relation_type IN (
                            'supersedes',
                            'contradicts',
                            'derives_from',
                            'reinforces',
                            'references',
                            'generalizes',
                            'corrects'
                        )
                    ),
    strength        REAL NOT NULL DEFAULT 1.0
                        CHECK (strength >= 0.0 AND strength <= 1.0),
    evidence_json   TEXT DEFAULT '{}',
    created_at      TEXT NOT NULL,
    metadata_json   TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON memory_relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON memory_relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type   ON memory_relationships(relation_type);

-- ============================================================
-- Tags (flat tagging system)
-- ============================================================
CREATE TABLE IF NOT EXISTS tags (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    tag_id      TEXT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, tag_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_memtag_tag ON memory_tags(tag_id);

-- ============================================================
-- Consolidation Run Records
-- ============================================================
CREATE TABLE IF NOT EXISTS consolidation_runs (
    id                      TEXT PRIMARY KEY,
    started_at              TEXT NOT NULL,
    completed_at            TEXT,
    trigger                 TEXT NOT NULL CHECK (trigger IN ('manual', 'threshold', 'schedule')),
    candidates_count        INTEGER NOT NULL DEFAULT 0,
    groups_formed           INTEGER NOT NULL DEFAULT 0,
    summaries_generated     INTEGER NOT NULL DEFAULT 0,
    duplicates_merged       INTEGER NOT NULL DEFAULT 0,
    memories_pruned         INTEGER NOT NULL DEFAULT 0,
    patterns_extracted      INTEGER NOT NULL DEFAULT 0,
    avg_intracluster_dist   REAL,
    compression_ratio       REAL,
    status                  TEXT NOT NULL DEFAULT 'running'
                                CHECK (status IN ('running', 'completed', 'failed')),
    error_message           TEXT,
    metadata_json           TEXT DEFAULT '{}'
);

-- ============================================================
-- Evolution: Learned Rules
-- ============================================================
CREATE TABLE IF NOT EXISTS evolution_rules (
    id                  TEXT PRIMARY KEY,
    rule_type           TEXT NOT NULL CHECK (
                            rule_type IN (
                                'correction_pattern',
                                'strategy_heuristic',
                                'inferred_knowledge',
                                'procedural_template'
                            )
                        ),
    label               TEXT,
    condition_json      TEXT NOT NULL,
    action_json         TEXT NOT NULL,
    confidence          REAL NOT NULL DEFAULT 0.0
                            CHECK (confidence >= 0.0 AND confidence <= 1.0),
    support_count       INTEGER NOT NULL DEFAULT 0,
    contradiction_count INTEGER NOT NULL DEFAULT 0,
    last_fired_at       TEXT,
    last_evaluated_at   TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'deprecated', 'superseded', 'invalidated')),
    superseded_by       TEXT,
    metadata_json       TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_evorule_type   ON evolution_rules(rule_type, status);
CREATE INDEX IF NOT EXISTS idx_evorule_conf   ON evolution_rules(confidence DESC);

-- ============================================================
-- Evolution: Fitness Metrics Over Time
-- ============================================================
CREATE TABLE IF NOT EXISTS evolution_metrics (
    id              TEXT PRIMARY KEY,
    metric_name     TEXT NOT NULL,
    metric_value    REAL NOT NULL,
    recorded_at     TEXT NOT NULL,
    session_id      TEXT,
    dimension_json  TEXT DEFAULT '{}',
    metadata_json   TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_evometric_name_time
    ON evolution_metrics(metric_name, recorded_at);

-- ============================================================
-- Plugin State (key-value for plugins to persist state)
-- ============================================================
CREATE TABLE IF NOT EXISTS plugin_state (
    plugin_name     TEXT NOT NULL,
    state_key       TEXT NOT NULL,
    state_value_json TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (plugin_name, state_key)
) WITHOUT ROWID;

-- ============================================================
-- Bootstrap: Insert initial schema version
-- ============================================================
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial schema: memories, relationships, tags, consolidation, evolution');
