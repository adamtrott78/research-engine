BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS works (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    publication_year INTEGER,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS work_versions (
    id TEXT PRIMARY KEY,
    work_id TEXT NOT NULL REFERENCES works(id),
    version_label TEXT,
    published_at TIMESTAMPTZ,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS work_versions_work_idx ON work_versions(work_id);

CREATE TABLE IF NOT EXISTS external_identifiers (
    id BIGSERIAL PRIMARY KEY,
    work_id TEXT REFERENCES works(id),
    work_version_id TEXT REFERENCES work_versions(id),
    scheme TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ext_identifier_owner CHECK (work_id IS NOT NULL OR work_version_id IS NOT NULL),
    CONSTRAINT ext_identifier_unique UNIQUE (scheme, value)
);

CREATE TABLE IF NOT EXISTS source_artifacts (
    id TEXT PRIMARY KEY,
    work_version_id TEXT REFERENCES work_versions(id),
    sha256 CHAR(64) NOT NULL UNIQUE,
    byte_size BIGINT NOT NULL CHECK (byte_size > 0),
    media_type TEXT NOT NULL,
    format TEXT NOT NULL,
    access_class TEXT NOT NULL,
    artifact_uri TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS source_artifacts_version_idx ON source_artifacts(work_version_id);

CREATE TABLE IF NOT EXISTS discovery_runs (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    query TEXT NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    raw_result_artifact_id TEXT REFERENCES source_artifacts(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS acquisition_attempts (
    id TEXT PRIMARY KEY,
    work_version_id TEXT REFERENCES work_versions(id),
    source_uri TEXT NOT NULL,
    executor TEXT NOT NULL,
    result TEXT NOT NULL,
    failure_detail TEXT,
    byte_size BIGINT,
    content_hash CHAR(64),
    source_artifact_id TEXT REFERENCES source_artifacts(id),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    id TEXT PRIMARY KEY,
    work_version_id TEXT REFERENCES work_versions(id),
    source_artifact_id TEXT REFERENCES source_artifacts(id),
    source_level TEXT NOT NULL,
    extractor_identity TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'COMPLETED',
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_locators (
    id TEXT PRIMARY KEY,
    source_artifact_id TEXT NOT NULL REFERENCES source_artifacts(id),
    source_level TEXT NOT NULL,
    locator_quality TEXT NOT NULL,
    page_number INTEGER CHECK (page_number IS NULL OR page_number > 0),
    pdf_bbox DOUBLE PRECISION[],
    section_path TEXT[],
    raw_anchor TEXT,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS source_locators_artifact_idx ON source_locators(source_artifact_id);

CREATE TABLE IF NOT EXISTS evidence_objects (
    id TEXT PRIMARY KEY,
    evidence_type TEXT NOT NULL,
    work_id TEXT NOT NULL REFERENCES works(id),
    work_version_id TEXT REFERENCES work_versions(id),
    source_artifact_id TEXT REFERENCES source_artifacts(id),
    source_locator_id TEXT REFERENCES source_locators(id),
    source_level TEXT NOT NULL,
    raw_source_form TEXT,
    normalized_form TEXT NOT NULL,
    extraction_run_id TEXT REFERENCES extraction_runs(id),
    extractor_identity TEXT,
    verification_state TEXT NOT NULL,
    supersedes_id TEXT REFERENCES evidence_objects(id),
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_document TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(normalized_form, '') || ' ' || coalesce(raw_source_form, ''))
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS evidence_objects_work_idx ON evidence_objects(work_id);
CREATE INDEX IF NOT EXISTS evidence_objects_type_idx ON evidence_objects(evidence_type);
CREATE INDEX IF NOT EXISTS evidence_objects_verification_idx ON evidence_objects(verification_state);
CREATE INDEX IF NOT EXISTS evidence_objects_search_idx ON evidence_objects USING GIN(search_document);
CREATE INDEX IF NOT EXISTS evidence_objects_attributes_idx ON evidence_objects USING GIN(attributes);
CREATE INDEX IF NOT EXISTS evidence_objects_supersedes_idx ON evidence_objects(supersedes_id);

CREATE TABLE IF NOT EXISTS evidence_relations (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES evidence_objects(id),
    target_id TEXT NOT NULL REFERENCES evidence_objects(id),
    relation_type TEXT NOT NULL,
    context_compatibility TEXT,
    verification_state TEXT NOT NULL,
    rationale TEXT NOT NULL,
    supersedes_id TEXT REFERENCES evidence_relations(id),
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT relation_not_self CHECK (source_id <> target_id)
);
CREATE INDEX IF NOT EXISTS evidence_relations_source_idx ON evidence_relations(source_id);
CREATE INDEX IF NOT EXISTS evidence_relations_target_idx ON evidence_relations(target_id);
CREATE INDEX IF NOT EXISTS evidence_relations_source_type_idx ON evidence_relations(source_id, relation_type);
CREATE INDEX IF NOT EXISTS evidence_relations_target_type_idx ON evidence_relations(target_id, relation_type);
CREATE INDEX IF NOT EXISTS evidence_relations_supersedes_idx ON evidence_relations(supersedes_id);

CREATE TABLE IF NOT EXISTS verification_events (
    id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    previous_state TEXT,
    new_state TEXT NOT NULL,
    verifier_type TEXT NOT NULL,
    verifier_identity TEXT NOT NULL,
    rationale TEXT,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS verification_record_idx ON verification_events(record_type, record_id, created_at);

CREATE TABLE IF NOT EXISTS knowledge_checkpoints (
    id TEXT PRIMARY KEY,
    fingerprint CHAR(64) NOT NULL UNIQUE,
    scope JSONB NOT NULL,
    manifest JSONB NOT NULL,
    record_counts JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS synthesis_artifacts (
    id TEXT PRIMARY KEY,
    scope_class TEXT NOT NULL,
    question TEXT NOT NULL,
    knowledge_checkpoint TEXT NOT NULL,
    body JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'CURRENT',
    supersedes_id TEXT REFERENCES synthesis_artifacts(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS inferences (
    id TEXT PRIMARY KEY,
    inference_type TEXT NOT NULL,
    statement TEXT NOT NULL,
    basis_evidence_ids TEXT[] NOT NULL DEFAULT '{}',
    verification_state TEXT NOT NULL,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS gap_candidates (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    gap_type TEXT NOT NULL,
    knowledge_checkpoint_id TEXT NOT NULL REFERENCES knowledge_checkpoints(id),
    motivating_evidence_ids TEXT[] NOT NULL DEFAULT '{}',
    novelty_status TEXT NOT NULL,
    rationale TEXT NOT NULL,
    human_review_state TEXT NOT NULL DEFAULT 'NOT_REVIEWED',
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS novelty_challenges (
    id TEXT PRIMARY KEY,
    gap_candidate_id TEXT NOT NULL REFERENCES gap_candidates(id),
    knowledge_checkpoint_id TEXT NOT NULL REFERENCES knowledge_checkpoints(id),
    result TEXT NOT NULL,
    search_provenance JSONB NOT NULL,
    counterevidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    reasoning TEXT NOT NULL,
    human_approval JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS gap_candidates_checkpoint_idx
    ON gap_candidates(knowledge_checkpoint_id, novelty_status);
CREATE INDEX IF NOT EXISTS novelty_challenges_candidate_idx
    ON novelty_challenges(gap_candidate_id, created_at);
CREATE INDEX IF NOT EXISTS novelty_challenges_checkpoint_idx
    ON novelty_challenges(knowledge_checkpoint_id);

CREATE TABLE IF NOT EXISTS research_programs (
    id TEXT PRIMARY KEY,
    source_gap_candidate_id TEXT REFERENCES gap_candidates(id),
    status TEXT NOT NULL,
    body JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS funding_opportunities (
    id TEXT PRIMARY KEY,
    agency TEXT NOT NULL,
    official_identifier TEXT NOT NULL,
    authoritative_source_uri TEXT NOT NULL,
    body JSONB NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL,
    UNIQUE(agency, official_identifier)
);

CREATE TABLE IF NOT EXISTS solicitation_requirements (
    id TEXT PRIMARY KEY,
    funding_opportunity_id TEXT NOT NULL REFERENCES funding_opportunities(id),
    category TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    normalized_rule TEXT NOT NULL,
    satisfaction_state TEXT NOT NULL,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS proposal_compilations (
    id TEXT PRIMARY KEY,
    research_program_id TEXT NOT NULL REFERENCES research_programs(id),
    funding_opportunity_id TEXT REFERENCES funding_opportunities(id),
    status TEXT NOT NULL,
    body JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS execution_bridge_plans (
    id TEXT PRIMARY KEY,
    research_program_id TEXT NOT NULL REFERENCES research_programs(id),
    instantiable BOOLEAN NOT NULL,
    body JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    status TEXT NOT NULL,
    current_phase TEXT NOT NULL,
    policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    knowledge_checkpoint_id TEXT REFERENCES knowledge_checkpoints(id),
    human_gate_state TEXT NOT NULL DEFAULT 'NOT_REQUIRED',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS research_runs_status_idx ON research_runs(status, updated_at);

CREATE TABLE IF NOT EXISTS research_workers (
    id TEXT PRIMARY KEY,
    logical_agent_id TEXT,
    supported_tiers TEXT[] NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    state TEXT NOT NULL DEFAULT 'ACTIVE',
    host_name TEXT,
    process_id INTEGER,
    code_sha TEXT,
    backend TEXT,
    model TEXT,
    reasoning_effort TEXT,
    last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE research_workers
    ADD COLUMN IF NOT EXISTS logical_agent_id TEXT,
    ADD COLUMN IF NOT EXISTS host_name TEXT,
    ADD COLUMN IF NOT EXISTS process_id INTEGER,
    ADD COLUMN IF NOT EXISTS code_sha TEXT,
    ADD COLUMN IF NOT EXISTS backend TEXT,
    ADD COLUMN IF NOT EXISTS model TEXT,
    ADD COLUMN IF NOT EXISTS reasoning_effort TEXT;
CREATE INDEX IF NOT EXISTS research_workers_logical_agent_idx
    ON research_workers(logical_agent_id, created_at);

CREATE TABLE IF NOT EXISTS research_provider_backoffs (
    provider TEXT PRIMARY KEY,
    blocked_until TIMESTAMPTZ NOT NULL,
    reason TEXT NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS research_provider_backoffs_until_idx
    ON research_provider_backoffs(blocked_until);

CREATE TABLE IF NOT EXISTS research_jobs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL,
    executor_tier TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    idempotency_key TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    depends_on TEXT[] NOT NULL DEFAULT '{}',
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    lease_owner TEXT REFERENCES research_workers(id),
    lease_expires_at TIMESTAMPTZ,
    available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_error JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(run_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS research_jobs_queue_idx
    ON research_jobs(status, executor_tier, available_at, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS research_jobs_run_type_idx
    ON research_jobs(run_id, job_type, status);
CREATE INDEX IF NOT EXISTS research_jobs_lease_idx
    ON research_jobs(status, lease_expires_at);

CREATE TABLE IF NOT EXISTS research_job_attempts (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES research_jobs(id) ON DELETE CASCADE,
    attempt_no INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    executor_tier TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(job_id, attempt_no)
);
CREATE INDEX IF NOT EXISTS research_job_attempts_job_idx
    ON research_job_attempts(job_id, attempt_no);

CREATE TABLE IF NOT EXISTS document_sidecar_events (
    id BIGSERIAL PRIMARY KEY,
    source_artifact_id TEXT NOT NULL REFERENCES source_artifacts(id) ON DELETE CASCADE,
    source_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('COMPLETED','CACHE_HIT','FAILED_DERIVED_SIDECAR')
    ),
    cache_path TEXT,
    compiler_version TEXT NOT NULL,
    native_status TEXT,
    standard_status TEXT,
    formula_pages INTEGER NOT NULL DEFAULT 0 CHECK (formula_pages >= 0),
    fast_empty_tables INTEGER NOT NULL DEFAULT 0 CHECK (fast_empty_tables >= 0),
    accurate_pages INTEGER NOT NULL DEFAULT 0 CHECK (accurate_pages >= 0),
    paddle_fallbacks INTEGER NOT NULL DEFAULT 0 CHECK (paddle_fallbacks >= 0),
    wall_time_ms BIGINT NOT NULL DEFAULT 0 CHECK (wall_time_ms >= 0),
    routing_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    local_inference_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_detail TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS document_sidecar_events_artifact_idx
    ON document_sidecar_events(source_artifact_id, created_at);
CREATE INDEX IF NOT EXISTS document_sidecar_events_status_idx
    ON document_sidecar_events(status, created_at);

CREATE TABLE IF NOT EXISTS research_usage_events (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    job_id TEXT NOT NULL UNIQUE REFERENCES research_jobs(id) ON DELETE CASCADE,
    job_type TEXT,
    executor_tier TEXT NOT NULL,
    worker_id TEXT REFERENCES research_workers(id),
    logical_agent_id TEXT,
    attempt_no INTEGER,
    provider TEXT NOT NULL,
    model TEXT,
    reasoning_effort TEXT,
    input_tokens BIGINT NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    cached_input_tokens BIGINT NOT NULL DEFAULT 0 CHECK (cached_input_tokens >= 0),
    output_tokens BIGINT NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    reasoning_output_tokens BIGINT NOT NULL DEFAULT 0 CHECK (reasoning_output_tokens >= 0),
    wall_time_ms BIGINT CHECK (wall_time_ms IS NULL OR wall_time_ms >= 0),
    queue_wait_ms BIGINT CHECK (queue_wait_ms IS NULL OR queue_wait_ms >= 0),
    web_search_count INTEGER NOT NULL DEFAULT 0 CHECK (web_search_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE research_usage_events
    ADD COLUMN IF NOT EXISTS job_type TEXT,
    ADD COLUMN IF NOT EXISTS worker_id TEXT REFERENCES research_workers(id),
    ADD COLUMN IF NOT EXISTS logical_agent_id TEXT,
    ADD COLUMN IF NOT EXISTS attempt_no INTEGER,
    ADD COLUMN IF NOT EXISTS reasoning_effort TEXT,
    ADD COLUMN IF NOT EXISTS cached_input_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS reasoning_output_tokens BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS wall_time_ms BIGINT,
    ADD COLUMN IF NOT EXISTS queue_wait_ms BIGINT,
    ADD COLUMN IF NOT EXISTS web_search_count INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS research_usage_run_idx
    ON research_usage_events(run_id, created_at);
CREATE INDEX IF NOT EXISTS research_usage_tier_idx
    ON research_usage_events(run_id, executor_tier, created_at);
CREATE INDEX IF NOT EXISTS research_usage_agent_idx
    ON research_usage_events(logical_agent_id, created_at);
CREATE INDEX IF NOT EXISTS research_usage_worker_idx
    ON research_usage_events(worker_id, created_at);

CREATE TABLE IF NOT EXISTS research_allowance_observations (
    id BIGSERIAL PRIMARY KEY,
    window_type TEXT NOT NULL CHECK (window_type IN ('FIVE_HOUR','WEEKLY')),
    percent_used NUMERIC(5,2) NOT NULL CHECK (percent_used >= 0 AND percent_used <= 100),
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reset_at TIMESTAMPTZ,
    source TEXT NOT NULL DEFAULT 'MANUAL_UI',
    approximate BOOLEAN NOT NULL DEFAULT FALSE,
    percent_quantum NUMERIC(5,2) CHECK (percent_quantum IS NULL OR percent_quantum > 0),
    reset_precision_seconds INTEGER CHECK (
        reset_precision_seconds IS NULL OR reset_precision_seconds > 0
    ),
    note TEXT,
    usage_event_highwater BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE research_allowance_observations
    ADD COLUMN IF NOT EXISTS percent_quantum NUMERIC(5,2)
        CHECK (percent_quantum IS NULL OR percent_quantum > 0),
    ADD COLUMN IF NOT EXISTS reset_precision_seconds INTEGER
        CHECK (reset_precision_seconds IS NULL OR reset_precision_seconds > 0);
CREATE INDEX IF NOT EXISTS research_allowance_observations_window_idx
    ON research_allowance_observations(window_type, observed_at);

CREATE TABLE IF NOT EXISTS research_calibration_events (
    id BIGSERIAL PRIMARY KEY,
    calibration_type TEXT NOT NULL,
    source_run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    source_job_id TEXT NOT NULL REFERENCES research_jobs(id) ON DELETE CASCADE,
    executor_tier TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    reasoning_effort TEXT,
    code_sha TEXT,
    configuration JSONB NOT NULL DEFAULT '{}'::jsonb,
    context_page_count INTEGER NOT NULL DEFAULT 0 CHECK (context_page_count >= 0),
    context_char_count BIGINT NOT NULL DEFAULT 0 CHECK (context_char_count >= 0),
    evidence_count INTEGER NOT NULL DEFAULT 0 CHECK (evidence_count >= 0),
    historical_input_tokens BIGINT NOT NULL DEFAULT 0 CHECK (historical_input_tokens >= 0),
    historical_cached_input_tokens BIGINT NOT NULL DEFAULT 0 CHECK (historical_cached_input_tokens >= 0),
    historical_output_tokens BIGINT NOT NULL DEFAULT 0 CHECK (historical_output_tokens >= 0),
    replay_input_tokens BIGINT NOT NULL DEFAULT 0 CHECK (replay_input_tokens >= 0),
    replay_cached_input_tokens BIGINT NOT NULL DEFAULT 0 CHECK (replay_cached_input_tokens >= 0),
    replay_output_tokens BIGINT NOT NULL DEFAULT 0 CHECK (replay_output_tokens >= 0),
    replay_reasoning_output_tokens BIGINT NOT NULL DEFAULT 0 CHECK (replay_reasoning_output_tokens >= 0),
    wall_time_ms BIGINT NOT NULL DEFAULT 0 CHECK (wall_time_ms >= 0),
    outcome_agreement JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS research_calibration_source_idx
    ON research_calibration_events(source_run_id, source_job_id, created_at);
CREATE INDEX IF NOT EXISTS research_calibration_type_idx
    ON research_calibration_events(calibration_type, created_at);

-- Migration-safe observability backfill from preserved authoritative records.
UPDATE research_workers
SET logical_agent_id = COALESCE(
        logical_agent_id,
        metadata->>'logical_agent_id',
        CASE
          WHEN id ~ ':deterministic:[0-9]+$'
          THEN 'deterministic:' || substring(id from ':deterministic:([0-9]+)$')
          WHEN id ~ ':cheap:[0-9]+$'
          THEN 'semantic:cheap:' || substring(id from ':cheap:([0-9]+)$')
          WHEN id ~ ':standard:[0-9]+$'
          THEN 'semantic:standard:' || substring(id from ':standard:([0-9]+)$')
          WHEN id ~ ':strong:[0-9]+$'
          THEN 'semantic:strong:' || substring(id from ':strong:([0-9]+)$')
          ELSE NULL
        END
    ),
    host_name = COALESCE(host_name, metadata->>'host_name'),
    process_id = COALESCE(
        process_id,
        CASE
          WHEN (metadata->>'process_id') ~ '^[0-9]+$'
          THEN (metadata->>'process_id')::integer
          ELSE NULL
        END
    ),
    code_sha = COALESCE(code_sha, metadata->>'code_sha'),
    backend = COALESCE(backend, metadata->>'backend'),
    model = COALESCE(model, metadata->>'model'),
    reasoning_effort = COALESCE(
        reasoning_effort,
        metadata->>'reasoning_effort'
    );

WITH completed_attempt AS (
    SELECT DISTINCT ON (a.job_id)
           a.job_id,
           a.attempt_no,
           a.worker_id,
           a.started_at,
           a.finished_at
    FROM research_job_attempts a
    WHERE a.status='COMPLETED'
    ORDER BY a.job_id, a.attempt_no DESC
)
UPDATE research_usage_events u
SET job_type = COALESCE(u.job_type, j.job_type),
    worker_id = COALESCE(u.worker_id, a.worker_id),
    logical_agent_id = COALESCE(
        u.logical_agent_id,
        w.logical_agent_id,
        w.metadata->>'logical_agent_id'
    ),
    attempt_no = COALESCE(u.attempt_no, a.attempt_no),
    reasoning_effort = COALESCE(
        u.reasoning_effort,
        j.result #>> '{_execution,reasoning_effort}'
    ),
    cached_input_tokens = CASE
        WHEN u.cached_input_tokens = 0
        THEN COALESCE(
            (j.result #>> '{_execution,usage,cached_input_tokens}')::bigint,
            0
        )
        ELSE u.cached_input_tokens
    END,
    reasoning_output_tokens = CASE
        WHEN u.reasoning_output_tokens = 0
        THEN COALESCE(
            (j.result #>> '{_execution,usage,reasoning_output_tokens}')::bigint,
            0
        )
        ELSE u.reasoning_output_tokens
    END,
    wall_time_ms = COALESCE(
        u.wall_time_ms,
        CASE WHEN a.finished_at IS NOT NULL
             THEN GREATEST(
                 0,
                 ROUND(EXTRACT(EPOCH FROM (a.finished_at-a.started_at))*1000)
             )::bigint
             ELSE NULL
        END
    ),
    queue_wait_ms = COALESCE(
        u.queue_wait_ms,
        CASE WHEN a.started_at IS NOT NULL
             THEN GREATEST(
                 0,
                 ROUND(EXTRACT(EPOCH FROM (a.started_at-j.available_at))*1000)
             )::bigint
             ELSE NULL
        END
    ),
    web_search_count = CASE
        WHEN u.web_search_count = 0
        THEN COALESCE(
            jsonb_array_length(
                COALESCE(
                    j.result #> '{_execution,web_search,events}',
                    '[]'::jsonb
                )
            ),
            0
        )
        ELSE u.web_search_count
    END
FROM research_jobs j
LEFT JOIN completed_attempt a ON a.job_id=j.id
LEFT JOIN research_workers w ON w.id=a.worker_id
WHERE u.job_id=j.id;

CREATE TABLE IF NOT EXISTS semantic_materializations (
    job_id TEXT PRIMARY KEY REFERENCES research_jobs(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','COMPLETED')),
    semantic_result JSONB NOT NULL,
    materialized_result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS semantic_materializations_status_idx
    ON semantic_materializations(status, updated_at);

CREATE TABLE IF NOT EXISTS human_review_requests (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    job_id TEXT NOT NULL UNIQUE REFERENCES research_jobs(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    payload JSONB NOT NULL,
    decision JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS human_review_requests_status_idx
    ON human_review_requests(status, created_at);

CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    record_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    model TEXT NOT NULL,
    model_version TEXT,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    source_content_hash TEXT NOT NULL,
    embedding VECTOR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(record_type, record_id, model, model_version, source_content_hash)
);
CREATE INDEX IF NOT EXISTS embeddings_record_idx ON embeddings(record_type, record_id);
-- Intentionally no HNSW/IVFFlat index in v1. Exact NN is the baseline.

COMMIT;
