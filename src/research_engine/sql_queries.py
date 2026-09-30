"""PostgreSQL query templates. Authoritative records live in relational tables."""

PROVENANCE_PATH_SQL = r"""
WITH RECURSIVE walk AS (
    SELECT
        er.source_id,
        er.target_id,
        er.relation_type,
        ARRAY[er.source_id, er.target_id]::text[] AS path,
        1 AS depth
    FROM evidence_relations er
    WHERE er.source_id = %(start_id)s

    UNION ALL

    SELECT
        er.source_id,
        er.target_id,
        er.relation_type,
        w.path || er.target_id,
        w.depth + 1
    FROM evidence_relations er
    JOIN walk w ON er.source_id = w.target_id
    WHERE w.depth < %(max_depth)s
      AND NOT er.target_id = ANY(w.path)
)
SELECT * FROM walk WHERE target_id = %(goal_id)s ORDER BY depth LIMIT 1;
"""

CLAIM_SUPPORT_SQL = r"""
SELECT
    claim.id AS claim_id,
    claim.normalized_form AS claim,
    result.id AS result_id,
    result.normalized_form AS result,
    loc.page_number,
    loc.pdf_bbox,
    art.sha256 AS source_sha256,
    result.verification_state
FROM evidence_relations rel
JOIN evidence_objects result ON result.id = rel.source_id
JOIN evidence_objects claim ON claim.id = rel.target_id
LEFT JOIN source_locators loc ON loc.id = result.source_locator_id
LEFT JOIN source_artifacts art ON art.id = result.source_artifact_id
WHERE rel.relation_type = 'SUPPORTS'
  AND claim.id = %(claim_id)s
ORDER BY result.id;
"""
