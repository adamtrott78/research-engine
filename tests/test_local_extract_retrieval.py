from types import SimpleNamespace

from research_engine.local_extract_retrieval import (
    PreservedExtractRetrievalBenchmark,
    evaluate_teacher_retrieval,
    exact_source_regions,
    rank_regions,
    retrieval_frontier,
    retrieval_query,
)


def test_exact_source_regions_preserve_authoritative_offsets():
    text = (
        "Methods\n"
        "We train the detector on known transmitters.\n"
        "Results\n"
        "The detector identifies unseen transmitters with 90 percent accuracy.\n"
    )
    rows = exact_source_regions(
        [{"page_number": 3, "text": text}],
        max_chars=96,
    )
    assert rows
    for row in rows:
        assert row["source_layer"] == "AUTHORITATIVE_PDF_TEXT"
        assert row["raw_source_form"] == text[
            row["start_char"] : row["end_char"]
        ]
        assert row["page_number"] == 3
    assert any(row["section_hint"] == "Methods" for row in rows)
    assert any(row["section_hint"] == "Results" for row in rows)


def test_rank_regions_puts_semantically_matching_lexical_region_first():
    pages = [
        {
            "page_number": 1,
            "text": (
                "Background\n"
                "The weather was mild during collection.\n"
            ),
        },
        {
            "page_number": 4,
            "text": (
                "Results\n"
                "The detector identifies unseen transmitters in RF observations.\n"
            ),
        },
    ]
    rows = exact_source_regions(pages, max_chars=96)
    ranked = rank_regions(
        "RF detector unseen transmitters",
        rows,
        top_k=2,
    )
    assert ranked[0]["page_number"] == 4
    assert "unseen transmitters" in ranked[0]["display_text"]
    assert ranked[0]["retrieval_score"] > ranked[1]["retrieval_score"]


def test_teacher_metrics_require_page_and_quote_region_hit():
    regions = [
        {
            "region_id": "r1",
            "page_number": 2,
            "raw_source_form": (
                "The proposed method detects unseen transmitters "
                "under dynamic RF conditions."
            ),
            "rank": 1,
        },
        {
            "region_id": "r2",
            "page_number": 7,
            "raw_source_form": "An unrelated limitation is discussed.",
            "rank": 2,
        },
    ]
    teacher = [
        {
            "id": "e1",
            "type": "RESULT",
            "page_number": 2,
            "raw_source_form": "method detects unseen transmitters",
        },
        {
            "id": "e2",
            "type": "LIMITATION",
            "page_number": 9,
            "raw_source_form": "not retrieved anywhere",
        },
    ]
    result = evaluate_teacher_retrieval(
        teacher_evidence=teacher,
        ranked_regions=regions,
        quote_threshold=0.8,
    )
    assert result["teacher_evidence_count"] == 2
    assert result["candidate_page_recall"] == 0.5
    assert result["teacher_quote_region_recall"] == 0.5
    assert result["teacher_rows"][0]["quote_region_hit"] is True
    assert result["teacher_rows"][1]["quote_region_hit"] is False


def test_query_uses_only_topic_title_and_retained_rationale():
    query = retrieval_query(
        topic="open-set RF recognition",
        item={
            "id": "paper:1",
            "title": "Unknown Transmitter Detection",
            "rationale": "Relevant to RF novelty detection",
            "secret": "must not appear",
        },
    )
    assert query == (
        "open-set RF recognition Unknown Transmitter Detection "
        "Relevant to RF novelty detection"
    )
    assert "must not appear" not in query


def test_facet_interleave_recovers_dataset_evidence_early():
    regions = [
        {
            "region_id": "method:1",
            "page_number": 2,
            "start_char": 0,
            "end_char": 100,
            "raw_source_form": (
                "The proposed contrastive learning framework learns "
                "domain invariant representations."
            ),
            "display_text": (
                "The proposed contrastive learning framework learns "
                "domain invariant representations."
            ),
            "section_hint": "Methodology",
        },
        {
            "region_id": "method:2",
            "page_number": 2,
            "start_char": 101,
            "end_char": 200,
            "raw_source_form": (
                "The training architecture uses a base encoder and "
                "contrastive objective."
            ),
            "display_text": (
                "The training architecture uses a base encoder and "
                "contrastive objective."
            ),
            "section_hint": "Methodology",
        },
        {
            "region_id": "dataset:1",
            "page_number": 2,
            "start_char": 201,
            "end_char": 320,
            "raw_source_form": (
                "The wireless testbed used fifteen transmitters and "
                "an Ettus USRP receiver for data collection."
            ),
            "display_text": (
                "The wireless testbed used fifteen transmitters and "
                "an Ettus USRP receiver for data collection."
            ),
            "section_hint": "WiFi Testbed",
        },
        {
            "region_id": "result:1",
            "page_number": 4,
            "start_char": 0,
            "end_char": 90,
            "raw_source_form": (
                "The evaluation reports improved classification accuracy."
            ),
            "display_text": (
                "The evaluation reports improved classification accuracy."
            ),
            "section_hint": "Results",
        },
    ]

    ranked = rank_regions(
        "contrastive learning domain invariant RF device fingerprinting",
        regions,
        top_k=4,
    )

    ids = [row["region_id"] for row in ranked]
    assert "dataset:1" in ids[:4]
    dataset = next(row for row in ranked if row["region_id"] == "dataset:1")
    assert dataset["retrieval_channel"] == "facet:dataset"


def test_retrieval_frontier_reports_recall_and_source_chars():
    ranked = [
        {
            "region_id": "r1",
            "page_number": 2,
            "raw_source_form": "alpha beta gamma",
        },
        {
            "region_id": "r2",
            "page_number": 4,
            "raw_source_form": "delta epsilon zeta",
        },
    ]
    teacher = [
        {
            "id": "e1",
            "type": "method",
            "page_number": 2,
            "raw_source_form": "alpha beta",
        },
        {
            "id": "e2",
            "type": "result",
            "page_number": 4,
            "raw_source_form": "delta epsilon",
        },
    ]

    frontier = retrieval_frontier(
        teacher_evidence=teacher,
        ranked_regions=ranked,
        cutoffs=(1, 2),
        quote_threshold=0.8,
    )

    assert frontier[0]["candidate_page_recall"] == 0.5
    assert frontier[0]["teacher_quote_region_recall"] == 0.5
    assert frontier[1]["candidate_page_recall"] == 1.0
    assert frontier[1]["teacher_quote_region_recall"] == 1.0
    assert frontier[1]["selected_source_chars"] > frontier[0][
        "selected_source_chars"
    ]


def test_preserved_benchmark_is_read_only_and_aggregates_teacher_recall():
    teacher_job = SimpleNamespace(
        id="extract:1",
        payload={
            "item": {
                "id": "paper:1",
                "title": "Unknown Transmitter Detection",
                "rationale": "RF novelty",
            }
        },
        result={
            "evidence": [
                {
                    "id": "ev:1",
                    "type": "RESULT",
                    "page_number": 2,
                    "raw_source_form": (
                        "detects previously unseen transmitters"
                    ),
                    "normalized_form": "Detects unseen transmitters.",
                }
            ]
        },
    )

    class Planner:
        def select_run(self, *, expected_papers):
            assert expected_papers == 1
            return "run:1"

        def completed_jobs(self, run_id, job_type):
            assert run_id == "run:1"
            assert str(job_type) == "EXTRACT"
            return [teacher_job]

    class Hydrator:
        def hydrate(self, job):
            payload = dict(job.payload)
            payload["_research_context"] = {
                "source_artifact": {
                    "id": "artifact:1",
                    "sha256": "a" * 64,
                },
                "pages": [
                    {
                        "page_number": 1,
                        "text": "Background\nNoise was measured.\n",
                    },
                    {
                        "page_number": 2,
                        "text": (
                            "Results\n"
                            "The method detects previously unseen transmitters "
                            "in RF observations.\n"
                        ),
                    },
                ],
            }
            return SimpleNamespace(
                id=job.id,
                payload=payload,
                result=job.result,
            )

    class NoWriteStore:
        def connect(self):
            raise AssertionError("store must not be touched with topic provider")

    service = PreservedExtractRetrievalBenchmark(
        store=NoWriteStore(),
        artifact_root="/unused",
        planner=Planner(),
        hydrator=Hydrator(),
        topic_provider=lambda run_id: "RF unseen transmitter detection",
    )
    result = service.run(
        expected_papers=1,
        top_k=2,
        quote_threshold=0.8,
        max_region_chars=96,
    )

    assert result["source_run_id"] == "run:1"
    assert result["teacher_evidence_count"] == 1
    assert result["candidate_page_recall_at_k"] == 1.0
    assert result["teacher_quote_region_recall_at_k"] == 1.0
    assert result["retrieval_strategy"] == "FACET_INTERLEAVED_BM25_V1"
    assert result["cases"][0]["retrieval_frontier"]
    assert result["semantic_model_calls"] == 0
    assert result["scientific_state_materialized"] is False
    assert result["production_routing_changed"] is False
