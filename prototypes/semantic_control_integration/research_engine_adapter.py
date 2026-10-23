from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Protocol

_ALLOWED_METHODS = {
    "Q1_CLAIM_SUPPORT": "claim_support",
    "Q2_METHOD_QUALIFIERS": "qualifiers_for_method",
    "Q3_SOURCE_LEVEL_COUNTS": "source_level_counts",
    "Q4_PROVENANCE_PATH": "provenance_path",
}

class ReadOnlyResearchStore(Protocol):
    def claim_support(self, claim_id: str) -> list[dict]: ...
    def qualifiers_for_method(self, method_id: str) -> list[dict]: ...
    def source_level_counts(self) -> dict[str, int]: ...
    def provenance_path(self, start_id: str, goal_id: str, max_depth: int = 8) -> list[dict]: ...

def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def _provenance(*, query_id: str, runtime_repository_commit: str, baseline_contract: dict[str, Any], baseline_contract_blob_sha: str) -> list[dict[str, Any]]:
    sources = baseline_contract["authoritative_sources"]
    return [
        {"source_kind":"RUNTIME_REPOSITORY_CHECKPOINT","repository":"adamtrott78/research-engine","commit_sha":runtime_repository_commit},
        {"source_kind":"STORE_INTERFACE","repository":"adamtrott78/research-engine","commit_sha":baseline_contract["repository_checkpoint"],"path":sources["postgres_store"]["path"],"git_blob_sha":sources["postgres_store"]["git_blob_sha"]},
        {"source_kind":"QUERY_TEMPLATE","repository":"adamtrott78/research-engine","commit_sha":baseline_contract["repository_checkpoint"],"path":sources["sql_queries"]["path"],"git_blob_sha":sources["sql_queries"]["git_blob_sha"],"query_id":query_id},
        {"source_kind":"FIXTURE","repository":"adamtrott78/research-engine","commit_sha":baseline_contract["repository_checkpoint"],"path":sources["fixture"]["path"],"git_blob_sha":sources["fixture"]["git_blob_sha"]},
        {"source_kind":"BASELINE_QUERY_CONTRACT","repository":"adamtrott78/research-engine","commit_sha":runtime_repository_commit,"path":"benchmarks/semantic_control_integration/relational-baseline-contract.v1.json","git_blob_sha":baseline_contract_blob_sha,"query_id":query_id},
    ]

def _observation(*, observation_id: str, subject_type: str, subject_id: str, predicate: str, object_value: dict[str, Any], provenance: list[dict[str, Any]], attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version":1,
        "observation_id":observation_id,
        "layer":"EPISTEMIC",
        "subject":{"entity_type":subject_type,"entity_id":subject_id},
        "predicate":predicate,
        "object":object_value,
        "assertion_state":"ACTIVE",
        "provenance":deepcopy(provenance),
        "verification":{"state":"AUTHORITATIVE_RECORD","basis":["read from pinned authoritative PostgresStore query path"]},
        "temporal":{"observed_at":None,"valid_from":None,"valid_to":None},
        "supersedes":[],
        "contradicts":[],
        "attributes":deepcopy(attributes),
    }

def project_query_result(*, query: dict[str, Any], result: Any, runtime_repository_commit: str, baseline_contract: dict[str, Any], baseline_contract_blob_sha: str) -> list[dict[str, Any]]:
    query_id=query["query_id"]
    provenance=_provenance(query_id=query_id,runtime_repository_commit=runtime_repository_commit,baseline_contract=baseline_contract,baseline_contract_blob_sha=baseline_contract_blob_sha)
    observations=[]

    if query_id=="Q1_CLAIM_SUPPORT":
        for row_index,row in enumerate(result):
            observations.append(_observation(
                observation_id=f'sem:claim-support:{row["claim_id"]}:{row["result_id"]}',
                subject_type="Claim",subject_id=row["claim_id"],predicate="supportedBy",
                object_value={"kind":"ENTITY","entity_type":"Evidence","entity_id":row["result_id"]},
                provenance=provenance,
                attributes={"query_id":query_id,"row_index":row_index,"relational_answer_fragment":deepcopy(row)},
            ))
        return observations

    if query_id=="Q2_METHOD_QUALIFIERS":
        for row_index,row in enumerate(result):
            observations.append(_observation(
                observation_id=f'sem:method-qualifier:{row["method_id"]}:{row["qualifier_id"]}',
                subject_type="Method",subject_id=row["method_id"],predicate="qualifiedBy",
                object_value={"kind":"ENTITY","entity_type":"Evidence","entity_id":row["qualifier_id"]},
                provenance=provenance,
                attributes={"query_id":query_id,"row_index":row_index,"relational_answer_fragment":deepcopy(row)},
            ))
        return observations

    if query_id=="Q3_SOURCE_LEVEL_COUNTS":
        for row_index,source_level in enumerate(sorted(result)):
            observations.append(_observation(
                observation_id=f"sem:source-level-count:{source_level}",
                subject_type="AuthoritativeEvidenceStore",subject_id="research-engine-postgres",predicate="sourceLevelCount",
                object_value={"kind":"LITERAL","value":result[source_level]},
                provenance=provenance,
                attributes={"query_id":query_id,"row_index":row_index,"source_level":source_level,"relational_answer_fragment":{"source_level":source_level,"count":result[source_level]}},
            ))
        return observations

    if query_id=="Q4_PROVENANCE_PATH":
        start_id=query["arguments"]["start_id"]
        goal_id=query["arguments"]["goal_id"]
        for row_index,row in enumerate(result):
            observations.append(_observation(
                observation_id=f'sem:provenance-path:{start_id}:{goal_id}:{row["depth"]}',
                subject_type="Evidence",subject_id=start_id,predicate="hasProvenancePath",
                object_value={"kind":"ENTITY","entity_type":"Evidence","entity_id":goal_id},
                provenance=provenance,
                attributes={"query_id":query_id,"row_index":row_index,"relational_answer_fragment":deepcopy(row)},
            ))
        return observations

    raise ValueError(f"unsupported query_id: {query_id}")

def reconstruct_relational_answer(query_id: str, observations: list[dict[str, Any]]) -> Any:
    selected=[item for item in observations if item["attributes"].get("query_id")==query_id]
    selected.sort(key=lambda item:(item["attributes"].get("row_index",0),item["observation_id"]))
    if query_id=="Q3_SOURCE_LEVEL_COUNTS":
        return {
            item["attributes"]["relational_answer_fragment"]["source_level"]:
            item["attributes"]["relational_answer_fragment"]["count"]
            for item in selected
        }
    return [deepcopy(item["attributes"]["relational_answer_fragment"]) for item in selected]

class ReadOnlyResearchEngineAdapter:
    def __init__(self, *, store: ReadOnlyResearchStore, baseline_contract: dict[str, Any], baseline_result: dict[str, Any], runtime_repository_commit: str, baseline_contract_blob_sha: str):
        self.store=store
        self.baseline_contract=deepcopy(baseline_contract)
        self.baseline_result=deepcopy(baseline_result)
        self.runtime_repository_commit=runtime_repository_commit
        self.baseline_contract_blob_sha=baseline_contract_blob_sha
        self._baseline_hashes={item["query_id"]:item["canonical_result_sha256"] for item in baseline_result["query_results"]}

    def _read(self, query: dict[str, Any]) -> Any:
        query_id=query["query_id"]
        method_name=_ALLOWED_METHODS.get(query_id)
        if method_name is None or query["store_method"]!=method_name:
            raise ValueError(f"query {query_id} is not bound to an allowed read method")
        return getattr(self.store,method_name)(**query["arguments"])

    def read_and_project_query(self, query: dict[str, Any]) -> dict[str, Any]:
        raw=self._read(query)
        raw_sha256=canonical_sha256(raw)
        expected_sha256=self._baseline_hashes[query["query_id"]]
        if raw_sha256!=expected_sha256:
            raise ValueError(f'{query["query_id"]} relational answer drift: {raw_sha256} != {expected_sha256}')
        observations=project_query_result(
            query=query,
            result=raw,
            runtime_repository_commit=self.runtime_repository_commit,
            baseline_contract=self.baseline_contract,
            baseline_contract_blob_sha=self.baseline_contract_blob_sha,
        )
        reconstructed=reconstruct_relational_answer(query["query_id"],observations)
        reconstructed_sha256=canonical_sha256(reconstructed)
        if reconstructed_sha256!=raw_sha256:
            raise ValueError(f'{query["query_id"]} semantic reconstruction drift')
        return {
            "query_id":query["query_id"],
            "arguments":deepcopy(query["arguments"]),
            "relational_result_sha256":raw_sha256,
            "semantic_reconstruction_sha256":reconstructed_sha256,
            "semantic_observation_ids":sorted(item["observation_id"] for item in observations),
            "observations":observations,
        }

    def build_manifest(self) -> dict[str, Any]:
        query_records=[self.read_and_project_query(query) for query in self.baseline_contract["queries"]]
        observations=[observation for record in query_records for observation in record["observations"]]
        observations.sort(key=lambda item:item["observation_id"])
        records=[{key:value for key,value in record.items() if key!="observations"} for record in query_records]
        body={
            "schema_version":1,
            "adapter_id":"read-only-research-engine-adapter-v1",
            "runtime_repository_commit":self.runtime_repository_commit,
            "baseline_id":self.baseline_contract["baseline_id"],
            "query_records":records,
            "observations":observations,
        }
        return {**body,"semantic_manifest_sha256":canonical_sha256(body)}
