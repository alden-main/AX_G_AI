"""File names of the generated corpus (``text_data/corpus``): the contract between build, loader and runtime."""

from __future__ import annotations

CHUNKS_FILE = "chunks.jsonl"
DOSE_FILE = "dose_rules.jsonl"
# Rule records the runtime evaluates (stage 2, and stage 3 for dose rules).
RULE_FILES = ("diagnosis.jsonl", "workup.jsonl", "referral.jsonl", "monitoring.jsonl", "lab_thresholds.jsonl", DOSE_FILE)
# Every record file the loader stores, including records the runtime does not evaluate (treatment, formulas).
RECORD_FILES = (
    "diagnosis.jsonl", "workup.jsonl", "treatment.jsonl", DOSE_FILE, "lab_thresholds.jsonl",
    "formulas.jsonl", "monitoring.jsonl", "referral.jsonl",
)
CONCEPT_FILES = ("test_concepts.jsonl", "drug_concepts.jsonl", "drug_classes.jsonl")

# Model id that keys stored passage vectors (``corpus/embeddings/<id>.jsonl``, ``evidence_embedding_1024.model_id``).
DEFAULT_EMBEDDING_MODEL = "kure-v1"
