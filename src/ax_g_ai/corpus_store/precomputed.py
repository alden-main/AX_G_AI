"""Precomputed passage embeddings shipped with the corpus (``corpus/embeddings/<model_id>.jsonl``).

The loader inserts these instead of calling the embedding server, so the first
deployment fills the vector store in seconds. Passages whose embedding input changed
(different SHA-256) are not in the file and are embedded by the server as before.

File format (JSONL)::

    {"format": "ax-g-embeddings/1", "model_id": "kure-v1", "hf_model": "nlpai-lab/KURE-v1",
     "hf_revision": "...", "dimensions": 1024, "normalized": true, "generator": "...", "generated_at": "..."}
    {"sha256": "<sha256 of embedding_input(chunk)>", "vector": "<base64 little-endian float32>"}
    ...

Check coverage (used by the Docker build)::

    python -m ax_g_ai.corpus_store.precomputed --check text_data/corpus
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ax_g_ai.corpus.files import CHUNKS_FILE, DEFAULT_EMBEDDING_MODEL
from ax_g_ai.jsonl import parse_jsonl

FORMAT = "ax-g-embeddings/1"


class PrecomputedEmbeddingError(ValueError):
    """The precomputed embedding file is malformed or belongs to another model."""


def embedding_input(chunk: Mapping[str, Any]) -> str:
    """Text sent to the embedding model; its SHA-256 keys the stored vector."""
    return " > ".join(chunk.get("title_path") or ()) + "\n" + chunk.get("text", "")


def embedding_key(chunk: Mapping[str, Any]) -> str:
    return hashlib.sha256(embedding_input(chunk).encode("utf-8")).hexdigest()


def embeddings_path(corpus_dir: Path, model_id: str = DEFAULT_EMBEDDING_MODEL) -> Path:
    return corpus_dir / "embeddings" / f"{model_id}.jsonl"


def encode_vector(vector: Sequence[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def decode_vector(text: str, dimensions: int) -> list[float]:
    raw = base64.b64decode(text, validate=True)
    if len(raw) != 4 * dimensions:
        raise PrecomputedEmbeddingError("vector has the wrong dimension")
    vector = list(struct.unpack(f"<{dimensions}f", raw))
    if not all(math.isfinite(value) for value in vector):
        raise PrecomputedEmbeddingError("vector has non-finite values")
    return vector


def read_precomputed(path: Path, *, model_id: str, dimensions: int) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Header and sha256 → vector. Raises when the file is for another model or malformed."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise PrecomputedEmbeddingError("empty file")
    try:
        header = json.loads(lines[0])
        rows = [json.loads(line) for line in lines[1:]]
    except json.JSONDecodeError as error:
        raise PrecomputedEmbeddingError(f"invalid JSON ({error.msg})") from error
    if header.get("format") != FORMAT:
        raise PrecomputedEmbeddingError(f"unsupported format {header.get('format')!r}")
    if header.get("model_id") != model_id or header.get("dimensions") != dimensions:
        raise PrecomputedEmbeddingError(
            f"file is for {header.get('model_id')!r}/{header.get('dimensions')}, expected {model_id!r}/{dimensions}"
        )
    vectors: dict[str, list[float]] = {}
    for row in rows:
        try:
            vectors[row["sha256"]] = decode_vector(row["vector"], dimensions)
        except (KeyError, TypeError, ValueError) as error:
            raise PrecomputedEmbeddingError(f"bad vector row ({type(error).__name__})") from error
    return header, vectors


def write_precomputed(path: Path, header: Mapping[str, Any], vectors: Mapping[str, Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"format": FORMAT, **header}, ensure_ascii=False, sort_keys=True)]
    lines += [json.dumps({"sha256": key, "vector": encode_vector(vectors[key])}) for key in sorted(vectors)]
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def chunk_keys(corpus_dir: Path) -> set[str]:
    # A missing chunks file raises (as before): the build check must not pass on an empty corpus.
    return {embedding_key(row) for row in parse_jsonl((corpus_dir / CHUNKS_FILE).read_text(encoding="utf-8"))}


def coverage(corpus_dir: Path, model_id: str = DEFAULT_EMBEDDING_MODEL, dimensions: int = 1024) -> tuple[int, set[str], str | None]:
    """(passage count, keys missing from the file, error). A missing file means every key is missing."""
    keys = chunk_keys(corpus_dir)
    path = embeddings_path(corpus_dir, model_id)
    if not path.is_file():
        return len(keys), keys, "file not found"
    try:
        _header, vectors = read_precomputed(path, model_id=model_id, dimensions=dimensions)
    except PrecomputedEmbeddingError as error:
        return len(keys), keys, str(error)
    return len(keys), keys - set(vectors), None


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = list(a), list(b)
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check precomputed passage embeddings cover the corpus.")
    parser.add_argument("--check", type=Path, required=True, metavar="CORPUS_DIR")
    parser.add_argument("--model-id", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args(argv)
    total, missing, error = coverage(args.check, args.model_id)
    if error or missing:
        print(f"precomputed embeddings ({args.model_id}) incomplete: {len(missing)}/{total} passages missing"
              + (f" — {error}" if error else "")
              + ". Run text_data/tools/build_embeddings.py after build_corpus.py.", file=sys.stderr)
        return 1
    print(f"precomputed embeddings ({args.model_id}) cover all {total} passages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
