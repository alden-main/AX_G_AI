"""말뭉치와 함께 배포하는 사전 계산 문단 임베딩(``corpus/embeddings/<model_id>.jsonl``) 읽기·쓰기·검사 모듈.

설계상 위치: 말뭉치 저장 계층의 적재 보조. 적재기(``loader``)가 벡터를 읽고, 임베딩 생성 도구가 파일을 쓰며,
Docker 빌드가 ``--check``로 범위를 검사한다.

역할: 적재기는 임베딩 서버를 부르는 대신 이 파일의 벡터를 넣으므로 첫 배포에서 벡터 저장소가 몇 초 만에 채워진다.
임베딩 입력이 바뀐(SHA-256이 다른) 문단은 파일에 없으므로 이전처럼 서버가 임베딩한다.

파일 형식(JSONL)::

    {"format": "ax-g-embeddings/1", "model_id": "kure-v1", "hf_model": "nlpai-lab/KURE-v1",
     "hf_revision": "...", "dimensions": 1024, "normalized": true, "generator": "...", "generated_at": "..."}
    {"sha256": "<sha256 of embedding_input(chunk)>", "vector": "<base64 little-endian float32>"}
    ...

범위 검사(Docker 빌드에서 사용)::

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
    """사전 계산 임베딩 파일 형식이 잘못됐거나 다른 모델용일 때의 오류.

    역할: 적재기가 이 파일을 무시하고 경고만 남기도록 구분되는 예외를 제공한다.
    """


def embedding_input(chunk: Mapping[str, Any]) -> str:
    """임베딩 모델에 보내는 텍스트(제목 경로 ``" > "`` 연결 + 줄바꿈 + 본문). 이 텍스트의 SHA-256이 저장 벡터의 키다."""
    return " > ".join(chunk.get("title_path") or ()) + "\n" + chunk.get("text", "")


def embedding_key(chunk: Mapping[str, Any]) -> str:
    """문단의 ``embedding_input`` SHA-256 16진 문자열."""
    return hashlib.sha256(embedding_input(chunk).encode("utf-8")).hexdigest()


def embeddings_path(corpus_dir: Path, model_id: str = DEFAULT_EMBEDDING_MODEL) -> Path:
    """모델의 사전 계산 임베딩 파일 경로(``<corpus_dir>/embeddings/<model_id>.jsonl``)."""
    return corpus_dir / "embeddings" / f"{model_id}.jsonl"


def encode_vector(vector: Sequence[float]) -> str:
    """벡터를 리틀 엔디언 float32 바이트의 base64 문자열로 바꾼다."""
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def decode_vector(text: str, dimensions: int) -> list[float]:
    """base64 리틀 엔디언 float32 문자열을 벡터로 되돌린다.

    역할: 바이트 길이로 차원을 확인하고 유한하지 않은 값을 거부한다.
    입력:
        text: ``encode_vector`` 형식의 문자열.
        dimensions: 기대 차원 수.
    출력: ``float`` 목록.
    예외: 차원이 다르거나 NaN·무한대가 있으면 ``PrecomputedEmbeddingError``. base64가 잘못되면 ``binascii.Error``.
    """
    raw = base64.b64decode(text, validate=True)
    if len(raw) != 4 * dimensions:
        raise PrecomputedEmbeddingError("vector has the wrong dimension")
    vector = list(struct.unpack(f"<{dimensions}f", raw))
    if not all(math.isfinite(value) for value in vector):
        raise PrecomputedEmbeddingError("vector has non-finite values")
    return vector


def read_precomputed(path: Path, *, model_id: str, dimensions: int) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """사전 계산 임베딩 파일을 읽어 헤더와 {sha256: 벡터}를 돌려준다.

    역할: 첫 줄 헤더의 형식·모델·차원을 확인하고 나머지 줄의 벡터를 해독한다.
    입력:
        path: 임베딩 JSONL 파일.
        model_id: 기대 모델 ID.
        dimensions: 기대 차원 수.
    출력: (헤더 dict, {sha256: 벡터}).
    예외: 빈 파일, 잘못된 JSON, 다른 형식·모델·차원, 잘못된 벡터 행이면 ``PrecomputedEmbeddingError``.
    """
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
    """헤더와 벡터를 키 순서로 파일에 쓴다(``.tmp``에 쓴 뒤 교체).

    역할: 첫 줄에 ``format``을 더한 헤더, 이후 줄마다 ``sha256``·base64 벡터를 기록한다.
    입력:
        path: 출력 파일 경로(상위 폴더가 없으면 만든다).
        header: 헤더 필드(``model_id``·``dimensions`` 등).
        vectors: {sha256: 벡터}.
    출력: 없음.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"format": FORMAT, **header}, ensure_ascii=False, sort_keys=True)]
    lines += [json.dumps({"sha256": key, "vector": encode_vector(vectors[key])}) for key in sorted(vectors)]
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def chunk_keys(corpus_dir: Path) -> set[str]:
    """말뭉치 문단 파일의 모든 문단에 대한 임베딩 키 집합."""
    # 문단 파일이 없으면 (이전처럼) 예외가 난다: 빈 말뭉치로 빌드 검사가 통과하면 안 된다.
    return {embedding_key(row) for row in parse_jsonl((corpus_dir / CHUNKS_FILE).read_text(encoding="utf-8"))}


def coverage(corpus_dir: Path, model_id: str = DEFAULT_EMBEDDING_MODEL, dimensions: int = 1024) -> tuple[int, set[str], str | None]:
    """사전 계산 파일이 말뭉치 문단을 얼마나 덮는지 계산한다.

    역할: 문단 키와 파일 키를 비교한다. 파일이 없거나 읽을 수 없으면 모든 키를 누락으로 본다.
    입력:
        corpus_dir: 말뭉치 폴더.
        model_id: 모델 ID.
        dimensions: 기대 차원 수.
    출력: (문단 수, 파일에 없는 키 집합, 오류 메시지 또는 None).
    """
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
    """두 벡터의 코사인 유사도(노름이 0이면 0.0)."""
    a, b = list(a), list(b)
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def main(argv: list[str] | None = None) -> int:
    """``--check`` CLI 진입점: 사전 계산 임베딩이 말뭉치 문단을 모두 덮는지 검사한다.

    역할: ``coverage`` 결과를 출력하고, 누락이나 오류가 있으면 재생성 방법을 안내한다.
    입력:
        argv: 명령행 인자(``None``이면 ``sys.argv``).
    출력: 종료 코드. 모두 덮으면 0, 누락·오류가 있으면 1.
    """
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
