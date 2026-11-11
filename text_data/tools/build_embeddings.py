"""corpus/chunks.jsonl → corpus/embeddings/<model_id>.jsonl (사전 계산 임베딩, 이미지에 포함).

build_corpus.py 다음에 실행한다. 이미 있는 벡터는 재사용하고 새·변경 문단만 임베딩 서버로 계산하며,
말뭉치에서 사라진 문단의 벡터는 지운다. 임베딩 서버는 운영과 같은 모델이어야 한다
(TEI + nlpai-lab/KURE-v1). 적재기는 서버에서 표본 문단으로 일치 여부(코사인 ≥ 0.99)를 다시 확인한다.

실행(프로젝트 루트, 임베딩 서버가 떠 있을 때):
    uv run python text_data/tools/build_embeddings.py --embedding-url http://127.0.0.1:18080 \\
        --generator "TEI cpu-1.9.4"
    uv run python text_data/tools/build_embeddings.py --check      # 누락 확인만(서버 불필요)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))

from ax_g_ai.corpus_store.embedding import EmbeddingError, TeiEmbeddingClient  # noqa: E402
from ax_g_ai.corpus_store.precomputed import (  # noqa: E402
    DEFAULT_MODEL_ID, PrecomputedEmbeddingError, embedding_input, embedding_key, embeddings_path, main as check_main,
    read_precomputed, write_precomputed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "corpus")
    parser.add_argument("--embedding-url", help="TEI 호환 서버(POST /embed)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-model", default="nlpai-lab/KURE-v1")
    parser.add_argument("--hf-revision", default="8b418a58414668e75532ed045c22d9ca018ae2b2")
    parser.add_argument("--generator", default="unspecified", help="계산에 쓴 서버·버전(기록용)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check_main(["--check", str(args.corpus_dir), "--model-id", args.model_id])
    if not args.embedding_url:
        parser.error("--embedding-url 이 필요합니다(또는 --check)")

    embedder = TeiEmbeddingClient(args.embedding_url, timeout_seconds=300)
    chunks = [json.loads(line) for line in (args.corpus_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = {embedding_key(chunk): embedding_input(chunk) for chunk in chunks}
    path = embeddings_path(args.corpus_dir, args.model_id)
    existing: dict[str, list[float]] = {}
    if path.is_file():
        try:
            header, existing = read_precomputed(path, model_id=args.model_id, dimensions=embedder.dimensions)
        except PrecomputedEmbeddingError as error:
            print(f"기존 파일을 쓰지 않습니다: {error}", file=sys.stderr)
        else:
            if header.get("hf_revision") != args.hf_revision:
                print(f"모델 리비전이 달라 전부 다시 계산합니다: {header.get('hf_revision')} → {args.hf_revision}", file=sys.stderr)
                existing = {}
    todo = [key for key in wanted if key not in existing]
    try:
        computed = embedder.embed([wanted[key] for key in todo]) if todo else []
    except EmbeddingError as error:
        print(f"임베딩 실패: {error}", file=sys.stderr)
        return 1
    vectors = {key: existing[key] for key in wanted if key in existing}
    vectors.update(zip(todo, computed))
    write_precomputed(path, {
        "model_id": args.model_id, "hf_model": args.hf_model, "hf_revision": args.hf_revision,
        "dimensions": embedder.dimensions, "normalized": True, "generator": args.generator,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, vectors)
    removed = len(set(existing) - set(wanted))
    print(f"{path.relative_to(ROOT.parent)}: 문단 {len(vectors)}개(재사용 {len(vectors) - len(todo)}, 새로 계산 {len(todo)}, 삭제 {removed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
