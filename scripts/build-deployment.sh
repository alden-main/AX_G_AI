#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
platform=${AX_G_PLATFORM:-linux/amd64}
tag=${AX_G_IMAGE_TAG:-ax-g-ai:1.0.0}
arch=${platform#linux/}
case "$platform" in linux/amd64|linux/arm64) ;; *) echo "지원하지 않는 플랫폼: $platform" >&2; exit 1;; esac
output=${AX_G_PACKAGE_PATH:-"$root/ax-g-ai-deploy-${arch}.tar.gz"}
case "$output" in /*) ;; *) output="$PWD/$output";; esac

# 말뭉치 DB(PostgreSQL + pgvector) 이미지를 함께 넣는다. 넣지 않으려면 AX_G_BUNDLE_POSTGRES=0.
bundle_postgres=${AX_G_BUNDLE_POSTGRES:-1}
postgres_image=${AX_G_POSTGRES_IMAGE:-pgvector/pgvector:0.8.7-pg16}
# 임베딩 서버(TEI 이미지 + KURE-v1 모델)를 함께 넣어 서버가 외부 접속 없이 임베딩을 기동하게 한다.
# 넣지 않으려면 AX_G_BUNDLE_EMBEDDING=0 (서버가 최초 기동 때 이미지와 모델을 내려받는다).
bundle_embedding=${AX_G_BUNDLE_EMBEDDING:-1}
case "$arch" in
  amd64) default_embedding_image=ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4 ;;
  arm64) default_embedding_image=ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-1.9.4 ;;
esac
embedding_image=${AX_G_EMBEDDING_IMAGE:-$default_embedding_image}
# Hugging Face 캐시 형식(models--<org>--<name>)의 모델 디렉터리. 없으면 아래 안내대로 먼저 받는다.
model_cache=${AX_G_EMBEDDING_MODEL_CACHE:-"$root/hf-cache/models--nlpai-lab--KURE-v1"}

if [[ $bundle_embedding == 1 ]]; then
  revision=$(cat "$model_cache/refs/main" 2>/dev/null || true)
  snapshot="$model_cache/snapshots/$revision"
  for f in config.json tokenizer.json model.safetensors modules.json 1_Pooling/config.json; do
    if [[ -z $revision || ! -s "$snapshot/$f" ]]; then
      printf '임베딩 모델 파일이 없습니다: %s\n먼저 받으세요: uvx --from huggingface_hub hf download nlpai-lab/KURE-v1 --cache-dir "%s"\n' \
        "$snapshot/$f" "$(dirname "$model_cache")" >&2
      exit 1
    fi
  done
fi

staging=$(mktemp -d "${TMPDIR:-/tmp}/ax-g-deploy.XXXXXX")
trap 'rm -rf "$staging"' EXIT
mkdir -p "$staging/ax-g-ai-deploy"
bundle="$staging/ax-g-ai-deploy"
printf '%s\n' "$tag" > "$bundle/image-tag.txt"
cp "$root/compose.yaml" "$root/compose.vllm.yaml" "$root/compose.redis.yaml" "$root/.env.example" "$root/scripts/start-deployment.sh" "$bundle/"
docker buildx build --platform "$platform" --tag "$tag" --output "type=docker,dest=$bundle/image.tar" "$root"

# 외부 이미지를 <이름>-image.tar로 저장하고, 서버가 같은 이름으로 쓰도록 <이름>-image.txt에 이미지 이름을 남긴다.
bundle_image() {
  docker pull --platform "$platform" "$2"
  docker save --platform "$platform" --output "$bundle/$1-image.tar" "$2"
  chmod 644 "$bundle/$1-image.tar"
  printf '%s\n' "$2" > "$bundle/$1-image.txt"
}
if [[ $bundle_postgres == 1 ]]; then bundle_image postgres "$postgres_image"; fi
if [[ $bundle_embedding == 1 ]]; then
  bundle_image embedding "$embedding_image"
  mkdir -p "$bundle/hf-cache"
  # 스냅샷 파일은 blobs를 가리키는 상대 심볼릭 링크이므로 링크 그대로 복사한다. 내려받기용 .lock 파일은 뺀다.
  cp -RP "$model_cache" "$bundle/hf-cache/"
  find "$bundle/hf-cache" -name '*.lock' -type f -delete
fi

# macOS 확장 속성(._* 파일)이 섞이지 않게 한다.
COPYFILE_DISABLE=1 tar -czf "$output" -C "$staging" ax-g-ai-deploy
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$(dirname "$output")" && sha256sum "$(basename "$output")" > "$(basename "$output").sha256")
else
  (cd "$(dirname "$output")" && shasum -a 256 "$(basename "$output")" > "$(basename "$output").sha256")
fi
printf '배포 파일: %s\n체크섬: %s.sha256\n' "$output" "$output"
