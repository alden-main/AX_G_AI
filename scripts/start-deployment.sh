#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode=${1:-openai}
case "$mode" in
  openai) compose=compose.yaml ;;
  vllm) compose=compose.vllm.yaml ;;
  *) echo "사용법: $0 [openai|vllm] [--redis]" >&2; exit 1 ;;
esac
args=(-f "$compose")
if [[ ${2:-} == --redis ]]; then args+=(-f compose.redis.yaml); elif [[ $# -gt 1 ]]; then echo "사용법: $0 [openai|vllm] [--redis]" >&2; exit 1; fi
if [[ $# -gt 2 ]]; then echo "사용법: $0 [openai|vllm] [--redis]" >&2; exit 1; fi
if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  printf '\n.env를 생성했습니다. OPENAI_API_KEY와 CORPUS_DB_PASSWORD를 입력한 뒤 이 스크립트를 다시 실행하세요.\nRedis 사용 시 REDIS_PASSWORD도 설정하세요.\n' >&2
  exit 1
fi
image=$(cat image-tag.txt)
# 패키지 이미지와 모든 Compose 서비스가 같은 태그를 사용하도록 셸 환경에서 우선 지정한다.
export AX_G_AI_IMAGE="$image"

# 셸이나 .env에서 이미 지정한 값은 그대로 둔다.
is_set() { [[ -n ${!1:-} ]] || grep -Eq "^[[:space:]]*(export[[:space:]]+)?$1=" .env; }
# 압축파일에 들어 있는 외부 이미지(PostgreSQL, 임베딩 서버)와 KURE-v1 모델을 쓴다(외부 접속 없이 기동).
bundled_images=()
for name in postgres embedding; do
  [[ -f $name-image.tar ]] || continue
  bundled_images+=("$name-image.tar")
  var=$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')_IMAGE
  is_set "$var" || export "$var=$(cat "$name-image.txt")"
done
if [[ -d hf-cache/models--nlpai-lab--KURE-v1 ]] && ! is_set EMBEDDING_CACHE_DIR; then
  export EMBEDDING_CACHE_DIR="$PWD/hf-cache"
  is_set EMBEDDING_HF_HUB_OFFLINE || export EMBEDDING_HF_HUB_OFFLINE=1
fi

docker compose "${args[@]}" config --quiet
docker load --input image.tar
for tarball in ${bundled_images[@]+"${bundled_images[@]}"}; do docker load --input "$tarball"; done
docker compose "${args[@]}" up -d --no-build
docker compose "${args[@]}" ps -a
printf '\n상태 확인: curl --fail http://127.0.0.1:8100/health (vLLM에서 AX_G_PORT를 변경했다면 해당 포트 사용)\n로그: docker compose'
printf ' %q' "${args[@]}"
printf ' logs --tail=100 ax-g-ai corpus-loader dur-loader\n'
