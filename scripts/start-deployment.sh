#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# openai: 외부 AI 제공자 / local: LLM_MODEL을 보고 vLLM·SGLang 중 하나를 고름 / vllm·sglang: 엔진 지정
usage() { echo "사용법: $0 [openai|local|vllm|sglang] [--redis]" >&2; exit 1; }
mode=${1:-openai}
case "$mode" in openai|local|vllm|sglang) ;; *) usage ;; esac
if [[ $# -gt 2 ]]; then usage; fi
redis=0
if [[ $# -eq 2 ]]; then [[ $2 == --redis ]] || usage; redis=1; fi

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
# 셸 값, 없으면 .env의 마지막 값(따옴표 제거)을 출력한다.
env_value() {
  if [[ -n ${!1:-} ]]; then printf '%s' "${!1}"; return; fi
  sed -nE "s/^[[:space:]]*(export[[:space:]]+)?$1=//p" .env | tail -n 1 | sed -E "s/^[\"'](.*)[\"'][[:space:]]*$/\1/"
}

# SGLang으로만 띄울 수 있는 모델(Hugging Face 모델 ID의 셸 glob). 그 밖의 모델은 vLLM으로 띄운다.
# learning-unit/L1-*: GravityMoE 구조(vLLM 미지원, SGLang fork 필요 → docker/sglang-gravity).
SGLANG_MODEL_PATTERNS=('learning-unit/L1-*')
engine_for_model() {
  local pattern
  for pattern in "${SGLANG_MODEL_PATTERNS[@]}"; do
    # shellcheck disable=SC2053  # 의도적인 glob 비교
    if [[ $1 == $pattern ]]; then echo sglang; return; fi
  done
  echo vllm
}

args=(-f compose.yaml)
engine=
if [[ $mode != openai ]]; then
  model=$(env_value LLM_MODEL)
  [[ -n $model ]] || model=$(env_value VLLM_MODEL)  # 이전 설정 이름
  case "$mode" in
    local) engine=$(engine_for_model "${model:-Qwen/Qwen3-32B-AWQ}") ;;
    *)
      engine=$mode
      if [[ -n $model && $(engine_for_model "$model") != "$engine" ]]; then
        printf '경고: %s 모델은 보통 %s 엔진으로 띄웁니다. 지정한 %s 엔진으로 계속합니다.\n' "$model" "$(engine_for_model "$model")" "$engine" >&2
      fi
      ;;
  esac
  if [[ -z $model ]]; then
    case "$engine" in vllm) model=Qwen/Qwen3-32B-AWQ ;; sglang) model=learning-unit/L1-16B-A3B ;; esac
  fi
  export LLM_MODEL="$model"
  # 응답 모델 이름: 지정이 없으면 모델 ID의 마지막 부분을 소문자로 쓴다(Qwen/Qwen3-32B-AWQ → qwen3-32b-awq).
  if [[ -z $(env_value AI_MODEL) ]]; then
    served=${model##*/}
    export AI_MODEL="$(printf '%s' "$served" | tr '[:upper:]' '[:lower:]')"
  fi
  args+=(-f "compose.$engine.yaml")
  printf '추론 서버: %s (모델 %s, 응답 이름 %s)\n' "$engine" "$LLM_MODEL" "$(env_value AI_MODEL)"

  # compose.<엔진>.yaml의 `!override`는 Docker Compose 2.24.4 이상에서 동작한다.
  compose_version=$(docker compose version --short 2>/dev/null || true)
  compose_version=${compose_version#v}
  if [[ -z $compose_version ]] || [[ $(printf '%s\n%s\n' 2.24.4 "$compose_version" | sort -V | head -n 1) != 2.24.4 ]]; then
    printf 'Docker Compose 2.24.4 이상이 필요합니다(현재: %s).\n' "${compose_version:-확인 불가}" >&2
    exit 1
  fi
fi
if [[ $redis == 1 ]]; then args+=(-f compose.redis.yaml); fi

# 압축파일에 들어 있는 외부 이미지(PostgreSQL, 임베딩 서버, SGLang)와 KURE-v1 모델을 쓴다(외부 접속 없이 기동).
# SGLang 이미지는 크므로 SGLang으로 띄울 때만 적재한다.
bundled_images=()
names=(postgres embedding)
if [[ $engine == sglang ]]; then names+=(sglang); fi
for name in "${names[@]}"; do
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
# 압축에 SGLang 이미지가 없고 서버에도 없으면 docker/sglang-gravity로 한 번 빌드한다(베이스 이미지 다운로드 필요).
if [[ $engine == sglang ]]; then
  sglang_image=$(docker compose "${args[@]}" config --images sglang)
  if ! docker image inspect "$sglang_image" >/dev/null 2>&1; then
    printf 'SGLang 이미지(%s)가 없어 빌드합니다.\n' "$sglang_image"
    docker compose "${args[@]}" build sglang
  fi
fi
docker compose "${args[@]}" up -d --no-build
docker compose "${args[@]}" ps -a
printf '\n상태 확인: curl --fail http://127.0.0.1:8100/health (vLLM·SGLang에서 AX_G_PORT를 변경했다면 해당 포트 사용)\n로그: docker compose'
printf ' %q' "${args[@]}"
printf ' logs --tail=100 ax-g-ai corpus-loader dur-loader%s\n' "${engine:+ $engine}"
