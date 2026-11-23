#!/usr/bin/env bash
# build-deployment.sh가 만든 배포 압축 파일을 검사한다(CI와 로컬 공용).
#   1. 체크섬(.sha256이 있으면)
#   2. 필수 파일, 그리고 <이름>-image.txt마다 짝이 되는 <이름>-image.tar
#   3. 앱 이미지를 불러와 띄우고 /health·/ready 확인. 접속하지 않는 AI 주소를 넣어 외부 호출 없이
#      "AI 설정 있음 + 파일 말뭉치(근거·규칙) 로드"까지 확인한다(DB·Redis 없이 단독 기동).
# 사용법: scripts/verify-deployment.sh ax-g-ai-deploy-amd64.tar.gz
set -euo pipefail

bundle=${1:?사용법: $0 <배포 압축 파일>}
[[ -s $bundle ]] || { echo "압축 파일이 없습니다: $bundle" >&2; exit 1; }
port=${AX_G_VERIFY_PORT:-18100}
fail() { echo "검사 실패: $*" >&2; exit 1; }

if [[ -f "$bundle.sha256" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$(dirname "$bundle")" && sha256sum -c "$(basename "$bundle").sha256")
  else
    (cd "$(dirname "$bundle")" && shasum -a 256 -c "$(basename "$bundle").sha256")
  fi
fi

listing=$(tar -tzf "$bundle")
for name in image.tar image-tag.txt compose.yaml compose.redis.yaml compose.vllm.yaml compose.sglang.yaml \
            compose.observability.yaml .env.example start-deployment.sh docker/sglang-gravity/Dockerfile; do
  grep -qxF "ax-g-ai-deploy/$name" <<<"$listing" || fail "필수 파일 없음: $name"
done
# 함께 넣은 외부 이미지: 이름 파일과 이미지 파일이 짝을 이뤄야 서버가 같은 이름으로 불러온다.
while IFS= read -r label; do
  component=${label#ax-g-ai-deploy/}
  component=${component%-image.txt}
  grep -qxF "ax-g-ai-deploy/$component-image.tar" <<<"$listing" || fail "$component-image.txt에 짝이 되는 이미지 파일 없음"
  echo "포함된 이미지: $component"
done < <(grep -E '^ax-g-ai-deploy/[a-z]+-image\.txt$' <<<"$listing" || true)
if grep -qxF "ax-g-ai-deploy/embedding-image.tar" <<<"$listing"; then
  grep -qE '^ax-g-ai-deploy/hf-cache/models--nlpai-lab--KURE-v1/refs/main$' <<<"$listing" || fail "임베딩 이미지는 있는데 모델(hf-cache)이 없음"
  grep -qE '^ax-g-ai-deploy/hf-cache/models--nlpai-lab--KURE-v1/snapshots/[0-9a-f]+/model\.safetensors$' <<<"$listing" \
    || fail "임베딩 모델 가중치(model.safetensors) 없음"
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/ax-g-verify.XXXXXX")
container=""
cleanup() {
  if [[ -n $container ]]; then docker rm -f "$container" >/dev/null 2>&1 || true; fi
  rm -rf "$work"
}
trap cleanup EXIT
tar -xzf "$bundle" -C "$work" ax-g-ai-deploy/image.tar ax-g-ai-deploy/image-tag.txt
tag=$(tr -d '[:space:]' < "$work/ax-g-ai-deploy/image-tag.txt")
docker load --input "$work/ax-g-ai-deploy/image.tar"
rm -f "$work/ax-g-ai-deploy/image.tar"

# AI_BASE_URL은 접속하지 않는 주소다: 기동과 /ready는 AI를 호출하지 않는다.
# 말뭉치 설정은 compose.yaml과 같다(현재 근거·규칙이 모두 미검수라, 끄면 0개가 된다).
container=$(docker run -d -p "127.0.0.1:$port:8100" \
  -e AI_PROVIDER=openai_compatible -e AI_BASE_URL=http://127.0.0.1:9/v1 -e AI_MODEL=verify \
  -e AX_G_EVIDENCE_INCLUDE_UNREVIEWED=true -e AX_G_CLINICAL_DOSES_ENABLED=true -e AX_G_DUR_ENABLED=false "$tag")
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$port/health" >/dev/null || { docker logs "$container" >&2 || true; fail "/health 응답 없음"; }
ready=$(curl -sS "http://127.0.0.1:$port/ready")
echo "/ready: $ready"
python3 - "$ready" <<'PY' || { docker logs "$container" >&2 || true; fail "/ready 결과가 예상과 다름"; }
import json, sys
body = json.loads(sys.argv[1])
corpus = body["corpus"]
checks = {
    "status=ready": body["status"] == "ready",
    "ai_provider=configured": body["ai_provider"] == "configured",
    "corpus.backend=files": corpus["backend"] == "files",
    "evidence_chunks>0": corpus["evidence_chunks"] > 0,
    "rules>0": corpus["rules"] > 0,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    sys.exit("불일치: " + ", ".join(failed))
print(f"앱 이미지 기동 확인: 근거 문단 {corpus['evidence_chunks']}개, 규칙 {corpus['rules']}개")
PY
echo "검사 통과: $bundle (이미지 $tag)"
