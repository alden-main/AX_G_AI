#!/usr/bin/env bash
# 합성 EMR 벤치마크 일괄 실행: 검증 → (SSH 터널) → 서버 확인 → 실행 → 검토 시트 → (judge 제안·미확정 점수) → 보고서 → (비교).
# 개발 PC에서 실행하고, 모델은 서버 API(GPU 서버의 vLLM 등)를 그대로 사용한다. 합성 데이터만 전송한다.
# 사용법: benchmarks/run-benchmark.sh --help
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
py="$root/.venv/bin/python"

usage() {
  cat <<'EOF'
사용법: benchmarks/run-benchmark.sh --label <이름> [옵션]

서버 연결 (둘 중 하나)
  --server <user@host>     SSH 터널로 서버의 127.0.0.1:<remote-port> API에 연결 (키 인증 필요)
  --remote-port <port>     서버 API 포트 (기본 8500, 서버 .env의 AX_G_PORT)
  --local-port <port>      터널의 로컬 포트 (기본 18100)
  --base-url <url>         터널 없이 직접 접속할 API 주소 (기본 http://127.0.0.1:8500)

실행 범위
  --label <이름>           실행 이름. 이미지 태그·모델명 등 비밀이 아닌 값 (필수)
  --suite core|recommendation|all   (기본 all)
  --split development|validation|holdout   (기본 development)
  --allow-holdout          holdout 실행 허용 (프롬프트 조정에 쓰지 말 것)
  --repetitions <n>        반복 횟수 (기본 1)
  --timeout <초>           요청 제한 시간 (기본 120)
  --quiet                  사례별 진행 표시를 끔 (기본: 업무마다 상태·소요 시간·남은 예상 시간 출력)

judge 제안 (선택, 둘 다 지정하면 실행; core 실행에만 적용)
  --judge-url <url>        OpenAI 호환 /v1/chat/completions 주소
  --judge-model <모델>     평가 대상과 다른 모델 권장
  인증이 필요하면 환경 변수 BENCHMARK_JUDGE_API_KEY에 키를 넣는다 (옵션으로 넘기지 않는다).

사전 점검
  --skip-readiness         말뭉치 규칙이 없어도 recommendation 실행을 진행 (기본: 중단)

비교 (선택)
  --compare-with <run_dir> 기준이 되는 이전 core 실행 폴더. 이번 core 실행과 compare(사람 판정)와
                           judge-compare(judge 제안, 양쪽 모두 같은 judge일 때)를 수행

예시
  benchmarks/run-benchmark.sh --server deploy@gpu01 --label img-1.0.0-qwen3-32b \
    --judge-url https://api.openai.com/v1/chat/completions --judge-model gpt-4.1
EOF
}

server="" remote_port=8500 local_port=18100 base_url="" label="" suite=all split=development
allow_holdout="" repetitions=1 timeout=120 judge_url="" judge_model="" compare_with="" skip_readiness="" quiet=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) server=$2; shift 2 ;;
    --remote-port) remote_port=$2; shift 2 ;;
    --local-port) local_port=$2; shift 2 ;;
    --base-url) base_url=$2; shift 2 ;;
    --label) label=$2; shift 2 ;;
    --suite) suite=$2; shift 2 ;;
    --split) split=$2; shift 2 ;;
    --allow-holdout) allow_holdout=--allow-holdout; shift ;;
    --repetitions) repetitions=$2; shift 2 ;;
    --timeout) timeout=$2; shift 2 ;;
    --judge-url) judge_url=$2; shift 2 ;;
    --judge-model) judge_model=$2; shift 2 ;;
    --compare-with) compare_with=$2; shift 2 ;;
    --skip-readiness) skip_readiness=1; shift ;;
    --quiet) quiet=--quiet; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "알 수 없는 옵션: $1 (--help 참고)" >&2; exit 2 ;;
  esac
done

session=""
fail() {
  echo "오류: $*" >&2
  [[ -n $session ]] && printf '실패: %s\n' "$*" > "$session/summary.txt"
  exit 1
}
[[ -n $label ]] || fail "--label이 필요합니다."
[[ -x $py ]] || fail ".venv가 없습니다. 프로젝트 루트에서 'uv sync'를 먼저 실행하세요."
case "$suite" in core) suites="core" ;; recommendation) suites="recommendation" ;; all) suites="core recommendation" ;; *) fail "--suite 값: $suite" ;; esac
[[ -n $server && -n $base_url ]] && fail "--server와 --base-url은 함께 쓸 수 없습니다."
if [[ -n $judge_url || -n $judge_model ]] && [[ -z $judge_url || -z $judge_model ]]; then
  fail "--judge-url과 --judge-model은 함께 지정해야 합니다."
fi
[[ -z $compare_with || -f $compare_with/manifest.json ]] || fail "비교 기준 실행 폴더가 아닙니다: $compare_with"

stamp=$(date +%Y%m%d-%H%M%S)
session="$root/benchmarks/output/session-$stamp-$label"
mkdir -p "$session"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$session/session.log"; }

if [[ -n $judge_url ]] && [[ " $suites " != *" core "* ]]; then
  log "경고: judge는 core 실행에만 적용됩니다. 이번 설정(--suite $suite)에서는 judge를 실행하지 않습니다."
fi
log "1/5 데이터 검증"
"$py" -m benchmarks validate > "$session/validate.json"

tunnel_pid=""
cleanup() {
  status=$?
  [[ -n $tunnel_pid ]] && kill "$tunnel_pid" 2>/dev/null || true
  # A failure that did not go through fail() (a command error under set -e, Ctrl-C) still marks the session.
  if [[ $status -ne 0 && -n $session && ! -f $session/summary.txt ]]; then
    printf '실패(종료 코드 %s): session.log와 터미널 출력을 확인하세요\n' "$status" > "$session/summary.txt"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
if [[ -n $server ]]; then
  log "2/5 SSH 터널: 127.0.0.1:$local_port → $server:127.0.0.1:$remote_port"
  # BatchMode: 비밀번호 입력을 기다리지 않고 실패한다(백그라운드 터널이므로 키 인증이 필요).
  ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L "127.0.0.1:$local_port:127.0.0.1:$remote_port" "$server" &
  tunnel_pid=$!
  base_url="http://127.0.0.1:$local_port"
else
  base_url=${base_url:-http://127.0.0.1:8500}
  log "2/5 직접 접속: $base_url"
fi

log "3/5 서버 상태 확인: $base_url/health"
ok=""
for _ in $(seq 1 30); do
  if curl -fsS --max-time 5 "$base_url/health" > "$session/health.json" 2>/dev/null; then ok=1; break; fi
  if [[ -n $tunnel_pid ]] && ! kill -0 "$tunnel_pid" 2>/dev/null; then fail "SSH 터널이 종료되었습니다(키 인증·포트 사용 여부 확인)."; fi
  sleep 1
done
[[ -n $ok ]] || fail "서버 API에 연결하지 못했습니다: $base_url/health"

# /health passes without corpus rules or DUR; the rule listing tells whether recommendations (and chat DUR) can work.
code=$(curl -sS --max-time 10 -o "$session/findings.json" -w '%{http_code}' "$base_url/internal/v1/clinical-recommendations/findings" 2>/dev/null) || code=000
readiness=$("$py" - "$code" "$session/findings.json" <<'PY'
import json
import sys

code, path = sys.argv[1], sys.argv[2]
if code == '200':
    data = json.load(open(path))
    dur = '사용 추정(pregnancy 소견 있음)' if 'pregnancy' in data.get('findings', []) else '미사용 추정(pregnancy 소견 없음)'
    print(f"ok 말뭉치 규칙 {data.get('rules')}개(미검수 {data.get('unreviewed_rules')}), DUR {dur}")
elif code == '503':
    print('no_rules 말뭉치 규칙 없음(503): 추천 API 불가, 채팅에는 DUR·규칙 결과가 들어가지 않음')
else:
    print(f'unknown 규칙 목록 확인 실패(HTTP {code})')
PY
)
state=${readiness%% *}
printf '%s\n' "${readiness#* }" > "$session/readiness.txt"
log "  준비 상태: ${readiness#* }"
if [[ $state == no_rules && " $suites " == *" recommendation "* && -z $skip_readiness ]]; then
  fail "말뭉치 규칙이 없어 recommendation 실행이 모두 실패합니다. 서버 적재(corpus-loader, dur-loader)를 확인하거나 --suite core 또는 --skip-readiness를 쓰세요."
fi
if [[ $state != ok ]]; then
  log "경고: core 채팅의 DUR 사실은 누락으로 판정될 수 있습니다."
fi

core_run=""
runs=""
log "4/5 벤치마크 실행 (SSE, split=$split, 반복=$repetitions)"
for s in $suites; do
  run_label="$label-$s"
  log "  실행: $run_label"
  run_dir=$("$py" -m benchmarks run --live --base-url "$base_url" --label "$run_label" --split "$split" $allow_holdout \
    --suite "$s" --repetitions "$repetitions" --timeout "$timeout" $quiet | tail -n 1)
  [[ -f $run_dir/manifest.json ]] || fail "실행 결과 폴더를 찾지 못했습니다: $run_dir"
  runs="$runs $run_dir"
  echo "$s $run_dir" >> "$session/runs.txt"
  if [[ $s == core ]]; then
    core_run=$run_dir
    "$py" -m benchmarks review "$run_dir" > /dev/null
    if [[ -n $judge_url ]]; then
      log "  judge 제안: $judge_model"
      "$py" -m benchmarks judge "$run_dir" --live --url "$judge_url" --model "$judge_model" --timeout "$timeout" > /dev/null
      "$py" -m benchmarks judge-report "$run_dir" > "$run_dir/judge-report.json"
    fi
  fi
  "$py" -m benchmarks report "$run_dir" > "$run_dir/report.json"
done

if [[ -n $compare_with ]]; then
  if [[ -z $core_run ]]; then
    log "비교 생략: 이번 세션에 core 실행이 없습니다."
  else
    log "비교: $compare_with ↔ $core_run"
    "$py" -m benchmarks compare "$compare_with" "$core_run" > "$session/compare.json"
    if ! ls "$compare_with"/judge-*.meta.json > /dev/null 2>&1; then
      log "judge 비교 생략: 기준 실행($compare_with)에 judge 결과가 없습니다."
    elif ! ls "$core_run"/judge-*.meta.json > /dev/null 2>&1; then
      log "judge 비교 생략: 이번 실행에 judge 결과가 없습니다(--judge-url/--judge-model 미지정)."
    else
      "$py" -m benchmarks judge-compare "$compare_with" "$core_run" > "$session/judge-compare.json"
    fi
  fi
fi

log "5/5 요약"
"$py" - "$session" $runs <<'PY' | tee "$session/summary.txt"
import json
import sys
from pathlib import Path


def value(r):
    return 'N/A' if not r or r.get('value') is None else f"{r['value']:.3f} ({r['numerator']}/{r['denominator']})"


def ms(x):
    if x is None:
        return 'N/A'
    return f'{x:.0f}ms' if x < 1000 else f'{x / 1000:.1f}s'


session, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
readiness = session / 'readiness.txt'
if readiness.exists():
    print(f'서버 준비 상태: {readiness.read_text().strip()}')
WARNINGS = {'dur_unavailable': '응답에 DUR 결과가 없음(서버 DUR 미적재)',
            'dur_dataset_differs': '서버 DUR 데이터셋이 기대값과 다름(이미지·스냅샷 버전 확인)'}
for run in runs:
    manifest = json.loads((run / 'manifest.json').read_text())
    report = json.loads((run / 'report.json').read_text())
    print(f"\n== {manifest['label']} ({run.name})")
    for task, data in report['tasks'].items():
        latency = data['latency_ms']
        line = f"  {task:<15} 성공 {value(data['success'])}  p50 {ms(latency['p50'])}  p95 {ms(latency['p95'])}"
        if 'streaming' in data:
            line += f"  첫 텍스트 p50 {ms(data['streaming']['first_text_ms']['p50'])}"
        print(line)
    if 'recommendation_checks' in report:
        checks = report['recommendation_checks']
        print(f"  [자동 대조] 상태 {checks['statuses']}")
        print(f"  완전 일치 {value(checks['exact_cases'])}  경고 재현율 {value(checks['alert_recall'])}  정밀도 {value(checks['alert_precision'])}")
        print(f"  임신 판정 {value(checks['pregnancy_reading'])}  충돌 표시 {value(checks['conflict_reporting'])}")
        for status, count in checks['statuses'].items():
            if status != 'compared':
                reason = WARNINGS.get(status, 'API 호출 실패' if status.startswith('run_') else status)
                print(f'  경고: {count}건 비교 불가 — {status}: {reason}')
    if 'quality' in report:
        quality = report['quality']
        print(f"  [사람 확정] 판정 완료율 {value(quality['fact_completion'])} — 검토 후 'report'를 다시 실행하세요")
    judge = run / 'judge-report.json'
    if judge.exists():
        j = json.loads(judge.read_text())
        print(f"  [judge 미확정] {j['judge_model']}  제안 {j['coverage']['proposed']}건, judge 오류 {j['coverage']['judge_errors']}건")
        print(f"  중요 사실 재현율 {value(j['important_fact_recall'])}  critical {value(j['critical_fact_recall'])}"
              f"  왜곡 {value(j['important_distortion'])}  금지 주장 위반 {value(j['forbidden_claim_violation_cases'])}")
compare = session / 'judge-compare.json'
if compare.exists():
    c = json.loads(compare.read_text())
    print('\n== judge 제안 비교 (기준 → 이번)')
    if not c['comparable']:
        print(f"  비교 불가: {c['judge_differences']}")
    else:
        for key, pair in c['summary'].items():
            print(f"  {key:<34} {value(pair['left'])} → {value(pair['right'])}")
print(f'\n결과 폴더: {session}')
PY
