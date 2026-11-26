from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks import judge_metrics
from benchmarks.history import commit, effective
from benchmarks.judge import evaluate
from benchmarks.metrics import compare, compare_quality, report
from benchmarks.recommendation import score
from benchmarks.review import create
from benchmarks.runner import SUITES, run, stderr_progress
from benchmarks.schema import DATA, read_jsonl, validate, validate_recommendations


def main():
    parser = argparse.ArgumentParser(description='Synthetic-only development benchmark')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('validate')
    p = sub.add_parser('run')
    p.add_argument('--base-url', required=True)
    p.add_argument('--split', choices=['development', 'validation', 'holdout'], default='development')
    p.add_argument('--allow-holdout', action='store_true')
    p.add_argument('--live', action='store_true', help='explicitly allow API requests')
    p.add_argument('--label', required=True, help='non-secret model/configuration label')
    p.add_argument('--repetitions', type=int, default=1)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--quiet', action='store_true', help='do not print per-task progress to stderr')
    p.add_argument('--suite', choices=sorted(SUITES), default='core',
                   help='core: summary and chat (human review); recommendation: DUR and pregnancy checks (automatic)')
    p.add_argument('--output', default='benchmarks/output')
    for name in ('review', 'report'):
        s = sub.add_parser(name)
        s.add_argument('run_dir', type=Path)
    s = sub.add_parser('compare')
    s.add_argument('left', type=Path)
    s.add_argument('right', type=Path)
    s = sub.add_parser('commit-review', help='save a validated, versioned review snapshot')
    s.add_argument('run_dir', type=Path)
    s.add_argument('--file', type=Path, required=True)
    s.add_argument('--actor', required=True, help='reviewer or coordinator, without secrets')
    s = sub.add_parser('judge', help='request unverified fact judgments from a separate judge model')
    s.add_argument('run_dir', type=Path)
    s.add_argument('--url', required=True, help='OpenAI-compatible chat completions endpoint')
    s.add_argument('--model', required=True)
    s.add_argument('--timeout', type=float, default=120)
    s.add_argument('--live', action='store_true', help='send synthetic EMR, answer and references to judge')
    s = sub.add_parser('judge-report', help='provisional scores from judge proposals (not reviewer-confirmed)')
    s.add_argument('run_dir', type=Path)
    s.add_argument('--job', help='judge job ID (default: the latest)')
    s = sub.add_parser('judge-compare', help='pair two runs by judge proposals (same judge model and prompt only)')
    s.add_argument('left', type=Path)
    s.add_argument('right', type=Path)
    s.add_argument('--left-job')
    s.add_argument('--right-job')
    args = parser.parse_args()
    if args.command == 'validate':
        print(json.dumps({'core': validate(read_jsonl(DATA / 'inputs.jsonl'), read_jsonl(DATA / 'gold.jsonl')),
                          'recommendation': validate_recommendations(read_jsonl(DATA / 'recommendation_inputs.jsonl'),
                                                                     read_jsonl(DATA / 'recommendation_gold.jsonl'))},
                         ensure_ascii=False))
    elif args.command == 'run':
        if not args.live:
            parser.error('run requires --live; no requests are made by default')
        print(run(args.base_url, args.split, args.output, label=args.label, repetitions=args.repetitions,
                  timeout=args.timeout, allow_holdout=args.allow_holdout, suite=args.suite,
                  progress=None if args.quiet else stderr_progress))
    elif args.command == 'review':
        print(create(args.run_dir))
    elif args.command == 'report':
        suite = json.loads((args.run_dir / 'manifest.json').read_text(encoding='utf-8')).get('suite', 'core')
        if suite == 'recommendation':
            result = report(read_jsonl(args.run_dir / 'responses.jsonl'), [])
            result.pop('quality')
            result['recommendation_checks'] = score(args.run_dir)
        else:
            reviews, source = effective(args.run_dir)
            result = report(read_jsonl(args.run_dir / 'responses.jsonl'), reviews)
            result['review_source'] = source
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'commit-review':
        print(commit(args.run_dir, args.file, args.actor))
    elif args.command == 'judge':
        if not args.live:
            parser.error('judge requires --live; no judge requests are made by default')
        print(evaluate(args.run_dir, url=args.url, model=args.model, timeout=args.timeout))
    elif args.command == 'judge-report':
        print(json.dumps(judge_metrics.report(args.run_dir, args.job), ensure_ascii=False, indent=2))
    elif args.command == 'judge-compare':
        print(json.dumps(judge_metrics.compare(args.left, args.right, args.left_job, args.right_job), ensure_ascii=False, indent=2))
    elif args.command == 'compare':
        left_meta = json.loads((args.left / 'manifest.json').read_text(encoding='utf-8'))
        right_meta = json.loads((args.right / 'manifest.json').read_text(encoding='utf-8'))
        differences = {k: [left_meta.get(k), right_meta.get(k)] for k in
                       ('input_sha256', 'split', 'repetitions', 'commit', 'working_tree_dirty', 'model', 'provider',
                        'prompt_version', 'corpus_version', 'dur_version', 'timeout_seconds', 'mode')
                       if left_meta.get(k) != right_meta.get(k) or left_meta.get(k) == 'unknown'}
        left_reviews, left_source = effective(args.left)
        right_reviews, right_source = effective(args.right)
        left_review_meta = args.left / 'review_metadata.json'
        right_review_meta = args.right / 'review_metadata.json'
        compatible = (left_meta.get('input_sha256') == right_meta.get('input_sha256') and
                      left_review_meta.exists() and right_review_meta.exists() and
                      json.loads(left_review_meta.read_text())['gold_sha256'] == json.loads(right_review_meta.read_text())['gold_sha256'])
        print(json.dumps({'configuration_differences_or_unknowns': differences,
                          'pairs': compare(read_jsonl(args.left / 'responses.jsonl'), read_jsonl(args.right / 'responses.jsonl')),
                          'quality': compare_quality(left_reviews, right_reviews) if compatible else None,
                          'quality_note': 'matching input and reference hashes required' if not compatible else 'human judgments only',
                          'review_sources': [left_source, right_source]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
