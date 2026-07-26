#!/usr/bin/env python3
"""
EMRB Evaluation CLI: Run LLM evaluation on benchmark problems.

Usage:
  python evaluate.py --level L3 --id EMRB_L3_4000
  python evaluate.py --level L1 --all --workers 8
  python evaluate.py --level L3 --n 10 --workers 4 --skip-existing
  python evaluate.py --level L3 --all --score-only --workers 8
"""
import argparse
import hashlib
import json
import os
import glob
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# The evaluators import NumPy/SciPy verifiers in the parent process.  Keep
# their native thread pools bounded before those imports so concurrent agent
# runs do not stall before reaching the model API.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_env] = "1"

from evaluation.runner import run_problem
from evaluation.runner import protocol_fingerprint as runner_protocol_fingerprint
from evaluation.pipeline_runner import run_pipeline
from evaluation.pipeline_runner import (
    protocol_fingerprint as pipeline_protocol_fingerprint,
)
from evaluation.l1_verifier import SCORER_VERSION as L1_SCORER_VERSION
from evaluation.l1_verifier import score_l1_question
from evaluation.l2_verifier import SCORER_VERSION as L2_SCORER_VERSION
from evaluation.l2_verifier import score_l2_question
from evaluation.l3_verifier import SCORER_VERSION as L3_SCORER_VERSION
from evaluation.l3_verifier import score_l3_question
from evaluation.l4_verifier import SCORER_VERSION as L4_SCORER_VERSION
from evaluation.l4_verifier import score_repaired_question
from evaluation.l4_generic_verifier import (
    SCORER_VERSION as L4_GENERIC_SCORER_VERSION,
    score_l4_generic_question,
)

# bump whenever the fingerprint algorithm changes: dicts from another
# algorithm version are treated as absent, never as comparable
PROVENANCE_VERSION = 2


def sample_provenance(meta, level, sample_id):
    """Fingerprint of the task exactly as a model would see it.

    ``prompt_md5`` hashes meta['question'] — the complete prompt handed to
    the runner, including top-level instructions and answer-schema markers
    that the per-question strings do not repeat (§10.1). ``questions_md5``
    additionally pins the per-question strings the scorers read, so a
    generator inconsistency between the two can never go unnoticed."""
    prompt_md5 = hashlib.md5(meta['question'].encode()).hexdigest()
    questions_md5 = hashlib.md5('\n'.join(
        q['question'] for q in meta['questions']).encode()).hexdigest()
    npy_path = os.path.join(LEVEL_DIRS[level], f'{sample_id}.npy')
    with open(npy_path, 'rb') as f:
        npy_md5 = hashlib.md5(f.read()).hexdigest()
    return {'v': PROVENANCE_VERSION, 'prompt_md5': prompt_md5,
            'questions_md5': questions_md5, 'npy_md5': npy_md5}


def stored_provenance(existing):
    """A stored result's provenance, or None when it is missing or was
    computed by a different fingerprint algorithm (not comparable)."""
    stored = (existing or {}).get('provenance')
    if isinstance(stored, dict) and stored.get('v') == PROVENANCE_VERSION:
        return stored
    return None


# single source of truth for the per-level deterministic scorer versions,
# used both in scoring_context and in the stored result's scorer_version
DETERMINISTIC_SCORER_VERSIONS = {
    'L1': L1_SCORER_VERSION,
    'L2': L2_SCORER_VERSION,
    'L3': L3_SCORER_VERSION,
    'L4': f'{L4_SCORER_VERSION}+{L4_GENERIC_SCORER_VERSION}',
}
from evaluation.l5_verifier import SCORER_VERSION as L5_SCORER_VERSION
from evaluation.l5_verifier import score_l5_response

LEVEL_DIRS = {
    'L1': 'data/L1',
    'L2': 'data/L2',
    'L3': 'data/L3',
    'L4': 'data/L4',
    'L5': 'data/L5',
}

RESULTS_DIR = 'eval_results'
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def load_problem(level, sample_id):
    d = LEVEL_DIRS[level]
    with open(os.path.join(d, f"{sample_id}.json")) as f:
        return json.load(f)


def get_all_ids(level):
    d = LEVEL_DIRS[level]
    files = sorted(glob.glob(os.path.join(d, f"EMRB_{level}_*.json")))
    return [os.path.basename(f).replace('.json', '') for f in files]


def _model_dir(model, pipeline=False):
    name = f"{model}-pipeline" if pipeline else model
    return name


def save_result(result, level, sample_id, model, pipeline=False):
    out_dir = os.path.join(RESULTS_DIR, _model_dir(model, pipeline), level)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sample_id}.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return path


def load_result(level, sample_id, model, pipeline=False):
    path = os.path.join(RESULTS_DIR, _model_dir(model, pipeline), level, f"{sample_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def score_problem(meta, response, level):
    """Score one response with fully deterministic per-level verifiers."""
    if level == 'L5':
        return score_l5_response(meta, response)

    scores = {}
    total_score = 0
    total_max = 0

    def _score_l4(q, r):
        repaired = score_repaired_question(q, r)
        if repaired is not None:
            return repaired
        return score_l4_generic_question(
            q, r, meta['generation_params']['signals'])

    level_verifiers = {
        'L1': lambda q, r: score_l1_question(
            q, r, meta['generation_params']['signals']),
        'L2': lambda q, r: score_l2_question(
            q, r, meta['generation_params']['signals']),
        'L3': lambda q, r: score_l3_question(
            q, r, meta['generation_params']['signals']),
        'L4': _score_l4,
    }

    verifier = level_verifiers[level]
    for q in meta['questions']:
        qid = q['id']
        rubric = q.get('rubric', {})
        q_max = rubric.get('points', 20)
        total_max += q_max

        result = verifier(q, response)
        if result is not None:
            scores[qid] = result
            total_score += result['total_score']
        else:
            scores[qid] = {'total_score': 0, 'error': 'verifier returned None'}

    scoring_context = {
        'scorer': DETERMINISTIC_SCORER_VERSIONS[level],
        'questions': [q['id'] for q in meta['questions']],
    }
    return scores, round(total_score, 1), total_max, scoring_context


def expected_scorer_version(level):
    """The scorer_version string current results must carry."""
    return (L5_SCORER_VERSION if level == 'L5'
            else DETERMINISTIC_SCORER_VERSIONS[level])


def run_protocol(use_pipeline, max_turns, model):
    fingerprint_fn = (pipeline_protocol_fingerprint if use_pipeline
                      else runner_protocol_fingerprint)
    return fingerprint_fn(model, max_turns)


def evaluate_one(level, sample_id, model, max_turns, skip_existing=False,
                 score_only=False, verbose=True, use_pipeline=False):
    """Evaluate one problem end-to-end. Thread-safe."""
    meta = load_problem(level, sample_id)
    sample_dir = os.path.abspath(LEVEL_DIRS[level])
    existing = load_result(level, sample_id, model, pipeline=use_pipeline)

    provenance = sample_provenance(meta, level, sample_id)
    protocol = run_protocol(use_pipeline, max_turns, model)

    if score_only:
        if not existing or 'response' not in existing:
            raise FileNotFoundError(
                f"No existing response for {model}/{level}/{sample_id}; "
                "score-only will not run model inference"
            )
        stored = stored_provenance(existing)
        if stored is not None and stored != provenance:
            raise ValueError(
                f"{sample_id}: the stored response was generated for a "
                f"different question/waveform version — re-run inference "
                f"instead of --score-only")
        protocol = existing.get('protocol', protocol)
        response = existing['response']
        run_info = existing.get('run_info', {})
    elif (skip_existing and existing and 'response' in existing
          and stored_provenance(existing) == provenance
          and existing.get('protocol') == protocol):
        if existing.get('scorer_version') == expected_scorer_version(level):
            log(f"[{sample_id}] skip (exists; provenance, protocol and "
                f"scorer version verified)")
            return existing
        log(f"[{sample_id}] stored scorer_version "
            f"{existing.get('scorer_version')!r} is stale — rescoring")
        response = existing['response']
        run_info = existing.get('run_info', {})
    else:
        if skip_existing and existing and 'response' in existing:
            log(f"[{sample_id}] existing result is stale "
                f"(task or protocol changed) — re-running")
        runner_fn = run_pipeline if use_pipeline else run_problem
        mode_tag = "pipeline" if use_pipeline else "free-form"
        log(f"[{sample_id}] running {model} ({mode_tag})...")
        run_result = runner_fn(
            meta['question'], sample_dir, model=model,
            max_turns=max_turns, verbose=verbose,
        )
        response = run_result['response']
        run_info = {
            'turns': run_result['turns'],
            'code_calls': run_result['code_calls'],
            'elapsed_s': run_result['elapsed_s'],
        }
        if use_pipeline:
            run_info['pipeline'] = True
            run_info['stage_info'] = run_result.get('stage_info', {})
        conv_dir = os.path.join(RESULTS_DIR, _model_dir(model, use_pipeline), level, 'conversations')
        os.makedirs(conv_dir, exist_ok=True)
        with open(os.path.join(conv_dir, f"{sample_id}_conv.json"), 'w') as f:
            json.dump(run_result['messages'], f, indent=2,
                      ensure_ascii=False, default=str)

    scores, total_score, total_max, scoring_context = score_problem(
        meta, response, level
    )

    final = {
        'sample_id': sample_id, 'level': level, 'model': model,
        'response': response, 'run_info': run_info,
        'provenance': provenance,
        'protocol': protocol,
        'scores': scores, 'total_score': round(total_score, 1),
        'total_max': total_max,
        'scorer_version': expected_scorer_version(level),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if scoring_context is not None:
        final['scoring_context'] = scoring_context

    save_result(final, level, sample_id, model, pipeline=use_pipeline)
    pct = total_score / total_max * 100 if total_max else 0
    log(f"[{sample_id}] {total_score:.1f}/{total_max} ({pct:.0f}%) "
        f"turns={run_info.get('turns', '?')} time={run_info.get('elapsed_s', '?')}s")
    return final


def print_summary(results):
    if not results:
        return
    print(f"\n{'='*65}")
    print(f"{'Sample ID':<20} {'Score':>6} {'Max':>5} {'%':>6} {'Turns':>6} {'Time':>7}")
    print(f"{'-'*65}")
    total_s, total_m = 0, 0
    for r in sorted(results, key=lambda x: x['sample_id']):
        s = r.get('total_score', 0)
        m = r.get('total_max', 100)
        pct = s / m * 100 if m > 0 else 0
        turns = r.get('run_info', {}).get('turns', '?')
        elapsed = r.get('run_info', {}).get('elapsed_s', '?')
        t_str = f"{elapsed}s" if isinstance(elapsed, (int, float)) else elapsed
        print(f"{r['sample_id']:<20} {s:>6.1f} {m:>5} {pct:>5.1f}% {str(turns):>6} {t_str:>7}")
        total_s += s
        total_m += m
    n = len(results)
    print(f"{'-'*65}")
    avg = total_s / total_m * 100 if total_m > 0 else 0
    print(f"{'AVERAGE (' + str(n) + ')':<20} {total_s/n:>6.1f} {total_m//n:>5} {avg:>5.1f}%")
    print(f"{'='*65}")


def main():
    global RESULTS_DIR
    parser = argparse.ArgumentParser(description='EMRB Benchmark Evaluation')
    parser.add_argument('--level', required=True,
                        choices=['L1', 'L2', 'L3', 'L4', 'L5'])
    parser.add_argument('--id', help='Specific sample ID')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--n', type=int, help='Evaluate first N problems')
    parser.add_argument('--model', default='gpt-4o')
    parser.add_argument('--max-turns', type=int, default=15)
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers (default: 1)')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--pipeline', action='store_true',
                        help='Use structured 3-stage pipeline instead of free-form agent')
    parser.add_argument(
        '--results-dir', default=RESULTS_DIR,
        help='Output root for result JSON and conversations')
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir

    verbose = args.workers == 1

    if args.id:
        evaluate_one(args.level, args.id, args.model, args.max_turns,
                     score_only=args.score_only, verbose=True,
                     use_pipeline=args.pipeline)
        return

    if not (args.all or args.n):
        parser.error("Specify --id, --all, or --n")

    ids = get_all_ids(args.level)
    if args.n:
        ids = ids[:args.n]

    mode = "pipeline" if args.pipeline else "free-form"
    print(f"Evaluating {len(ids)} {args.level} problems with {args.model} "
          f"({args.workers} workers, {mode})")
    t0 = time.time()
    results = []

    if args.workers <= 1:
        for i, sid in enumerate(ids):
            try:
                r = evaluate_one(args.level, sid, args.model, args.max_turns,
                                 skip_existing=args.skip_existing,
                                 score_only=args.score_only,
                                 verbose=True,
                                 use_pipeline=args.pipeline)
                results.append(r)
            except Exception as e:
                log(f"[{sid}] FAILED: {e}")
    else:
        def _run(sid):
            return evaluate_one(
                args.level, sid, args.model, args.max_turns,
                skip_existing=args.skip_existing,
                score_only=args.score_only,
                verbose=False,
                use_pipeline=args.pipeline,
            )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run, sid): sid for sid in ids}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    log(f"[{sid}] FAILED: {e}")

    elapsed = time.time() - t0
    print_summary(results)
    print(f"Total time: {elapsed:.0f}s ({elapsed/max(len(results),1):.0f}s/problem)")


if __name__ == '__main__':
    main()
