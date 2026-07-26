import glob
import json
from collections import Counter

import evaluate
from evaluation.l4_verifier import (
    SCORER_VERSION,
    format_reference_response,
    is_repaired_question,
    reference_payload,
    score_repaired_question,
)
from evaluation.runner import _force_answer_prompts


def _metadata(sample_id):
    with open(f'data/L4/{sample_id}.json') as handle:
        return json.load(handle)


def _question(sample_id, question_type):
    return next(
        question for question in _metadata(sample_id)['questions']
        if question['question_type'] == question_type
        and is_repaired_question(question)
    )


def _response(question, payload):
    return (
        f"===ANSWERS===\n{question['id']}: "
        + json.dumps(payload)
        + "\n===END==="
    )


def _criterion(result, identifier):
    return next(item for item in result['sub_scores'] if item['id'] == identifier)


def test_repaired_inventory_preserves_all_l4_problems_and_question_slots():
    problem_count = 0
    repaired_problem_count = 0
    repaired_types = Counter()
    for path in sorted(glob.glob('data/L4/EMRB_L4_*.json')):
        with open(path) as handle:
            metadata = json.load(handle)
        problem_count += 1
        assert len(metadata['questions']) == 5
        assert metadata['total_points'] == 100
        repaired = [
            question for question in metadata['questions']
            if is_repaired_question(question)
        ]
        if repaired:
            repaired_problem_count += 1
            assert metadata['answer_schema_version'] == 'emrb-l4-repaired-v1'
            assert 'emrb-l4-repaired-v1' in metadata['question']
            repaired_types.update(q['question_type'] for q in repaired)
        else:
            assert 'answer_schema_version' not in metadata
    assert problem_count == 40
    assert repaired_problem_count == 21
    assert repaired_types == {'QT05': 10, 'QT07': 16}


def test_every_repaired_reference_answer_scores_full_credit():
    checked = 0
    for path in sorted(glob.glob('data/L4/EMRB_L4_*.json')):
        with open(path) as handle:
            metadata = json.load(handle)
        for question in metadata['questions']:
            if not is_repaired_question(question):
                continue
            result = score_repaired_question(
                question, format_reference_response(question)
            )
            assert result['total_score'] == 20.0
            assert result['total_max'] == 20.0
            assert result['method'] == SCORER_VERSION
            checked += 1
    assert checked == 26


def test_gap_design_fields_are_scored_independently():
    question = _question('EMRB_L4_1005', 'QT05')
    payload = reference_payload(question)
    payload['raw_gap_bounds_MHz'] = [-1, 1]
    payload['raw_gap_MHz'] = 2
    payload['symbol_rate_ksps'] *= 0.4
    payload['minimum_received_power_dBm'] += 10
    result = score_repaired_question(question, _response(question, payload))
    assert _criterion(result, 'gap_raw_bounds')['score'] == 0
    assert _criterion(result, 'gap_raw_bandwidth')['score'] == 0
    assert _criterion(result, 'gap_symbol_rate')['score'] == 0
    assert _criterion(result, 'gap_minimum_power')['score'] == 0
    assert result['total_score'] < result['total_max']


def test_interference_pair_order_is_not_part_of_the_answer():
    question = _question('EMRB_L4_1000', 'QT07')
    payload = reference_payload(question)
    payload['pair'].reverse()
    result = score_repaired_question(question, _response(question, payload))
    assert result['total_score'] == 20.0


def test_wrong_spectral_relation_gates_the_corresponding_geometry():
    question = _question('EMRB_L4_1000', 'QT07')
    payload = reference_payload(question)
    payload['spectral_relation'] = 'overlapping'
    payload['overlap_MHz'] = 0
    result = score_repaired_question(question, _response(question, payload))
    assert _criterion(result, 'interference_relation')['score'] == 0
    assert _criterion(result, 'interference_overlap_or_gap')['score'] == 0


def test_overlap_isolation_claim_is_checked_against_hidden_geometry():
    question = _question('EMRB_L4_1035', 'QT07')
    assert question['ground_truth']['spectral_relation'] == 'overlapping'
    payload = reference_payload(question)
    payload['full_band_isolation_possible'] = True
    payload['nonoverlapped_target_bandwidth_MHz'] = 99
    payload['overlapped_target_fraction_pct'] = 0
    result = score_repaired_question(question, _response(question, payload))
    assert _criterion(result, 'interference_full_isolation')['score'] == 0
    assert _criterion(result, 'interference_isolation_geometry')['score'] == 0


def test_invalid_question_level_json_scores_zero_without_fallback():
    question = _question('EMRB_L4_1005', 'QT05')
    result = score_repaired_question(
        question,
        f"===ANSWERS===\n{question['id']}: not-json\n===END===",
    )
    assert result['total_score'] == 0
    assert result['parse_error']


def test_integrated_evaluator_uses_deterministic_path(monkeypatch):
    from evaluation.l4_generic_verifier import (
        SCORER_VERSION as GENERIC_VERSION,
        is_l4_generic_question,
        reference_response_lines,
    )
    metadata = _metadata('EMRB_L4_1030')
    repaired = [q for q in metadata['questions'] if is_repaired_question(q)]
    generic = [q for q in metadata['questions'] if is_l4_generic_question(q)]
    assert len(repaired) + len(generic) == len(metadata['questions'])
    answer_lines = [
        f"{q['id']}: {json.dumps(reference_payload(q))}" for q in repaired
    ]
    for q in generic:
        answer_lines += reference_response_lines(q)
    response = "===ANSWERS===\n" + "\n".join(answer_lines) + "\n===END==="

    scores, total, maximum, context = evaluate.score_problem(
        metadata, response, 'L4'
    )
    assert total == 100.0
    assert maximum == 100
    assert context == {
        'scorer': f'{SCORER_VERSION}+{GENERIC_VERSION}',
        'questions': [q['id'] for q in metadata['questions']],
    }
    # evaluate_one derives the stored scorer_version from this same constant,
    # so the top-level version cannot silently omit the generic scorer again
    assert evaluate.DETERMINISTIC_SCORER_VERSIONS['L4'] == \
        f'{SCORER_VERSION}+{GENERIC_VERSION}'
    assert all(scores[q['id']]['method'] == SCORER_VERSION for q in repaired)
    assert all(scores[q['id']]['method'] == GENERIC_VERSION for q in generic)


def test_force_answer_prompt_preserves_question_level_json_objects():
    prompts = _force_answer_prompts('format emrb-l4-repaired-v1')
    assert all('Q1' in prompt and 'Q5' in prompt for prompt in prompts)
    assert all('JSON' in prompt for prompt in prompts)
