import json
from pathlib import Path

from evaluate import score_problem
from evaluation.l5_verifier import (
    SCORER_VERSION,
    extract_answer_json,
    format_reference_response,
    reference_answer,
    score_l5_response,
)
from evaluation.runner import _force_answer_prompts


def _metadata():
    with open('data/L5/EMRB_L5_2000.json') as handle:
        return json.load(handle)


def _response(answer):
    return 'analysis before final\n===ANSWERS===\n' + json.dumps(answer) + '\n===END==='


def _criterion(scores, question_id, subquestion_id, criterion_id):
    subscore = next(
        item for item in scores[question_id]['sub_scores']
        if item['id'] == subquestion_id
    )
    return next(item for item in subscore['criteria'] if item['id'] == criterion_id)


def test_reference_answer_scores_exactly_100_without_judge():
    meta = _metadata()
    scores, total, maximum, context = score_l5_response(
        meta, format_reference_response(meta)
    )
    assert total == 100.0
    assert maximum == 100.0
    assert {key: value['total_score'] for key, value in scores.items()} == {
        'Q1': 34.0, 'Q2': 33.0, 'Q3': 33.0,
    }
    assert context['parse_error'] is None
    assert all(value['method'] == SCORER_VERSION for value in scores.values())


def test_main_evaluation_path_uses_same_deterministic_verifier():
    meta = _metadata()
    direct = score_l5_response(meta, format_reference_response(meta))
    integrated = score_problem(
        meta, format_reference_response(meta), 'L5'
    )
    assert integrated == direct


def test_empty_or_malformed_json_scores_zero():
    meta = _metadata()
    for response in ('', '===ANSWERS===\nnot json\n===END==='):
        _, total, maximum, context = score_l5_response(meta, response)
        assert total == 0.0
        assert maximum == 100.0
        assert context['parse_error']


def test_wrong_schema_version_scores_zero():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['schema_version'] = 'obsolete-schema'
    _, total, maximum, context = score_l5_response(meta, _response(answer))
    assert total == 0.0
    assert maximum == 100.0
    assert 'unsupported' in context['parse_error']


def test_l5_prompts_define_tasks_without_exposing_verifier_formulas():
    forbidden_fragments = (
        'R_max =',
        'P_chirp + 10log10',
        'P_comm - G_processing',
        'U(alpha) =',
        'sum_i B_i log2',
        'P_i = B_i',
        "If fraction r of a signal's occupied interval",
        'standard monostatic radar equation',
        'maximum range',
        'weighted geometric mean',
    )
    for path in Path('data/L5').glob('EMRB_L5_*.json'):
        metadata = json.loads(path.read_text())
        question = metadata['question']
        assert metadata['answer_schema_version'] == 'emrb-l5-verifiable-v5'
        assert not any(fragment in question for fragment in forbidden_fragments)
        assert 'symbol_anchor_index' not in question
        assert 'symbol_windows_iq' not in question


def test_force_answer_prompt_preserves_current_l5_json_schema():
    prompt = _force_answer_prompts('schema emrb-l5-verifiable-v5')[1]
    assert 'emrb-l5-verifiable-v5' in prompt
    assert 'Q1a: [value]' not in prompt


def test_parser_uses_answer_block_and_ignores_reasoning_json():
    answer = {'schema_version': 'test', 'Q1a': {'signals': []}}
    response = '{"decoy": true}\n===ANSWERS===\n' + json.dumps(answer) + '\n===END==='
    assert extract_answer_json(response) == answer


def test_missing_and_hallucinated_catalog_entries_reduce_detection_score():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q1a']['signals'].pop()
    answer['Q1a']['signals'].append({
        'id': 'fake',
        'center_MHz': 99,
        'bandwidth_MHz': 1,
        'modulation': 'QPSK',
        'power_dBm': -20,
    })
    scores, total, _, _ = score_l5_response(meta, _response(answer))
    detection = _criterion(scores, 'Q1', 'Q1a', 'Q1a_detection')
    assert detection['score'] < detection['max']
    assert total < 100.0


def test_q1_design_is_recomputed_instead_of_trusting_claimed_metrics():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q1c']['passband_MHz'] = [-10, 10]
    answer['Q1c']['post_sir_dB'] = 999
    answer['Q1c']['improvement_dB'] = 999
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    objective = _criterion(scores, 'Q1', 'Q1c', 'Q1c_objective')
    reported = _criterion(scores, 'Q1', 'Q1c', 'Q1c_reported_metrics')
    assert objective['score'] < objective['max']
    assert reported['score'] == 0


def test_wrong_q1_pair_gates_pair_dependent_measurements_and_filter_design():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q1b']['pair_ids'] = [
        answer['Q2a']['signal_id'], answer['Q1b']['target_id']
    ]
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    assert _criterion(scores, 'Q1', 'Q1b', 'Q1b_overlap')['score'] == 0
    assert _criterion(scores, 'Q1', 'Q1b', 'Q1b_sir')['score'] == 0
    assert _criterion(scores, 'Q1', 'Q1c', 'Q1c_feasibility')['score'] == 0
    assert _criterion(scores, 'Q1', 'Q1c', 'Q1c_objective')['score'] == 0


def test_q1_channel_centers_must_satisfy_guards():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q1d']['additional_centers_MHz'] = [0.0] * answer['Q1d']['additional_channel_count']
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    placement = _criterion(scores, 'Q1', 'Q1d', 'Q1d_placement')
    assert placement['score'] < placement['max']


def test_incomplete_q1_deployment_does_not_receive_placement_credit():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q1d']['additional_centers_MHz'].pop()
    answer['Q1d']['additional_channel_count'] -= 1
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    assert _criterion(scores, 'Q1', 'Q1d', 'Q1d_count')['score'] == 0
    assert _criterion(scores, 'Q1', 'Q1d', 'Q1d_placement')['score'] == 0


def test_wrong_q3_target_gates_modulation_recommendation():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q3b']['worst_id'] = next(
        link['id'] for link in answer['Q3a']['links']
        if link['id'] != answer['Q3b']['worst_id']
    )
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    for criterion_id in ('Q3b_modulation', 'Q3b_bits', 'Q3b_improvement'):
        assert _criterion(scores, 'Q3', 'Q3b', criterion_id)['score'] == 0


def test_incomplete_digital_link_set_gates_global_water_filling():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q3a']['links'].pop()
    scores, _, _, context = score_l5_response(meta, _response(answer))
    assert not context['prerequisites']['q3_complete_digital_link_set']
    for criterion in next(
        item for item in scores['Q3']['sub_scores'] if item['id'] == 'Q3c'
    )['criteria']:
        assert criterion['score'] == 0


def test_wrong_chirp_identity_gates_all_chirp_dependent_q2_work():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q2a']['signal_id'] = next(
        signal['id'] for signal in answer['Q1a']['signals']
        if signal['id'] != answer['Q2a']['signal_id']
    )
    scores, _, _, context = score_l5_response(meta, _response(answer))
    assert not context['prerequisites']['q2_chirp_identified']
    q2a = next(item for item in scores['Q2']['sub_scores'] if item['id'] == 'Q2a')
    assert all(
        criterion['score'] == 0
        for criterion in q2a['criteria']
        if criterion['id'] != 'Q2a_identity'
    )
    for subquestion_id in ('Q2b', 'Q2c', 'Q2d'):
        subquestion = next(
            item for item in scores['Q2']['sub_scores']
            if item['id'] == subquestion_id
        )
        assert subquestion['score'] == 0


def test_wrong_q2_modulation_loses_recovery_points_without_faking_dependency():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q2b']['modulation'] = 'QPSK'
    scores, total, _, context = score_l5_response(meta, _response(answer))
    modulation = _criterion(scores, 'Q2', 'Q2b', 'Q2b_modulation')
    assert modulation['score'] == 0
    assert context['prerequisites']['q2_victim_measurements_valid']
    assert next(
        item for item in scores['Q2']['sub_scores'] if item['id'] == 'Q2c'
    )['score'] == 4.0
    assert next(
        item for item in scores['Q2']['sub_scores'] if item['id'] == 'Q2d'
    )['score'] == 10.0
    assert total == 97.0


def test_wrong_q2_measurement_does_not_erase_independent_recovery_credit():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q2b']['center_MHz'] += 1.0
    scores, total, _, context = score_l5_response(meta, _response(answer))
    assert not context['prerequisites']['q2_victim_measurements_valid']
    for subquestion_id in ('Q2c', 'Q2d'):
        subquestion = next(
            item for item in scores['Q2']['sub_scores']
            if item['id'] == subquestion_id
        )
        assert subquestion['score'] == subquestion['max']
    assert total == 99.0


def test_q2_crossing_times_are_scored_from_the_hidden_trajectory():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q2c']['entry_time_ms'] += 0.5
    answer['Q2c']['duration_ms'] = 0
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    assert _criterion(scores, 'Q2', 'Q2c', 'Q2c_entry_time')['score'] == 0
    assert _criterion(scores, 'Q2', 'Q2c', 'Q2c_duration')['score'] == 0
    assert _criterion(scores, 'Q2', 'Q2c', 'Q2c_exit_time')['score'] > 0


def test_symbol_recovery_accepts_common_scale_phase_and_conjugation():
    meta = _metadata()
    answer = reference_answer(meta)
    transformed = []
    rotation = complex(1.7, -0.8)
    for real, imag in answer['Q2d']['symbols_iq']:
        value = rotation * complex(real, -imag)
        transformed.append([value.real, value.imag])
    answer['Q2d']['symbols_iq'] = transformed
    scores, total, _, _ = score_l5_response(meta, _response(answer))
    symbols = _criterion(scores, 'Q2', 'Q2d', 'Q2d_symbols')
    assert symbols['score'] == symbols['max']
    assert total == 100.0


def test_symbol_recovery_accepts_small_timing_alignment_error():
    meta = _metadata()
    answer = reference_answer(meta)
    q2_ground_truth = next(
        question['ground_truth'] for question in meta['questions']
        if question['id'] == 'Q2'
    )
    answer['Q2d']['symbols_iq'] = q2_ground_truth['Q2d']['symbol_windows_iq'][0]
    scores, total, _, _ = score_l5_response(meta, _response(answer))
    assert _criterion(scores, 'Q2', 'Q2d', 'Q2d_symbols')['score'] == 9.0
    assert total == 100.0


def test_wrong_symbol_order_does_not_receive_recovery_credit():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q2d']['symbols_iq'].reverse()
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    symbols = _criterion(scores, 'Q2', 'Q2d', 'Q2d_symbols')
    assert symbols['score'] < 3.0


def test_short_symbol_block_receives_only_proportional_credit():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q2d']['symbols_iq'] = answer['Q2d']['symbols_iq'][:16]
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    symbols = _criterion(scores, 'Q2', 'Q2d', 'Q2d_symbols')
    assert 0 < symbols['score'] <= 4.5


def test_wrong_q2d_victim_identity_gates_hidden_symbol_credit():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q2d']['victim_id'] = next(
        signal['id'] for signal in answer['Q1a']['signals']
        if signal['id'] != answer['Q2d']['victim_id']
    )
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    assert _criterion(scores, 'Q2', 'Q2d', 'Q2d_victim')['score'] == 0
    assert _criterion(scores, 'Q2', 'Q2d', 'Q2d_symbols')['score'] == 0


def test_ofdm_claims_do_not_override_power_and_leakage_constraints():
    meta = _metadata()
    answer = reference_answer(meta)
    answer['Q3d']['power_dBm'] = 0
    answer['Q3d']['rate_Mbps'] = 999
    answer['Q3d']['adjacent_leakage_dBm'] = -100
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    feasibility = _criterion(scores, 'Q3', 'Q3d', 'Q3d_feasibility')
    metrics = _criterion(scores, 'Q3', 'Q3d', 'Q3d_reported_metrics')
    assert feasibility['score'] == 0
    assert metrics['score'] < metrics['max']


def test_nonunique_ofdm_center_is_accepted_when_constraints_hold():
    meta = _metadata()
    answer = reference_answer(meta)
    q3d = answer['Q3d']
    gap_lo, gap_hi = q3d['gap_MHz']
    bandwidth = q3d['occupied_bandwidth_MHz']
    q3d['center_MHz'] = gap_lo + 0.10 + bandwidth / 2 + 0.01
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    feasibility = _criterion(scores, 'Q3', 'Q3d', 'Q3d_feasibility')
    assert feasibility['score'] == feasibility['max']


def test_ofdm_power_rounded_to_two_decimals_remains_feasible():
    meta = _metadata()
    answer = reference_answer(meta)
    q3d = answer['Q3d']
    q3d['power_dBm'] = round(q3d['power_dBm'], 2)
    q3d['adjacent_leakage_dBm'] = round(q3d['adjacent_leakage_dBm'], 2)
    scores, _, _, _ = score_l5_response(meta, _response(answer))
    feasibility = _criterion(scores, 'Q3', 'Q3d', 'Q3d_feasibility')
    assert feasibility['score'] == feasibility['max']
