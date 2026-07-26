import json


def test_all_l2_questions_marked_and_reference_gets_full_credit():
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from evaluation import l2_verifier as v
    paths = sorted((root / 'data/L2').glob('EMRB_L2_*.json'))
    assert len(paths) == 40
    checked = 0
    for path in paths:
        meta = json.loads(path.read_text())
        response = v.reference_response(meta)
        signals = meta['generation_params']['signals']
        for question in meta['questions']:
            assert v.is_l2_deterministic_question(question), \
                (meta['sample_id'], question['id'])
            result = v.score_l2_question(question, response, signals)
            assert result['total_score'] == 20.0, \
                (meta['sample_id'], question['id'], result['sub_scores'])
            checked += 1
    assert checked == 200


def test_l2_q1_wrong_window_choice_scores_zero():
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from evaluation import l2_verifier as v
    meta = json.loads(sorted(
        (root / 'data/L2').glob('EMRB_L2_*.json'))[0].read_text())
    response = v.reference_response(meta).replace(
        'Hamming window is more suitable',
        'The rectangular window is more suitable')
    q1 = meta['questions'][0]
    result = v.score_l2_question(q1, response)
    choice = next(c for c in result['sub_scores']
                  if c['id'] == 'window_choice')
    assert choice['score'] == 0.0
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.l2_verifier import (
    SCORER_VERSION,
    format_reference_response,
    is_l2_deterministic_question,
    reference_payload,
    score_l2_question,
)


def _all_metadata():
    paths = sorted((ROOT / 'data/L2').glob('EMRB_L2_*.json'))
    assert len(paths) == 40
    return [json.loads(path.read_text()) for path in paths]


def _q3(meta):
    question = meta['questions'][2]
    assert question['id'] == 'Q3'
    return question


def _response(question, payload):
    return (f"===ANSWERS===\n"
            f"{question['id']}: {json.dumps(payload)}\n"
            f"===END===")


def test_every_sample_uses_the_deterministic_q3():
    for meta in _all_metadata():
        question = _q3(meta)
        assert is_l2_deterministic_question(question)
        assert question['ground_truth']['schema'] == 'emrb-l2-autocorr-v1'
        assert question['rubric']['points'] == 20


def test_reference_answer_receives_full_credit_on_all_samples():
    for meta in _all_metadata():
        question = _q3(meta)
        result = score_l2_question(question, format_reference_response(question))
        assert result['total_score'] == 20.0, meta['sample_id']
        assert result['method'] == SCORER_VERSION


def test_ground_truth_is_consistent_with_generation_params():
    for meta in _all_metadata():
        question = _q3(meta)
        gt = question['ground_truth']
        signals = meta['generation_params']['signals']
        src_type = {'FM': 'FM', 'AM': 'AM-DSB'}[gt['source_signal']]
        sources = [s for s in signals if s['type'] == src_type]
        assert len(sources) == 1, meta['sample_id']
        if gt['source_signal'] == 'AM':
            assert not any(s['type'] == 'FM' for s in signals)
        theory_us = 1e3 / sources[0]['modulating_frequency_kHz']
        assert abs(gt['comb_spacing_us'] - theory_us) <= 0.03 * theory_us
        assert gt['comb_persists_after_filtering'] is \
            (gt['filter_target'] == 'source')
        assert gt['n_comb_peaks'] >= 3
        assert 0.0 < gt['max_R_magnitude'] <= 1.0


def test_q3_categorical_answers_are_diverse():
    sources, persists = set(), set()
    for meta in _all_metadata():
        gt = _q3(meta)['ground_truth']
        sources.add(gt['source_signal'])
        persists.add(gt['comb_persists_after_filtering'])
    assert sources == {'FM', 'AM'}, sources
    assert persists == {True, False}, persists


def test_wrong_comb_spacing_loses_spacing_points_only():
    question = _q3(_all_metadata()[0])
    payload = reference_payload(question)
    payload['comb_spacing_us'] = payload['comb_spacing_us'] * 2
    result = score_l2_question(question, _response(question, payload))
    assert result['total_score'] == 14.0


def test_spacing_within_partial_tolerance_gets_half_credit():
    question = _q3(_all_metadata()[0])
    payload = reference_payload(question)
    payload['comb_spacing_us'] = payload['comb_spacing_us'] * 1.10
    result = score_l2_question(question, _response(question, payload))
    assert result['total_score'] == 17.0


def test_unit_mistake_in_spacing_gets_no_spacing_credit():
    question = _q3(_all_metadata()[0])
    payload = reference_payload(question)
    payload['comb_spacing_us'] = payload['comb_spacing_us'] / 1000  # reported in ms
    result = score_l2_question(question, _response(question, payload))
    assert result['total_score'] == 14.0


def test_contradictory_source_signal_gets_no_credit():
    question = _q3(_all_metadata()[0])
    payload = reference_payload(question)
    payload['source_signal'] = 'either FM or AM'
    result = score_l2_question(question, _response(question, payload))
    assert result['total_score'] == 17.0


def test_boolean_requires_exact_token():
    question = _q3(_all_metadata()[0])
    gt_value = question['ground_truth']['comb_persists_after_filtering']
    for bad in ('insufficient', 'the comb mostly disappears', '不可以', 1, 0):
        payload = reference_payload(question)
        payload['comb_persists_after_filtering'] = bad
        result = score_l2_question(question, _response(question, payload))
        assert result['total_score'] == 16.0, bad
    payload = reference_payload(question)
    payload['comb_persists_after_filtering'] = 'false' if not gt_value else 'true'
    result = score_l2_question(question, _response(question, payload))
    assert result['total_score'] == 20.0


def test_max_r_tolerance_boundaries():
    question = _q3(_all_metadata()[0])
    gt_value = question['ground_truth']['max_R_magnitude']
    for delta, expected in ((0.08, 20.0), (0.12, 18.0), (0.20, 16.0)):
        payload = reference_payload(question)
        payload['max_R_magnitude'] = gt_value + delta
        result = score_l2_question(question, _response(question, payload))
        assert result['total_score'] == expected, delta


def test_spacing_tolerance_boundaries():
    question = _q3(_all_metadata()[0])
    gt_value = question['ground_truth']['comb_spacing_us']
    for factor, expected in ((1.05, 20.0), (1.15, 17.0), (1.16, 14.0)):
        payload = reference_payload(question)
        payload['comb_spacing_us'] = gt_value * factor
        result = score_l2_question(question, _response(question, payload))
        assert result['total_score'] == expected, factor


def test_non_finite_and_junk_values_earn_nothing():
    question = _q3(_all_metadata()[0])
    for bad in (float('nan'), float('inf'), 'N/A', None, [1, 2]):
        payload = reference_payload(question)
        payload['comb_spacing_us'] = bad
        result = score_l2_question(question, _response(question, payload))
        assert result['total_score'] == 14.0, bad


def test_native_boolean_answers_are_accepted():
    question = _q3(_all_metadata()[0])
    payload = reference_payload(question)
    payload['comb_persists_after_filtering'] = False
    assert score_l2_question(
        question, _response(question, payload))['total_score'] == 20.0
    payload['comb_persists_after_filtering'] = True
    assert score_l2_question(
        question, _response(question, payload))['total_score'] == 16.0


def test_missing_fields_earn_nothing():
    question = _q3(_all_metadata()[0])
    result = score_l2_question(question, _response(question, {}))
    assert result['total_score'] == 0.0


def test_malformed_answer_scores_zero_with_parse_error():
    question = _q3(_all_metadata()[0])
    result = score_l2_question(question, "===ANSWERS===\nQ3: no json here\n===END===")
    assert result['total_score'] == 0.0
    assert 'parse_error' in result


def test_every_l2_question_is_claimed_and_unmarked_is_not():
    meta = _all_metadata()[0]
    q1 = meta['questions'][0]
    # the deterministic L2 scorer claims all five questions, so Q1 is claimed
    result = score_l2_question(q1, "===ANSWERS===\nQ1: 42\n===END===")
    assert result is not None and result['total_score'] < 20.0
    # an unmarked question is still left to other scorers
    unmarked = {**q1, 'rubric': {k: v for k, v in q1['rubric'].items()
                                 if k != 'scoring'}}
    assert score_l2_question(unmarked, "===ANSWERS===\nQ1: 42\n===END===") \
        is None


# --- §10.3 / §10.4 / §10.8: strict prose criteria -------------------------


def _part_response(line):
    return f"===ANSWERS===\n{line}\n===END==="


def _sub(result, cid):
    return next(c for c in result['sub_scores'] if c['id'] == cid)


def _q4_samples():
    burst = continuous = None
    for meta in _all_metadata():
        if meta['questions'][3]['ground_truth']['has_burst']:
            burst = burst or meta
        else:
            continuous = continuous or meta
    return burst, continuous


def test_q1_window_reason_requires_the_actual_tradeoff():
    q1 = _all_metadata()[0]['questions'][0]
    frag = score_l2_question(q1, _part_response('Q1b: Hamming sidelobe'))
    assert _sub(frag, 'window_choice')['score'] == 3.0
    assert _sub(frag, 'window_reason')['score'] == 0.0
    full = score_l2_question(q1, _part_response(
        'Q1b: Hamming: lower sidelobes suppress leakage that would mask '
        'weak signals, at the cost of a wider main lobe'))
    assert _sub(full, 'window_reason')['score'] == 3.0


def test_q2_planning_answer_requires_definition_distinction():
    q2 = _all_metadata()[0]['questions'][1]
    frag = score_l2_question(q2, _part_response('Q2d: occupied energy'))
    assert _sub(frag, 'planning_choice')['score'] == 0.0
    assert _sub(frag, 'definitions')['score'] == 0.0
    assert _sub(frag, 'planning_reason')['score'] == 0.0
    full = score_l2_question(q2, _part_response(
        'Q2d: the 3 dB bandwidth spans the half-power points, null-to-null '
        'spans the main lobe between spectral nulls, and the 99% bandwidth '
        'contains 99% of the energy; for planning use the 99% occupied '
        'bandwidth to limit adjacent-channel interference'))
    assert _sub(full, 'planning_choice')['score'] == 2.0
    assert _sub(full, 'definitions')['score'] == 2.0
    assert _sub(full, 'planning_reason')['score'] == 1.0


def test_q4_features_require_identity_binding():
    _, meta = _q4_samples()
    q4 = meta['questions'][3]
    sigs = meta['generation_params']['signals']
    frag = score_l2_question(
        q4, _part_response('Q4a: diagonal horizontal'), sigs)
    for cid in ('features_chirp', 'features_digital', 'features_analog'):
        assert _sub(frag, cid)['score'] == 0.0, cid
    bound = score_l2_question(q4, _part_response(
        'Q4a: the chirp is a diagonal line; the digital signal is a '
        'wideband stripe; the analog signal is a narrow horizontal line'),
        sigs)
    for cid, expected in (('features_chirp', 2.0), ('features_digital', 1.0),
                          ('features_analog', 1.0)):
        assert _sub(bound, cid)['score'] == expected, cid


def test_q4_no_burst_requires_explicit_verdict():
    _, meta = _q4_samples()
    q4 = meta['questions'][3]
    assert _sub(score_l2_question(q4, _part_response('Q4c: not')),
                'burst_timing')['score'] == 0.0
    assert _sub(score_l2_question(q4, _part_response(
        'Q4c: no burst; all signals are continuous')),
        'burst_timing')['score'] == 6.0


def test_q4_burst_times_must_be_bound():
    meta, _ = _q4_samples()
    q4 = meta['questions'][3]
    gt = q4['ground_truth']
    unlabeled = score_l2_question(q4, _part_response(
        f"Q4c: {gt['burst_end_ms']} ms {gt['burst_start_ms']} ms"))
    assert _sub(unlabeled, 'burst_timing')['score'] == 0.0  # no verdict
    verdict_only = score_l2_question(
        q4, _part_response('Q4c: yes, a burst is present'))
    assert _sub(verdict_only, 'burst_timing')['score'] == 2.0
    bag = score_l2_question(q4, _part_response(
        f"Q4c: burst present, times {gt['burst_end_ms']} and "
        f"{gt['burst_start_ms']}"))
    assert _sub(bag, 'burst_timing')['score'] == 2.0  # times stay unbound
    pair = score_l2_question(q4, _part_response(
        f"Q4c: there is a burst from {gt['burst_start_ms']} to "
        f"{gt['burst_end_ms']} ms"))
    assert _sub(pair, 'burst_timing')['score'] == 6.0
    labeled = score_l2_question(q4, _part_response(
        f"Q4c: burst present, start = {gt['burst_start_ms']} ms, "
        f"end = {gt['burst_end_ms']} ms"))
    assert _sub(labeled, 'burst_timing')['score'] == 6.0


def test_q5_unlabeled_number_bag_earns_nothing():
    meta = _all_metadata()[0]
    q5 = meta['questions'][4]
    entries = q5['ground_truth']['energy_per_signal']
    bag = '; '.join(f"{e['power_mW']} mW, {e['energy_J']} J"
                    for e in reversed(entries))
    result = score_l2_question(q5, _part_response(f'Q5a: {bag}'),
                               meta['generation_params']['signals'])
    assert _sub(result, 'power_energy')['score'] == 0.0


def test_q5_requires_power_and_energy_per_signal():
    meta = _all_metadata()[0]
    q5 = meta['questions'][4]
    entries = q5['ground_truth']['energy_per_signal']
    sigs = meta['generation_params']['signals']
    energy_only = '; '.join(f"{e['type']}: {e['energy_J']} J"
                            for e in entries)
    result = score_l2_question(
        q5, _part_response(f'Q5a: {energy_only}'), sigs)
    assert _sub(result, 'power_energy')['score'] == 2.5
    both = '; '.join(f"{e['type']}: {e['power_mW']} mW, {e['energy_J']} J"
                     for e in entries)
    result = score_l2_question(q5, _part_response(f'Q5a: {both}'), sigs)
    assert _sub(result, 'power_energy')['score'] == 5.0


def test_q5_unanchored_signal_earns_nothing_for_that_signal():
    meta = _all_metadata()[0]
    q5 = meta['questions'][4]
    entries = q5['ground_truth']['energy_per_signal']
    sigs = meta['generation_params']['signals']
    lines = [f"{e['type']}: {e['power_mW']} mW, {e['energy_J']} J"
             for e in entries[:-1]]
    lines.append(f"{entries[-1]['power_mW']} mW, "
                 f"{entries[-1]['energy_J']} J")
    result = score_l2_question(
        q5, _part_response('Q5a: ' + '; '.join(lines)), sigs)
    expected = round(5.0 * (len(entries) - 1) / len(entries), 2)
    assert _sub(result, 'power_energy')['score'] == expected


def test_q5_explanation_requires_relation_and_verification():
    meta = _all_metadata()[0]
    q5 = meta['questions'][4]
    gt = q5['ground_truth']
    frag = score_l2_question(q5, _part_response('Q5d: log linear'))
    assert _sub(frag, 'explanation')['score'] == 0.0
    full = score_l2_question(q5, _part_response(
        f"Q5d: dBm is logarithmic; convert each power to mW, add them in "
        f"the linear domain, then convert back: total = "
        f"{gt['total_received_power_dBm']} dBm"))
    assert _sub(full, 'explanation')['score'] == 5.0


# --- §12.4: negated vocabulary earns no qualitative credit -----------------


def test_q1_negated_reasoning_earns_nothing():
    q1 = _all_metadata()[0]['questions'][0]
    negated = score_l2_question(q1, _part_response(
        'Q1b: Hamming has not lower sidelobes, its main lobe is not wider, '
        'and it does not reduce leakage'))
    assert _sub(negated, 'window_reason')['score'] == 0.0


def test_q1_reasoning_binds_each_direction_to_the_right_quantity():
    q1 = _all_metadata()[0]['questions'][0]
    swapped = score_l2_question(q1, _part_response(
        'Q1b: Hamming has lower main-lobe width and wider sidelobe '
        'spacing, while sidelobe leakage is higher'))
    assert _sub(swapped, 'window_choice')['score'] == 3.0
    assert _sub(swapped, 'window_reason')['score'] == 0.0


def test_q1_not_only_is_an_affirmative_construction():
    q1 = _all_metadata()[0]['questions'][0]
    natural = score_l2_question(q1, _part_response(
        'Q1b: Hamming has not only lower sidelobes but also a wider main '
        'lobe, reducing leakage'))
    assert _sub(natural, 'window_choice')['score'] == 3.0
    assert _sub(natural, 'window_reason')['score'] == 3.0


def test_q2_definition_words_without_the_relations_earn_no_credit():
    q2 = _all_metadata()[0]['questions'][1]
    wrong = score_l2_question(q2, _part_response(
        'Q2d: 3 dB is unrelated to half power; 99% excludes energy; '
        'null-to-null avoids spectral nulls; use the 99% occupied bandwidth '
        'for planning because adjacent interference exists'))
    assert _sub(wrong, 'planning_choice')['score'] == 2.0
    assert _sub(wrong, 'definitions')['score'] == 0.0
    assert _sub(wrong, 'planning_reason')['score'] == 0.0


def test_q4_negated_morphology_earns_nothing():
    _, meta = _q4_samples()
    q4 = meta['questions'][3]
    sigs = meta['generation_params']['signals']
    negated = score_l2_question(q4, _part_response(
        'Q4a: the chirp is not diagonal; the digital signal is not '
        'wideband; the analog signal is not narrowband'), sigs)
    for cid in ('features_chirp', 'features_digital', 'features_analog'):
        assert _sub(negated, cid)['score'] == 0.0, cid


def test_q5_negated_explanation_caps_at_verification_only():
    meta = _all_metadata()[0]
    q5 = meta['questions'][4]
    gt = q5['ground_truth']
    negated = score_l2_question(q5, _part_response(
        f"Q5d: dBm is not logarithmic; do not convert or sum powers; "
        f"total = {gt['total_received_power_dBm']} dBm"))
    assert _sub(negated, 'explanation')['score'] == round(5 / 3, 2)
