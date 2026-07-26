import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import l1_verifier, l3_verifier
from evaluation.answer_parsing import polarity, score_scalar


def _all_metadata(level):
    paths = sorted((ROOT / 'data' / f'L{level}').glob(
        f'EMRB_L{level}_*.json'))
    metas = [json.loads(path.read_text()) for path in paths]
    assert len(metas) == 40
    return metas


def _score_all(meta, response):
    module = l1_verifier if meta['level'] == 'L1' else l3_verifier
    scorer = (module.score_l1_question if meta['level'] == 'L1'
              else module.score_l3_question)
    signals = meta['generation_params']['signals']
    return {q['id']: scorer(q, response, signals) for q in meta['questions']}


def test_every_question_is_marked_deterministic():
    for level, module in ((1, l1_verifier), (3, l3_verifier)):
        checker = (module.is_l1_deterministic_question if level == 1
                   else module.is_l3_deterministic_question)
        for meta in _all_metadata(level):
            assert len(meta['questions']) == 5
            for question in meta['questions']:
                assert checker(question), (meta['sample_id'], question['id'])


def test_reference_answers_receive_full_credit_on_all_samples():
    for level, module in ((1, l1_verifier), (3, l3_verifier)):
        for meta in _all_metadata(level):
            response = module.reference_response(meta)
            for qid, result in _score_all(meta, response).items():
                assert result['total_score'] == 20.0, \
                    (meta['sample_id'], qid, result['sub_scores'])


def test_l1_ground_truth_consistency():
    for meta in _all_metadata(1):
        q1 = meta['questions'][0]['ground_truth']
        assert q1['strongest_signal_MHz'] in q1['strongest_accepted_MHz']
        assert q1['weakest_signal_MHz'] in q1['weakest_accepted_MHz']
        q5 = meta['questions'][4]['ground_truth']
        for three_db, occupied, accepted in zip(
                q5['bandwidths_3dB_kHz'], q5['bandwidths_kHz'],
                q5['bandwidth_categories_accepted']):
            assert three_db > 0
            assert three_db <= occupied * 1.2
            assert accepted


def test_l3_tie_sets_contain_the_extremes():
    tie_count = 0
    for meta in _all_metadata(3):
        gt = meta['questions'][2]['ground_truth']
        assert gt['best_for_PA']['signal'] in gt['best_for_PA_accepted']
        assert gt['worst_for_PA']['signal'] in gt['worst_for_PA_accepted']
        tie_count += len(gt['best_for_PA_accepted']) > 1
    assert tie_count > 0  # the audited FM/Chirp ties must be represented


def test_l1_missing_answers_block_scores_zero():
    meta = _all_metadata(1)[0]
    for result in _score_all(meta, 'no block here').values():
        assert result['total_score'] == 0.0
        assert 'parse_error' in result


def test_l1_wrong_count_loses_only_count_points():
    meta = _all_metadata(1)[0]
    module = l1_verifier
    response = module.reference_response(meta)
    gt = meta['questions'][0]['ground_truth']
    response = response.replace(
        f"Q1a: {gt['signal_count']} signals",
        f"Q1a: {gt['signal_count'] + 1} signals")
    assert _score_all(meta, response)['Q1']['total_score'] == 16.0


def test_l1_unit_variant_still_scores():
    meta = _all_metadata(1)[0]
    gt = meta['questions'][2]['ground_truth']
    response = l1_verifier.reference_response(meta).replace(
        f"Q3c: {gt['delta_f_Hz']} Hz",
        f"Q3c: {gt['delta_f_Hz'] / 1e3} kHz")
    assert _score_all(meta, response)['Q3']['total_score'] == 20.0


def test_l1_swapped_labeled_snrs_lose_points():
    for meta in _all_metadata(1):
        gt = meta['questions'][1]['ground_truth']
        snrs = gt['SNR_per_signal_dB']
        if max(snrs) - min(snrs) <= 6.5:
            continue
        signals = meta['generation_params']['signals']
        i_hi = snrs.index(max(snrs))
        i_lo = snrs.index(min(snrs))
        swapped = list(snrs)
        swapped[i_hi], swapped[i_lo] = swapped[i_lo], swapped[i_hi]
        original = '; '.join(f"{s['type']}: {p} dB"
                             for s, p in zip(signals, snrs))
        modified = '; '.join(f"{s['type']}: {p} dB"
                             for s, p in zip(signals, swapped))
        response = l1_verifier.reference_response(meta).replace(
            f"Q2d: {original}", f"Q2d: {modified}")
        assert response != l1_verifier.reference_response(meta)
        assert _score_all(meta, response)['Q2']['total_score'] < 20.0
        return
    raise AssertionError('no sample with a large SNR spread found')


def test_l1_flipped_noise_boolean_loses_points():
    meta = _all_metadata(1)[0]
    response = l1_verifier.reference_response(meta).replace(
        'Q4d: No, the noise floor would remain the same',
        'Q4d: Yes, the noise floor would change')
    assert _score_all(meta, response)['Q4']['total_score'] == 16.0


def test_l1_chinese_qualitative_answers_receive_credit():
    meta = _all_metadata(1)[0]
    response = l1_verifier.reference_response(meta)
    response = response.replace('Strongest:', '最强信号：')
    response = response.replace('Weakest:', '最弱信号：')
    response = response.replace(
        'increase N (the number of samples), keep fs unchanged',
        '增加采样点数 N，保持采样率 fs 不变')
    response = response.replace(
        'Yes, the PSD is flat — white noise',
        '是白噪声，功率谱密度均匀且平坦')
    response = response.replace(
        'No, the noise floor would remain the same',
        '不会变化，噪声底保持不变')
    response = response.replace('Narrowband', '窄带')
    response = response.replace('Midband', '中带')
    response = response.replace('Wideband', '宽带')
    scores = _score_all(meta, response)
    for qid in ('Q1', 'Q3', 'Q4', 'Q5'):
        assert scores[qid]['total_score'] == 20.0, scores[qid]


def test_l1_constant_envelope_explanation_handles_semantic_negation():
    meta = _all_metadata(1)[0]
    gt = meta['questions'][4]['ground_truth']
    true_types = [entry['signal'] for entry in gt['constant_envelope']
                  if entry['constant_envelope']]
    false_types = [entry['signal'] for entry in gt['constant_envelope']
                   if not entry['constant_envelope']]
    original = ', '.join(true_types) + ' are constant-envelope'
    chinese = '；'.join(
        [f'{signal} 调制不改变幅度' for signal in true_types]
        + [f'{signal} 非恒包络' for signal in false_types]
    )
    response = l1_verifier.reference_response(meta).replace(original, chinese)
    result = _score_all(meta, response)['Q5']
    const = next(c for c in result['sub_scores']
                 if c['id'] == 'const_envelope')
    assert const['score'] == const['max'], const


def test_l3_family_only_modulation_gets_half_credit():
    meta = _all_metadata(3)[0]
    gt = meta['questions'][0]['ground_truth']
    family = 'PSK' if 'PSK' in gt['type'] else 'QAM'
    response = l3_verifier.reference_response(meta).replace(
        f"Q1a: {gt['type']},", f"Q1a: some kind of {family} signal,")
    assert _score_all(meta, response)['Q1']['total_score'] == 17.0


def test_l3_wrong_feasibility_loses_points():
    meta = _all_metadata(3)[0]
    gt = meta['questions'][1]['ground_truth']
    right = ('Yes, the requirement is met' if gt['feasible']
             else 'No, the requirement is not met')
    wrong = ('No, the requirement is not met' if gt['feasible']
             else 'Yes, the requirement is met')
    response = l3_verifier.reference_response(meta).replace(
        f"Q2d: {right};", f"Q2d: {wrong};")
    assert _score_all(meta, response)['Q2']['total_score'] == 18.0


def test_l3_decimation_factor_of_two_is_not_full_credit():
    meta = _all_metadata(3)[0]
    gt = meta['questions'][4]['ground_truth']
    response = l3_verifier.reference_response(meta).replace(
        f"decimation factor = {gt['decimation']};",
        f"decimation factor = {gt['decimation'] * 2};")
    result = _score_all(meta, response)['Q5']
    decim = next(c for c in result['sub_scores'] if c['id'] == 'decim')
    assert decim['score'] == 0.0


def test_l3_consistent_alternative_decimation_design_scores():
    meta = _all_metadata(3)[0]
    gt = meta['questions'][4]['ground_truth']
    fs_khz = gt['new_rate_kHz'] * gt['decimation']
    cutoff = gt['LPF_cutoff_kHz'] * 0.8
    decimation = int(fs_khz // (2 * cutoff))
    new_rate = fs_khz / decimation
    response = l3_verifier.reference_response(meta)
    response = response.replace(
        f"Q5b: {gt['LPF_cutoff_kHz']} kHz", f"Q5b: {cutoff} kHz")
    response = response.replace(
        f"Q5c: decimation factor = {gt['decimation']}; "
        f"new sampling rate = {gt['new_rate_kHz']} kHz",
        f"Q5c: decimation factor = {decimation}; "
        f"new sampling rate = {new_rate} kHz")
    response = response.replace(
        f"Q5d: data volume reduced by a factor of {gt['data_reduction']}",
        f"Q5d: data volume reduced by a factor of {decimation}")
    assert _score_all(meta, response)['Q5']['total_score'] == 20.0


def test_l3_ten_bit_verdict_requires_polarity():
    meta = _all_metadata(3)[0]
    gt = meta['questions'][3]['ground_truth']
    right = ('10 bits is sufficient' if gt['ten_bit_ok']
             else '10 bits is not sufficient')
    wrong = ('10 bits is not sufficient' if gt['ten_bit_ok']
             else '10 bits is sufficient')
    response = l3_verifier.reference_response(meta).replace(right, wrong)
    assert _score_all(meta, response)['Q4']['total_score'] == 18.0


def test_l3_chinese_qualitative_answers_receive_credit():
    meta = _all_metadata(3)[0]
    gt2 = meta['questions'][1]['ground_truth']
    gt3 = meta['questions'][2]['ground_truth']
    gt4 = meta['questions'][3]['ground_truth']
    response = l3_verifier.reference_response(meta)

    english_feasible = ('Yes, the requirement is met' if gt2['feasible']
                        else 'No, the requirement is not met')
    chinese_feasible = '满足要求' if gt2['feasible'] else '不满足要求'
    response = response.replace(english_feasible, chinese_feasible)
    response = response.replace('Best suited:', '最适合：')
    response = response.replace('least suited:', '最不适合：')
    if gt3['exceeds_P1dB']:
        response = response.replace(
            'peak exceeds P1dB, distortion occurs',
            '峰值超过 P1dB，会产生失真')
    else:
        response = response.replace(
            'peak stays below P1dB, no distortion occurs',
            '峰值低于 P1dB，不会产生失真')
    response = response.replace(
        'larger back-off reduces PA efficiency',
        '较大的回退会降低功放效率')
    response = response.replace('Strongest =', '最强信号 =')
    response = response.replace('weakest =', '最弱信号 =')
    english_ten = ('10 bits is sufficient' if gt4['ten_bit_ok']
                   else '10 bits is not sufficient')
    chinese_ten = '10位足够' if gt4['ten_bit_ok'] else '10位不够'
    response = response.replace(english_ten, chinese_ten)

    scores = _score_all(meta, response)
    for qid in ('Q2', 'Q3', 'Q4'):
        assert scores[qid]['total_score'] == 20.0, scores[qid]


def test_l3_chinese_swapped_power_labels_do_not_oracle_match():
    meta = next(
        item for item in _all_metadata(3)
        if abs(item['questions'][3]['ground_truth']['strongest_dBm']
               - item['questions'][3]['ground_truth']['weakest_dBm']) > 6.0
    )
    q4 = meta['questions'][3]
    gt = q4['ground_truth']
    response = (
        '===ANSWERS===\n'
        f"Q4a: 最强信号 = {gt['weakest_dBm']} dBm；"
        f"最弱信号 = {gt['strongest_dBm']} dBm\n"
        '===END==='
    )
    result = l3_verifier.score_l3_question(
        q4, response, meta['generation_params']['signals'])
    subs = {item['id']: item['score'] for item in result['sub_scores']}
    assert subs['strongest_power'] == 0.0
    assert subs['weakest_power'] == 0.0


def test_exact_int_rejects_non_integer_answers():
    assert score_scalar('2.51 signals', 3, 'none', None,
                        0, None, mode='exact_int')[0] == 0.0
    assert score_scalar('3 signals', 3, 'none', None,
                        0, None, mode='exact_int')[0] == 1.0
    assert score_scalar('3.0 signals', 3, 'none', None,
                        0, None, mode='exact_int')[0] == 1.0


def test_explicit_final_correction_supersedes_earlier_numeric_value():
    text = ('10 dB was the scratch value; final correction: 100 dB, '
            'the earlier value is wrong')
    assert score_scalar(text, 10, 'db', 'dB', 1.0, 2.0, mode='abs',
                        candidate_policy='asserted')[0] == 0.0
    assert score_scalar(text, 100, 'db', 'dB', 1.0, 2.0, mode='abs',
                        candidate_policy='asserted')[0] == 1.0


def test_explicitly_retracted_numeric_value_is_not_eligible():
    text = 'final answer: 100 dB; 10 dB was explicitly retracted'
    assert score_scalar(text, 10, 'db', 'dB', 1.0, 2.0, mode='abs',
                        candidate_policy='asserted')[0] == 0.0


def test_l1_verifier_honors_final_numeric_correction():
    meta = _all_metadata(1)[0]
    gt = meta['questions'][2]['ground_truth']
    response = l1_verifier.reference_response(meta).replace(
        f"Q3c: {gt['delta_f_Hz']} Hz",
        f"Q3c: {gt['delta_f_Hz']} Hz; final correction: 9999 Hz")
    result = _score_all(meta, response)['Q3']
    delta_f = next(c for c in result['sub_scores'] if c['id'] == 'delta_f')
    assert delta_f['score'] == 0.0


def test_l3_verifier_honors_final_numeric_correction():
    meta = _all_metadata(3)[0]
    gt = meta['questions'][1]['ground_truth']
    response = l3_verifier.reference_response(meta).replace(
        f"Q2c: {gt['EbN0_required_dB']} dB",
        f"Q2c: {gt['EbN0_required_dB']} dB; final correction: 99 dB")
    result = _score_all(meta, response)['Q2']
    required = next(c for c in result['sub_scores'] if c['id'] == 'required')
    assert required['score'] == 0.0


def test_l1_bandwidth_full_credit_matches_declared_thirty_percent():
    meta = _all_metadata(1)[0]
    gt = meta['questions'][4]['ground_truth']
    signals = meta['generation_params']['signals']
    original = '; '.join(
        f"{s['type']}: {b} kHz"
        for s, b in zip(signals, gt['bandwidths_3dB_kHz']))
    changed = list(gt['bandwidths_3dB_kHz'])
    changed[0] *= 1.35
    modified = '; '.join(
        f"{s['type']}: {b} kHz" for s, b in zip(signals, changed))
    response = l1_verifier.reference_response(meta).replace(original, modified)
    result = _score_all(meta, response)['Q5']
    bandwidth = next(c for c in result['sub_scores'] if c['id'] == 'bw_estimate')
    assert 0.0 < bandwidth['score'] < bandwidth['max']


def test_l3_spectral_efficiency_above_twenty_five_percent_is_partial():
    meta = _all_metadata(3)[0]
    gt = meta['questions'][0]['ground_truth']
    response = l3_verifier.reference_response(meta).replace(
        f"Q1c: {gt['spectral_efficiency_bps_Hz']} bps/Hz",
        f"Q1c: {gt['spectral_efficiency_bps_Hz'] * 1.30} bps/Hz")
    result = _score_all(meta, response)['Q1']
    efficiency = next(c for c in result['sub_scores']
                      if c['id'] == 'spectral_eff')
    assert efficiency['score'] == 2.0


def test_l3_adjacent_enob_is_partial_not_full():
    meta = _all_metadata(3)[0]
    gt = meta['questions'][3]['ground_truth']
    response = l3_verifier.reference_response(meta).replace(
        f"minimum {gt['min_ENOB']} bits",
        f"minimum {gt['min_ENOB'] - 1} bits")
    result = _score_all(meta, response)['Q4']
    enob = next(c for c in result['sub_scores'] if c['id'] == 'min_ENOB')
    assert enob['score'] == 2.0


def test_polarity_contradictory_lead_returns_none():
    pos = ('meets', 'met', 'feasible', 'sufficient')
    neg = ('fails', 'infeasible', 'insufficient')
    assert polarity('Yes, the requirement is not met', pos, neg) is None
    assert polarity('No, the requirement is met', pos, neg) is None
    assert polarity('Yes, the requirement is met', pos, neg) is True
    # mixed prose does not override an explicit lead token
    assert polarity('Yes, it is feasible although margin nearly fails',
                    pos, neg) is True


def test_polarity_negation_stops_at_clause_boundary():
    pos = ('feasible', 'met')
    neg = ('infeasible',)
    assert polarity('No distortion occurs; the requirement is feasible',
                    pos, neg) is True
    assert polarity('The requirement is not feasible', pos, neg) is False


def test_l1_contradictory_lead_boolean_scores_zero():
    meta = _all_metadata(1)[0]
    response = l1_verifier.reference_response(meta).replace(
        'Q4d: No, the noise floor would remain the same',
        'Q4d: Yes, the noise floor would remain the same')
    assert _score_all(meta, response)['Q4']['total_score'] == 16.0


def test_l1_swapped_type_labels_lose_points():
    for meta in _all_metadata(1):
        signals = meta['generation_params']['signals']
        tokens = [set(l1_verifier._expected_label_tokens(s['type']))
                  for s in signals]
        if any(tokens[i] & tokens[j] for i in range(len(tokens))
               for j in range(i + 1, len(tokens))):
            continue  # e.g. two digital signals both answer to 'digital'
        gt = meta['questions'][4]['ground_truth']
        labels = [label['label'] for label in gt['signal_types']]
        rolled = labels[1:] + labels[:1]
        original = '; '.join(
            f'{l1_verifier.signal_center_mhz(s):+.2f} MHz: {label}'
            for s, label in zip(signals, labels))
        modified = '; '.join(
            f'{l1_verifier.signal_center_mhz(s):+.2f} MHz: {label}'
            for s, label in zip(signals, rolled))
        response = l1_verifier.reference_response(meta).replace(
            original, modified)
        assert response != l1_verifier.reference_response(meta)
        result = _score_all(meta, response)['Q5']
        type_id = next(c for c in result['sub_scores'] if c['id'] == 'type_id')
        # a chirp sweeping through another signal's center keeps some rolled
        # labels defensible, so demand a loss rather than zero
        assert type_id['score'] < type_id['max'], meta['sample_id']
        return
    raise AssertionError('no sample with disjoint type tokens found')


def test_l1_extra_constant_envelope_claim_loses_points():
    for meta in _all_metadata(1):
        gt = meta['questions'][4]['ground_truth']
        entries = gt['constant_envelope']
        true_types = [e['signal'] for e in entries if e['constant_envelope']]
        false_types = [e['signal'] for e in entries
                       if not e['constant_envelope']]
        if not true_types or not false_types:
            continue
        response = l1_verifier.reference_response(meta).replace(
            ', '.join(true_types) + ' are constant-envelope',
            ', '.join(true_types + false_types[:1]) + ' are constant-envelope')
        result = _score_all(meta, response)['Q5']
        const = next(c for c in result['sub_scores']
                     if c['id'] == 'const_envelope')
        assert const['score'] < const['max'], meta['sample_id']
        return
    raise AssertionError('no sample with mixed constant-envelope GT found')


def test_db_and_dbm_are_not_interchangeable():
    assert score_scalar('10 dBm', 10, 'db', 'dB', 1.0, 2.0,
                        mode='abs')[0] == 0.0
    assert score_scalar('-30 dBJ', -30, 'dbm', 'dBm', 1.0, 2.0,
                        mode='abs')[0] == 0.0
    assert score_scalar('10 dB', 10, 'db', 'dB', 1.0, 2.0,
                        mode='abs')[0] == 1.0
    assert score_scalar('10', 10, 'db', 'dB', 1.0, 2.0,
                        mode='abs')[0] == 1.0  # unitless assumed correct


def test_l3_swapped_noise_and_snr_fields_score_zero():
    from evaluation.l3_verifier import score_l3_question
    meta = _all_metadata(3)[0]
    signals = meta['generation_params']['signals']
    q2 = meta['questions'][1]
    gt = q2['ground_truth']
    swapped = l3_verifier.reference_response(meta).replace(
        f"Noise in band = {gt['noise_in_band_dBm']} dBm; "
        f"SNR = {gt['SNR_dB']} dB",
        f"Noise in band = {gt['SNR_dB']} dBm; "
        f"SNR = {gt['noise_in_band_dBm']} dB")
    result = score_l3_question(q2, swapped, signals)
    for cid in ('noise_in_band', 'SNR'):
        criterion = next(c for c in result['sub_scores'] if c['id'] == cid)
        assert criterion['score'] == 0.0, criterion


def test_l3_contradictory_selection_gets_nothing():
    meta = _all_metadata(3)[0]
    gt = meta['questions'][2]['ground_truth']
    wrong = next(t for t in ('64QAM', '16QAM', 'QPSK', 'BPSK', '8PSK')
                 if t not in gt['best_for_PA_accepted'])
    response = re.sub(r'Q3b: Best suited: [^;]+;',
                      f'Q3b: Best suited: {wrong};',
                      l3_verifier.reference_response(meta))
    result = _score_all(meta, response)['Q3']
    best = next(c for c in result['sub_scores'] if c['id'] == 'pa_best')
    assert best['score'] == 0.0


def test_l1_psd_and_absolute_power_are_not_interchangeable():
    """§10.5: a dBm-labeled total power must not satisfy the dBm/Hz PSD
    criterion, and a dBm/Hz-labeled density must not satisfy the absolute
    in-band power criterion."""
    meta = _all_metadata(1)[0]
    q4 = next(q for q in meta['questions'] if 'psd' in q['rubric'])
    gt = q4['ground_truth']
    signals = meta['generation_params']['signals']
    qid = q4['id']

    def _score(a_unit, b_unit):
        response = (f"===ANSWERS===\n"
                    f"{qid}a: {gt['noise_psd_dBm_Hz']} {a_unit}\n"
                    f"{qid}b: {gt['noise_1MHz_dBm']} {b_unit}\n"
                    f"===END===")
        result = l1_verifier.score_l1_question(q4, response, signals)
        subs = {c['id']: c['score'] for c in result['sub_scores']}
        return subs['psd'], subs['noise_1mhz']

    assert _score('dBm/Hz', 'dBm') == (6.0, 5.0)
    assert _score('dBm', 'dBm/Hz') == (0.0, 0.0)


def test_l1_negated_method_choice_earns_nothing():
    """§12.4: 'do not increase N' names the vocabulary while asserting the
    opposite."""
    meta = _all_metadata(1)[0]
    q3 = meta['questions'][2]
    gt = q3['ground_truth']
    signals = meta['generation_params']['signals']
    qid = q3['id']
    response = (f"===ANSWERS===\n{qid}d: do not increase N; "
                f"{gt['N_for_half_delta_f']} samples\n===END===")
    result = l1_verifier.score_l1_question(q3, response, signals)
    subs = {c['id']: c['score'] for c in result['sub_scores']}
    assert subs['halving_choice'] == 0.0
    response = (f"===ANSWERS===\n{qid}d: increase N to "
                f"{gt['N_for_half_delta_f']} samples\n===END===")
    result = l1_verifier.score_l1_question(q3, response, signals)
    subs = {c['id']: c['score'] for c in result['sub_scores']}
    assert subs['halving_choice'] == 4.0


def test_l3_negated_efficiency_direction_earns_nothing():
    """§12.4: 'does not reduce PA efficiency' asserts the wrong relation."""
    meta = _all_metadata(3)[0]
    q3 = meta['questions'][2]
    signals = meta['generation_params']['signals']
    qid = q3['id']
    negated = (f"===ANSWERS===\n{qid}d: larger back-off does not reduce "
               f"PA efficiency\n===END===")
    result = l3_verifier.score_l3_question(q3, negated, signals)
    subs = {c['id']: c['score'] for c in result['sub_scores']}
    assert subs['ibo_efficiency_direction'] == 0.0
    asserted = (f"===ANSWERS===\n{qid}d: larger back-off reduces PA "
                f"efficiency\n===END===")
    result = l3_verifier.score_l3_question(q3, asserted, signals)
    subs = {c['id']: c['score'] for c in result['sub_scores']}
    assert subs['ibo_efficiency_direction'] == 2.0
