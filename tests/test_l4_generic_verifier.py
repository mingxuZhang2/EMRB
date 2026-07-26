import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import l4_generic_verifier as v
from evaluation.l4_verifier import is_repaired_question


def _all_metadata():
    paths = sorted((ROOT / 'data/L4').glob('EMRB_L4_*.json'))
    metas = [json.loads(path.read_text()) for path in paths]
    assert len(metas) == 40
    return metas


def _generic_questions(meta):
    return [q for q in meta['questions'] if v.is_l4_generic_question(q)]


def _reference_response(meta):
    lines = ['===ANSWERS===']
    for question in _generic_questions(meta):
        lines += v.reference_response_lines(question)
    lines.append('===END===')
    return '\n'.join(lines)


def test_every_l4_question_is_marked_deterministic():
    generic = repaired = 0
    for meta in _all_metadata():
        for question in meta['questions']:
            if is_repaired_question(question):
                repaired += 1
            elif v.is_l4_generic_question(question):
                generic += 1
            else:
                raise AssertionError(
                    (meta['sample_id'], question['id'], 'unmarked'))
    assert repaired == 26
    assert generic == 174


def test_reference_answers_receive_full_credit():
    checked = 0
    for meta in _all_metadata():
        response = _reference_response(meta)
        signals = meta['generation_params']['signals']
        for question in _generic_questions(meta):
            result = v.score_l4_generic_question(question, response, signals)
            assert result['total_score'] == 20.0, \
                (meta['sample_id'], question['id'], result['sub_scores'])
            checked += 1
    assert checked == 174


def test_missing_answers_block_scores_zero():
    meta = _all_metadata()[0]
    signals = meta['generation_params']['signals']
    for question in _generic_questions(meta):
        result = v.score_l4_generic_question(question, 'nothing here', signals)
        assert result['total_score'] == 0.0
        assert 'parse_error' in result


def test_qt02_multitone_carries_both_conventions():
    dual = 0
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            if question['question_type'] != 'QT02':
                continue
            gt = question['ground_truth']
            accepted = gt['modulation_index_accepted']
            assert gt['modulation_index'] in accepted
            assert gt['carson_bandwidth_kHz'] in \
                gt['carson_bandwidth_accepted_kHz']
            dual += len(accepted) > 1
    assert dual > 0  # the 21 audited multitone instances must be represented


def test_qt02_alternative_beta_convention_gets_credit():
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            if question['question_type'] != 'QT02':
                continue
            gt = question['ground_truth']
            accepted = gt['modulation_index_accepted']
            if len(accepted) < 2:
                continue
            alt = [b for b in accepted if b != gt['modulation_index']][0]
            response = _reference_response(meta).replace(
                f"beta = {gt['modulation_index']}", f"beta = {alt}")
            result = v.score_l4_generic_question(
                question, response, meta['generation_params']['signals'])
            beta = next(c for c in result['sub_scores']
                        if c['id'] == 'mod_index')
            assert beta['score'] == beta['max'], meta['sample_id']
            return
    raise AssertionError('no multitone QT02 sample found')


def test_qt10_full_average_convention_gets_full_credit():
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            if question['question_type'] != 'QT10':
                continue
            gt = question['ground_truth']
            conventions = gt.get('accepted_conventions', [])
            if len(conventions) != 2:
                continue
            average = conventions[1]
            qid = question['id']
            response = _reference_response(meta).replace(
                f"{qid}a: SNR = {gt['snr_dB']} dB\n"
                f"{qid}b: {gt['shannon_capacity_Mbps']} Mbps\n"
                f"{qid}c: {gt['spectral_efficiency_gap']} bps/Hz",
                f"{qid}a: SNR = {average['snr_dB']} dB\n"
                f"{qid}b: {average['shannon_capacity_Mbps']} Mbps\n"
                f"{qid}c: {average['spectral_efficiency_gap'][0]} bps/Hz")
            result = v.score_l4_generic_question(
                question, response, meta['generation_params']['signals'])
            assert result['total_score'] == 20.0, \
                (meta['sample_id'], result['sub_scores'])
            return
    raise AssertionError('no burst QT10 sample found')


def test_qt10_mixed_conventions_do_not_get_full_credit():
    # synthetic conventions far outside every tolerance window: an answer
    # mixing convention A's SNR with convention B's capacity/gap must be
    # scored against one convention, never a per-field best-of
    question = {
        'id': 'Q9', 'question_type': 'QT10',
        'rubric': {'points': 20, 'scoring': v.SCORER_VERSION},
        'ground_truth': {
            'snr_dB': 40.0, 'shannon_capacity_Mbps': 10.0,
            'spectral_efficiency_gap': 8.0,
            'accepted_conventions': [
                {'snr_dB': 40.0, 'shannon_capacity_Mbps': 10.0,
                 'spectral_efficiency_gap': [8.0]},
                {'snr_dB': 20.0, 'shannon_capacity_Mbps': 2.0,
                 'spectral_efficiency_gap': [3.0]},
            ],
        },
    }
    consistent = ('===ANSWERS===\nQ9a: SNR = 20 dB\nQ9b: 2 Mbps\n'
                  'Q9c: 3 bps/Hz\n===END===')
    mixed = ('===ANSWERS===\nQ9a: SNR = 20 dB\nQ9b: 10 Mbps\n'
             'Q9c: 8 bps/Hz\n===END===')
    assert v.score_l4_generic_question(
        question, consistent, [])['total_score'] == 20.0
    assert v.score_l4_generic_question(
        question, mixed, [])['total_score'] < 20.0


def test_qt02_carson_accepted_uses_only_the_highest_tone():
    # Carson's rule is defined with the highest modulating frequency; the
    # fundamental-based variant must not be an accepted answer
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            if question['question_type'] != 'QT02':
                continue
            gt = question['ground_truth']
            carson = 2 * (gt['frequency_deviation_kHz']
                          + gt['max_modulating_freq_kHz'])
            for value in gt['carson_bandwidth_accepted_kHz']:
                assert abs(value - carson) <= 0.02 * carson, \
                    (meta['sample_id'], value, carson)


def test_qt02_contradicted_carson_verdict_scores_zero():
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            if question['question_type'] != 'QT02':
                continue
            response = _reference_response(meta).replace(
                'consistent with the observed bandwidth',
                'not consistent with the observed bandwidth')
            result = v.score_l4_generic_question(
                question, response, meta['generation_params']['signals'])
            verdict = next(c for c in result['sub_scores']
                           if c['id'] == 'carson_verdict')
            assert verdict['score'] == 0.0, meta['sample_id']
            return
    raise AssertionError('no QT02 sample found')


def test_qt04_snr_loss_is_sign_agnostic_and_duty_accepts_percent():
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            if question['question_type'] != 'QT04':
                continue
            gt = question['ground_truth']
            signals = meta['generation_params']['signals']
            response = _reference_response(meta).replace(
                f"SNR loss = {gt['snr_loss_dB']} dB",
                f"SNR loss = {abs(gt['snr_loss_dB'])} dB").replace(
                f"duty cycle = {gt['duty_cycle']}",
                f"duty cycle = {gt['duty_cycle'] * 100:.0f}%")
            result = v.score_l4_generic_question(question, response, signals)
            assert result['total_score'] == 20.0, \
                (meta['sample_id'], result['sub_scores'])
            return
    raise AssertionError('no QT04 sample found')


def test_qt01_family_only_modulation_gets_half_credit():
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            gt = question['ground_truth']
            if question['question_type'] != 'QT01' or 'signal_1' in gt:
                continue
            family = 'PSK' if 'PSK' in gt['type'] else 'QAM'
            response = _reference_response(meta).replace(
                f"ksps, {gt['type']},", f"ksps, some {family} signal,")
            result = v.score_l4_generic_question(
                question, response, meta['generation_params']['signals'])
            mod = next(c for c in result['sub_scores']
                       if c['id'] == 'mod_order')
            assert mod['score'] == mod['max'] / 2, meta['sample_id']
            return
    raise AssertionError('no single-signal QT01 sample found')


def test_qt01_index_referenced_answers_bind_by_ascending_frequency():
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            gt = question['ground_truth']
            if question['question_type'] != 'QT01' or 'signal_1' not in gt:
                continue
            qid = question['id']
            pair = [gt['signal_1'], gt['signal_2']]
            low, high = sorted(pair,
                               key=lambda s: s['center_frequency_MHz'])
            response = '\n'.join([
                '===ANSWERS===',
                f"{qid}a: 信号1符号速率: {low['symbol_rate_ksps']} ksym/s",
                f"{qid}b: 信号1调制: {low['type']}",
                f"{qid}c: 信号2符号速率: {high['symbol_rate_ksps']} ksym/s",
                f"{qid}d: 信号2调制: {high['type']}, 通过带宽和星座区分",
                '===END===',
            ])
            result = v.score_l4_generic_question(
                question, response, meta['generation_params']['signals'])
            assert result['total_score'] == 20.0, \
                (meta['sample_id'], result['sub_scores'])
            return
    raise AssertionError('no two-signal QT01 sample found')


def test_qt01_two_signal_swapped_rates_lose_points():
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            gt = question['ground_truth']
            if question['question_type'] != 'QT01' or 'signal_1' not in gt:
                continue
            r1 = gt['signal_1']['symbol_rate_ksps']
            r2 = gt['signal_2']['symbol_rate_ksps']
            if min(r1, r2) / max(r1, r2) > 0.55:
                continue  # inside the half-credit window either way
            response = _reference_response(meta).replace(
                f"symbol rate = {r1} ksps", "symbol rate = TMP").replace(
                f"symbol rate = {r2} ksps",
                f"symbol rate = {r1} ksps").replace(
                "symbol rate = TMP", f"symbol rate = {r2} ksps")
            result = v.score_l4_generic_question(
                question, response, meta['generation_params']['signals'])
            rates = next(c for c in result['sub_scores']
                         if c['id'] == 'symbol_rates')
            assert rates['score'] < rates['max'], meta['sample_id']
            return
    raise AssertionError('no two-signal QT01 sample with distinct rates')


def test_qt01_negated_method_earns_no_reasoning_credit():
    """§12.4: declaring the analysis method inapplicable must not satisfy
    the reasoning criterion."""
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            if (question['question_type'] != 'QT01'
                    or 'signal_1' not in question['ground_truth']):
                continue
            signals = meta['generation_params']['signals']
            qid = question['id']
            negated = (f"===ANSWERS===\n{qid}: 100 ksps QPSK and 200 ksps "
                       f"16QAM; constellation analysis is not applicable"
                       f"\n===END===")
            result = v.score_l4_generic_question(question, negated, signals)
            reasoning = next(c for c in result['sub_scores']
                             if c['id'] == 'reasoning')
            assert reasoning['score'] == 0.0, meta['sample_id']
            asserted = (f"===ANSWERS===\n{qid}: 100 ksps QPSK and 200 ksps "
                        f"16QAM, measured via constellation clustering"
                        f"\n===END===")
            result = v.score_l4_generic_question(question, asserted, signals)
            reasoning = next(c for c in result['sub_scores']
                             if c['id'] == 'reasoning')
            assert reasoning['score'] == reasoning['max'], meta['sample_id']
            return
    raise AssertionError('no two-signal QT01 sample found')


def test_qt04_negated_derivation_earns_no_credit():
    """§12.4: 'do not use 10log10(duty)' names the formula while rejecting
    it."""
    for meta in _all_metadata():
        for question in _generic_questions(meta):
            if question['question_type'] != 'QT04':
                continue
            gt = question['ground_truth']
            signals = meta['generation_params']['signals']
            qid = question['id']
            negated = (f"===ANSWERS===\n{qid}c: SNR loss = "
                       f"{gt['snr_loss_dB']} dB, but do not use "
                       f"10log10(duty) here\n===END===")
            result = v.score_l4_generic_question(question, negated, signals)
            derivation = next(c for c in result['sub_scores']
                              if c['id'] == 'snr_loss_derivation')
            assert derivation['score'] == 0.0, meta['sample_id']
            asserted = (f"===ANSWERS===\n{qid}c: SNR loss = "
                        f"{gt['snr_loss_dB']} dB from 10log10(duty "
                        f"cycle)\n===END===")
            result = v.score_l4_generic_question(question, asserted, signals)
            derivation = next(c for c in result['sub_scores']
                              if c['id'] == 'snr_loss_derivation')
            assert derivation['score'] == derivation['max'], \
                meta['sample_id']
            return
    raise AssertionError('no QT04 sample found')
