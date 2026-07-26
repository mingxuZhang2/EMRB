"""Deterministic per-question-type scoring for EMRB L3.

Same contract as l1_verifier: scores the existing labeled-line answer format
(no prompt change), binding each requested output to its own sub-answer with
unit-aware tolerances, tie-aware accepted sets, and negation-resolved booleans.
"""
import math

from evaluation.answer_parsing import (
    asserted_quantities_in_family,
    asserted_token,
    TYPE_TOKENS,
    has_token,
    polarity,
    score_per_signal as _shared_score_per_signal,
    score_scalar as _shared_score_scalar,
    score_scalar_any as _shared_score_scalar_any,
)
from evaluation.auto_scorer import parse_answer_block
from evaluation.l1_verifier import (
    STRONGEST_KEYWORDS,
    WEAKEST_KEYWORDS,
    _criterion,
    _keyword_segment,
    part_text,
)

SCORER_VERSION = 'l3-deterministic-v5'


def score_scalar(*args, **kwargs):
    kwargs.setdefault('candidate_policy', 'asserted')
    return _shared_score_scalar(*args, **kwargs)


def score_scalar_any(*args, **kwargs):
    kwargs.setdefault('candidate_policy', 'asserted')
    return _shared_score_scalar_any(*args, **kwargs)


def score_per_signal(*args, **kwargs):
    kwargs.setdefault('candidate_policy', 'asserted')
    return _shared_score_per_signal(*args, **kwargs)


def _preferred_value(text, family, target_unit):
    """The model's asserted result, independent of the ground truth.

    Explicitly unit-bearing values take precedence over bare derivation
    constants.  Within the final asserted scope, the last value is the result
    convention used by the requested calculation format.
    """
    values = asserted_quantities_in_family(
        text, family, target_unit, explicit_only=True)
    if not values:
        values = asserted_quantities_in_family(text, family, target_unit)
    return values[-1][0] if values else None


def _score_preferred(text, reference, family, target_unit,
                     full, half, mode='rel'):
    value = _preferred_value(text, family, target_unit)
    if value is None:
        return 0.0, f'no {target_unit or "numeric"} value found'
    return score_scalar(str(value), reference, 'none', None,
                        full, half, mode=mode)

_MOD_FAMILY_TOKEN = {'BPSK': 'psk', 'QPSK': 'psk', '8PSK': 'psk',
                     '16QAM': 'qam', '64QAM': 'qam'}


def is_l3_deterministic_question(question):
    return question.get('rubric', {}).get('scoring') == SCORER_VERSION


def _result(criteria, parse_error=None):
    maximum = sum(item['max'] for item in criteria)
    assert math.isclose(maximum, 20.0)
    result = {
        'total_score': round(sum(item['score'] for item in criteria), 2),
        'total_max': 20.0,
        'objective_max': 20.0,
        'subjective_max': 0.0,
        'rubric_total': 20.0,
        'sub_scores': criteria,
        'method': SCORER_VERSION,
    }
    if parse_error:
        result['parse_error'] = parse_error
    return result


def _first_type_token(text):
    """Earliest modulation-type token in a text segment, or None."""
    best = None
    from evaluation.answer_parsing import _token_positions
    for sig_type, tokens in TYPE_TOKENS.items():
        positions = _token_positions(text, tokens)
        if positions and (best is None or min(positions) < best[0]):
            best = (min(positions), sig_type)
    return best[1] if best else None


def _score_type_selection(text, keywords, stop_keywords, accepted_types):
    segment = _keyword_segment(text, keywords, stop_keywords)
    found = _first_type_token(segment)
    return 1.0 if found is not None and found in accepted_types else 0.0


def _score_power_extremes(text, strongest, weakest):
    """Bind strongest/weakest powers by labels, then requested order.

    Falling back to two independent ground-truth searches over the whole
    sub-answer lets swapped labels score full credit.  When labels are absent,
    Q4(a)'s requested strongest-then-weakest order is the only valid fallback.
    """
    strongest_segment = _keyword_segment(
        text, STRONGEST_KEYWORDS, WEAKEST_KEYWORDS)
    weakest_segment = _keyword_segment(text, WEAKEST_KEYWORDS, ())
    if strongest_segment and weakest_segment:
        ratio_strong, _ = score_scalar(
            strongest_segment, strongest, 'dbm', 'dBm',
            3.0, 6.0, mode='abs')
        ratio_weak, _ = score_scalar(
            weakest_segment, weakest, 'dbm', 'dBm',
            3.0, 6.0, mode='abs')
        return ratio_strong, ratio_weak, 'label-bound'

    values = asserted_quantities_in_family(
        text, 'dbm', 'dBm', explicit_only=True)
    if len(values) < 2:
        values = asserted_quantities_in_family(text, 'dbm', 'dBm')
    ratios = []
    for index, reference in enumerate((strongest, weakest)):
        if index >= len(values):
            ratios.append(0.0)
            continue
        ratio, _ = score_scalar(
            str(values[index][0]), reference, 'none', None,
            3.0, 6.0, mode='abs')
        ratios.append(ratio)
    return ratios[0], ratios[1], 'ordered positional'


def _score_q1(question, response):
    gt = question['ground_truth']
    qid = question['id']
    text_a = part_text(response, qid, 'a')
    # the label must be asserted: "not QPSK" is a rejection (§12.4)
    exact = asserted_token(text_a, TYPE_TOKENS[gt['type']])
    family = asserted_token(text_a, (_MOD_FAMILY_TOKEN[gt['type']],))
    ratio_type = 1.0 if exact else 0.5 if family else 0.0
    ratio_rs, _ = score_scalar(text_a, gt['symbol_rate_ksps'],
                               'symrate', 'ksps', 0.15, 0.30)
    ratio_rb, _ = score_scalar(part_text(response, qid, 'b'),
                               gt['bit_rate_kbps'], 'bitrate', 'kbps',
                               0.15, 0.30)
    ratio_eta, _ = _score_preferred(
        part_text(response, qid, 'c'), gt['spectral_efficiency_bps_Hz'],
        'eff', 'bps/Hz', 0.25, 0.40)
    ratio_64, _ = score_scalar(part_text(response, qid, 'd'),
                               gt['bit_rate_64QAM_kbps'], 'bitrate', 'kbps',
                               0.15, 0.30)
    return _result([
        _criterion('mod_type', 6, ratio_type, f"expected {gt['type']}"),
        _criterion('sym_rate', 4, ratio_rs,
                   f"ref={gt['symbol_rate_ksps']} ksps"),
        _criterion('bit_rate', 3, ratio_rb, f"ref={gt['bit_rate_kbps']} kbps"),
        _criterion('spectral_eff', 4, ratio_eta,
                   'bandwidth-definition freedom: ±25%'),
        _criterion('64qam', 3, ratio_64,
                   f"ref={gt['bit_rate_64QAM_kbps']} kbps"),
    ])


def _score_q2(question, response):
    gt = question['ground_truth']
    qid = question['id']
    text_a = part_text(response, qid, 'a')
    # two quantities share sub-answer (a): bind each to its field label so a
    # swapped "Noise = <SNR> dBm; SNR = <noise> dB" earns nothing (§8.1.5)
    noise_seg = _keyword_segment(text_a, ('noise', '噪声'), ('snr', '信噪比'))
    snr_seg = _keyword_segment(text_a, ('snr', '信噪比'), ())
    ratio_nib, _ = score_scalar(noise_seg or text_a, gt['noise_in_band_dBm'],
                                'dbm', 'dBm', 3.0, 6.0, mode='abs')
    ratio_snr, _ = score_scalar(snr_seg or text_a, gt['SNR_dB'],
                                'db', 'dB', 3.0, 6.0, mode='abs')
    ratio_ebn0, _ = score_scalar(part_text(response, qid, 'b'),
                                 gt['EbN0_dB'], 'db', 'dB',
                                 3.0, 6.0, mode='abs')
    ratio_req, _ = score_scalar(part_text(response, qid, 'c'),
                                gt['EbN0_required_dB'], 'db', 'dB',
                                1.0, 2.0, mode='abs')
    text_d = part_text(response, qid, 'd')
    ratio_margin, _ = score_scalar(text_d, gt['margin_dB'],
                                   'db', 'dB', 3.0, 6.0, mode='abs')
    feasible = polarity(text_d,
                        ('meets', 'met', 'feasible', 'sufficient',
                         'satisfied', 'satisfies', 'achievable', '满足',
                         '达到', '可行', '足够', '充分', '可以实现'),
                        ('fails', 'infeasible', 'insufficient',
                         'unachievable', '不满足', '未满足', '无法满足',
                         '不可行', '不足', '不够', '无法实现'))
    ratio_ok = 1.0 if feasible is not None \
        and feasible == bool(gt['feasible']) else 0.0
    return _result([
        _criterion('noise_in_band', 2, ratio_nib,
                   f"ref={gt['noise_in_band_dBm']} dBm"),
        _criterion('SNR', 3, ratio_snr, f"ref={gt['SNR_dB']} dB"),
        _criterion('EbN0', 5, ratio_ebn0, f"ref={gt['EbN0_dB']} dB"),
        _criterion('required', 5, ratio_req,
                   f"ref={gt['EbN0_required_dB']} dB"),
        _criterion('margin', 3, ratio_margin, f"ref={gt['margin_dB']} dB"),
        _criterion('feasible', 2, ratio_ok, f"expected {gt['feasible']}"),
    ])


def _score_q3(question, response, signals):
    gt = question['ground_truth']
    qid = question['id']
    paprs = [entry['PAPR_dB'] for entry in gt['papr_per_signal']]
    ratio_papr, reason_papr = score_per_signal(
        part_text(response, qid, 'a'), signals, paprs,
        'db', 'dB', 2.0, 4.0, mode='abs')
    text_b = part_text(response, qid, 'b')
    best_accepted = gt.get('best_for_PA_accepted',
                           [gt['best_for_PA']['signal']])
    worst_accepted = gt.get('worst_for_PA_accepted',
                            [gt['worst_for_PA']['signal']])
    ratio_best = _score_type_selection(
        text_b, ('best', 'suitable', 'suited', '最适合', '最合适', '最适宜'),
        ('least', 'worst', 'unsuited', 'not suited', '最不适合',
         '最不合适', '最差'), best_accepted)
    ratio_worst = _score_type_selection(
        text_b, ('least', 'worst', 'unsuited', 'not suited', '最不适合',
                 '最不合适', '最差'), (),
        worst_accepted)
    text_c = part_text(response, qid, 'c')
    ratio_peak, _ = score_scalar(text_c, gt['digital_peak_dBm'],
                                 'dbm', 'dBm', 2.0, 4.0, mode='abs')
    exceeds = polarity(text_c,
                       ('exceeds', 'exceed', 'above', 'clip', 'clipping',
                        'distortion', 'distort', '超过', '高于', '削顶',
                        '失真'),
                       ('below', 'under', 'headroom', 'safe', '低于',
                        '小于', '未超过', '不超过', '有余量', '安全'))
    ratio_exceeds = 1.0 if exceeds is not None \
        and exceeds == bool(gt['exceeds_P1dB']) else 0.0
    text_d = part_text(response, qid, 'd')
    ratio_ibo, _ = score_scalar(text_d, gt['IBO_dB'],
                                'db', 'dB', 2.0, 4.0, mode='abs')
    # the efficiency-cost relation must be asserted, not negated
    # (remediation log §12.4)
    mentions_eff = asserted_token(text_d, ('efficiency', 'efficient', '效率'))
    mentions_down = asserted_token(text_d, (
        'lower', 'lowers', 'reduce', 'reduces', 'reduced', 'reducing',
        'decrease', 'decreases', 'decreased', 'degrade', 'degrades',
        'degraded', 'drop', 'drops', 'less', 'poorer', 'worse', 'sacrific'))
    mentions_down = mentions_down or asserted_token(
        text_d, ('降低', '下降', '减小', '变低', '牺牲', '损失'))
    ratio_dir = 1.0 if (mentions_eff and mentions_down) else 0.0
    return _result([
        _criterion('papr_estimation', 5, ratio_papr, reason_papr),
        _criterion('pa_best', 2.5, ratio_best, f'accepted {best_accepted}'),
        _criterion('pa_worst', 2.5, ratio_worst, f'accepted {worst_accepted}'),
        _criterion('digital_peak', 3, ratio_peak,
                   f"ref={gt['digital_peak_dBm']} dBm"),
        _criterion('exceeds_P1dB', 2, ratio_exceeds,
                   f"expected {gt['exceeds_P1dB']}"),
        _criterion('IBO', 3, ratio_ibo, f"ref={gt['IBO_dB']} dB"),
        _criterion('ibo_efficiency_direction', 2, ratio_dir,
                   'larger back-off must be linked to lower PA efficiency'),
    ])


def _score_q4(question, response):
    gt = question['ground_truth']
    qid = question['id']
    text_a = part_text(response, qid, 'a')
    ratio_pmax, ratio_pmin, binding = _score_power_extremes(
        text_a, gt['strongest_dBm'], gt['weakest_dBm'])
    ratio_peak, _ = score_scalar(part_text(response, qid, 'b'),
                                 gt['effective_peak_dBm'], 'dbm', 'dBm',
                                 3.0, 6.0, mode='abs')
    ratio_dr, _ = score_scalar(part_text(response, qid, 'c'),
                               gt['total_DR_dB'], 'db', 'dB',
                               5.0, 10.0, mode='abs')
    text_d = part_text(response, qid, 'd')
    ratio_enob, _ = score_scalar(text_d, gt['min_ENOB'], 'count', None,
                                 0.0, 1.0, mode='abs')
    ten_ok = polarity(text_d,
                      ('sufficient', 'enough', 'adequate', 'suffices',
                       '足够', '充分', '满足', '够用'),
                      ('insufficient', 'inadequate', '不足', '不够',
                       '无法满足', '不充分'))
    ratio_ten = 1.0 if ten_ok is not None \
        and ten_ok == bool(gt['ten_bit_ok']) else 0.0
    return _result([
        _criterion('strongest_power', 2, ratio_pmax,
                   f"ref={gt['strongest_dBm']} dBm; {binding}"),
        _criterion('weakest_power', 2, ratio_pmin,
                   f"ref={gt['weakest_dBm']} dBm; {binding}"),
        _criterion('effective_peak', 5, ratio_peak,
                   f"ref={gt['effective_peak_dBm']} dBm"),
        _criterion('total_DR', 5, ratio_dr, f"ref={gt['total_DR_dB']} dB"),
        _criterion('min_ENOB', 4, ratio_enob, f"ref={gt['min_ENOB']} bits"),
        _criterion('ten_bit_ok', 2, ratio_ten,
                   f"expected {gt['ten_bit_ok']}"),
    ])


def _score_q5(question, response):
    gt = question['ground_truth']
    qid = question['id']
    shift = gt['freq_shift_MHz']
    ratio_shift, _ = score_scalar_any(part_text(response, qid, 'a'),
                                      [shift, -shift], 'freq', 'MHz',
                                      0.2, 0.5, mode='abs')
    text_b = part_text(response, qid, 'b')
    ratio_lpf, _ = _score_preferred(
        text_b, gt['LPF_cutoff_kHz'], 'freq', 'kHz', 0.30, 0.50)
    lpf_value = _preferred_value(text_b, 'freq', 'kHz')

    text_c = part_text(response, qid, 'c')
    decim_text = _keyword_segment(
        text_c, ('decimation factor', 'decimation', 'decimate', 'factor',
                 '抽取因子', '抽取', '降采样因子', '降采样倍数'),
        ('new sampling rate', 'new rate', 'sampling rate', '新采样率',
         '新的采样率', '输出采样率')) or text_c
    rate_text = _keyword_segment(
        text_c, ('new sampling rate', 'new rate', 'sampling rate',
                 '新采样率', '新的采样率', '输出采样率'), ()) or text_c
    reduction_text = part_text(response, qid, 'd')

    decim_value = _preferred_value(decim_text, 'count', None)
    rate_value = _preferred_value(rate_text, 'freq', 'kHz')
    reduction_value = _preferred_value(reduction_text, 'count', None)

    def direct_ratio(value, reference):
        if value is None:
            return 0.0
        return score_scalar(str(value), reference, 'none', None,
                            0.20, 0.40, mode='rel')[0]

    ratio_decim = direct_ratio(decim_value, gt['decimation'])
    ratio_rate = direct_ratio(rate_value, gt['new_rate_kHz'])
    ratio_red = direct_ratio(reduction_value, gt['data_reduction'])

    # A different but valid cutoff estimate changes the maximum decimation.
    # Credit such a design only when all downstream values are internally
    # consistent and the cutoff itself remains within its accepted window.
    if lpf_value and lpf_value > 0 and decim_value and decim_value > 0:
        fs_khz = gt['new_rate_kHz'] * gt['decimation']
        max_decim = max(1, math.floor(fs_khz / (2.0 * lpf_value)))
        decim_error = abs(decim_value - max_decim)
        consistent_decim = (1.0 if decim_error <= 1.0 else
                            0.5 if (decim_value <= max_decim
                                    and decim_value >= 0.8 * max_decim)
                            else 0.0)
        design_decim = min(ratio_lpf, consistent_decim)
        ratio_decim = max(ratio_decim, design_decim)

        expected_rate = fs_khz / decim_value
        if rate_value is not None:
            consistent_rate = score_scalar(
                str(rate_value), expected_rate, 'none', None,
                0.05, 0.10, mode='rel')[0]
            ratio_rate = max(ratio_rate, min(design_decim, consistent_rate))
        if reduction_value is not None:
            consistent_reduction = score_scalar(
                str(reduction_value), decim_value, 'none', None,
                0.05, 0.10, mode='rel')[0]
            ratio_red = max(
                ratio_red, min(design_decim, consistent_reduction))
    return _result([
        _criterion('shift', 4, ratio_shift, f'accepted ±{abs(shift)} MHz'),
        _criterion('lpf', 5, ratio_lpf, f"ref={gt['LPF_cutoff_kHz']} kHz"),
        _criterion('decim', 3, ratio_decim, f"ref={gt['decimation']}"),
        _criterion('new_rate', 3, ratio_rate, f"ref={gt['new_rate_kHz']} kHz"),
        _criterion('reduction', 5, ratio_red, f"ref={gt['data_reduction']}"),
    ])


_KIND_KEYS = {
    'mod_type': _score_q1,
    'EbN0': _score_q2,
    'papr_estimation': _score_q3,
    'ENOB': _score_q4,
    'decim': _score_q5,
}


def _question_kind(rubric):
    for key in _KIND_KEYS:
        if key in rubric:
            return key
    return None


def score_l3_question(question, response, signals):
    """Return a deterministic score, or None for an unmarked question."""
    if not is_l3_deterministic_question(question):
        return None
    if '===ANSWERS===' not in (response or ''):
        return _result(
            [_criterion('missing_answers', 20, 0, 'no ===ANSWERS=== block')],
            parse_error='missing ===ANSWERS=== block')
    kind = _question_kind(question['rubric'])
    if kind is None:
        raise ValueError(f"unrecognized L3 rubric for {question['id']}")
    scorer = _KIND_KEYS[kind]
    if kind == 'papr_estimation':
        return scorer(question, response, signals)
    return scorer(question, response)


# --- canonical reference answers ---------------------------------------------


def reference_response(meta):
    """Render the ground truth as a full labeled answer block (for replay)."""
    signals = meta['generation_params']['signals']
    lines = ['===ANSWERS===']
    for question in meta['questions']:
        gt = question['ground_truth']
        qid = question['id']
        kind = _question_kind(question['rubric'])
        if kind == 'mod_type':
            lines += [
                f"{qid}a: {gt['type']}, Rs = {gt['symbol_rate_ksps']} ksps",
                f"{qid}b: Rb = {gt['bit_rate_kbps']} kbps",
                f"{qid}c: {gt['spectral_efficiency_bps_Hz']} bps/Hz",
                f"{qid}d: {gt['bit_rate_64QAM_kbps']} kbps",
            ]
        elif kind == 'EbN0':
            verdict = ('Yes, the requirement is met'
                       if gt['feasible'] else 'No, the requirement is not met')
            lines += [
                f"{qid}a: Noise in band = {gt['noise_in_band_dBm']} dBm; "
                f"SNR = {gt['SNR_dB']} dB",
                f"{qid}b: Eb/N0 = {gt['EbN0_dB']} dB",
                f"{qid}c: {gt['EbN0_required_dB']} dB",
                f"{qid}d: {verdict}; margin = {gt['margin_dB']} dB",
            ]
        elif kind == 'papr_estimation':
            best = gt.get('best_for_PA_accepted', [gt['best_for_PA']['signal']])
            worst = gt.get('worst_for_PA_accepted',
                           [gt['worst_for_PA']['signal']])
            verdict = ('peak exceeds P1dB, distortion occurs'
                       if gt['exceeds_P1dB']
                       else 'peak stays below P1dB, no distortion occurs')
            lines += [
                f"{qid}a: " + '; '.join(
                    f"{entry['type']}: {entry['PAPR_dB']} dB"
                    for entry in gt['papr_per_signal']),
                f"{qid}b: Best suited: {', '.join(best)}; "
                f"least suited: {', '.join(worst)}",
                f"{qid}c: Peak = {gt['digital_peak_dBm']} dBm, {verdict}",
                f"{qid}d: IBO = {gt['IBO_dB']} dB; larger back-off reduces "
                f"PA efficiency",
            ]
        elif kind == 'ENOB':
            verdict = ('10 bits is sufficient' if gt['ten_bit_ok']
                       else '10 bits is not sufficient')
            lines += [
                f"{qid}a: Strongest = {gt['strongest_dBm']} dBm; "
                f"weakest = {gt['weakest_dBm']} dBm",
                f"{qid}b: Effective peak = {gt['effective_peak_dBm']} dBm",
                f"{qid}c: Total dynamic range = {gt['total_DR_dB']} dB",
                f"{qid}d: minimum {gt['min_ENOB']} bits; {verdict}",
            ]
        elif kind == 'decim':
            lines += [
                f"{qid}a: shift by {gt['freq_shift_MHz']} MHz",
                f"{qid}b: {gt['LPF_cutoff_kHz']} kHz",
                f"{qid}c: decimation factor = {gt['decimation']}; "
                f"new sampling rate = {gt['new_rate_kHz']} kHz",
                f"{qid}d: data volume reduced by a factor of "
                f"{gt['data_reduction']}",
            ]
    lines.append('===END===')
    return '\n'.join(lines)
