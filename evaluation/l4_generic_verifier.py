"""Deterministic per-question-type scoring for the generic EMRB L4 instances.

Covers the eight non-repaired question types (QT01/02/03/04/05-legacy/06/08/10)
with the same contract as the L1/L3 verifiers: the existing labeled-line
answer format is parsed as-is (no prompt change, stored responses stay
score-only re-evaluable), every criterion binds one requested output of one
sub-question to one ground-truth value with unit-aware tolerances, ambiguous
conventions carry accepted sets (multitone FM beta, burst-average SNR), and
verdicts use negation-resolved polarity.
"""
import math

from evaluation.answer_parsing import (
    asserted_token,
    DIGITAL_TOKENS,
    TYPE_TOKENS,
    _INDEX_RE,
    has_token,
    normalize_text,
    polarity,
    score_per_signal,
    score_scalar,
    score_scalar_any,
    signal_anchor,
)
from evaluation.l1_verifier import part_text

SCORER_VERSION = 'l4-generic-v3'

_DIGITAL_TYPES = ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM')


def is_l4_generic_question(question):
    return question.get('rubric', {}).get('scoring') == SCORER_VERSION


def _criterion(identifier, maximum, ratio, reason):
    ratio = min(1.0, max(0.0, float(ratio)))
    return {
        'id': identifier,
        'score': round(maximum * ratio, 2),
        'max': float(maximum),
        'reason': reason,
    }


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


def _block(response, qid):
    """All sub-answers of one question joined (for questions whose prose has
    no fixed letter->output mapping, e.g. QT01)."""
    parts = [part_text(response, qid, letter) for letter in 'abcd']
    parts.append(part_text(response, qid, ''))
    seen, out = set(), []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return '\n'.join(out)


def _type_ratio(text, expected_type, family_ratios):
    """Exact-type / family / generic-digital credit for a modulation label.
    The label must be asserted: "not QPSK" is a rejection, not an
    identification (remediation log §12.4)."""
    if asserted_token(text, TYPE_TOKENS.get(expected_type, ())):
        return 1.0
    for family_token, ratio in family_ratios:
        if asserted_token(text, (family_token,)):
            return ratio
    if expected_type in _DIGITAL_TYPES and asserted_token(text, DIGITAL_TOKENS):
        return 0.25
    return 0.0


# --- QT01: symbol rate + modulation order --------------------------------


def _score_qt01(question, response, signals):
    gt = question['ground_truth']
    qid = question['id']
    text = _block(response, qid)
    family_of = {'BPSK': 'psk', 'QPSK': 'psk', '8PSK': 'psk',
                 '16QAM': 'qam', '64QAM': 'qam'}
    if 'signal_1' in gt:
        targets = [s for s in signals if s['type'] in _DIGITAL_TYPES][:2]
        refs = [gt['signal_1']['symbol_rate_ksps'],
                gt['signal_2']['symbol_rate_ksps']]
        norm = normalize_text(text)
        # "Signal 1"/"信号1" references number the pair by ascending frequency
        ascending = sorted(range(2),
                           key=lambda i: targets[i]['center_frequency_MHz'])
        index_anchor = {}
        for m in _INDEX_RE.finditer(norm):
            k = int(m.group(1) or m.group(2)) - 1
            if 0 <= k < 2 and ascending[k] not in index_anchor:
                index_anchor[ascending[k]] = m.start()
        if len(index_anchor) == 2:
            anchors = [index_anchor[0], index_anchor[1]]
        else:
            anchors = [signal_anchor(text, s, targets,
                                     s.get('center_frequency_MHz'))
                       for s in targets]
        ratios_rs = []
        ratios_mod = []
        if all(a is not None for a in anchors) and len(set(anchors)) == 2:
            order = sorted(range(2), key=lambda i: anchors[i])
            bounds = [anchors[i] for i in order] + [len(norm)]
            segments = {i: norm[bounds[rank]:bounds[rank + 1]]
                        for rank, i in enumerate(order)}
            for i in range(2):
                ratio, _ = score_scalar(segments[i], refs[i],
                                        'symrate', 'ksps', 0.15, 0.30)
                ratios_rs.append(ratio)
            ratio_rs = sum(ratios_rs) / 2
            reason_rs = 'anchored ' + ' '.join(
                f'{gt[f"signal_{i + 1}"]["type"]}:{r:g}'
                for i, r in enumerate(ratios_rs))
        else:
            segments = {0: norm, 1: norm}
            ratio_rs, reason_rs = score_per_signal(
                text, targets, refs, 'symrate', 'ksps', 0.15, 0.30)
        for i, target in enumerate(targets):
            expected = gt[f'signal_{i + 1}']['type']
            ratios_mod.append(_type_ratio(
                segments[i], expected, [(family_of[expected], 0.5)]))
        ratio_mod = sum(ratios_mod) / 2
        # 'symbol'/'rate' are excluded: restating the requested symbol rate
        # must not satisfy the reasoning criterion (§8.1.8); the method must
        # be asserted, not declared inapplicable (§12.4)
        method = asserted_token(text, (
            'bandwidth', 'spectral', 'spectrum', 'constellation',
            'cyclostation', 'null', 'envelope', 'eye', 'cluster',
            '带宽', '频谱', '星座', '包络'))
        return _result([
            _criterion('symbol_rates', 8, ratio_rs, reason_rs),
            _criterion('mod_orders', 8, ratio_mod,
                       f"expected {gt['signal_1']['type']}"
                       f"/{gt['signal_2']['type']}"),
            _criterion('reasoning', 4, 1.0 if method else 0.0,
                       'names an analysis method'),
        ])
    ratio_rs, reason_rs = score_scalar(text, gt['symbol_rate_ksps'],
                                       'symrate', 'ksps', 0.15, 0.30)
    ratio_mod = _type_ratio(text, gt['type'], [(family_of[gt['type']], 0.5)])
    return _result([
        _criterion('symbol_rate', 10, ratio_rs, reason_rs),
        _criterion('mod_order', 10, ratio_mod, f"expected {gt['type']}"),
    ])


# --- QT02: FM parameters ---------------------------------------------------


def _score_qt02(question, response):
    gt = question['ground_truth']
    qid = question['id']
    ratio_dev, _ = score_scalar(part_text(response, qid, 'a'),
                                gt['frequency_deviation_kHz'],
                                'freq', 'kHz', 0.20, 0.40)
    ratio_mf, _ = score_scalar(part_text(response, qid, 'b'),
                               gt['max_modulating_freq_kHz'],
                               'freq', 'kHz', 0.30, 0.60)
    beta_accepted = gt.get('modulation_index_accepted',
                           [gt['modulation_index']])
    ratio_beta, _ = score_scalar_any(part_text(response, qid, 'c'),
                                     beta_accepted, 'none', None, 0.30, 0.60)
    text_d = part_text(response, qid, 'd')
    carson_accepted = gt.get('carson_bandwidth_accepted_kHz',
                             [gt['carson_bandwidth_kHz']])
    ratio_cv, _ = score_scalar_any(text_d, carson_accepted,
                                   'freq', 'kHz', 0.20, 0.40)
    verdict = polarity(text_d,
                       ('consistent', 'matches', 'match', 'agrees', 'agree',
                        'confirms', 'confirmed', 'holds', '一致', '符合'),
                       ('inconsistent', 'violates', 'violated', 'disagrees',
                        '不一致', '不符'))
    ratio_verdict = 1.0 if verdict is True else 0.0
    return _result([
        _criterion('deviation', 6, ratio_dev,
                   f"ref={gt['frequency_deviation_kHz']} kHz"),
        _criterion('mod_freq', 5, ratio_mf,
                   f"ref={gt['max_modulating_freq_kHz']} kHz"),
        _criterion('mod_index', 4, ratio_beta, f'accepted {beta_accepted}'),
        _criterion('carson_value', 3, ratio_cv,
                   f'accepted {carson_accepted} kHz'),
        _criterion('carson_verdict', 2, ratio_verdict,
                   'bandwidth is Carson-consistent by construction'),
    ])


# --- QT03: chirp radar -----------------------------------------------------


def _score_qt03(question, response):
    gt = question['ground_truth']
    qid = question['id']
    ratio_tbp, _ = score_scalar(part_text(response, qid, 'a'), gt['TBP'],
                                'none', None, 0.10, 0.20)
    ratio_pg, _ = score_scalar(part_text(response, qid, 'b'),
                               gt['processing_gain_dB'], 'db', 'dB',
                               1.0, 2.0, mode='abs')
    ratio_rr, _ = score_scalar(part_text(response, qid, 'c'),
                               gt['range_resolution_m'], 'length', 'm',
                               0.10, 0.20)
    return _result([
        _criterion('TBP', 7, ratio_tbp, f"ref={gt['TBP']}"),
        _criterion('processing_gain', 7, ratio_pg,
                   f"ref={gt['processing_gain_dB']} dB"),
        _criterion('range_resolution', 6, ratio_rr,
                   f"ref={gt['range_resolution_m']} m"),
    ])


# --- QT04: burst analysis --------------------------------------------------


def _score_qt04(question, response):
    gt = question['ground_truth']
    qid = question['id']
    ratio_mod = _type_ratio(part_text(response, qid, 'a'), gt['modulation'],
                            [('psk', 0.6), ('qam', 0.6), ('fsk', 0.6)])
    ratio_duty, _ = score_scalar(part_text(response, qid, 'b'),
                                 gt['duty_cycle'], 'ratio', None,
                                 0.05, 0.10, mode='abs')
    text_c = part_text(response, qid, 'c')
    loss = gt['snr_loss_dB']
    ratio_loss, _ = score_scalar_any(text_c, [loss, -loss], 'db', 'dB',
                                     1.0, 2.0, mode='abs')
    # the derivation must be asserted, not negated (§12.4)
    derivation = asserted_token(text_c, ('10log', '10 log', 'log10', 'log₁₀',
                                         'duty', '占空比'))
    return _result([
        _criterion('modulation', 5, ratio_mod, f"expected {gt['modulation']}"),
        _criterion('duty_cycle', 5, ratio_duty, f"ref={gt['duty_cycle']}"),
        _criterion('snr_loss_value', 6, ratio_loss,
                   f"ref={loss} dB (sign-agnostic)"),
        _criterion('snr_loss_derivation', 4, 1.0 if derivation else 0.0,
                   'invokes 10log10(duty cycle)'),
    ])


# --- QT05 legacy: spectral gap link budget ---------------------------------


def _score_qt05_legacy(question, response):
    gt = question['ground_truth']
    qid = question['id']
    ratio_bw, _ = score_scalar(part_text(response, qid, 'a'),
                               gt['available_gap_MHz'], 'freq', 'MHz',
                               0.05, 0.25, mode='abs')
    ratio_dr, _ = score_scalar(part_text(response, qid, 'b'),
                               gt['data_rate_kbps'], 'bitrate', 'kbps',
                               0.20, 0.40)
    ratio_pw, _ = score_scalar(part_text(response, qid, 'c'),
                               gt['required_power_dBm'], 'dbm', 'dBm',
                               3.0, 6.0, mode='abs')
    return _result([
        _criterion('bandwidth', 6, ratio_bw,
                   f"ref={gt['available_gap_MHz']} MHz"),
        _criterion('data_rate', 7, ratio_dr,
                   f"ref={gt['data_rate_kbps']} kbps"),
        _criterion('power', 7, ratio_pw,
                   f"ref={gt['required_power_dBm']} dBm (received in-band)"),
    ])


# --- QT06: OFDM parameters --------------------------------------------------


def _score_qt06(question, response):
    gt = question['ground_truth']
    qid = question['id']
    ratio_sc, _ = score_scalar(part_text(response, qid, 'a'),
                               gt['subcarrier_spacing_kHz'], 'freq', 'kHz',
                               0.20, 0.40)
    ratio_cp, _ = score_scalar(part_text(response, qid, 'b'),
                               gt['cp_duration_us'], 'time', 'us',
                               0.30, 0.60)
    ratio_bw, _ = score_scalar(part_text(response, qid, 'c'),
                               gt['occupied_bandwidth_MHz'], 'freq', 'MHz',
                               0.20, 0.40)
    ratio_sym, _ = score_scalar(part_text(response, qid, 'd'),
                                gt['symbol_duration_us'], 'time', 'us',
                                0.20, 0.40)
    return _result([
        _criterion('sc_spacing', 6, ratio_sc,
                   f"ref={gt['subcarrier_spacing_kHz']} kHz"),
        _criterion('cp_duration', 5, ratio_cp,
                   f"ref={gt['cp_duration_us']} us"),
        _criterion('occupied_bw', 5, ratio_bw,
                   f"ref={gt['occupied_bandwidth_MHz']} MHz"),
        _criterion('sym_duration', 4, ratio_sym,
                   f"ref={gt['symbol_duration_us']} us"),
    ])


# --- QT08: AM parameters -----------------------------------------------------


def _score_qt08(question, response):
    gt = question['ground_truth']
    qid = question['id']
    ratio_depth, _ = score_scalar(part_text(response, qid, 'a'),
                                  gt['modulation_depth'], 'ratio', None,
                                  0.15, 0.30)
    ratio_mf, _ = score_scalar(part_text(response, qid, 'b'),
                               gt['modulating_freq_kHz'], 'freq', 'kHz',
                               0.20, 0.40)
    ratio_eff, _ = score_scalar(part_text(response, qid, 'c'),
                                gt['efficiency'], 'ratio', None, 0.30, 0.60)
    ratio_bw, _ = score_scalar(part_text(response, qid, 'd'),
                               gt['bandwidth_kHz'], 'freq', 'kHz',
                               0.20, 0.40)
    return _result([
        _criterion('mod_depth', 5, ratio_depth,
                   f"ref={gt['modulation_depth']}"),
        _criterion('mod_freq', 5, ratio_mf,
                   f"ref={gt['modulating_freq_kHz']} kHz"),
        _criterion('efficiency', 5, ratio_eff, f"ref={gt['efficiency']}"),
        _criterion('bandwidth', 5, ratio_bw, f"ref={gt['bandwidth_kHz']} kHz"),
    ])


# --- QT10: Shannon capacity --------------------------------------------------


def _score_qt10(question, response):
    gt = question['ground_truth']
    qid = question['id']
    conventions = gt.get('accepted_conventions') or [{
        'snr_dB': gt['snr_dB'],
        'shannon_capacity_Mbps': gt['shannon_capacity_Mbps'],
        'spectral_efficiency_gap': [gt['spectral_efficiency_gap']],
    }]
    text_a = part_text(response, qid, 'a')
    text_b = part_text(response, qid, 'b')
    text_c = part_text(response, qid, 'c')
    # score the whole answer against ONE self-consistent convention (active-
    # or average-power), never a per-field best-of across conventions
    best = None
    for convention in conventions:
        gaps = convention['spectral_efficiency_gap']
        if not isinstance(gaps, (list, tuple)):
            gaps = [gaps]
        ratio_snr, _ = score_scalar(text_a, convention['snr_dB'],
                                    'db', 'dB', 3.0, 6.0, mode='abs')
        ratio_cap, _ = score_scalar(text_b,
                                    convention['shannon_capacity_Mbps'],
                                    'bitrate', 'Mbps', 0.20, 0.40)
        ratio_gap, _ = score_scalar_any(text_c, gaps, 'eff', 'bps/Hz',
                                        0.30, 0.60)
        total = 6 * ratio_snr + 8 * ratio_cap + 6 * ratio_gap
        if best is None or total > best[0]:
            best = (total, ratio_snr, ratio_cap, ratio_gap, convention)
    _, ratio_snr, ratio_cap, ratio_gap, convention = best
    return _result([
        _criterion('snr', 6, ratio_snr, f"ref={convention['snr_dB']} dB "
                   f"({len(conventions)} convention(s))"),
        _criterion('capacity', 8, ratio_cap,
                   f"ref={convention['shannon_capacity_Mbps']} Mbps"),
        _criterion('gap_analysis', 6, ratio_gap,
                   f"accepted {convention['spectral_efficiency_gap']} bps/Hz"),
    ])


_SCORERS = {
    'QT01': _score_qt01,
    'QT02': _score_qt02,
    'QT03': _score_qt03,
    'QT04': _score_qt04,
    'QT05': _score_qt05_legacy,
    'QT06': _score_qt06,
    'QT08': _score_qt08,
    'QT10': _score_qt10,
}


def score_l4_generic_question(question, response, signals):
    """Return a deterministic score, or None for an unmarked question."""
    if not is_l4_generic_question(question):
        return None
    if '===ANSWERS===' not in (response or ''):
        return _result(
            [_criterion('missing_answers', 20, 0, 'no ===ANSWERS=== block')],
            parse_error='missing ===ANSWERS=== block')
    qt = question.get('question_type')
    if qt not in _SCORERS:
        raise ValueError(f'unsupported generic L4 question type: {qt}')
    if qt == 'QT01':
        return _SCORERS[qt](question, response, signals)
    return _SCORERS[qt](question, response)


# --- canonical reference answers ---------------------------------------------


def reference_response_lines(question):
    """Canonical labeled answer lines for one generic L4 question."""
    gt = question['ground_truth']
    qid = question['id']
    qt = question.get('question_type')
    if qt == 'QT01':
        if 'signal_1' in gt:
            return [
                f"{qid}a: {gt['signal_1']['center_frequency_MHz']:+.1f} MHz "
                f"signal: symbol rate = {gt['signal_1']['symbol_rate_ksps']} "
                f"ksps, {gt['signal_1']['type']}",
                f"{qid}b: {gt['signal_2']['center_frequency_MHz']:+.1f} MHz "
                f"signal: symbol rate = {gt['signal_2']['symbol_rate_ksps']} "
                f"ksps, {gt['signal_2']['type']}",
                f"{qid}c: distinguished by occupied bandwidth and "
                f"constellation clustering",
            ]
        return [f"{qid}a: symbol rate = {gt['symbol_rate_ksps']} ksps, "
                f"{gt['type']}, M = {gt['M']}"]
    if qt == 'QT02':
        return [
            f"{qid}a: {gt['frequency_deviation_kHz']} kHz",
            f"{qid}b: {gt['max_modulating_freq_kHz']} kHz",
            f"{qid}c: beta = {gt['modulation_index']}",
            f"{qid}d: Carson bandwidth = {gt['carson_bandwidth_kHz']} kHz, "
            f"consistent with the observed bandwidth",
        ]
    if qt == 'QT03':
        return [
            f"{qid}a: TBP = {gt['TBP']}",
            f"{qid}b: {gt['processing_gain_dB']} dB",
            f"{qid}c: {gt['range_resolution_m']} m",
        ]
    if qt == 'QT04':
        return [
            f"{qid}a: {gt['modulation']}",
            f"{qid}b: duty cycle = {gt['duty_cycle']}",
            f"{qid}c: SNR loss = {gt['snr_loss_dB']} dB, "
            f"derived from 10log10(duty cycle)",
        ]
    if qt == 'QT05':
        return [
            f"{qid}a: {gt['available_gap_MHz']} MHz",
            f"{qid}b: data rate = {gt['data_rate_kbps']} kbps "
            f"(symbol rate {gt['symbol_rate_ksps']} ksps)",
            f"{qid}c: {gt['required_power_dBm']} dBm",
        ]
    if qt == 'QT06':
        return [
            f"{qid}a: {gt['subcarrier_spacing_kHz']} kHz",
            f"{qid}b: {gt['cp_duration_us']} us",
            f"{qid}c: {gt['occupied_bandwidth_MHz']} MHz",
            f"{qid}d: {gt['symbol_duration_us']} us",
        ]
    if qt == 'QT08':
        return [
            f"{qid}a: modulation depth = {gt['modulation_depth']}",
            f"{qid}b: {gt['modulating_freq_kHz']} kHz",
            f"{qid}c: efficiency = {gt['efficiency']}",
            f"{qid}d: {gt['bandwidth_kHz']} kHz",
        ]
    if qt == 'QT10':
        return [
            f"{qid}a: SNR = {gt['snr_dB']} dB",
            f"{qid}b: {gt['shannon_capacity_Mbps']} Mbps",
            f"{qid}c: {gt['spectral_efficiency_gap']} bps/Hz below the "
            f"Shannon limit",
        ]
    raise ValueError(f'unsupported generic L4 question type: {qt}')
