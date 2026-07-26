"""Deterministic per-question-type scoring for EMRB L1.

Scores the existing labeled-line answer format (no prompt change, so stored
model responses remain score-only re-evaluable). Each criterion binds one
requested output of one sub-question to one ground-truth value: values are
read only from their own sub-answer, converted into the requested unit,
per-signal values are bound by signal identity or position, selections accept
tied alternatives, and booleans use negation-resolved polarity.
"""
import math
import re

from evaluation.answer_parsing import (
    DIGITAL_TOKENS,
    TYPE_TOKENS,
    _DIGITAL_TYPES,
    asserted_token,
    claimed_indices,
    freq_anchor_positions,
    has_token,
    index_reference,
    normalize_text,
    polarity,
    quantities_in_family,
    score_per_signal as _shared_score_per_signal,
    score_scalar as _shared_score_scalar,
    score_scalar_any as _shared_score_scalar_any,
    signal_anchor,
    signal_center_mhz,
)
from evaluation.auto_scorer import parse_answer_block

SCORER_VERSION = 'l1-deterministic-v6'

STRONGEST_KEYWORDS = (
    'strongest', 'strong', 'highest power', 'most powerful',
    '最强', '功率最大', '最大功率', '功率最高',
)
WEAKEST_KEYWORDS = (
    'weakest', 'weak', 'lowest power', 'least powerful',
    '最弱', '功率最小', '最小功率', '功率最低',
)


def score_scalar(*args, **kwargs):
    kwargs.setdefault('candidate_policy', 'asserted')
    return _shared_score_scalar(*args, **kwargs)


def score_scalar_any(*args, **kwargs):
    kwargs.setdefault('candidate_policy', 'asserted')
    return _shared_score_scalar_any(*args, **kwargs)


def score_per_signal(*args, **kwargs):
    kwargs.setdefault('candidate_policy', 'asserted')
    return _shared_score_per_signal(*args, **kwargs)


def is_l1_deterministic_question(question):
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


def part_text(response, question_id, letter):
    """Sub-answer for e.g. Q1(a): the 'Q1a' line, else the whole 'Q1' block."""
    answers = parse_answer_block(response)
    return answers.get(f'{question_id}{letter}') or answers.get(question_id, '')


def _keyword_segment(text, keywords, stop_keywords):
    """Text following the first keyword, up to the first stop keyword."""
    lowered = normalize_text(text).lower()
    starts = [m.end() for kw in keywords
              for m in re.finditer(re.escape(kw), lowered)]
    if not starts:
        return ''
    start = min(starts)
    stops = [m.start() for kw in stop_keywords
             for m in re.finditer(re.escape(kw), lowered)
             if m.start() > start]
    return normalize_text(text)[start:min(stops) if stops else len(lowered)]


def _score_frequency_list(text, references_mhz, full_abs, half_abs):
    """Nearest-match scoring of a requested frequency list.

    Each reference claims its nearest candidate that no other reference has
    already claimed, so supporting values the answer legitimately mentions
    (a sweep range, a coarse and a refined estimate of the same carrier, an
    enumeration label) cannot displace the requested estimates. Rank-position
    alignment was replaced here because it made one extra in-family number
    shift every later slot, zeroing answers whose values were correct.

    Reference frequencies in this level are separated by far more than
    ``full_abs``, so the claim order does not create ambiguity; the claimed
    set still prevents one value from covering two references, and a
    reference left without any candidate still scores zero.
    """
    values = [v for v, _ in quantities_in_family(text, 'freq', 'MHz',
                                                 explicit_only=True)]
    if len(values) < len(references_mhz):
        values = [v for v, _ in quantities_in_family(text, 'freq', 'MHz')]
    claimed = set()
    ratios = []
    for reference in sorted(references_mhz):
        best = None
        for index, value in enumerate(values):
            if index in claimed:
                continue
            error = abs(value - reference)
            if best is None or error < best[0]:
                best = (error, index)
        if best is None:
            ratios.append(0.0)
            continue
        claimed.add(best[1])
        error = best[0]
        ratios.append(1.0 if error <= full_abs else 0.5 if error <= half_abs
                      else 0.0)
    return sum(ratios) / len(references_mhz)


def _score_selection(text, keywords, stop_keywords, accepted_mhz, tol=0.5):
    segment = _keyword_segment(text, keywords, stop_keywords)
    for value, _ in quantities_in_family(segment, 'freq', 'MHz'):
        if any(abs(value - accepted) <= tol for accepted in accepted_mhz):
            return 1.0
    return 0.0


def _score_category_per_signal(text, signals, accepted_categories):
    """Per-signal bandwidth-category check, identity-anchored."""
    category_tokens = {'Narrowband': ('narrowband', 'narrow', '窄带'),
                       'Midband': ('midband', 'mid-band', 'mid', '中带'),
                       'Wideband': ('wideband', 'wide', 'broadband', '宽带')}
    anchors = [signal_anchor(text, s, signals, signal_center_mhz(s))
               for s in signals]
    n = len(signals)
    norm = normalize_text(text)
    ratios = []
    if all(a is not None for a in anchors) and len(set(anchors)) == n:
        order = sorted(range(n), key=lambda i: anchors[i])
        bounds = [anchors[i] for i in order] + [len(norm)]
        for rank, i in enumerate(order):
            segment = norm[bounds[rank]:bounds[rank + 1]]
            accepted = accepted_categories[i]
            hit = any(has_token(segment, category_tokens[c])
                      for c in accepted)
            wrong = any(has_token(segment, tokens)
                        for cat, tokens in category_tokens.items()
                        if cat not in accepted)
            ratios.append(1.0 if hit and not wrong else 0.0)
        return sum(ratios) / n
    # positional: k-th category token belongs to the k-th signal
    found = re.findall(r'(narrowband|narrow|midband|mid-band|mid|wideband'
                       r'|wide|broadband|窄带|中带|宽带)', norm.lower())
    canon = {'narrowband': 'Narrowband', 'narrow': 'Narrowband',
             '窄带': 'Narrowband',
             'midband': 'Midband', 'mid-band': 'Midband', 'mid': 'Midband',
             '中带': 'Midband',
             'wideband': 'Wideband', 'wide': 'Wideband',
             'broadband': 'Wideband', '宽带': 'Wideband'}
    for i in range(n):
        ratios.append(1.0 if i < len(found)
                      and canon[found[i]] in accepted_categories[i] else 0.0)
    return sum(ratios) / n


# --- per-question scoring -----------------------------------------------------


def _score_q1(question, response):
    gt = question['ground_truth']
    qid = question['id']
    ratio_a, reason_a = score_scalar(part_text(response, qid, 'a'),
                                     gt['signal_count'], 'none', None,
                                     0, None, mode='exact_int')
    ratio_b = _score_frequency_list(part_text(response, qid, 'b'),
                                    gt['center_frequencies_MHz'],
                                    0.5, 1.0)
    text_c = part_text(response, qid, 'c')
    strongest = gt.get('strongest_accepted_MHz', [gt['strongest_signal_MHz']])
    weakest = gt.get('weakest_accepted_MHz', [gt['weakest_signal_MHz']])
    ratio_c = 0.5 * _score_selection(
        text_c, STRONGEST_KEYWORDS, WEAKEST_KEYWORDS, strongest)
    ratio_c += 0.5 * _score_selection(
        text_c, WEAKEST_KEYWORDS, (), weakest)
    ratio_d, reason_d = score_scalar(part_text(response, qid, 'd'),
                                     gt['power_difference_dB'], 'db', 'dB',
                                     2.0, 4.0, mode='abs')
    return _result([
        _criterion('count', 4, ratio_a, reason_a),
        _criterion('frequencies', 8, ratio_b, 'nearest-match ±0.5 MHz'),
        _criterion('strongest_weakest', 4, ratio_c, 'tie-aware selection'),
        _criterion('power_diff', 4, ratio_d, reason_d),
    ])


def _score_q2(question, response, signals):
    gt = question['ground_truth']
    qid = question['id']
    ratio_a, reason_a = score_per_signal(
        part_text(response, qid, 'a'), signals, gt['powers_dBm'],
        'dbm', 'dBm', 3.0, 6.0, mode='abs')
    # (b) tests the dBm->mW conversion of the model's OWN (a) estimate, so the
    # linear-domain window mirrors (a)'s ±3 dB tolerance (x2 full, x4 half).
    ratio_b, reason_b = score_per_signal(
        part_text(response, qid, 'b'), signals, gt['powers_mW'],
        'power_lin', 'mW', 1.0, 3.0, mode='factor')
    text_c = part_text(response, qid, 'c')
    ratio_c_dbm, _ = score_scalar(text_c, gt['total_power_dBm'],
                                  'dbm', 'dBm', 1.0, 2.0, mode='abs')
    ratio_c_mw, _ = score_scalar(text_c, gt['total_power_mW'],
                                 'power_lin', 'mW', 0.26, 0.60, mode='rel')
    ratio_d, reason_d = score_per_signal(
        part_text(response, qid, 'd'), signals, gt['SNR_per_signal_dB'],
        'db', 'dB', 3.0, 6.0, mode='abs')
    return _result([
        _criterion('power_est', 6, ratio_a, reason_a),
        _criterion('dbm_to_mw', 4, ratio_b, reason_b),
        _criterion('total_power', 4, max(ratio_c_dbm, ratio_c_mw),
                   'accepted in dBm or mW'),
        _criterion('snr', 6, ratio_d, reason_d),
    ])


def _score_q3(question, response):
    gt = question['ground_truth']
    qid = question['id']
    ratio_a, _ = score_scalar(part_text(response, qid, 'a'),
                              gt['duration_ms'], 'time', 'ms',
                              0.01, 0.05, mode='rel')
    ratio_b, _ = score_scalar(part_text(response, qid, 'b'),
                              gt['Ts_ns'], 'time', 'ns',
                              0.01, 0.05, mode='rel')
    ratio_c, _ = score_scalar(part_text(response, qid, 'c'),
                              gt['delta_f_Hz'], 'freq', 'Hz',
                              0.01, 0.05, mode='rel')
    text_d = part_text(response, qid, 'd')
    ratio_n, _ = score_scalar(text_d, gt['N_for_half_delta_f'],
                              'count', None, 0.01, None, mode='rel')
    # the method must be asserted: "do not increase N" names the
    # vocabulary while asserting the opposite (remediation log §12.4)
    mentions_samples = asserted_token(
        text_d,
        ('n', 'samples', 'sample', '采样点数', '采样点', '样本数', '样本'),
    )
    mentions_grow = asserted_token(text_d, ('increase', 'increasing',
                                            'double', 'doubling', 'doubled',
                                            'raise', 'change n', 'longer',
                                            '增加', '增大', '加倍', '翻倍',
                                            '延长'))
    ratio_choice = 1.0 if (mentions_samples and mentions_grow) else 0.0
    return _result([
        _criterion('duration', 4, ratio_a, f"ref={gt['duration_ms']} ms"),
        _criterion('Ts', 4, ratio_b, f"ref={gt['Ts_ns']} ns"),
        _criterion('delta_f', 4, ratio_c, f"ref={gt['delta_f_Hz']} Hz"),
        _criterion('halving_N', 4, ratio_n,
                   f"ref={gt['N_for_half_delta_f']} samples"),
        _criterion('halving_choice', 4, ratio_choice,
                   'must increase the number of samples N'),
    ])


def _score_q4(question, response):
    gt = question['ground_truth']
    qid = question['id']
    # (a) asks for a spectral density and (b) for an integrated power: a
    # dBm-labeled value cannot answer (a), a dBm/Hz-labeled value cannot
    # answer (b) (remediation log §10.5)
    ratio_a, _ = score_scalar(part_text(response, qid, 'a'),
                              gt['noise_psd_dBm_Hz'], 'psd', 'dBm/Hz',
                              2.0, 4.0, mode='abs')
    ratio_b, _ = score_scalar(part_text(response, qid, 'b'),
                              gt['noise_1MHz_dBm'], 'dbm', 'dBm',
                              2.0, 4.0, mode='abs')
    white = polarity(part_text(response, qid, 'c'),
                     ('white', 'uniform', 'flat', '白噪声', '均匀', '平坦'),
                     ('colored', 'coloured', 'non-white', '有色噪声',
                      '非白噪声', '不均匀', '不平坦'))
    ratio_c = 1.0 if white is not None and white == bool(gt['is_white_noise']) \
        else 0.0
    changes = polarity(part_text(response, qid, 'd'),
                       ('change', 'changes', 'increase', 'decrease', 'rise',
                        'drop', '变化', '改变', '升高', '降低', '上升', '下降'),
                       ('unchanged', 'same', 'constant', 'unaffected',
                        'remain', 'remains', 'independent', '保持不变',
                        '不变', '相同', '无影响'))
    ratio_d = 1.0 if changes is not None \
        and changes == bool(gt['noise_changes_without_signals']) else 0.0
    return _result([
        _criterion('psd', 6, ratio_a, f"ref={gt['noise_psd_dBm_Hz']} dBm/Hz"),
        _criterion('noise_1mhz', 5, ratio_b, f"ref={gt['noise_1MHz_dBm']} dBm"),
        _criterion('white_noise', 5, ratio_c,
                   f"expected white={gt['is_white_noise']}"),
        _criterion('removal', 4, ratio_d,
                   f"expected changes={gt['noise_changes_without_signals']}"),
    ])


_LABEL_TOKENS = {
    'digital': DIGITAL_TOKENS + tuple(
        token for t in ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM')
        for token in TYPE_TOKENS[t]),
    'FM': ('fm', 'frequency modulation', '调频'),
    'AM-DSB': ('am', 'amplitude modulation', '调幅'),
    'Chirp (LFM)': ('chirp', 'swept', 'lfm', '扫频', '线性调频'),
}


def _expected_label_tokens(signal_type):
    if signal_type in ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'):
        return DIGITAL_TOKENS + TYPE_TOKENS[signal_type]
    return _LABEL_TOKENS.get(signal_type, TYPE_TOKENS.get(signal_type, ()))


def _score_type_per_signal(text, signals):
    """Per-signal type identification. Type tokens cannot anchor identity here
    (the label IS the answer), so bind by frequency; a list with one chunk per
    signal binds by per-chunk frequency/index reference, then position; only
    unstructured prose falls back to global token presence."""
    n = len(signals)
    norm = normalize_text(text)
    expected = [_expected_label_tokens(s['type']) for s in signals]
    anchors = []
    for signal in signals:
        positions = freq_anchor_positions(text, signal)
        anchors.append(min(positions) if positions else None)
    if all(a is not None for a in anchors) and len(set(anchors)) == n:
        order = sorted(range(n), key=lambda i: anchors[i])
        bounds = [anchors[i] for i in order] + [len(norm)]
        return sum(
            1.0 if has_token(norm[bounds[rank]:bounds[rank + 1]], expected[i])
            else 0.0
            for rank, i in enumerate(order)) / n
    chunks = [c for c in re.split(r'[;,\n；，。]', norm) if c.strip()]
    if len(chunks) == n:
        ascending = sorted(range(n), key=lambda i: signal_center_mhz(signals[i]))
        credited = [False] * n
        for k, chunk in enumerate(chunks):
            freq_hits = [i for i in range(n)
                         if freq_anchor_positions(chunk, signals[i])]
            if freq_hits:
                # a frequency names the signal; the label must match one of
                # the frequency's candidates, never another signal's tokens
                for i in freq_hits:
                    credited[i] = credited[i] or has_token(chunk, expected[i])
                continue
            indexed = index_reference(chunk, n)
            if indexed is not None:
                # models number signals by ascending frequency (Q1's order)
                i = ascending[indexed]
                credited[i] = credited[i] or has_token(chunk, expected[i])
                continue
            token_hits = [i for i in range(n) if has_token(chunk, expected[i])]
            if len(token_hits) == 1:
                # a name-labeled list ("扫频信号: 扫频") binds by its tokens
                credited[token_hits[0]] = True
            elif has_token(chunk, expected[k]):
                credited[k] = True
        return sum(credited) / n
    if n == 1:
        return 1.0 if has_token(norm, expected[0]) else 0.0
    # no binding information: unbound type mentions earn no identity credit
    # (remediation log §8.1.9 — presence-anywhere let cyclic misassignments
    # score full marks)
    return 0.0


# "amplitude varies" rejects constant-envelope; "phase/frequency varies"
# affirms it, and "包络变化7.32%" is a measurement, not a verdict
_ENVELOPE_NEGATOR_PATTERNS = (
    r'(?<![0-9a-z])amplitude\s+(?:vari|fluctuat)',
    r'(?<![0-9a-z])envelope\s+(?:vari|fluctuat)',
    r'幅度(?:变化|波动|起伏)(?!\s*[≈~约<>=0-9.])',
    r'包络(?:变化|波动|起伏)(?!\s*[≈~约<>=0-9.])',
)

_CONSTANT_ENVELOPE_AFFIRMATIONS = (
    r'(?i)(?:does?|will|would|can)?\s*not\s+'
    r'(?:change|vary|modulate|affect)\s+(?:the\s+)?(?:amplitude|envelope)',
    r'(?i)(?:amplitude|envelope)\s+(?:does?|will|would|can)?\s*not\s+'
    r'(?:change|vary)',
    r'不(?:会)?(?:改变|调制|影响)(?:信号的?)?(?:幅度|包络)',
    r'(?:幅度|包络)(?:保持)?(?:恒定|不变)',
)


def _normalize_constant_envelope_affirmations(text):
    """Remove semantic negators from phrases that assert constant envelope.

    In "FM does not change amplitude" / "FM 不改变幅度", the negation
    applies to amplitude variation, not to the FM signal selection.  Generic
    proximity-based negation cannot distinguish that relation on its own.
    """
    normalized = str(text or '')
    for pattern in _CONSTANT_ENVELOPE_AFFIRMATIONS:
        normalized = re.sub(pattern, ' constant-envelope ', normalized)
    return normalized


def _score_constant_envelope(text, gt_entries, signals):
    """Set match of claimed constant-envelope signals (negation- and
    hedge-aware) against the ground-truth set; extra wrong claims cost
    credit (Jaccard)."""
    types = [entry['signal'] for entry in gt_entries]
    digital_count = sum(t in _DIGITAL_TYPES for t in types)
    token_sets = []
    for t in types:
        tokens = TYPE_TOKENS.get(t, (t.lower(),))
        if t in _DIGITAL_TYPES and digital_count == 1:
            tokens = tokens + DIGITAL_TOKENS
        token_sets.append(tokens)
    aligned = (len(signals) == len(types)
               and all(s['type'] == t for s, t in zip(signals, types)))
    true_idx = {i for i, entry in enumerate(gt_entries)
                if entry['constant_envelope']}
    if not true_idx:
        verdict = polarity(text, ('constant',), ('none',))
        return (1.0 if verdict is False else 0.0), 'expected: none'
    claimed = claimed_indices(_normalize_constant_envelope_affirmations(text),
                              token_sets,
                              signals=signals if aligned else None,
                              extra_negator_patterns=_ENVELOPE_NEGATOR_PATTERNS)
    ratio = len(claimed & true_idx) / len(claimed | true_idx)
    return ratio, (f'claimed {sorted(types[i] for i in claimed)} '
                   f'vs true {sorted(types[i] for i in true_idx)}')


def _score_q5(question, response, signals):
    gt = question['ground_truth']
    qid = question['id']
    accepted_categories = gt.get(
        'bandwidth_categories_accepted',
        [[c['category'].split(' ')[0]] for c in
         gt['bandwidth_classifications']])
    accepted_categories = [
        [c.split(' ')[0] for c in accepted] for accepted in accepted_categories
    ]
    ratio_a = _score_category_per_signal(part_text(response, qid, 'a'),
                                         signals, accepted_categories)
    # (b) asks explicitly for the 3 dB bandwidth, so only the 3 dB oracle is
    # accepted (remediation log §8.1.6).  Full credit follows the rubric's
    # declared ±30% window; a 30--50% error receives partial credit.
    bw_3db = gt.get('bandwidths_3dB_kHz', gt['bandwidths_kHz'])
    ratio_b, reason_b = score_per_signal(
        part_text(response, qid, 'b'), signals, bw_3db,
        'freq', 'kHz', 0.30, 0.50)
    text_c = part_text(response, qid, 'c')
    ratio_c, reason_c = _score_constant_envelope(text_c,
                                                 gt['constant_envelope'],
                                                 signals)
    ratio_d = _score_type_per_signal(part_text(response, qid, 'd'), signals)
    return _result([
        _criterion('bw_category', 5, ratio_a,
                   'accepted under 3dB or occupied definition'),
        _criterion('bw_estimate', 6, ratio_b, reason_b),
        _criterion('const_envelope', 5, ratio_c, reason_c),
        _criterion('type_id', 4, ratio_d,
                   'per-signal type identity (freq-anchored)'),
    ])


_SCORERS = {
    'count': _score_q1,
    'power_est': _score_q2,
    'duration': _score_q3,
    'psd': _score_q4,
    'bw_category': _score_q5,
}


def _question_kind(rubric):
    for key in _SCORERS:
        if key in rubric:
            return key
    return None


def score_l1_question(question, response, signals):
    """Return a deterministic score, or None for an unmarked question."""
    if not is_l1_deterministic_question(question):
        return None
    if '===ANSWERS===' not in (response or ''):
        return _result(
            [_criterion('missing_answers', 20, 0, 'no ===ANSWERS=== block')],
            parse_error='missing ===ANSWERS=== block')
    kind = _question_kind(question['rubric'])
    if kind is None:
        raise ValueError(f"unrecognized L1 rubric for {question['id']}")
    scorer = _SCORERS[kind]
    if kind in ('power_est', 'bw_category'):
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
        if kind == 'count':
            strongest = gt.get('strongest_accepted_MHz',
                               [gt['strongest_signal_MHz']])[0]
            weakest = gt.get('weakest_accepted_MHz',
                             [gt['weakest_signal_MHz']])[0]
            lines += [
                f"{qid}a: {gt['signal_count']} signals",
                f"{qid}b: " + ', '.join(
                    f'{f:+.2f} MHz' for f in gt['center_frequencies_MHz']),
                f"{qid}c: Strongest: {strongest:+.2f} MHz; "
                f"Weakest: {weakest:+.2f} MHz",
                f"{qid}d: {gt['power_difference_dB']} dB",
            ]
        elif kind == 'power_est':
            lines += [
                f"{qid}a: " + '; '.join(
                    f"{s['type']}: {p} dBm"
                    for s, p in zip(signals, gt['powers_dBm'])),
                f"{qid}b: " + '; '.join(
                    f"{s['type']}: {p} mW"
                    for s, p in zip(signals, gt['powers_mW'])),
                f"{qid}c: {gt['total_power_dBm']} dBm",
                f"{qid}d: " + '; '.join(
                    f"{s['type']}: {p} dB"
                    for s, p in zip(signals, gt['SNR_per_signal_dB'])),
            ]
        elif kind == 'duration':
            lines += [
                f"{qid}a: {gt['duration_ms']} ms",
                f"{qid}b: {gt['Ts_ns']} ns",
                f"{qid}c: {gt['delta_f_Hz']} Hz",
                f"{qid}d: {gt['N_for_half_delta_f']} samples; increase N "
                f"(the number of samples), keep fs unchanged",
            ]
        elif kind == 'psd':
            lines += [
                f"{qid}a: {gt['noise_psd_dBm_Hz']} dBm/Hz",
                f"{qid}b: {gt['noise_1MHz_dBm']} dBm",
                f"{qid}c: Yes, the PSD is flat — white noise",
                f"{qid}d: No, the noise floor would remain the same",
            ]
        elif kind == 'bw_category':
            accepted = gt.get('bandwidth_categories_accepted')
            categories = ([a[0] for a in accepted] if accepted else
                          [c['category'] for c in
                           gt['bandwidth_classifications']])
            bw_3db = gt.get('bandwidths_3dB_kHz', gt['bandwidths_kHz'])
            constant = [e['signal'] for e in gt['constant_envelope']
                        if e['constant_envelope']]
            lines += [
                f"{qid}a: " + '; '.join(
                    f"{s['type']}: {c.split(' ')[0]}"
                    for s, c in zip(signals, categories)),
                f"{qid}b: " + '; '.join(
                    f"{s['type']}: {b} kHz"
                    for s, b in zip(signals, bw_3db)),
                f"{qid}c: " + (', '.join(constant) + ' are constant-envelope'
                               if constant else
                               'None of the signals are constant-envelope'),
                f"{qid}d: " + '; '.join(
                    f"{signal_center_mhz(s):+.2f} MHz: {label['label']}"
                    for s, label in zip(signals, gt['signal_types'])),
            ]
    lines.append('===END===')
    return '\n'.join(lines)
