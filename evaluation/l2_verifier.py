"""Deterministic scoring for all five EMRB L2 questions.

Q3 (schema emrb-l2-autocorr-v1) requests one JSON object with five
machine-checkable fields. Q1/Q2/Q4/Q5 score the labeled-line prose format:
every criterion binds one requested output of one sub-question to one
ground-truth value or to an explicit set of answer components — values are
read only from their own sub-answer, per-signal values require an identity
anchor (type token, identifying frequency, or listing-order index reference;
never a whole-answer number bag), and qualitative credit requires the
specific comparisons the question asks for, not keyword fragments
(remediation log §10.3/§10.4/§10.8). No LLM judge is involved.
"""
import json
import math
import re

from evaluation.l4_verifier import _answer_json, _criterion, _number

SCORER_VERSION = 'l2-deterministic-v4'
ANSWER_SCHEMA = 'emrb-l2-autocorr-v1'

_BOOL_TOKENS = {'true': True, 'yes': True, 'false': False, 'no': False}


def is_l2_deterministic_question(question):
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


def _ratio_abs(value, reference, full, partial):
    value = _number(value)
    if value is None or reference is None:
        return 0.0
    error = abs(value - reference)
    if error <= full + 1e-12:
        return 1.0
    if error <= partial + 1e-12:
        return 0.5
    return 0.0


def _ratio_rel(value, reference, full=0.05, partial=0.15):
    value = _number(value)
    if value is None or reference is None or abs(reference) < 1e-12:
        return 0.0
    error = abs(value - reference) / abs(reference)
    if error <= full + 1e-12:
        return 1.0
    if error <= partial + 1e-12:
        return 0.5
    return 0.0


def _boolean(value):
    """Strict boolean: native JSON booleans or exact yes/no/true/false tokens."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _BOOL_TOKENS.get(value.strip().lower())
    return None


def validate_q3_payload_structure(payload):
    """Return whether an L2 Q3 payload is complete and type-valid.

    This deliberately checks answer structure rather than correctness. It is
    used by ReconPilot when choosing between successive final-answer blocks,
    so an all-null rewrite cannot replace an earlier scoreable answer.
    """
    if not isinstance(payload, dict):
        return False
    numeric_fields = (
        'max_R_magnitude',
        'comb_spacing_us',
        'modulating_freq_kHz',
    )
    if any(_number(payload.get(field)) is None for field in numeric_fields):
        return False
    if not isinstance(payload.get('source_signal'), str) \
            or not payload['source_signal'].strip():
        return False
    if _boolean(payload.get('comb_persists_after_filtering')) is None:
        return False
    return (isinstance(payload.get('explanation'), str)
            and bool(payload['explanation'].strip()))


def _modulation_family(value):
    """Map an answer string to 'FM' or 'AM'; None if absent or contradictory."""
    text = re.sub(r'[^A-Z]', '', str(value or '').upper())
    has_fm = 'FM' in text
    has_am = 'AM' in text
    if has_fm and not has_am:
        return 'FM'
    if has_am and not has_fm:
        return 'AM'
    return None


def score_l2_question(question, response, signals=None):
    """Return a deterministic score, or None for a non-deterministic L2 question.

    ``signals`` (generation_params['signals'], listing order) enables
    frequency-anchored identity binding in Q4/Q5; type-token and index
    anchors work without it."""
    if not is_l2_deterministic_question(question):
        return None
    rubric = question.get('rubric', {})
    for key, scorer in _PROSE_SCORERS.items():
        if key in rubric:
            if '===ANSWERS===' not in (response or ''):
                return _result(
                    [_criterion('missing_answers', 20, 0,
                                'no ===ANSWERS=== block')],
                    parse_error='missing ===ANSWERS=== block')
            return scorer(question, response, signals)
    return _score_q3(question, response)


def _score_q3(question, response):
    try:
        payload = _answer_json(response, question['id'])
    except ValueError as exc:
        return _result(
            [_criterion('invalid_answer', 20, 0, 'valid question-level JSON')],
            parse_error=str(exc),
        )

    gt = question['ground_truth']
    gt_max_r = _number(gt['max_R_magnitude'])
    gt_spacing = _number(gt['comb_spacing_us'])
    gt_fmod = _number(gt['modulating_freq_kHz'])
    gt_family = _modulation_family(gt['source_signal'])
    gt_persists = _boolean(gt['comb_persists_after_filtering'])
    if gt_persists is None:
        raise ValueError(
            f"{question['id']}: malformed ground truth boolean "
            f"{gt['comb_persists_after_filtering']!r}"
        )

    family = _modulation_family(payload.get('source_signal'))
    persists = _boolean(payload.get('comb_persists_after_filtering'))

    criteria = [
        _criterion('max_R_magnitude', 4,
                   _ratio_abs(payload.get('max_R_magnitude'), gt_max_r, 0.08, 0.16),
                   f'expected ≈ {gt_max_r}'),
        _criterion('comb_spacing_us', 6,
                   _ratio_rel(payload.get('comb_spacing_us'), gt_spacing),
                   f'expected ≈ {gt_spacing} μs'),
        _criterion('source_signal', 3,
                   1.0 if family is not None and family == gt_family else 0.0,
                   f'expected {gt_family}'),
        _criterion('modulating_freq_kHz', 3,
                   _ratio_rel(payload.get('modulating_freq_kHz'), gt_fmod),
                   f'expected ≈ {gt_fmod} kHz'),
        _criterion('comb_persists_after_filtering', 4,
                   1.0 if persists is not None and persists == gt_persists else 0.0,
                   f'expected {gt_persists}'),
    ]
    return _result(criteria)


# --- strict prose scorers for Q1/Q2/Q4/Q5 (labeled Q<n><letter>: lines) ----


_SIDELOBE_TOKENS = ('sidelobe', 'sidelobes', 'side-lobe', 'side-lobes',
                    'side lobe', 'side lobes', '旁瓣')
_LOWER_TOKENS = ('lower', 'low', 'much lower', '-42', 'smaller', 'reduce',
                 'reduces', 'reduced', 'reducing', 'reduction', 'suppress',
                 'suppresses', 'suppressed', 'suppressing', 'suppression',
                 'attenuate', 'attenuates', 'attenuated', 'attenuation',
                 'drop', 'drops', 'fall', 'falls', 'better', '低', '小',
                 '抑制', '下降')
_MAINLOBE_TOKENS = ('main lobe', 'mainlobe', 'main-lobe', '主瓣')
_WIDER_TOKENS = ('wider', 'wide', 'widen', 'widens', 'widened', 'broader',
                 'broad', 'broaden', 'broadens', 'broadened', '1.81',
                 'increase', 'increases', 'increased', 'coarser', 'worse',
                 'degrade', 'degrades', 'degraded', '宽', '增', '变差')
_LEAKAGE_TOKENS = ('leakage', 'leak', 'leaks', '泄漏', 'mask', 'masks',
                   'masked', 'masking', '淹没', '遮蔽', 'dynamic range',
                   '动态范围', 'weak', 'weaker', '弱', 'swamp', 'swamps',
                   'bury', 'buried', 'obscure', 'obscures', 'obscured')
_LEAKAGE_REDUCTION_TOKENS = (
    'reduce', 'reduces', 'reduced', 'reducing', 'suppress', 'suppresses',
    'suppressed', 'suppressing', 'attenuate', 'attenuates', 'attenuated',
    'minimize', 'minimizes', 'prevent', 'prevents', 'avoid', 'avoids',
    'protect', 'protects', 'lower', 'less', '抑制', '降低', '减少', '避免',
)
_LEAKAGE_INCREASE_TOKENS = (
    'higher', 'increase', 'increases', 'increased', 'more', 'worse',
    'amplify', 'amplifies', '高', '增加', '更差',
)


def _affirmative_relational_text(text):
    """Remove negation syntax that is affirmative as a complete idiom."""
    text = re.sub(r'\bnot\s+only\b', 'also', str(text or ''),
                  flags=re.IGNORECASE)
    return text.replace('不仅', '同时')


def _asserted_positions(text, tokens):
    from evaluation.answer_parsing import _marker_hits, _token_positions
    positions = []
    for token in tokens:
        token_positions = _token_positions(text, (token,))
        assertions = _marker_hits(text, (token,))
        positions.extend(
            position for position, asserted in zip(
                token_positions, assertions)
            if asserted
        )
    return positions


def _paired_relation(text, left_tokens, right_tokens, *,
                     max_distance=64, invalid_tokens=()):
    """Whether two concepts are affirmatively linked within one clause."""
    from evaluation.answer_parsing import _CLAUSE_RE, has_token, normalize_text
    normalized = normalize_text(_affirmative_relational_text(text))
    for clause in _CLAUSE_RE.split(normalized):
        if invalid_tokens and has_token(clause, invalid_tokens):
            continue
        left = _asserted_positions(clause, left_tokens)
        right = _asserted_positions(clause, right_tokens)
        if any(abs(a - b) <= max_distance for a in left for b in right):
            return True
    return False


def _nearest_direction_relation(text, subject_tokens, desired_tokens,
                                competing_tokens, max_distance=32):
    """Bind a comparison direction to the nearest physical quantity."""
    from evaluation.answer_parsing import _CLAUSE_RE, normalize_text
    normalized = normalize_text(_affirmative_relational_text(text))
    for clause in _CLAUSE_RE.split(normalized):
        subjects = _asserted_positions(clause, subject_tokens)
        desired = _asserted_positions(clause, desired_tokens)
        competing = _asserted_positions(clause, competing_tokens)
        for subject in subjects:
            desired_distance = min(
                (abs(subject - position) for position in desired),
                default=float('inf'),
            )
            competing_distance = min(
                (abs(subject - position) for position in competing),
                default=float('inf'),
            )
            if (desired_distance <= max_distance
                    and desired_distance < competing_distance):
                return True
    return False


def _score_q1_spectral(question, response, signals=None):
    from evaluation.answer_parsing import has_token, polarity, score_scalar
    from evaluation.l1_verifier import part_text
    gt = question['ground_truth']
    qid = question['id']
    ratio_df, _ = score_scalar(part_text(response, qid, 'a'),
                               gt['delta_f_Hz'], 'freq', 'Hz', 0.01, None)
    text_b = _affirmative_relational_text(part_text(response, qid, 'b'))
    picks_hamming = (has_token(text_b, ('hamming', '汉明'))
                     and polarity(text_b, ('hamming', '汉明'),
                                  ()) is not False)
    recommends_rect = polarity(
        text_b, ('rectangular is more suitable', 'rectangular is better',
                 '矩形窗更合适'), ()) is True
    ratio_choice = 1.0 if picks_hamming and not recommends_rect else 0.0
    # the question asks for the reasoning, not a keyword: credit the three
    # independently checkable components of the tradeoff, and only when they
    # are asserted rather than negated (§10.8, §12.4)
    comp_side = _nearest_direction_relation(
        text_b, _SIDELOBE_TOKENS, _LOWER_TOKENS, _WIDER_TOKENS)
    comp_main = _nearest_direction_relation(
        text_b, _MAINLOBE_TOKENS, _WIDER_TOKENS, _LOWER_TOKENS)
    comp_leak = _nearest_direction_relation(
        text_b, _LEAKAGE_TOKENS, _LEAKAGE_REDUCTION_TOKENS,
        _LEAKAGE_INCREASE_TOKENS, max_distance=48)
    ratio_reason = (comp_side + comp_main + comp_leak) / 3.0
    ratio_t, _ = score_scalar(part_text(response, qid, 'c'),
                              gt['min_observation_time_us'], 'time', 'us',
                              0.10, 0.20)
    text_d = part_text(response, qid, 'd')
    ratio_l, _ = score_scalar(text_d, gt['welch_segment_length'],
                              'count', None, 0.10, 0.20)
    ratio_var, _ = score_scalar(text_d, gt['variance_reduction_dB'],
                                'db', 'dB', 0.5, 1.0, mode='abs')
    return _result([
        _criterion('delta_f', 4, ratio_df, f"ref={gt['delta_f_Hz']} Hz"),
        _criterion('window_choice', 3, ratio_choice, 'must pick Hamming'),
        _criterion('window_reason', 3, ratio_reason,
                   'components: lower sidelobes / wider main lobe tradeoff '
                   '/ leakage consequence'),
        _criterion('min_obs_time', 4, ratio_t,
                   f"ref={gt['min_observation_time_us']} us"),
        _criterion('welch_segment', 3, ratio_l,
                   f"ref={gt['welch_segment_length']} samples"),
        _criterion('variance_reduction', 3, ratio_var,
                   f"ref={gt['variance_reduction_dB']} dB"),
    ])


_NULL_NAME_TOKENS = ('null-to-null', 'null to null', '零点到零点')
_NULL_BOUNDARY_TOKENS = ('spectral null', 'spectral nulls', 'first null',
                         'first nulls', 'first zero', 'first zeros',
                         'between nulls', 'between the nulls', '零点', '过零')
_RELATION_REJECTION_TOKENS = (
    'unrelated', 'independent', 'excludes', 'exclude', 'avoids', 'avoid',
    'without', 'does not contain', 'not related', '无关', '不包含', '避开',
)
_PLANNING_CHOICE_TOKENS = (
    'use', 'uses', 'choose', 'chooses', 'prefer', 'preferred',
    'appropriate', 'suitable', 'planning', 'allocate', 'allocation',
    '采用', '选择', '适合', '规划', '分配',
)
_MITIGATION_TOKENS = (
    'limit', 'limits', 'reduce', 'reduces', 'avoid', 'avoids', 'protect',
    'protects', 'guard', 'capture', 'captures', 'contain', 'contains',
    'include', 'includes', 'account', 'accounts', '限制', '减少', '避免',
    '保护', '包含', '覆盖',
)
_PLANNING_EFFECT_TOKENS = (
    'adjacent', 'interference', 'interfere', 'interferes', 'energy',
    'power', 'roll-off', 'rolloff', 'skirt', 'skirts', 'leakage',
    '邻道', '干扰', '能量', '功率', '滚降', '泄漏',
)


def _score_q2_bandwidths(question, response, signals=None):
    from evaluation.answer_parsing import score_scalar
    from evaluation.l1_verifier import part_text
    gt = question['ground_truth']
    qid = question['id']
    ratio_3db, _ = score_scalar(part_text(response, qid, 'a'),
                                gt['bw_3dB_kHz'], 'freq', 'kHz', 0.15, 0.30)
    ratio_99, _ = score_scalar(part_text(response, qid, 'b'),
                               gt['bw_99_kHz'], 'freq', 'kHz', 0.15, 0.30)
    ratio_null, _ = score_scalar(part_text(response, qid, 'c'),
                                 gt['bw_null_kHz'], 'freq', 'kHz', 0.15, 0.30)
    text_d = _affirmative_relational_text(part_text(response, qid, 'd'))
    picks_99 = _paired_relation(
        text_d, ('99%', '99 %', 'occupied', '占用'),
        _PLANNING_CHOICE_TOKENS, max_distance=56,
        invalid_tokens=('do not use', 'not appropriate', '不采用', '不适合'))
    # "why do they differ": credit each definition concept actually
    # linked — asserted vocabulary alone is insufficient (§14.3)
    concept_thresh = _paired_relation(
        text_d, ('3db', '3 db', '-3db', '-3 db'),
        ('half power', 'half-power', 'half the power', 'drops by half',
         '半功率'),
        invalid_tokens=_RELATION_REJECTION_TOKENS)
    concept_energy = _paired_relation(
        text_d, ('99%', '99 %', '99 percent'),
        ('energy', 'power', '能量', '功率'),
        invalid_tokens=_RELATION_REJECTION_TOKENS)
    concept_null = _paired_relation(
        text_d, _NULL_NAME_TOKENS, _NULL_BOUNDARY_TOKENS,
        invalid_tokens=_RELATION_REJECTION_TOKENS)
    ratio_defs = (concept_thresh + concept_energy + concept_null) / 3.0
    ratio_reason = 1.0 if (
        _paired_relation(
            text_d, _PLANNING_EFFECT_TOKENS, _MITIGATION_TOKENS,
            max_distance=56,
            invalid_tokens=('does not affect', 'is irrelevant', '无影响'))
    ) else 0.0
    return _result([
        _criterion('bw_3dB', 5, ratio_3db, f"ref={gt['bw_3dB_kHz']} kHz"),
        _criterion('bw_99', 5, ratio_99, f"ref={gt['bw_99_kHz']} kHz"),
        _criterion('bw_null', 5, ratio_null, f"ref={gt['bw_null_kHz']} kHz"),
        _criterion('planning_choice', 2, 1.0 if picks_99 else 0.0,
                   'must pick 99%/occupied bandwidth'),
        _criterion('definitions', 2, ratio_defs,
                   'must distinguish threshold / enclosed-energy / '
                   'spectral-null definitions'),
        _criterion('planning_reason', 1, ratio_reason,
                   'must justify via energy capture/adjacent interference'),
    ])


_DIAGONAL_TOKENS = ('diagonal', 'slant', 'slanted', 'slope', 'sloped',
                    'sloping', 'sweep', 'sweeps', 'sweeping', 'swept',
                    'ramp', 'tilted', 'oblique', 'linearly increasing',
                    'linearly decreasing', 'linearly rising',
                    'linearly falling', '斜', '对角')
_WIDEBAND_TOKENS = ('wideband', 'wide band', 'wide', 'broad', 'broadband',
                    'block', '宽')
_NARROWBAND_TOKENS = ('narrowband', 'narrow band', 'narrow', 'thin',
                      '窄', '细')
_BURST_RANGE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:ms|毫秒)?\s*(?:to|~|–|—|-|and|到|至)\s*'
    r'(\d+(?:\.\d+)?)\s*(?:ms|毫秒)')


def _clause_bound(text, id_tokens, feature_tokens, id_signals=()):
    """True when one clause both identifies the signal (type/class token or
    identifying frequency) and states the requested feature — a feature word
    floating anywhere in the answer is not a per-signal description."""
    from evaluation.answer_parsing import (_CLAUSE_RE, asserted_token,
                                           freq_anchor_positions, has_token,
                                           normalize_text)
    for clause in _CLAUSE_RE.split(normalize_text(text)):
        identified = has_token(clause, id_tokens) or any(
            freq_anchor_positions(clause, s) for s in id_signals)
        if identified and asserted_token(clause, feature_tokens):
            return True
    return False


def _q4_identity_sets(signals):
    """(chirp, digital, analog) identity token sets + freq-anchor signals."""
    from evaluation.answer_parsing import (DIGITAL_TOKENS, TYPE_TOKENS,
                                           _DIGITAL_TYPES)
    signals = signals or []
    digital = [s for s in signals if s['type'] in _DIGITAL_TYPES]
    analog = [s for s in signals if s['type'] in ('FM', 'AM-DSB')]
    chirp = [s for s in signals if s['type'] == 'Chirp (LFM)']
    digital_types = [s['type'] for s in digital] or list(_DIGITAL_TYPES)
    analog_types = [s['type'] for s in analog] or ['FM', 'AM-DSB']
    chirp_ids = TYPE_TOKENS['Chirp (LFM)']
    digital_ids = DIGITAL_TOKENS + tuple(
        tok for t in digital_types for tok in TYPE_TOKENS[t])
    analog_ids = ('analog', '模拟') + tuple(
        tok for t in analog_types for tok in TYPE_TOKENS[t])
    return (chirp_ids, chirp), (digital_ids, digital), (analog_ids, analog)


def _score_q4_stft(question, response, signals=None):
    from evaluation.answer_parsing import (has_token, normalize_text,
                                           polarity, score_scalar,
                                           score_scalar_any)
    from evaluation.l1_verifier import _keyword_segment, part_text
    gt = question['ground_truth']
    qid = question['id']
    # (a) morphology must be bound to the signal it describes (§10.8): the
    # chirp is the diagonal trace, the digital signal the wideband stripe,
    # the analog signal the narrowband line
    text_a = part_text(response, qid, 'a')
    (chirp_ids, chirp_sigs), (dig_ids, dig_sigs), (ana_ids, ana_sigs) = \
        _q4_identity_sets(signals)
    ratio_diag = 1.0 if _clause_bound(text_a, chirp_ids, _DIAGONAL_TOKENS,
                                      chirp_sigs) else 0.0
    ratio_wide = 1.0 if _clause_bound(text_a, dig_ids, _WIDEBAND_TOKENS,
                                      dig_sigs) else 0.0
    ratio_narrow = 1.0 if _clause_bound(text_a, ana_ids, _NARROWBAND_TOKENS,
                                        ana_sigs) else 0.0
    rate = gt['chirp_sweep_rate_MHz_per_ms']
    text_b = part_text(response, qid, 'b')
    if isinstance(rate, (int, float)):
        ratio_rate, _ = score_scalar_any(text_b, [rate, -rate],
                                         'sweeprate', 'MHz/ms', 0.10, 0.25)
    else:
        ratio_rate = 1.0 if polarity(text_b, ('no chirp', '没有chirp'),
                                     ()) is True else 0.0
    # (c) requires an explicit burst / no-burst verdict; for burst scenes the
    # start and end must be keyword-bound or given as an ordered "A to B ms"
    # pair — never fished out of the whole sub-answer (§10.4)
    text_c = part_text(response, qid, 'c')
    if gt['has_burst']:
        # "the other signals are continuous" must not poison an affirmative
        # burst verdict, so only explicit no-burst phrases vote against
        says_burst = polarity(text_c, ('burst', '突发'),
                              ('no burst', '没有突发', '无突发'))
        if says_burst is not True:
            ratio_burst = 0.0
        else:
            start_seg = _keyword_segment(
                text_c, ('start', 'begin', 'onset', '开始', '起始'),
                ('end', 'stop', 'until', '结束', '终止'))
            end_seg = _keyword_segment(
                text_c, ('end', 'stop', 'until', '结束', '终止'), ())
            pair = _BURST_RANGE_RE.search(normalize_text(text_c))
            if not start_seg and pair:
                start_seg = pair.group(1) + ' ms'
            if not end_seg and pair:
                end_seg = pair.group(2) + ' ms'
            ratio_start, _ = score_scalar(start_seg, gt['burst_start_ms'],
                                          'time', 'ms', 0.1, 0.2, mode='abs')
            ratio_end, _ = score_scalar(end_seg, gt['burst_end_ms'],
                                        'time', 'ms', 0.1, 0.2, mode='abs')
            # verdict 2 + start 2 + end 2 of the 6 burst points
            ratio_burst = (2.0 + 2.0 * ratio_start + 2.0 * ratio_end) / 6.0
    else:
        says_burst = polarity(text_c, ('burst', '突发'),
                              ('continuous', 'no burst', '连续', '没有突发',
                               '无突发'))
        ratio_burst = 1.0 if says_burst is False else 0.0
    text_d = part_text(response, qid, 'd')
    verdict = polarity(text_d,
                       ('possible', 'achievable', 'can be achieved', '可以'),
                       ('impossible', 'not possible', 'cannot', 'violates',
                        '不可能', '无法'))
    ratio_verdict = 1.0 if verdict is False else 0.0
    ratio_product, _ = score_scalar(text_d, gt['uncertainty_product'],
                                    'none', None, 0.01, None)
    return _result([
        _criterion('features_chirp', 2, ratio_diag,
                   'chirp identified as the diagonal trace'),
        _criterion('features_digital', 1, ratio_wide,
                   'digital signal identified as the wideband stripe'),
        _criterion('features_analog', 1, ratio_narrow,
                   'analog signal identified as the narrowband line'),
        _criterion('chirp_rate', 6, ratio_rate, f'ref={rate} MHz/ms'),
        _criterion('burst_timing', 6, ratio_burst,
                   f"burst={gt['has_burst']} "
                   f"[{gt['burst_start_ms']},{gt['burst_end_ms']}] ms; "
                   'explicit verdict required, times must be bound'),
        _criterion('uncertainty_verdict', 2, ratio_verdict,
                   'must say impossible'),
        _criterion('uncertainty_product', 2, ratio_product,
                   'must compute Δt·Δf = 0.1'),
    ])


_LOG_CONCEPT_TOKENS = ('logarithm', 'logarithmic', 'log', '对数',
                       'nonlinear', 'non-linear', '非线性')
_DBM_TOKENS = ('dbm',)
_CONVERT_TOKENS = ('convert', 'converts', 'converted', 'converting',
                   'conversion', 'express', 'expressed', 'transform',
                   '换算', '转换')
_LINEAR_POWER_TOKENS = ('linear', 'linear domain', 'linear power', 'mw',
                        'watts', 'watt', '线性', '毫瓦', '瓦特')
_SUM_TOKENS = ('sum', 'sums', 'summed', 'summing', 'add', 'adds', 'added',
               'adding', 'total', '相加', '求和', '加和', '叠加', '总')


def _score_q5_energy(question, response, signals=None):
    from evaluation.answer_parsing import score_scalar
    from evaluation.l1_verifier import part_text
    gt = question['ground_truth']
    qid = question['id']
    # (a) asks for BOTH the power (mW) and the energy (J) of EVERY signal;
    # each value is read only from its own signal's identity-anchored
    # segment — an unanchored signal earns nothing (§10.3)
    text_a = part_text(response, qid, 'a')
    entries = gt['energy_per_signal']
    segments = _entry_segments(text_a, entries, signals)
    per_signal = []
    for entry, segment in zip(entries, segments):
        if not segment:
            per_signal.append(0.0)
            continue
        ratio_p, _ = score_scalar(segment, float(entry['power_mW']),
                                  'power_lin', 'mW', 1.0, 3.0, mode='factor')
        ratio_e, _ = score_scalar(segment, float(entry['energy_J']),
                                  'energy', 'J', 1.0, 9.0, mode='factor')
        per_signal.append((ratio_p + ratio_e) / 2)
    ratio_pe = sum(per_signal) / len(per_signal)
    text_b = part_text(response, qid, 'b')
    ratio_eb_db, _ = score_scalar(text_b, gt['digital_signal']['Eb_dBJ'],
                                  'dbj', 'dBJ', 1.0, 3.0, mode='abs')
    ratio_eb_lin, _ = score_scalar(text_b, float(gt['digital_signal']['Eb_J']),
                                   'energy', 'J', 1.0, 3.0, mode='factor')
    text_c = part_text(response, qid, 'c')
    ratio_total, _ = score_scalar(text_c, gt['total_received_power_dBm'],
                                  'dbm', 'dBm', 1.0, 2.0, mode='abs')
    # (d) asks to derive AND verify: credit the log-domain concept, the
    # convert-sum-convert-back relation, and a numeric verification against
    # this problem's totals — 'log' + 'linear' alone is not the analysis
    text_d = part_text(response, qid, 'd')
    comp_log = 1.0 if _paired_relation(
        text_d, _DBM_TOKENS, _LOG_CONCEPT_TOKENS,
        invalid_tokens=('not logarithmic', 'is linear', '不是对数')) else 0.0
    comp_rel = 1.0 if (
        _paired_relation(text_d, _CONVERT_TOKENS, _LINEAR_POWER_TOKENS)
        and _paired_relation(text_d, _SUM_TOKENS, _LINEAR_POWER_TOKENS)
    ) else 0.0
    verify_refs = [
        score_scalar(text_d, gt['total_received_power_dBm'],
                     'dbm', 'dBm', 1.0, 2.0, mode='abs')[0],
        score_scalar(text_d, float(gt['total_received_power_mW']),
                     'power_lin', 'mW', 0.26, 0.60, mode='rel')[0],
        score_scalar(text_d, float(gt['total_signal_power_mW']),
                     'power_lin', 'mW', 0.26, 0.60, mode='rel')[0],
    ]
    ratio_expl = (comp_log + comp_rel + max(verify_refs)) / 3.0
    return _result([
        _criterion('power_energy', 5, ratio_pe,
                   'per-signal P in mW and E = P·T in J, identity-bound'),
        _criterion('Eb', 5, max(ratio_eb_db, ratio_eb_lin),
                   f"ref={gt['digital_signal']['Eb_dBJ']} dBJ"),
        _criterion('total_power', 5, ratio_total,
                   f"ref={gt['total_received_power_dBm']} dBm"),
        _criterion('explanation', 5, ratio_expl,
                   'components: log-domain concept / convert-sum-convert '
                   'relation / numeric verification'),
    ])


def _entry_segments(text, entries, signals=None):
    """Identity-anchored answer segment per Q5 signal entry, or None.

    Anchors: the signal's type token, an identifying frequency (when the
    listing-ordered ``signals`` metadata is supplied), or a 'Signal k'
    reference in listing order. There is deliberately no whole-answer
    fallback — an unlabeled bag of numbers matches no signal (§10.3)."""
    from evaluation.answer_parsing import (TYPE_TOKENS, _INDEX_RE,
                                           _token_positions,
                                           freq_anchor_positions,
                                           normalize_text)
    norm = normalize_text(text)
    n = len(entries)
    anchors = []
    for i, entry in enumerate(entries):
        positions = _token_positions(
            norm, TYPE_TOKENS.get(entry['type'], (entry['type'].lower(),)))
        if (signals and i < len(signals)
                and signals[i].get('type') == entry['type']):
            positions += freq_anchor_positions(norm, signals[i])
        for m in _INDEX_RE.finditer(norm):
            if int(m.group(1) or m.group(2)) - 1 == i:
                positions.append(m.start())
        anchors.append(min(positions) if positions else None)
    starts = sorted(p for p in anchors if p is not None)
    segments = []
    for anchor in anchors:
        if anchor is None:
            segments.append(None)
            continue
        later = [p for p in starts if p > anchor]
        segments.append(norm[anchor:later[0] if later else len(norm)])
    return segments


_PROSE_SCORERS = {
    'delta_f': _score_q1_spectral,
    'bw_3dB': _score_q2_bandwidths,
    'features': _score_q4_stft,
    'energy': _score_q5_energy,
}


def reference_payload(question):
    """Build the canonical machine-readable answer for the Q3 question."""
    gt = question['ground_truth']
    return {
        'max_R_magnitude': gt['max_R_magnitude'],
        'comb_spacing_us': gt['comb_spacing_us'],
        'source_signal': gt['source_signal'],
        'modulating_freq_kHz': gt['modulating_freq_kHz'],
        'comb_persists_after_filtering': gt['comb_persists_after_filtering'],
        'explanation': 'reference answer',
    }


def format_reference_response(question):
    """Render the canonical answer as a full ===ANSWERS=== response block."""
    return (f"===ANSWERS===\n"
            f"{question['id']}: {json.dumps(reference_payload(question))}\n"
            f"===END===")


def reference_response(meta):
    """Render every question's canonical answer as one response block."""
    from evaluation.answer_parsing import _DIGITAL_TYPES
    signals = meta.get('generation_params', {}).get('signals', [])
    digital_name = next(
        (s['type'] for s in signals if s['type'] in _DIGITAL_TYPES),
        'digital')
    analog_name = next(
        (s['type'] for s in signals if s['type'] in ('FM', 'AM-DSB')), 'FM')
    lines = ['===ANSWERS===']
    for question in meta['questions']:
        rubric = question.get('rubric', {})
        gt = question['ground_truth']
        qid = question['id']
        if 'delta_f' in rubric:
            lines += [
                f"{qid}a: {gt['delta_f_Hz']} Hz",
                f"{qid}b: Hamming window is more suitable: its sidelobes "
                f"are much lower (-42 dB vs -13 dB), which suppresses "
                f"spectral leakage that would otherwise mask weak signals; "
                f"the tradeoff is a wider main lobe (1.81×Δf)",
                f"{qid}c: {gt['min_observation_time_us']} us",
                f"{qid}d: {gt['welch_segment_length']} samples per segment; "
                f"variance reduced by {gt['variance_reduction_dB']} dB",
            ]
        elif 'bw_3dB' in rubric:
            lines += [
                f"{qid}a: {gt['bw_3dB_kHz']} kHz",
                f"{qid}b: {gt['bw_99_kHz']} kHz",
                f"{qid}c: {gt['bw_null_kHz']} kHz",
                f"{qid}d: the definitions measure different things: the "
                f"3 dB bandwidth spans the half-power points, the "
                f"null-to-null bandwidth spans the main lobe between the "
                f"first spectral nulls, and the 99% bandwidth contains 99% "
                f"of the signal energy including the roll-off skirts. For "
                f"frequency planning the 99% occupied bandwidth is most "
                f"appropriate: it captures nearly all energy and limits "
                f"adjacent-channel interference",
            ]
        elif 'features' in rubric:
            rate = gt['chirp_sweep_rate_MHz_per_ms']
            line_b = (f"{qid}b: sweep rate = {rate} MHz/ms"
                      if isinstance(rate, (int, float))
                      else f"{qid}b: no chirp signal present")
            if gt['has_burst']:
                line_c = (f"{qid}c: burst present, start = "
                          f"{gt['burst_start_ms']} ms, end = "
                          f"{gt['burst_end_ms']} ms")
            else:
                line_c = (f"{qid}c: no burst; all signals are continuous "
                          f"along the time axis")
            lines += [
                f"{qid}a: the chirp appears as a diagonal sweeping line; "
                f"the {digital_name} digital signal appears as a wideband "
                f"horizontal stripe; the {analog_name} signal appears as a "
                f"narrowband horizontal line at a fixed frequency",
                line_b,
                line_c,
                f"{qid}d: impossible: Δt×Δf = "
                f"{gt['uncertainty_product']} < 1 violates the uncertainty "
                f"principle",
            ]
        elif 'energy' in rubric:
            digital = gt['digital_signal']
            lines += [
                f"{qid}a: " + '; '.join(
                    f"{entry['type']}: {entry['power_mW']} mW, "
                    f"{entry['energy_J']} J"
                    for entry in gt['energy_per_signal']),
                f"{qid}b: Eb = {digital['Eb_J']} J = {digital['Eb_dBJ']} dBJ",
                f"{qid}c: {gt['total_received_power_dBm']} dBm",
                f"{qid}d: dBm is logarithmic and cannot be summed directly; "
                f"convert each signal to mW, add the powers in the linear "
                f"domain, then convert back: the total is "
                f"{gt['total_received_power_dBm']} dBm",
            ]
        else:
            lines.append(f"{qid}: {json.dumps(reference_payload(question))}")
    lines.append('===END===')
    return '\n'.join(lines)
