"""Deterministic scoring for the repaired EMRB L4 question instances."""

import json
import math
import re

from evaluation.auto_scorer import parse_answer_block


SCORER_VERSION = 'l4-deterministic-v1'


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _pair(value):
    values = _list(value)
    if len(values) != 2:
        return None
    first, second = _number(values[0]), _number(values[1])
    if first is None or second is None:
        return None
    return [min(first, second), max(first, second)]


def _quality(value, reference, *, full_abs=None, partial_abs=None,
             full_rel=None, partial_rel=None):
    value, reference = _number(value), _number(reference)
    if value is None or reference is None:
        return 0.0
    full = 0.0 if full_abs is None else float(full_abs)
    partial = full if partial_abs is None else float(partial_abs)
    if full_rel is not None:
        full = max(full, abs(reference) * full_rel)
    if partial_rel is not None:
        partial = max(partial, abs(reference) * partial_rel)
    error = abs(value - reference)
    if error <= full + 1e-12:
        return 1.0
    if error <= partial + 1e-12:
        return 0.5
    return 0.0


def _mean(values):
    return sum(values) / len(values) if values else 0.0


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
    score = round(sum(item['score'] for item in criteria), 2)
    result = {
        'total_score': score,
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


def _answer_json(response, question_id):
    answer_text = parse_answer_block(response).get(question_id, '')
    if not answer_text:
        raise ValueError(f'{question_id} JSON answer is missing')
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', answer_text):
        try:
            payload, _ = decoder.raw_decode(answer_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f'{question_id} does not contain a valid JSON object')


def _canonical_type(value):
    text = re.sub(r'[^A-Z0-9]+', '', str(value or '').upper())
    if not text:
        return None
    if 'CHIRP' in text or 'LFM' in text:
        return 'CHIRP'
    if 'OFDM' in text:
        return 'OFDM'
    for modulation in ('64QAM', '16QAM', '8PSK', 'QPSK', 'BPSK', '4FSK', '2FSK'):
        if modulation in text:
            return modulation
    if 'QAM' in text:
        return 'QAM'
    if 'PSK' in text:
        return 'PSK'
    if 'FSK' in text:
        return 'FSK'
    if 'FM' in text:
        return 'FM'
    if 'AM' in text:
        return 'AM'
    return text


def _type_family(value):
    canonical = _canonical_type(value)
    if canonical in ('BPSK', 'QPSK', '8PSK', 'PSK'):
        return 'PSK'
    if canonical in ('16QAM', '64QAM', 'QAM'):
        return 'QAM'
    if canonical in ('2FSK', '4FSK', 'FSK'):
        return 'FSK'
    return canonical


def _type_quality(value, reference):
    candidate = _canonical_type(value)
    expected = _canonical_type(reference)
    if candidate == expected:
        return 1.0
    if candidate and _type_family(candidate) == _type_family(expected):
        return 0.5
    return 0.0


def _relation(value):
    key = re.sub(r'[^a-z]+', '', str(value or '').lower())
    if key in ('overlap', 'overlapping', 'intersecting'):
        return 'overlapping'
    if key in ('separated', 'disjoint', 'nonoverlapping', 'nooverlap'):
        return 'separated'
    return key


def _boolean(value):
    if isinstance(value, bool):
        return value
    key = str(value or '').strip().lower()
    if key in ('true', 'yes'):
        return True
    if key in ('false', 'no'):
        return False
    return None


_GAP_SCHEMA_FIELDS = {
    'raw_gap_bounds_MHz',
    'raw_gap_MHz',
    'usable_gap_bounds_MHz',
    'usable_gap_MHz',
    'symbol_rate_ksps',
    'data_rate_kbps',
    'noise_in_link_band_dBm',
    'minimum_received_power_dBm',
}

_INTERFERENCE_SCHEMA_FIELDS = {
    'pair',
    'center_separation_MHz',
    'spectral_relation',
    'overlap_MHz',
    'guard_gap_MHz',
    'target_type',
    'target_to_other_power_ratio_dB',
    'full_band_isolation_possible',
    'available_transition_bandwidth_MHz',
    'nonoverlapped_target_bandwidth_MHz',
    'overlapped_target_fraction_pct',
}


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _valid_pair_entry(value):
    item = _dict(value)
    return (
        _nonempty_string(item.get('type'))
        and _number(item.get('center_MHz')) is not None
        and _pair(item.get('occupied_interval_MHz')) is not None
    )


def _valid_generic_required_field(name, value):
    """Fallback validation for a generated repaired-L4 field list.

    Current benchmark prompts use one of the two explicit schemas above.
    The fallback keeps synthetic or future numeric field lists strict without
    silently accepting ``null`` solely because a key is present.
    """
    if name == 'pair':
        return (isinstance(value, list) and len(value) == 2
                and all(_valid_pair_entry(item) for item in value))
    if name in ('spectral_relation', 'target_type'):
        return _nonempty_string(value)
    if name == 'full_band_isolation_possible':
        return _boolean(value) is not None
    if 'bounds' in name or 'interval' in name or name.endswith('passband_MHz'):
        return _pair(value) is not None
    return _number(value) is not None


def validate_repaired_payload_structure(payload, required_keys=None):
    """Validate the machine-readable shape of a repaired L4 answer.

    Values are checked for the primitive and nested types demanded by the
    prompt, but are not compared with ground truth.
    """
    if not isinstance(payload, dict):
        return False
    keys = set(payload)
    required = set(required_keys or ())
    if _GAP_SCHEMA_FIELDS.issubset(keys):
        return (
            _pair(payload.get('raw_gap_bounds_MHz')) is not None
            and _pair(payload.get('usable_gap_bounds_MHz')) is not None
            and all(
                _number(payload.get(field)) is not None
                for field in _GAP_SCHEMA_FIELDS
                if field not in ('raw_gap_bounds_MHz',
                                 'usable_gap_bounds_MHz')
            )
        )
    if _INTERFERENCE_SCHEMA_FIELDS.issubset(keys):
        pair = payload.get('pair')
        return (
            isinstance(pair, list)
            and len(pair) == 2
            and all(_valid_pair_entry(item) for item in pair)
            and _nonempty_string(payload.get('spectral_relation'))
            and _nonempty_string(payload.get('target_type'))
            and _boolean(payload.get('full_band_isolation_possible'))
            is not None
            and all(
                _number(payload.get(field)) is not None
                for field in _INTERFERENCE_SCHEMA_FIELDS
                if field not in (
                    'pair',
                    'spectral_relation',
                    'target_type',
                    'full_band_isolation_possible',
                )
            )
        )
    return bool(required) and required.issubset(keys) and all(
        _valid_generic_required_field(name, payload.get(name))
        for name in required
    )


def _score_gap_question(payload, ground_truth):
    raw_bounds = _pair(payload.get('raw_gap_bounds_MHz'))
    usable_bounds = _pair(payload.get('usable_gap_bounds_MHz'))
    expected_raw = sorted(ground_truth['raw_gap_bounds_MHz'])
    expected_usable = sorted(ground_truth['usable_gap_bounds_MHz'])
    raw_bound_quality = 0.0 if raw_bounds is None else _mean([
        _quality(raw_bounds[index], expected_raw[index],
                 full_abs=0.05, partial_abs=0.15)
        for index in range(2)
    ])
    usable_bound_quality = 0.0 if usable_bounds is None else _mean([
        _quality(usable_bounds[index], expected_usable[index],
                 full_abs=0.05, partial_abs=0.15)
        for index in range(2)
    ])
    usable_width = payload.get(
        'usable_gap_MHz', payload.get('available_gap_MHz')
    )
    return _result([
        _criterion('gap_raw_bounds', 2, raw_bound_quality,
                   'largest raw unoccupied interval boundaries'),
        _criterion('gap_raw_bandwidth', 2, _quality(
            payload.get('raw_gap_MHz'), ground_truth['raw_gap_MHz'],
            full_abs=0.05, partial_abs=0.15,
        ), 'raw unoccupied bandwidth'),
        _criterion('gap_usable_bounds', 2, usable_bound_quality,
                   'guarded usable interval boundaries'),
        _criterion('gap_usable_bandwidth', 2, _quality(
            usable_width, ground_truth['available_gap_MHz'],
            full_abs=0.05, partial_abs=0.15,
        ), 'guarded usable bandwidth'),
        _criterion('gap_symbol_rate', 3, _quality(
            payload.get('symbol_rate_ksps'), ground_truth['symbol_rate_ksps'],
            full_rel=0.15, partial_rel=0.30,
        ), 'maximum symbol rate'),
        _criterion('gap_data_rate', 3, _quality(
            payload.get('data_rate_kbps'), ground_truth['data_rate_kbps'],
            full_rel=0.15, partial_rel=0.30,
        ), 'raw data rate'),
        _criterion('gap_noise_power', 3, _quality(
            payload.get('noise_in_link_band_dBm'),
            ground_truth['noise_in_link_band_dBm'],
            full_abs=2.0, partial_abs=4.0,
        ), 'noise power in the deployed-link bandwidth'),
        _criterion('gap_minimum_power', 3, _quality(
            payload.get('minimum_received_power_dBm'),
            ground_truth['minimum_received_power_dBm'],
            full_abs=2.0, partial_abs=4.0,
        ), 'minimum received signal power'),
    ])


def _pair_entry(value):
    item = _dict(value)
    return {
        'type': item.get('type'),
        'center': _number(item.get(
            'center_MHz', item.get('center_frequency_MHz')
        )),
        'interval': _pair(item.get(
            'occupied_interval_MHz', item.get('interval_MHz')
        )),
    }


def _pair_match_quality(candidate_pair, reference_pair):
    candidates = [_pair_entry(item) for item in _list(candidate_pair)[:2]]
    if len(candidates) != 2:
        return 0.0, 0.0, 0.0
    references = [_pair_entry(item) for item in reference_pair]
    best = (0.0, 0.0, 0.0)
    for order in ((0, 1), (1, 0)):
        types, centers, intervals = [], [], []
        for candidate_index, reference_index in enumerate(order):
            candidate = candidates[candidate_index]
            reference = references[reference_index]
            types.append(_type_quality(candidate['type'], reference['type']))
            centers.append(_quality(
                candidate['center'], reference['center'],
                full_abs=0.10, partial_abs=0.25,
            ))
            if candidate['interval'] is None or reference['interval'] is None:
                intervals.append(0.0)
            else:
                intervals.append(_mean([
                    _quality(
                        candidate['interval'][edge], reference['interval'][edge],
                        full_abs=0.10, partial_abs=0.25,
                    )
                    for edge in range(2)
                ]))
        quality = (_mean(types), _mean(centers), _mean(intervals))
        if quality[0] + quality[1] + 2 * quality[2] > (
            best[0] + best[1] + 2 * best[2]
        ):
            best = quality
    return best


def _score_interference_question(payload, ground_truth):
    type_quality, center_quality, interval_quality = _pair_match_quality(
        payload.get('pair'), ground_truth['pair']
    )
    relation = _relation(payload.get('spectral_relation'))
    expected_relation = ground_truth['spectral_relation']
    relation_ok = relation == expected_relation
    if expected_relation == 'overlapping':
        relation_geometry = _quality(
            payload.get('overlap_MHz'), ground_truth['overlap_MHz'],
            full_abs=0.05, partial_abs=0.15,
        )
        isolation_geometry = _mean([
            _quality(
                payload.get('nonoverlapped_target_bandwidth_MHz'),
                ground_truth['nonoverlapped_target_bandwidth_MHz'],
                full_abs=0.05, partial_abs=0.15,
            ),
            _quality(
                payload.get('overlapped_target_fraction_pct'),
                ground_truth['overlapped_target_fraction_pct'],
                full_abs=10.0, partial_abs=20.0,
            ),
        ])
    else:
        relation_geometry = _quality(
            payload.get('guard_gap_MHz'), ground_truth['guard_gap_MHz'],
            full_abs=0.05, partial_abs=0.15,
        )
        isolation_geometry = _quality(
            payload.get('available_transition_bandwidth_MHz'),
            ground_truth['available_transition_bandwidth_MHz'],
            full_abs=0.05, partial_abs=0.15,
        )

    isolation = _boolean(payload.get('full_band_isolation_possible'))
    isolation_ok = isolation is ground_truth['full_band_isolation_possible']
    return _result([
        _criterion('interference_pair_types', 1, type_quality,
                   'critical-pair signal types'),
        _criterion('interference_pair_centers', 1, center_quality,
                   'critical-pair center frequencies'),
        _criterion('interference_pair_intervals', 2, interval_quality,
                   'critical-pair occupied intervals'),
        _criterion('interference_center_separation', 1, _quality(
            payload.get('center_separation_MHz'),
            ground_truth['center_separation_MHz'],
            full_abs=0.10, partial_abs=0.25,
        ), 'critical-pair center separation'),
        _criterion('interference_relation', 1, float(relation_ok),
                   'overlapping or separated classification'),
        _criterion('interference_overlap_or_gap', 4,
                   float(relation_ok) * relation_geometry,
                   'overlap bandwidth or guard gap'),
        _criterion('interference_target', 1, _type_quality(
            payload.get('target_type'), ground_truth['target_type']
        ), 'weaker target signal'),
        _criterion('interference_power_ratio', 3, _quality(
            payload.get('target_to_other_power_ratio_dB'),
            ground_truth['target_to_other_power_ratio_dB'],
            full_abs=2.0, partial_abs=4.0,
        ), 'target-to-other received-power ratio'),
        _criterion('interference_full_isolation', 2, float(isolation_ok),
                   'full-band frequency-selective isolation feasibility'),
        _criterion('interference_isolation_geometry', 4,
                   float(isolation_ok) * isolation_geometry,
                   'transition bandwidth or overlapped-target geometry'),
    ])


def is_repaired_question(question):
    return question.get('rubric', {}).get('scoring') == SCORER_VERSION


def score_repaired_question(question, response):
    """Return a deterministic score, or None for an untouched L4 question."""
    if not is_repaired_question(question):
        return None
    try:
        payload = _answer_json(response, question['id'])
    except ValueError as exc:
        empty = _result([
            _criterion('invalid_answer', 20, 0, 'valid question-level JSON')
        ], parse_error=str(exc))
        return empty

    ground_truth = question['ground_truth']
    if question.get('question_type') == 'QT05' and 'raw_gap_bounds_MHz' in ground_truth:
        return _score_gap_question(payload, ground_truth)
    if question.get('question_type') == 'QT07' and 'spectral_relation' in ground_truth:
        return _score_interference_question(payload, ground_truth)
    raise ValueError(
        f"unsupported repaired L4 question: {question.get('question_type')}"
    )


def reference_payload(question):
    """Build the canonical machine-readable answer for one repaired question."""
    ground_truth = question['ground_truth']
    if question.get('question_type') == 'QT05':
        return {
            'raw_gap_bounds_MHz': ground_truth['raw_gap_bounds_MHz'],
            'raw_gap_MHz': ground_truth['raw_gap_MHz'],
            'usable_gap_bounds_MHz': ground_truth['usable_gap_bounds_MHz'],
            'usable_gap_MHz': ground_truth['available_gap_MHz'],
            'symbol_rate_ksps': ground_truth['symbol_rate_ksps'],
            'data_rate_kbps': ground_truth['data_rate_kbps'],
            'noise_in_link_band_dBm': ground_truth['noise_in_link_band_dBm'],
            'minimum_received_power_dBm': ground_truth['minimum_received_power_dBm'],
        }
    if question.get('question_type') == 'QT07':
        return {
            'pair': [{
                'type': item['type'],
                'center_MHz': item['center_frequency_MHz'],
                'occupied_interval_MHz': item['occupied_interval_MHz'],
            } for item in ground_truth['pair']],
            'center_separation_MHz': ground_truth['center_separation_MHz'],
            'spectral_relation': ground_truth['spectral_relation'],
            'overlap_MHz': ground_truth['overlap_MHz'],
            'guard_gap_MHz': ground_truth['guard_gap_MHz'],
            'target_type': ground_truth['target_type'],
            'target_to_other_power_ratio_dB': ground_truth['target_to_other_power_ratio_dB'],
            'full_band_isolation_possible': ground_truth['full_band_isolation_possible'],
            'available_transition_bandwidth_MHz': ground_truth['available_transition_bandwidth_MHz'],
            'nonoverlapped_target_bandwidth_MHz': ground_truth['nonoverlapped_target_bandwidth_MHz'],
            'overlapped_target_fraction_pct': ground_truth['overlapped_target_fraction_pct'],
        }
    raise ValueError(f"unsupported question type: {question.get('question_type')}")


def format_reference_response(question):
    payload = json.dumps(reference_payload(question), separators=(',', ':'))
    return f"===ANSWERS===\n{question['id']}: {payload}\n===END==="


def reference_line(question):
    """The single labeled JSON line a repaired L4 question is scored from."""
    payload = json.dumps(reference_payload(question), separators=(',', ':'))
    return f"{question['id']}: {payload}"


def reference_response(meta):
    """Render the ground truth as a full labeled answer block (for replay).

    L4 problems mix the two scoring paths ``evaluate.score_problem`` routes
    through: repaired questions (``l4-deterministic-v1``, one JSON payload per
    question) and generic questions (``l4-generic-v3``, labeled prose lines).
    Both are emitted into one block, in the order the problem asks them.
    """
    # imported here: the generic verifier is a peer module and only this
    # replay helper needs it, so the runtime scoring path stays unchanged
    from evaluation.l4_generic_verifier import reference_response_lines

    lines = ['===ANSWERS===']
    for question in meta['questions']:
        if is_repaired_question(question):
            lines.append(reference_line(question))
        else:
            lines += reference_response_lines(question)
    lines.append('===END===')
    return '\n'.join(lines)
