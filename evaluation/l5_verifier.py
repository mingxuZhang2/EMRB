"""Deterministic verifier for the parameterized EMRB L5 tasks."""

import json
import math
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from generate_l5_batch import (
    MASK_TARGET_RETENTION,
    MODULATION_MARGIN_DB,
    MODULATION_OPTIONS,
    OFDM_CP_RATIOS,
    OFDM_GUARD_MHZ,
    OFDM_OOB_ATTENUATION_DB,
    OFDM_SPACINGS_KHZ,
    OFDM_SUBCARRIER_COUNTS,
    PACKING_GUARD_MHZ,
    SCHEMA_VERSION,
    evaluate_extraction_mask,
    occupied_intervals_mhz,
)


SCORER_VERSION = 'l5-deterministic-v6'

# every top-level section the deterministic verifier scores; used by the
# ReconPilot answer selector to require schema-complete candidates (§12.1)
REQUIRED_ANSWER_SECTIONS = ('Q1a', 'Q1b', 'Q1c', 'Q1d',
                            'Q2a', 'Q2b', 'Q2c', 'Q2d',
                            'Q3a', 'Q3b', 'Q3c', 'Q3d')
QUESTION_MAXIMA = {'Q1': 34.0, 'Q2': 33.0, 'Q3': 33.0}


def extract_answer_json(response):
    """Extract the last valid JSON object from an EMRB answer block."""
    text = response or ''
    blocks = re.findall(
        r'===ANSWERS===\s*(.*?)\s*===END===', text, flags=re.DOTALL
    )
    candidates = list(reversed(blocks)) if blocks else [text]
    decoder = json.JSONDecoder()
    errors = []
    for candidate in candidates:
        candidate = re.sub(
            r'^\s*```(?:json)?\s*|\s*```\s*$', '', candidate, flags=re.I
        )
        for match in re.finditer(r'\{', candidate):
            try:
                payload, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                continue
            if isinstance(payload, dict):
                return payload
    detail = errors[-1] if errors else 'no JSON object found'
    raise ValueError(f'invalid L5 answer JSON: {detail}')


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value):
    number = _number(value)
    if number is None or not math.isclose(number, round(number), abs_tol=1e-6):
        return None
    return int(round(number))


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


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _matches_structure(value, specification):
    """Small recursive schema checker used only for answer completeness."""
    if specification == 'number':
        return _number(value) is not None
    if specification == 'integer':
        return _integer(value) is not None
    if specification == 'string':
        return _nonempty_string(value)
    if specification == 'number_pair':
        return _pair(value) is not None
    if isinstance(specification, dict):
        return isinstance(value, dict) and all(
            key in value and _matches_structure(value[key], child)
            for key, child in specification.items()
        )
    if isinstance(specification, tuple) and specification[0] == 'list':
        _, child, minimum, maximum = specification
        if not isinstance(value, list):
            return False
        if len(value) < minimum or (
                maximum is not None and len(value) > maximum):
            return False
        return all(_matches_structure(item, child) for item in value)
    raise ValueError(f'unsupported answer schema node: {specification!r}')


_ANSWER_STRUCTURE = {
    'Q1a': {
        'signals': ('list', {
            'id': 'string',
            'center_MHz': 'number',
            'bandwidth_MHz': 'number',
            'modulation': 'string',
            'power_dBm': 'number',
        }, 1, None),
    },
    'Q1b': {
        'pair_ids': ('list', 'string', 2, 2),
        'target_id': 'string',
        'overlap_MHz': 'number',
        'sir_dB': 'number',
    },
    'Q1c': {
        'target_id': 'string',
        'passband_MHz': 'number_pair',
        'target_retained_fraction': 'number',
        'post_sir_dB': 'number',
        'improvement_dB': 'number',
    },
    'Q1d': {
        'total_occupied_MHz': 'number',
        'occupancy_pct': 'number',
        'additional_channel_count': 'integer',
        'additional_centers_MHz': ('list', 'number', 0, None),
    },
    'Q2a': {
        'signal_id': 'string',
        'sweep_start_MHz': 'number',
        'sweep_end_MHz': 'number',
        'bandwidth_MHz': 'number',
        'chirp_rate_MHz_per_ms': 'number',
        'tbp': 'number',
    },
    'Q2b': {
        'victim_id': 'string',
        'center_MHz': 'number',
        'bandwidth_MHz': 'number',
        'modulation': 'string',
        'symbol_rate_kHz': 'number',
        'rolloff': 'number',
        'power_dBm': 'number',
    },
    'Q2c': {
        'entry_time_ms': 'number',
        'exit_time_ms': 'number',
        'duration_ms': 'number',
    },
    'Q2d': {
        'victim_id': 'string',
        'symbols_iq': ('list', 'number_pair', 32, 32),
    },
    'Q3a': {
        'links': ('list', {
            'id': 'string',
            'snr_dB': 'number',
            'capacity_Mbps': 'number',
            'actual_efficiency_bps_per_Hz': 'number',
            'shannon_efficiency_bps_per_Hz': 'number',
        }, 1, None),
    },
    'Q3b': {
        'worst_id': 'string',
        'recommended_modulation': 'string',
        'bits_per_symbol': 'integer',
        'throughput_improvement_pct': 'number',
    },
    'Q3c': {
        'water_level_dBm_per_Hz': 'number',
        'allocations': ('list', {
            'id': 'string',
            'power_dBm': 'number',
        }, 1, None),
        'capacity_before_Mbps': 'number',
        'capacity_after_Mbps': 'number',
        'improvement_pct': 'number',
    },
    'Q3d': {
        'gap_MHz': 'number_pair',
        'center_MHz': 'number',
        'n_subcarriers': 'integer',
        'spacing_kHz': 'number',
        'cp_ratio': 'number',
        'modulation': 'string',
        'power_dBm': 'number',
        'occupied_bandwidth_MHz': 'number',
        'rate_Mbps': 'number',
        'adjacent_leakage_dBm': 'number',
    },
}


def validate_answer_structure(payload):
    """Return whether an L5 payload is complete and type-valid.

    The check is intentionally independent of ground truth. It prevents a
    later empty or partially rewritten JSON object from replacing an earlier
    answer that the deterministic verifier can actually score.
    """
    if not isinstance(payload, dict):
        return False
    if any(
            section not in payload
            or not _matches_structure(payload[section], specification)
            for section, specification in _ANSWER_STRUCTURE.items()):
        return False
    count = _integer(payload['Q1d']['additional_channel_count'])
    centers = payload['Q1d']['additional_centers_MHz']
    return count is not None and count >= 0 and len(centers) == count


def _key(value):
    return re.sub(r'[^a-z0-9]+', '', str(value or '').lower())


def _canonical_type(value):
    text = re.sub(r'[^A-Z0-9]+', '', str(value or '').upper())
    if not text:
        return None
    if 'CHIRP' in text or 'LFM' in text:
        return 'CHIRP'
    for family in ('FSK', 'QAM', 'PSK'):
        if family in text:
            match = re.search(rf'(\d+){family}', text)
            if match:
                return f'{match.group(1)}{family}'
            if family == 'PSK' and 'BPSK' in text:
                return 'BPSK'
            if family == 'PSK' and 'QPSK' in text:
                return 'QPSK'
            return family
    if 'FM' in text:
        return 'FM'
    if 'AM' in text:
        return 'AM'
    if 'OFDM' in text:
        return 'OFDM'
    return text


def _type_family(value):
    canonical = _canonical_type(value)
    if canonical in ('BPSK', 'QPSK') or str(canonical).endswith('PSK'):
        return 'PSK'
    if str(canonical).endswith('QAM'):
        return 'QAM'
    if str(canonical).endswith('FSK'):
        return 'FSK'
    return canonical


def _quality(value, reference, *, full_abs=None, partial_abs=None,
             full_rel=None, partial_rel=None):
    value, reference = _number(value), _number(reference)
    if value is None or reference is None:
        return 0.0
    full = 0.0 if full_abs is None else full_abs
    partial = full if partial_abs is None else partial_abs
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


def _subscore(identifier, maximum, criteria):
    assert math.isclose(sum(item['max'] for item in criteria), maximum)
    return {
        'id': identifier,
        'score': round(sum(item['score'] for item in criteria), 2),
        'max': float(maximum),
        'criteria': criteria,
    }


def _question_result(identifier, subscores):
    maximum = QUESTION_MAXIMA[identifier]
    assert math.isclose(sum(item['max'] for item in subscores), maximum)
    return {
        'sub_scores': subscores,
        'total_score': round(sum(item['score'] for item in subscores), 2),
        'total_max': maximum,
        'method': SCORER_VERSION,
    }


def _ground_truth(meta):
    return {
        key: value
        for question in meta.get('questions', [])
        for key, value in question.get('ground_truth', {}).items()
    }


def _normalize_catalog(payload):
    signals = []
    seen = set()
    for index, item in enumerate(_list(_dict(payload.get('Q1a')).get('signals'))[:20]):
        item = _dict(item)
        label = str(item.get('id') or f'candidate-{index + 1}').strip()
        label_key = _key(label)
        unique = label_key and label_key not in seen
        seen.add(label_key)
        signals.append({
            'label': label,
            'label_key': label_key,
            'unique_label': unique,
            'center_frequency_MHz': _number(item.get('center_MHz')),
            'bandwidth_MHz': _number(item.get('bandwidth_MHz')),
            'type': item.get('modulation'),
            'power_dBm': _number(item.get('power_dBm')),
        })
    return signals


def _match_catalog(candidates, references):
    """Match candidate IDs to physical reference signals without an LLM."""
    if not candidates:
        return {
            'signals': [],
            'label_to_reference': {},
            'by_reference': {},
            'unmatched_reference_ids': [r['signal_id'] for r in references],
        }
    unmatched_cost = 4.0
    invalid_cost = 100.0
    costs = []
    for candidate in candidates:
        row = []
        for reference in references:
            center = candidate['center_frequency_MHz']
            bandwidth = candidate['bandwidth_MHz']
            ref_center = float(reference['center_frequency_MHz'])
            ref_bandwidth = float(reference['bandwidth_MHz'])
            if center is None:
                row.append(invalid_cost)
                continue
            gate = max(0.30, 0.8 * (ref_bandwidth + (bandwidth or ref_bandwidth)))
            if abs(center - ref_center) > gate:
                row.append(invalid_cost)
                continue
            frequency_cost = abs(center - ref_center) / max(ref_bandwidth, 0.1)
            bandwidth_cost = (
                0.35 * abs(math.log(max(bandwidth, 1e-9) / ref_bandwidth))
                if bandwidth and bandwidth > 0 else 0.5
            )
            candidate_type = _canonical_type(candidate['type'])
            reference_type = _canonical_type(reference['type'])
            if candidate_type == reference_type:
                type_cost = 0.0
            elif _type_family(candidate_type) == _type_family(reference_type):
                type_cost = 0.4
            else:
                type_cost = 1.2
            power_cost = (
                min(0.8, abs(candidate['power_dBm'] - reference['power_dBm']) / 10)
                if candidate['power_dBm'] is not None else 0.3
            )
            row.append(frequency_cost + bandwidth_cost + type_cost + power_cost)
        row.extend(unmatched_cost for _ in candidates)
        costs.append(row)

    rows, columns = linear_sum_assignment(np.asarray(costs))
    assignments = dict(zip(rows.tolist(), columns.tolist()))
    mapped, label_to_reference, by_reference = [], {}, {}
    for row_index, candidate in enumerate(candidates):
        column = assignments.get(row_index)
        entry = dict(candidate)
        if (
            column is not None
            and column < len(references)
            and costs[row_index][column] < unmatched_cost
        ):
            reference_id = int(references[column]['signal_id'])
            entry['reference_id'] = reference_id
            entry['status'] = 'matched'
            by_reference[reference_id] = entry
            if candidate['unique_label'] and candidate['label_key']:
                label_to_reference[candidate['label_key']] = reference_id
        else:
            entry['reference_id'] = None
            entry['status'] = 'unmatched'
        mapped.append(entry)
    return {
        'signals': mapped,
        'label_to_reference': label_to_reference,
        'by_reference': by_reference,
        'unmatched_reference_ids': [
            int(reference['signal_id']) for reference in references
            if int(reference['signal_id']) not in by_reference
        ],
    }


def _resolve_id(value, mapping, _reference_count):
    key = _key(value)
    if key in mapping['label_to_reference']:
        return mapping['label_to_reference'][key]
    return None


def _canonical_side(value):
    key = _key(value)
    if 'communication' in key or key.startswith('comm'):
        return 'communication'
    if 'radar' in key:
        return 'radar'
    return key


def _q1b_selection(answer, gt, mapping, reference_count):
    section = _dict(answer.get('Q1b'))
    pair = [
        _resolve_id(value, mapping, reference_count)
        for value in _list(section.get('pair_ids'))[:2]
    ]
    expected_pair = {int(value) for value in gt['pair_ids']}
    pair_ok = len(pair) == 2 and None not in pair and set(pair) == expected_pair
    target = _resolve_id(section.get('target_id'), mapping, reference_count)
    target_ok = pair_ok and target == gt.get('target_id')
    return pair_ok, target_ok


def _resolved_item_ids(items, mapping, reference_count):
    resolved = set()
    for item in _list(items):
        reference_id = _resolve_id(_dict(item).get('id'), mapping, reference_count)
        if reference_id is not None:
            resolved.add(reference_id)
    return resolved


def _score_q1a(answer, references, mapping):
    candidates = mapping['signals']
    matched = len(mapping['by_reference'])
    recall = matched / len(references) if references else 0.0
    precision = matched / len(candidates) if candidates else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    frequency, bandwidth, modulation, power = [], [], [], []
    for reference in references:
        candidate = mapping['by_reference'].get(int(reference['signal_id']))
        if not candidate:
            frequency.append(0.0)
            bandwidth.append(0.0)
            modulation.append(0.0)
            power.append(0.0)
            continue
        frequency.append(_quality(
            candidate['center_frequency_MHz'], reference['center_frequency_MHz'],
            full_abs=0.10, partial_abs=0.25,
        ))
        bandwidth.append(_quality(
            candidate['bandwidth_MHz'], reference['bandwidth_MHz'],
            full_rel=0.20, partial_rel=0.40,
        ))
        candidate_type = _canonical_type(candidate['type'])
        reference_type = _canonical_type(reference['type'])
        modulation.append(
            1.0 if candidate_type == reference_type else
            0.5 if _type_family(candidate_type) == _type_family(reference_type) else
            0.0
        )
        power.append(_quality(
            candidate['power_dBm'], reference['power_dBm'],
            full_abs=3.0, partial_abs=6.0,
        ))
    return _subscore('Q1a', 12, [
        _criterion('Q1a_detection', 4, f1, f'{matched}/{len(references)} references matched'),
        _criterion('Q1a_frequency', 3, _mean(frequency), 'per-signal frequency accuracy'),
        _criterion('Q1a_bandwidth', 2, _mean(bandwidth), 'per-signal bandwidth accuracy'),
        _criterion('Q1a_modulation', 2, _mean(modulation), 'per-signal modulation accuracy'),
        _criterion('Q1a_power', 1, _mean(power), 'per-signal power accuracy'),
    ])


def _score_q1b(answer, gt, mapping, reference_count):
    section = _dict(answer.get('Q1b'))
    pair_ok, target_ok = _q1b_selection(
        answer, gt, mapping, reference_count
    )
    overlap = _quality(
        section.get('overlap_MHz'), gt['overlap_MHz'],
        full_abs=0.05, partial_abs=0.15,
    ) if pair_ok else 0.0
    sir = _quality(
        section.get('sir_dB'), gt['SIR_weak_dB'],
        full_abs=3.0, partial_abs=6.0,
    ) if target_ok else 0.0
    return _subscore('Q1b', 8, [
        _criterion('Q1b_pair', 2, float(pair_ok), 'largest-overlap pair identity'),
        _criterion('Q1b_target', 2, float(target_ok), 'weaker target identity'),
        _criterion('Q1b_overlap', 2, overlap, 'overlap bandwidth after pair identification'),
        _criterion('Q1b_sir', 2, sir, 'target SIR after pair and target identification'),
    ])


def _score_q1c(answer, gt, q1b_gt, signals, mapping, reference_count):
    section = _dict(answer.get('Q1c'))
    target_id = _resolve_id(section.get('target_id'), mapping, reference_count)
    pair_ok, q1b_target_ok = _q1b_selection(
        answer, q1b_gt, mapping, reference_count
    )
    target_ok = target_id == gt['target_id']
    prerequisite_ok = pair_ok and q1b_target_ok and target_ok
    passband = _pair(section.get('passband_MHz'))
    metrics = None
    if prerequisite_ok and passband and passband[1] > passband[0]:
        target = signals[int(gt['target_id']) - 1]
        interferer = signals[int(gt['interferer_id']) - 1]
        metrics = evaluate_extraction_mask(target, interferer, passband)
    retention_ratio = 0.0
    performance_ratio = 0.0
    reported_ratio = 0.0
    if metrics:
        retention_ratio = min(
            1.0, metrics['target_inband_fraction'] / MASK_TARGET_RETENTION
        )
        if metrics['target_inband_fraction'] + 5e-4 >= MASK_TARGET_RETENTION:
            objective_gap = max(
                0.0, gt['optimal_SIR_after_dB'] - metrics['SIR_after_dB']
            )
            performance_ratio = 1.0 if objective_gap <= 1 else 0.5 if objective_gap <= 4 else 0.0
        reported_ratio = _mean([
            _quality(section.get('target_retained_fraction'), metrics['target_inband_fraction'],
                     full_abs=0.03, partial_abs=0.10),
            _quality(section.get('post_sir_dB'), metrics['SIR_after_dB'],
                     full_abs=2.0, partial_abs=5.0),
            _quality(section.get('improvement_dB'), metrics['improvement_dB'],
                     full_abs=2.0, partial_abs=5.0),
        ])
    return _subscore('Q1c', 8, [
        _criterion('Q1c_target', 1, float(target_ok), 'target identity'),
        _criterion('Q1c_feasibility', 2, retention_ratio, 'target retention constraint'),
        _criterion('Q1c_objective', 3, performance_ratio, 'verified post-mask SIR versus optimum'),
        _criterion('Q1c_reported_metrics', 2, reported_ratio, 'reported mask metrics'),
    ])


def _feasible_packed_centers(centers, signals, fs, channel_bw):
    occupied = occupied_intervals_mhz(signals, fs)
    band_lo, band_hi = -fs / 2 / 1e6, fs / 2 / 1e6
    accepted = []
    tolerance = 5e-4
    for center in sorted(value for value in centers if value is not None):
        lo, hi = center - channel_bw / 2, center + channel_bw / 2
        if lo < band_lo + PACKING_GUARD_MHZ - tolerance:
            continue
        if hi > band_hi - PACKING_GUARD_MHZ + tolerance:
            continue
        if any(not (
            hi + PACKING_GUARD_MHZ <= occupied_lo + tolerance
            or lo >= occupied_hi + PACKING_GUARD_MHZ - tolerance
        ) for occupied_lo, occupied_hi in occupied):
            continue
        if accepted and lo < accepted[-1][1] + PACKING_GUARD_MHZ - tolerance:
            continue
        accepted.append((lo, hi))
    return len(accepted)


def _score_q1d(answer, gt, signals, fs):
    section = _dict(answer.get('Q1d'))
    packing = gt['packing']
    required = int(packing['required_count'])
    centers = [_number(value) for value in _list(section.get('additional_centers_MHz'))]
    feasible = _feasible_packed_centers(
        centers, signals, fs, packing['channel_bandwidth_MHz']
    )
    reported_count = _integer(section.get('additional_channel_count'))
    requested_count_ok = reported_count == required and len(centers) == required
    placement_ratio = min(feasible / required, 1.0) if requested_count_ok else 0.0
    return _subscore('Q1d', 6, [
        _criterion('Q1d_total_occupied', 1.5, _quality(
            section.get('total_occupied_MHz'), gt['total_bw_MHz'],
            full_abs=0.3, partial_abs=0.8,
        ), 'union occupied bandwidth'),
        _criterion('Q1d_occupancy', 1.5, _quality(
            section.get('occupancy_pct'), gt['occupancy_pct'],
            full_abs=2.0, partial_abs=5.0,
        ), 'union occupancy percentage'),
        _criterion('Q1d_count', 1, float(requested_count_ok),
                   f'required deployment count is {required}'),
        _criterion('Q1d_placement', 2, placement_ratio,
                   f'{feasible}/{required} centers satisfy all guards'),
    ])


def _score_q2a(answer, gt, mapping, reference_count, chirp_id):
    section = _dict(answer.get('Q2a'))
    identity_ok = (
        _resolve_id(section.get('signal_id'), mapping, reference_count)
        == chirp_id
    )
    gate = float(identity_ok)
    candidate_bounds = _pair([
        section.get('sweep_start_MHz'), section.get('sweep_end_MHz')
    ])
    expected_bounds = sorted([gt['sweep_start_MHz'], gt['sweep_end_MHz']])
    bound_scores = [0.0, 0.0] if not candidate_bounds else [
        _quality(candidate_bounds[index], expected_bounds[index],
                 full_abs=0.25, partial_abs=0.60)
        for index in range(2)
    ]
    return _subscore('Q2a', 7, [
        _criterion('Q2a_identity', 1, float(identity_ok), 'chirp identity'),
        _criterion('Q2a_sweep_bounds', 2, gate * _mean(bound_scores),
                   'chirp sweep bounds'),
        _criterion('Q2a_bandwidth', 1, gate * _quality(
            section.get('bandwidth_MHz'), gt['bandwidth_MHz'],
            full_rel=0.10, partial_rel=0.20,
        ), 'chirp bandwidth'),
        _criterion('Q2a_rate', 2, gate * _quality(
            section.get('chirp_rate_MHz_per_ms'), gt['chirp_rate_MHz_per_ms'],
            full_rel=0.10, partial_rel=0.25,
        ), 'chirp rate'),
        _criterion('Q2a_tbp', 1, gate * _quality(
            section.get('tbp'), gt['time_bandwidth_product'],
            full_rel=0.10, partial_rel=0.25,
        ), 'time-bandwidth product'),
    ])


def _score_q2b(answer, gt, mapping, reference_count, prerequisite_ok):
    section = _dict(answer.get('Q2b'))
    victim_ok = (
        prerequisite_ok
        and _resolve_id(section.get('victim_id'), mapping, reference_count)
        == gt['victim_id']
    )
    gate = float(victim_ok)
    modulation_ok = (
        _canonical_type(section.get('modulation'))
        == _canonical_type(gt['victim_type'])
    )
    return _subscore('Q2b', 12, [
        _criterion('Q2b_victim', 1, float(victim_ok),
                   'largest-overlap digital signal'),
        _criterion('Q2b_center', 1, gate * _quality(
            section.get('center_MHz'), gt['center_MHz'],
            full_abs=0.10, partial_abs=0.25,
        ), 'recovered center frequency'),
        _criterion('Q2b_bandwidth', 1, gate * _quality(
            section.get('bandwidth_MHz'), gt['bandwidth_MHz'],
            full_rel=0.20, partial_rel=0.40,
        ), 'recovered occupied bandwidth'),
        _criterion('Q2b_modulation', 3, gate * float(modulation_ok),
                   'recovered modulation type'),
        _criterion('Q2b_symbol_rate', 3, gate * _quality(
            section.get('symbol_rate_kHz'), gt['symbol_rate_kHz'],
            full_rel=0.10, partial_rel=0.25,
        ), 'recovered symbol rate'),
        _criterion('Q2b_rolloff', 2, gate * _quality(
            section.get('rolloff'), gt['rolloff'],
            full_abs=0.08, partial_abs=0.15,
        ), 'recovered pulse-shaping rolloff'),
        _criterion('Q2b_power', 1, gate * _quality(
            section.get('power_dBm'), gt['power_dBm'],
            full_abs=3.0, partial_abs=6.0,
        ), 'recovered signal power'),
    ])


def _score_q2c(answer, gt, prerequisite_ok):
    section = _dict(answer.get('Q2c'))
    gate = float(prerequisite_ok)
    return _subscore('Q2c', 4, [
        _criterion('Q2c_entry_time', 1.5, gate * _quality(
            section.get('entry_time_ms'), gt['entry_time_ms'],
            full_abs=0.05, partial_abs=0.12,
        ), 'chirp entry time'),
        _criterion('Q2c_exit_time', 1.5, gate * _quality(
            section.get('exit_time_ms'), gt['exit_time_ms'],
            full_abs=0.05, partial_abs=0.12,
        ), 'chirp exit time'),
        _criterion('Q2c_duration', 1, gate * _quality(
            section.get('duration_ms'), gt['duration_ms'],
            full_abs=0.03, partial_abs=0.08,
        ), 'chirp crossing duration'),
    ])


def _complex_symbol_sequence(value, maximum):
    sequence = []
    for pair in _list(value)[:maximum]:
        pair = _list(pair)
        if len(pair) != 2:
            return []
        real, imag = _number(pair[0]), _number(pair[1])
        if real is None or imag is None:
            return []
        sequence.append(complex(real, imag))
    return np.asarray(sequence, dtype=complex)


def _symbol_recovery_quality(submitted, gt):
    requested = int(gt['symbol_count'])
    candidate = _complex_symbol_sequence(submitted, requested)
    if len(candidate) < 4 or np.mean(np.abs(candidate) ** 2) <= 1e-12:
        return 0.0

    coverage = min(len(candidate) / requested, 1.0)
    minimum_distance = float(gt['minimum_constellation_distance'])
    best_accuracy = 0.0
    best_error = math.inf
    for raw_window in gt['symbol_windows_iq']:
        window = np.asarray(raw_window[:len(candidate)], dtype=float)
        reference = window[:, 0] + 1j * window[:, 1]
        for conjugated in (False, True):
            aligned_reference = np.conjugate(reference) if conjugated else reference
            denominator = np.vdot(aligned_reference, aligned_reference).real
            if denominator <= 1e-12:
                continue
            scale = np.vdot(aligned_reference, candidate) / denominator
            scaled_distance = abs(scale) * minimum_distance
            if scaled_distance <= 1e-12:
                continue
            errors = np.abs(candidate - scale * aligned_reference)
            accuracy = float(np.mean(errors <= 0.45 * scaled_distance))
            normalized_error = float(np.mean(errors / scaled_distance))
            if (accuracy, -normalized_error) > (best_accuracy, -best_error):
                best_accuracy = accuracy
                best_error = normalized_error

    chance_accuracy = 1.0 / int(gt['modulation_order'])
    corrected = max(
        0.0,
        (best_accuracy - chance_accuracy) / (1.0 - chance_accuracy),
    )
    return coverage * corrected


def _score_q2d(answer, gt, mapping, reference_count, prerequisite_ok):
    section = _dict(answer.get('Q2d'))
    victim_ok = (
        prerequisite_ok
        and _resolve_id(section.get('victim_id'), mapping, reference_count)
        == gt['victim_id']
    )
    sequence_quality = (
        _symbol_recovery_quality(section.get('symbols_iq'), gt)
        if victim_ok else 0.0
    )
    return _subscore('Q2d', 10, [
        _criterion('Q2d_victim', 1, float(victim_ok),
                   'recovered communication signal identity'),
        _criterion('Q2d_symbols', 9, sequence_quality,
                   'hidden transmitted-symbol sequence'),
    ])


def _score_q3a(answer, gt, mapping, reference_count):
    section = _dict(answer.get('Q3a'))
    links_by_reference = {}
    for item in _list(section.get('links')):
        item = _dict(item)
        reference_id = _resolve_id(item.get('id'), mapping, reference_count)
        if reference_id is not None and reference_id not in links_by_reference:
            links_by_reference[reference_id] = item
    coverage, snr, capacity, actual, shannon = [], [], [], [], []
    for reference in gt:
        item = links_by_reference.get(int(reference['id']))
        coverage.append(float(item is not None))
        if not item:
            snr.append(0.0); capacity.append(0.0); actual.append(0.0); shannon.append(0.0)
            continue
        snr.append(_quality(item.get('snr_dB'), reference['SNR_dB'],
                            full_abs=3.0, partial_abs=6.0))
        capacity.append(_quality(item.get('capacity_Mbps'), reference['cap_Mbps'],
                                 full_rel=0.10, partial_rel=0.25))
        actual.append(_quality(item.get('actual_efficiency_bps_per_Hz'), reference['se_actual'],
                               full_abs=0.15, partial_abs=0.40))
        shannon.append(_quality(item.get('shannon_efficiency_bps_per_Hz'), reference['se_shannon'],
                                full_abs=0.50, partial_abs=1.50))
    return _subscore('Q3a', 9, [
        _criterion('Q3a_coverage', 1, _mean(coverage), 'digital-link coverage'),
        _criterion('Q3a_snr', 2, _mean(snr), 'per-link SNR'),
        _criterion('Q3a_capacity', 2, _mean(capacity), 'per-link Shannon capacity'),
        _criterion('Q3a_actual_efficiency', 2, _mean(actual), 'per-link actual efficiency'),
        _criterion('Q3a_shannon_efficiency', 2, _mean(shannon), 'per-link Shannon efficiency'),
    ])


def _score_q3b(answer, gt, mapping, reference_count, coverage_ok):
    section = _dict(answer.get('Q3b'))
    worst_ok = (
        coverage_ok
        and _resolve_id(section.get('worst_id'), mapping, reference_count)
        == gt['worst_id']
    )
    gate = float(worst_ok)
    return _subscore('Q3b', 8, [
        _criterion('Q3b_worst_link', 2, float(worst_ok), 'largest efficiency-gap link'),
        _criterion('Q3b_modulation', 2, gate * float(
            _canonical_type(section.get('recommended_modulation')) == _canonical_type(gt['rec_mod'])
        ), 'highest supported modulation'),
        _criterion('Q3b_bits', 2, gate * _quality(
            section.get('bits_per_symbol'), gt['rec_bits'],
            full_abs=0.01, partial_abs=1.0,
        ), 'recommended bits per symbol'),
        _criterion('Q3b_improvement', 2, gate * _quality(
            section.get('throughput_improvement_pct'), gt['improvement_pct'],
            full_abs=5.0, partial_abs=15.0,
        ), 'throughput improvement'),
    ])


def _score_q3c(answer, gt, mapping, reference_count, coverage_ok):
    section = _dict(answer.get('Q3c'))
    allocations = {}
    for item in _list(section.get('allocations')):
        item = _dict(item)
        reference_id = _resolve_id(item.get('id'), mapping, reference_count)
        if reference_id is not None and reference_id not in allocations:
            allocations[reference_id] = item
    allocation_scores = []
    for reference in gt['allocation']:
        item = allocations.get(int(reference['id']))
        if not item or reference['new_dBm'] is None:
            allocation_scores.append(0.0)
        else:
            allocation_scores.append(_quality(
                item.get('power_dBm'), reference['new_dBm'],
                full_abs=2.0, partial_abs=5.0,
            ))
    expected_ids = {int(item['id']) for item in gt['allocation']}
    complete_problem = coverage_ok and expected_ids.issubset(allocations)
    gate = float(complete_problem)
    return _subscore('Q3c', 9, [
        _criterion('Q3c_water_level', 2, gate * _quality(
            section.get('water_level_dBm_per_Hz'), gt['water_level_dBm_per_Hz'],
            full_abs=2.0, partial_abs=5.0,
        ), 'water level'),
        _criterion('Q3c_allocations', 3, gate * _mean(allocation_scores),
                   'complete-set power allocation'),
        _criterion('Q3c_capacity_before', 1, gate * _quality(
            section.get('capacity_before_Mbps'), gt['cap_before_Mbps'],
            full_rel=0.10, partial_rel=0.25,
        ), 'capacity before water filling'),
        _criterion('Q3c_capacity_after', 1, gate * _quality(
            section.get('capacity_after_Mbps'), gt['cap_after_Mbps'],
            full_rel=0.10, partial_rel=0.25,
        ), 'capacity after water filling'),
        _criterion('Q3c_improvement', 2, gate * _quality(
            section.get('improvement_pct'), gt['improvement_pct'],
            full_abs=2.0, partial_abs=5.0,
        ), 'capacity improvement'),
    ])


def _score_q3d(answer, gt, noise_dbm, fs):
    section = _dict(answer.get('Q3d'))
    gap = _pair(section.get('gap_MHz'))
    expected_gap = sorted(gt['gap_MHz'])
    gap_ratio = 0.0 if not gap else _mean([
        _quality(gap[index], expected_gap[index], full_abs=0.20, partial_abs=0.50)
        for index in range(2)
    ])
    n_sc = _integer(section.get('n_subcarriers'))
    spacing = _number(section.get('spacing_kHz'))
    cp_ratio = _number(section.get('cp_ratio'))
    modulation = _canonical_type(section.get('modulation'))
    table = {
        _canonical_type(name): (bits, threshold)
        for name, bits, threshold in MODULATION_OPTIONS
    }
    allowed = (
        n_sc in OFDM_SUBCARRIER_COUNTS
        and any(math.isclose(spacing or -1, value, abs_tol=1e-6)
                for value in OFDM_SPACINGS_KHZ)
        and any(math.isclose(cp_ratio or -1, value, abs_tol=1e-6)
                for value in OFDM_CP_RATIOS)
        and modulation in table
    )
    feasible = False
    achieved_ratio = 0.0
    reported_ratio = 0.0
    if allowed:
        bits, threshold = table[modulation]
        bandwidth = n_sc * spacing / 1000
        rate = n_sc * bits * spacing * 1e3 / (1 + cp_ratio) / 1e6
        center = _number(section.get('center_MHz'))
        power = _number(section.get('power_dBm'))
        noise_inband = noise_dbm + 10 * math.log10(bandwidth * 1e6 / fs)
        minimum_power = noise_inband + threshold + MODULATION_MARGIN_DB
        leakage_limit = noise_dbm + 10 * math.log10(1e6 / fs)
        leakage = None if power is None else power - OFDM_OOB_ATTENUATION_DB
        placement_ok = (
            center is not None
            and center - bandwidth / 2 >= expected_gap[0] + OFDM_GUARD_MHZ - 1e-5
            and center + bandwidth / 2 <= expected_gap[1] - OFDM_GUARD_MHZ + 1e-5
        )
        power_ok = (
            power is not None
            and power >= minimum_power - 0.05
            and leakage <= leakage_limit + 0.05
        )
        feasible = placement_ok and power_ok and rate >= 2.0
        if feasible and gt.get('design'):
            ratio = rate / gt['design']['rate_Mbps']
            achieved_ratio = 1.0 if ratio >= 0.99 else 0.5 if ratio >= 0.75 else 0.0
        reported_ratio = _mean([
            _quality(section.get('occupied_bandwidth_MHz'), bandwidth,
                     full_rel=0.02, partial_rel=0.10),
            _quality(section.get('rate_Mbps'), rate,
                     full_rel=0.02, partial_rel=0.10),
            _quality(section.get('adjacent_leakage_dBm'), leakage,
                     full_abs=1.0, partial_abs=3.0),
        ])
    return _subscore('Q3d', 7, [
        _criterion('Q3d_gap', 1, gap_ratio, 'largest unoccupied gap'),
        _criterion('Q3d_parameters', 2, float(allowed), 'allowed OFDM parameter set'),
        _criterion('Q3d_feasibility', 2, float(feasible), 'guard, SNR, leakage, and rate constraints'),
        _criterion('Q3d_objective', 1, achieved_ratio, 'verified rate versus optimum'),
        _criterion('Q3d_reported_metrics', 1, reported_ratio, 'reported OFDM metrics'),
    ])


def score_l5_response(meta, response):
    """Score a complete L5 response using only deterministic calculations."""
    parse_error = None
    received_schema = None
    try:
        answer = extract_answer_json(response)
        received_schema = answer.get('schema_version')
        if received_schema != SCHEMA_VERSION:
            raise ValueError(
                f'unsupported L5 answer schema: {received_schema!r}'
            )
    except ValueError as exc:
        answer = {}
        parse_error = str(exc)
    gt = _ground_truth(meta)
    params = meta['generation_params']
    signals = params['signals']
    references = gt['Q1a']
    candidates = _normalize_catalog(answer)
    mapping = _match_catalog(candidates, references)
    reference_count = len(references)
    chirp_id = next(
        int(item['signal_id']) for item in references
        if _canonical_type(item['type']) == 'CHIRP'
    )
    q2a_section = _dict(answer.get('Q2a'))
    chirp_ok = (
        _resolve_id(q2a_section.get('signal_id'), mapping, reference_count)
        == chirp_id
    )
    q2a_bounds = _pair([
        q2a_section.get('sweep_start_MHz'), q2a_section.get('sweep_end_MHz')
    ])
    expected_q2a_bounds = sorted([
        gt['Q2a']['sweep_start_MHz'], gt['Q2a']['sweep_end_MHz']
    ])
    q2a_measurements_ok = bool(
        chirp_ok
        and q2a_bounds
        and all(
            _quality(
                q2a_bounds[index], expected_q2a_bounds[index],
                full_abs=0.25, partial_abs=0.60,
            ) == 1.0
            for index in range(2)
        )
        and _quality(
            q2a_section.get('bandwidth_MHz'), gt['Q2a']['bandwidth_MHz'],
            full_rel=0.10, partial_rel=0.20,
        ) == 1.0
    )
    q2b_section = _dict(answer.get('Q2b'))
    q2b_victim_ok = bool(
        chirp_ok
        and _resolve_id(
            q2b_section.get('victim_id'), mapping, reference_count
        ) == gt['Q2b']['victim_id']
    )
    q2b_measurements_ok = bool(
        q2b_victim_ok
        and _quality(
            q2b_section.get('center_MHz'), gt['Q2b']['center_MHz'],
            full_abs=0.10, partial_abs=0.25,
        ) > 0.0
        and _quality(
            q2b_section.get('bandwidth_MHz'), gt['Q2b']['bandwidth_MHz'],
            full_rel=0.20, partial_rel=0.40,
        ) > 0.0
    )
    q3_expected_ids = {int(item['id']) for item in gt['Q3a']}
    q3_submitted_ids = _resolved_item_ids(
        _dict(answer.get('Q3a')).get('links'), mapping, reference_count
    )
    q3_coverage_ok = q3_expected_ids.issubset(q3_submitted_ids)

    q1 = _question_result('Q1', [
        _score_q1a(answer, references, mapping),
        _score_q1b(answer, gt['Q1b'], mapping, reference_count),
        _score_q1c(
            answer, gt['Q1c'], gt['Q1b'], signals, mapping, reference_count
        ),
        _score_q1d(answer, gt['Q1d'], signals, params['fs']),
    ])
    q2 = _question_result('Q2', [
        _score_q2a(answer, gt['Q2a'], mapping, reference_count, chirp_id),
        _score_q2b(
            answer, gt['Q2b'], mapping, reference_count,
            chirp_ok,
        ),
        _score_q2c(answer, gt['Q2c'], q2b_victim_ok),
        _score_q2d(
            answer, gt['Q2d'], mapping, reference_count, chirp_ok,
        ),
    ])
    q3 = _question_result('Q3', [
        _score_q3a(answer, gt['Q3a'], mapping, reference_count),
        _score_q3b(
            answer, gt['Q3b'], mapping, reference_count, q3_coverage_ok
        ),
        _score_q3c(
            answer, gt['Q3c'], mapping, reference_count, q3_coverage_ok
        ),
        _score_q3d(answer, gt['Q3d'], params['noise_floor_dBm'], params['fs']),
    ])
    scores = {'Q1': q1, 'Q2': q2, 'Q3': q3}
    context = {
        'schema_version': received_schema,
        'expected_schema_version': SCHEMA_VERSION,
        'parse_error': parse_error,
        'prerequisites': {
            'q2_chirp_identified': chirp_ok,
            'q2_chirp_measurements_valid': q2a_measurements_ok,
            'q2_victim_identified': q2b_victim_ok,
            'q2_victim_measurements_valid': q2b_measurements_ok,
            'q3_complete_digital_link_set': q3_coverage_ok,
        },
        'student_signal_mapping': {
            'signals': mapping['signals'],
            'unmatched_reference_ids': mapping['unmatched_reference_ids'],
        },
    }
    return (
        scores,
        round(sum(item['total_score'] for item in scores.values()), 2),
        100.0,
        context,
    )


def reference_answer(meta):
    """Build a canonical full-credit answer for verifier regression tests."""
    gt = _ground_truth(meta)
    ids = {int(item['signal_id']): f"S{int(item['signal_id'])}" for item in gt['Q1a']}
    chirp_id = next(
        int(item['signal_id']) for item in gt['Q1a']
        if _canonical_type(item['type']) == 'CHIRP'
    )
    design = gt['Q3d']['design']
    return {
        'schema_version': SCHEMA_VERSION,
        'Q1a': {'signals': [{
            'id': ids[int(item['signal_id'])],
            'center_MHz': item['center_frequency_MHz'],
            'bandwidth_MHz': item['bandwidth_MHz'],
            'modulation': item['type'],
            'power_dBm': item['power_dBm'],
        } for item in gt['Q1a']]},
        'Q1b': {
            'pair_ids': [ids[int(value)] for value in gt['Q1b']['pair_ids']],
            'target_id': ids[int(gt['Q1c']['target_id'])],
            'overlap_MHz': gt['Q1b']['overlap_MHz'],
            'sir_dB': gt['Q1b']['SIR_weak_dB'],
        },
        'Q1c': {
            'target_id': ids[int(gt['Q1c']['target_id'])],
            'passband_MHz': gt['Q1c']['optimal_passband_MHz'],
            'target_retained_fraction': gt['Q1c']['target_inband_fraction'],
            'post_sir_dB': gt['Q1c']['optimal_SIR_after_dB'],
            'improvement_dB': gt['Q1c']['optimal_improvement_dB'],
        },
        'Q1d': {
            'total_occupied_MHz': gt['Q1d']['total_bw_MHz'],
            'occupancy_pct': gt['Q1d']['occupancy_pct'],
            'additional_channel_count': gt['Q1d']['packing']['required_count'],
            'additional_centers_MHz': gt['Q1d']['packing']['reference_centers_MHz'],
        },
        'Q2a': {
            'signal_id': ids[chirp_id],
            'sweep_start_MHz': gt['Q2a']['sweep_start_MHz'],
            'sweep_end_MHz': gt['Q2a']['sweep_end_MHz'],
            'bandwidth_MHz': gt['Q2a']['bandwidth_MHz'],
            'chirp_rate_MHz_per_ms': gt['Q2a']['chirp_rate_MHz_per_ms'],
            'tbp': gt['Q2a']['time_bandwidth_product'],
        },
        'Q2b': {
            'victim_id': ids[int(gt['Q2b']['victim_id'])],
            'center_MHz': gt['Q2b']['center_MHz'],
            'bandwidth_MHz': gt['Q2b']['bandwidth_MHz'],
            'modulation': gt['Q2b']['victim_type'],
            'symbol_rate_kHz': gt['Q2b']['symbol_rate_kHz'],
            'rolloff': gt['Q2b']['rolloff'],
            'power_dBm': gt['Q2b']['power_dBm'],
        },
        'Q2c': {
            'entry_time_ms': gt['Q2c']['entry_time_ms'],
            'exit_time_ms': gt['Q2c']['exit_time_ms'],
            'duration_ms': gt['Q2c']['duration_ms'],
        },
        'Q2d': {
            'victim_id': ids[int(gt['Q2d']['victim_id'])],
            'symbols_iq': gt['Q2d']['symbols_iq'],
        },
        'Q3a': {'links': [{
            'id': ids[int(item['id'])],
            'snr_dB': item['SNR_dB'],
            'capacity_Mbps': item['cap_Mbps'],
            'actual_efficiency_bps_per_Hz': item['se_actual'],
            'shannon_efficiency_bps_per_Hz': item['se_shannon'],
        } for item in gt['Q3a']]},
        'Q3b': {
            'worst_id': ids[int(gt['Q3b']['worst_id'])],
            'recommended_modulation': gt['Q3b']['rec_mod'],
            'bits_per_symbol': gt['Q3b']['rec_bits'],
            'throughput_improvement_pct': gt['Q3b']['improvement_pct'],
        },
        'Q3c': {
            'water_level_dBm_per_Hz': gt['Q3c']['water_level_dBm_per_Hz'],
            'allocations': [{
                'id': ids[int(item['id'])], 'power_dBm': item['new_dBm']
            } for item in gt['Q3c']['allocation']],
            'capacity_before_Mbps': gt['Q3c']['cap_before_Mbps'],
            'capacity_after_Mbps': gt['Q3c']['cap_after_Mbps'],
            'improvement_pct': gt['Q3c']['improvement_pct'],
        },
        'Q3d': {
            'gap_MHz': gt['Q3d']['gap_MHz'],
            'center_MHz': design['center_MHz'],
            'n_subcarriers': design['n_sc'],
            'spacing_kHz': design['sc_spacing_kHz'],
            'cp_ratio': design['cp_ratio'],
            'modulation': design['mod'],
            'power_dBm': design['power_dBm'],
            'occupied_bandwidth_MHz': design['bw_MHz'],
            'rate_Mbps': design['rate_Mbps'],
            'adjacent_leakage_dBm': design['adjacent_leakage_dBm'],
        },
    }


def format_reference_response(meta):
    return '===ANSWERS===\n' + json.dumps(reference_answer(meta)) + '\n===END==='


def reference_response(meta):
    """Render the ground truth as a full labeled answer block (for replay).

    L5 answers are a single JSON object rather than labeled lines, so the
    block simply wraps :func:`reference_answer`. Same contract as the
    L1/L3/L4 ``reference_response`` helpers: feed the returned string to
    ``evaluate.score_problem`` to obtain the maximum reachable score.
    """
    return format_reference_response(meta)
