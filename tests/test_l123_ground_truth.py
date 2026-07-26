import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation import generate_l1_batch, generate_l2_batch
from generation.signal_library import SIGNAL_GENERATORS, apply_burst


DIGITAL_TYPES = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'}


def _metadata(level):
    paths = sorted((ROOT / 'data' / f'L{level}').glob(f'EMRB_L{level}_*.json'))
    return [(path, json.loads(path.read_text())) for path in paths]


def _regenerate_waveforms(module, meta):
    params = meta['generation_params']
    fs = params['fs']
    sample_count = params['N']
    rng = np.random.RandomState(params['seed'])
    time = np.arange(sample_count) / fs
    archetype = module.ARCHETYPES[meta['archetype']]
    waveforms = []
    regenerated_meta = []

    for spec in archetype['signals']:
        generator_name, signal_params = module._make_params(rng, spec)
        waveform, signal_meta = SIGNAL_GENERATORS[generator_name](
            rng, fs, sample_count, time, signal_params
        )
        burst = archetype.get('burst')
        if spec.get('burst') and burst:
            start = round(rng.uniform(*burst['start_range']), 2)
            end = round(rng.uniform(*burst['end_range']), 2)
            waveform, burst_info = apply_burst(
                waveform, rng, sample_count,
                {'start_frac': start, 'end_frac': end},
            )
            signal_meta['is_burst'] = True
            signal_meta['burst_info'] = burst_info
        elif module is generate_l2_batch:
            signal_meta['is_burst'] = False
        waveforms.append(waveform)
        regenerated_meta.append(signal_meta)

    return rng, waveforms, regenerated_meta


def test_all_l1_l3_problems_remain_complete():
    for level in (1, 2, 3):
        items = _metadata(level)
        assert len(items) == 40
        for _, meta in items:
            assert meta['num_questions'] == 5
            assert meta['total_points'] == 100
            assert [question['id'] for question in meta['questions']] == [
                'Q1', 'Q2', 'Q3', 'Q4', 'Q5'
            ]


def test_l1_constant_envelope_labels_match_final_waveforms():
    psk_count = 0
    for _, meta in _metadata(1):
        _, waveforms, signals = _regenerate_waveforms(generate_l1_batch, meta)
        labels = meta['questions'][4]['ground_truth']['constant_envelope']
        assert len(labels) == len(signals)

        for label, signal, waveform in zip(labels, signals, waveforms):
            expected = signal['type'] in {'FM', 'Chirp (LFM)'}
            assert label['signal'] == signal['type']
            assert label['constant_envelope'] is expected
            if signal['type'] in {'BPSK', 'QPSK', '8PSK'}:
                psk_count += 1
                power = np.abs(waveform) ** 2
                papr_db = 10 * np.log10(power.max() / power.mean())
                assert papr_db > 1.0

    assert psk_count == 40


def test_l2_energy_answers_use_final_gated_waveforms():
    burst_count = 0
    for path, meta in _metadata(2):
        rng, waveforms, signals = _regenerate_waveforms(generate_l2_batch, meta)
        params = meta['generation_params']
        fs = params['fs']
        sample_count = params['N']
        duration = sample_count / fs

        noise_dbm = round(rng.uniform(-48, -53), 0)
        noise_power_w = 10 ** (noise_dbm / 10) / 1000
        noise = np.sqrt(noise_power_w / 2) * (
            rng.randn(sample_count) + 1j * rng.randn(sample_count)
        )
        regenerated = (sum(waveforms) + noise).astype(np.complex64)
        stored = np.load(path.with_suffix('.npy'))
        assert np.array_equal(regenerated, stored)
        assert signals == params['signals']

        ground_truth = meta['questions'][4]['ground_truth']
        powers_w = [float(np.mean(np.abs(waveform) ** 2)) for waveform in waveforms]
        for signal, power_w, answer in zip(signals, powers_w,
                                           ground_truth['energy_per_signal']):
            power_mw = power_w * 1000
            power_dbm = 10 * np.log10(power_mw)
            energy_j = power_w * duration
            assert answer['type'] == signal['type']
            assert answer['power_dBm'] == round(power_dbm, 1)
            assert answer['power_mW'] == round(power_mw, 6)
            assert answer['energy_J'] == f'{energy_j:.2e}'
            assert answer['energy_dBJ'] == round(10 * np.log10(energy_j), 1)
            burst_count += int(signal.get('is_burst', False))

        total_signal_mw = sum(powers_w) * 1000
        noise_mw = 10 ** (noise_dbm / 10)
        total_received_mw = total_signal_mw + noise_mw
        assert ground_truth['total_signal_power_mW'] == round(total_signal_mw, 6)
        assert ground_truth['noise_power_mW'] == round(noise_mw, 6)
        assert ground_truth['total_received_power_mW'] == round(
            total_received_mw, 6
        )
        assert ground_truth['total_received_power_dBm'] == round(
            10 * np.log10(total_received_mw), 1
        )

        primary_index = next(
            index for index, signal in enumerate(signals)
            if signal['type'] in DIGITAL_TYPES
        )
        primary = signals[primary_index]
        total_bits = int(
            primary['symbol_rate_kHz'] * 1e3
            * primary['bits_per_symbol'] * duration
        )
        energy_per_bit = powers_w[primary_index] * duration / total_bits
        digital_answer = ground_truth['digital_signal']
        assert digital_answer['total_bits'] == total_bits
        assert digital_answer['Eb_J'] == f'{energy_per_bit:.2e}'
        assert digital_answer['Eb_dBJ'] == round(
            10 * np.log10(energy_per_bit), 1
        )

    assert burst_count == 20


def test_l3_ground_truth_uses_native_json_types():
    boolean_paths = 0
    for _, meta in _metadata(3):
        q2 = meta['questions'][1]['ground_truth']
        q3 = meta['questions'][2]['ground_truth']
        q4 = meta['questions'][3]['ground_truth']

        assert type(q2['feasible']) is bool
        assert type(q3['P1dB_dBm']) in (int, float)
        assert type(q3['exceeds_P1dB']) is bool
        assert type(q4['ten_bit_ok']) is bool
        boolean_paths += 3
        for signal in q3['papr_per_signal']:
            assert type(signal['constant_envelope']) is bool
            boolean_paths += 1

    assert boolean_paths == 260
