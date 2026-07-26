"""
EMRB L4 Batch Generator: Generate 40 diverse multi-signal reasoning problems.

8 scenario archetypes × 5 problems each = 40 problems.
Each problem: 5 scored questions × 20 pts = 100 pts total.
"""
import numpy as np
import json
import os
import sys

from generation.signal_library import SIGNAL_GENERATORS, apply_burst, _set_power
from generation.question_library import QUESTION_GENERATORS


REPAIRED_ANSWER_SCHEMA = 'emrb-l4-repaired-v1'


# ============================================================
# Scenario archetypes
# ============================================================

ARCHETYPES = {
    'A_dense_digital': {
        'desc': 'Dense digital band with adjacent PSK/QAM signals',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (0.5e6, 2e6), 'sym_rate_range': (200e3, 400e3)},
            {'gen': '8PSK', 'fc_range': (1.5e6, 3e6), 'sym_rate_range': (150e3, 350e3)},
            {'gen': '16QAM', 'fc_range': (-3e6, -1e6), 'sym_rate_range': (200e3, 500e3)},
            {'gen': 'Chirp', 'sweep_range': (4e6, 9e6)},
        ],
        'question_pool': ['QT01', 'QT03', 'QT05', 'QT07', 'QT10'],
    },
    'B_mixed_analog_digital': {
        'desc': 'Mixed analog + digital coexistence',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (1e6, 3e6), 'sym_rate_range': (200e3, 500e3)},
            {'gen': 'FM', 'fc_range': (-5e6, -2e6), 'deviation_range': (100e3, 400e3)},
            {'gen': 'AM', 'fc_range': (5e6, 8e6), 'mod_depth_range': (0.3, 0.9)},
            {'gen': 'BPSK', 'fc_range': (-8e6, -5e6), 'sym_rate_range': (300e3, 600e3), 'burst': True},
        ],
        'question_pool': ['QT01', 'QT02', 'QT04', 'QT05', 'QT08'],
    },
    'C_radar_comms': {
        'desc': 'Radar + communications coexistence',
        'signals': [
            {'gen': 'Chirp', 'sweep_range': (3e6, 8e6)},
            {'gen': 'QPSK', 'fc_range': (-3e6, -1e6), 'sym_rate_range': (200e3, 400e3)},
            {'gen': '8PSK', 'fc_range': (-6e6, -4e6), 'sym_rate_range': (150e3, 300e3)},
            {'gen': 'BPSK', 'fc_range': (0.5e6, 2e6), 'sym_rate_range': (400e3, 800e3), 'burst': True},
        ],
        'question_pool': ['QT01', 'QT03', 'QT04', 'QT05', 'QT10'],
    },
    'D_ofdm_centric': {
        'desc': 'OFDM + narrowband interferers',
        'signals': [
            {'gen': 'OFDM', 'fc_range': (-2e6, 2e6), 'n_sc_choices': [64, 128], 'spacing_choices': [15.625e3, 31.25e3]},
            {'gen': 'FM', 'fc_range': (4e6, 7e6), 'deviation_range': (150e3, 350e3)},
            {'gen': 'BPSK', 'fc_range': (-7e6, -4e6), 'sym_rate_range': (200e3, 500e3)},
        ],
        'question_pool': ['QT01', 'QT02', 'QT05', 'QT06', 'QT10'],
    },
    'E_fsk_scenario': {
        'desc': 'FSK + mixed signals',
        'signals': [
            {'gen': '2FSK', 'fc_range': (-4e6, -1e6), 'sym_rate_range': (100e3, 300e3), 'dev_range': (100e3, 300e3)},
            {'gen': 'QPSK', 'fc_range': (2e6, 5e6), 'sym_rate_range': (200e3, 400e3)},
            {'gen': 'Chirp', 'sweep_range': (5e6, 9e6)},
            {'gen': 'FM', 'fc_range': (-8e6, -5e6), 'deviation_range': (100e3, 300e3)},
        ],
        'question_pool': ['QT01', 'QT02', 'QT03', 'QT05', 'QT07'],
    },
    'F_burst_heavy': {
        'desc': 'Multiple burst signals (TDMA-like)',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (1e6, 3e6), 'sym_rate_range': (250e3, 500e3), 'burst': True},
            {'gen': 'BPSK', 'fc_range': (-3e6, -1e6), 'sym_rate_range': (300e3, 600e3), 'burst': True},
            {'gen': '16QAM', 'fc_range': (4e6, 7e6), 'sym_rate_range': (200e3, 400e3)},
            {'gen': 'FM', 'fc_range': (-7e6, -4e6), 'deviation_range': (150e3, 350e3)},
        ],
        'question_pool': ['QT01', 'QT02', 'QT04', 'QT05', 'QT10'],
    },
    'G_high_density': {
        'desc': '5-6 signals, congested spectrum',
        'signals': [
            {'gen': 'BPSK', 'fc_range': (-8e6, -6e6), 'sym_rate_range': (300e3, 500e3)},
            {'gen': 'FM', 'fc_range': (-4e6, -2e6), 'deviation_range': (100e3, 300e3)},
            {'gen': 'QPSK', 'fc_range': (0.5e6, 2e6), 'sym_rate_range': (200e3, 400e3)},
            {'gen': '8PSK', 'fc_range': (2.5e6, 4e6), 'sym_rate_range': (150e3, 300e3)},
            {'gen': 'Chirp', 'sweep_range': (5e6, 9e6)},
        ],
        'question_pool': ['QT01', 'QT02', 'QT03', 'QT05', 'QT07'],
    },
    'H_interference': {
        'desc': 'Communication + interference/jamming',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (1e6, 3e6), 'sym_rate_range': (300e3, 500e3)},
            {'gen': '16QAM', 'fc_range': (1.5e6, 3.5e6), 'sym_rate_range': (200e3, 400e3)},  # overlapping!
            {'gen': 'Chirp', 'sweep_range': (5e6, 9e6)},
            {'gen': 'FM', 'fc_range': (-5e6, -2e6), 'deviation_range': (200e3, 400e3)},
        ],
        'question_pool': ['QT01', 'QT02', 'QT03', 'QT05', 'QT07'],
    },
}


def _make_signal_params(rng, sig_spec, fs, N):
    """Convert a signal spec into concrete parameters."""
    gen_type = sig_spec['gen']
    power_dbm = rng.uniform(-28, -38)
    params = {'power_dbm': round(power_dbm, 1)}

    if gen_type in ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'):
        lo, hi = sig_spec['fc_range']
        params['fc'] = rng.uniform(lo, hi)
        lo_sr, hi_sr = sig_spec['sym_rate_range']
        params['sym_rate'] = rng.choice(np.arange(lo_sr, hi_sr + 1, 50e3))
        params['rolloff'] = rng.choice([0.2, 0.25, 0.3, 0.35, 0.4])
    elif gen_type == 'FM':
        lo, hi = sig_spec['fc_range']
        params['fc'] = rng.uniform(lo, hi)
        lo_d, hi_d = sig_spec['deviation_range']
        params['deviation'] = rng.uniform(lo_d, hi_d)
        params['mod_freq'] = rng.uniform(8e3, 20e3)
        params['n_harmonics'] = rng.randint(1, 4)
    elif gen_type == 'AM':
        lo, hi = sig_spec['fc_range']
        params['fc'] = rng.uniform(lo, hi)
        lo_m, hi_m = sig_spec['mod_depth_range']
        params['mod_depth'] = round(rng.uniform(lo_m, hi_m), 2)
        params['mod_freq'] = rng.uniform(5e3, 25e3)
    elif gen_type == 'Chirp':
        lo, hi = sig_spec['sweep_range']
        mid = (lo + hi) / 2
        span = hi - lo
        actual_span = rng.uniform(span * 0.5, span)
        params['sweep_start'] = mid - actual_span / 2
        params['sweep_end'] = mid + actual_span / 2
    elif gen_type == 'OFDM':
        lo, hi = sig_spec['fc_range']
        params['fc'] = rng.uniform(lo, hi)
        params['n_subcarriers'] = rng.choice(sig_spec['n_sc_choices'])
        params['subcarrier_spacing'] = rng.choice(sig_spec['spacing_choices'])
        params['cp_ratio'] = rng.choice([0.125, 0.25, 0.5])
    elif gen_type in ('2FSK', '4FSK'):
        lo, hi = sig_spec['fc_range']
        params['fc'] = rng.uniform(lo, hi)
        lo_sr, hi_sr = sig_spec['sym_rate_range']
        params['sym_rate'] = rng.uniform(lo_sr, hi_sr)
        lo_d, hi_d = sig_spec['dev_range']
        params['freq_deviation'] = rng.uniform(lo_d, hi_d)

    return gen_type, params, sig_spec.get('burst', False)


def build_question_text(sample_id, fs, N, questions):
    """Build the model-facing prompt from an already validated question list."""
    duration = N / fs
    question_text = f"""You are an electromagnetic signal analysis expert. Below is I/Q signal data collected from a complex electromagnetic environment.

Signal file: {sample_id}.npy
Sample rate: {fs/1e6:.0f} MHz
Number of samples: {N}
Data format: complex64 (numpy)
Recording duration: {duration*1e3:.4f} ms

Multiple independent signal sources exist within this frequency band (spectral overlap and burst signals may be present).
Please first analyze the signal environment on your own, then answer the following questions:

"""
    for question in questions:
        question_text += f"{question['id']}. {question['question']}\n\n"
    question_text += "Please provide a complete reasoning process and numerical results for each question."
    if any(
        question.get('rubric', {}).get('scoring') == 'l4-deterministic-v1'
        for question in questions
    ):
        question_text += (
            f"\n\nAnswer format version: {REPAIRED_ANSWER_SCHEMA}. In the final "
            "===ANSWERS=== block, use one line per top-level question in the form "
            "Q1: value through Q5: value. For a question that requests a JSON "
            "object, place one valid single-line JSON object directly after its Q label."
        )
    return question_text


def generate_one_problem(seed, archetype_name, archetype, output_dir, fs=20e6, N=32768):
    """Generate one L4 problem from an archetype."""
    rng = np.random.RandomState(seed)
    t = np.arange(N) / fs
    duration = N / fs

    # Generate signals
    all_signals = []
    mixed = np.zeros(N, dtype=complex)
    for sig_spec in archetype['signals']:
        gen_type, params, is_burst = _make_signal_params(rng, sig_spec, fs, N)
        gen_func = SIGNAL_GENERATORS[gen_type]
        sig, meta = gen_func(rng, fs, N, t, params)
        if gen_type in ('2FSK', '4FSK'):
            # L5 uses this field, but adding it to legacy L4 metadata changes
            # question selection and otherwise untouched problem records.
            meta.pop('bits_per_symbol', None)

        if is_burst:
            start_f = rng.uniform(0.15, 0.4)
            end_f = rng.uniform(start_f + 0.2, min(start_f + 0.6, 0.85))
            sig, burst_meta = apply_burst(sig, rng, N, {'start_frac': start_f, 'end_frac': end_f})
            meta.update(burst_meta)
            meta['type'] = meta['type'] + ' (burst)'

        mixed += sig
        all_signals.append(meta)

    # Add noise
    noise_dbm = rng.choice([-45, -48, -50, -52, -55])
    noise_power = 10 ** (noise_dbm / 10) / 1000
    noise = np.sqrt(noise_power / 2) * (rng.randn(N) + 1j * rng.randn(N))
    received = mixed + noise

    # Save signal
    sample_id = f"EMRB_L4_{seed:04d}"
    np.save(os.path.join(output_dir, f"{sample_id}.npy"), received.astype(np.complex64))

    # Select and generate questions
    q_pool = archetype['question_pool']
    questions = []
    for qt_id in q_pool:
        gen = QUESTION_GENERATORS[qt_id]
        if qt_id in ('QT05', 'QT10'):
            result = gen(all_signals, noise_dbm, fs, rng)
        else:
            result = gen(all_signals, rng)
        if result is not None:
            q_text, gt, rubric = result
            questions.append({
                'id': f"Q{len(questions)+1}",
                'question_type': qt_id,
                'question': q_text,
                'ground_truth': gt,
                'rubric': rubric,
            })

    # Preserve the original five-question composition when a declared type is
    # inapplicable to a particular generated scene.
    while len(questions) < 5:
        for qt_id in ['QT01', 'QT02', 'QT03', 'QT04', 'QT05', 'QT06', 'QT07', 'QT08', 'QT10']:
            if qt_id not in [q['question_type'] for q in questions]:
                gen = QUESTION_GENERATORS[qt_id]
                if qt_id in ('QT05', 'QT10'):
                    result = gen(all_signals, noise_dbm, fs, rng)
                else:
                    result = gen(all_signals, rng)
                if result:
                    questions.append({
                        'id': f"Q{len(questions)+1}",
                        'question_type': qt_id,
                        'question': result[0],
                        'ground_truth': result[1],
                        'rubric': result[2],
                    })
                if len(questions) >= 5:
                    break
        break

    questions = questions[:5]

    question_text = build_question_text(sample_id, fs, N, questions)

    # Save metadata
    metadata = {
        'sample_id': sample_id,
        'level': 'L4',
        'archetype': archetype_name,
        'archetype_desc': archetype['desc'],
        'total_points': sum(q['rubric']['points'] for q in questions),
        'question': question_text,
        'questions': questions,
        'generation_params': {
            'fs': fs, 'N': N, 'duration_ms': round(duration * 1e3, 4),
            'signals': all_signals, 'noise_floor_dBm': noise_dbm, 'seed': seed,
        },
    }
    if any(
        question.get('rubric', {}).get('scoring') == 'l4-deterministic-v1'
        for question in questions
    ):
        metadata['answer_schema_version'] = REPAIRED_ANSWER_SCHEMA

    with open(os.path.join(output_dir, f"{sample_id}.json"), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    return metadata


def generate_batch(num_problems=40, seed_start=1000, output_dir="data/L4"):
    """Generate a batch of L4 problems."""
    os.makedirs(output_dir, exist_ok=True)

    archetype_names = list(ARCHETYPES.keys())
    problems_per_arch = num_problems // len(archetype_names)
    remainder = num_problems % len(archetype_names)

    manifest = {
        'total_problems': num_problems,
        'problems': [],
        'archetype_counts': {},
        'question_type_counts': {},
    }

    prob_idx = 0
    for arch_idx, arch_name in enumerate(archetype_names):
        n = problems_per_arch + (1 if arch_idx < remainder else 0)
        manifest['archetype_counts'][arch_name] = n

        for i in range(n):
            seed = seed_start + prob_idx
            print(f"[{prob_idx+1}/{num_problems}] {arch_name} seed={seed}...", end=' ')
            try:
                meta = generate_one_problem(seed, arch_name, ARCHETYPES[arch_name], output_dir)
                n_q = len(meta['questions'])
                n_sig = len(meta['generation_params']['signals'])
                print(f"OK ({n_sig} signals, {n_q} questions)")

                manifest['problems'].append({
                    'sample_id': meta['sample_id'],
                    'archetype': arch_name,
                    'num_signals': n_sig,
                    'num_questions': n_q,
                    'question_types': [q['question_type'] for q in meta['questions']],
                    'total_points': meta['total_points'],
                })

                for q in meta['questions']:
                    qt = q['question_type']
                    manifest['question_type_counts'][qt] = manifest['question_type_counts'].get(qt, 0) + 1

            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()

            prob_idx += 1

    # Save manifest
    with open(os.path.join(output_dir, 'batch_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {prob_idx} problems in {output_dir}/")
    print(f"Archetype distribution: {manifest['archetype_counts']}")
    print(f"Question type distribution: {manifest['question_type_counts']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=40)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--output', type=str, default='data/L4')
    args = parser.parse_args()

    generate_batch(args.num, args.seed_start, args.output)
