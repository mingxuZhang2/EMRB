"""
EMRB L1 Batch Generator: Basic observation, measurement, simple arithmetic.
5 questions x 20pts = 100pts.  Target GPT: 90+.  Easiest level.
8 archetypes x 5 = 40 problems.

Q1: Signal detection & frequency measurement
Q2: Power measurement & dB conversion
Q3: Sampling parameters & frequency resolution
Q4: Noise floor estimation
Q5: Signal feature classification
"""
import numpy as np
import json
import os

from scipy.signal import welch

from signal_library import SIGNAL_GENERATORS

# Marker consumed by evaluation/l1_verifier.py via rubric['scoring'];
# single-sourced so the stamped marker can never drift from the scorer.
from evaluation.l1_verifier import SCORER_VERSION as L1_SCORING

# ============================================================
# Archetypes: 2-3 simple, well-separated signals, high SNR
# ============================================================
ARCHETYPES = {
    'A_bpsk_fm_chirp': {
        'desc': 'BPSK + FM + Chirp',
        'signals': [
            {'gen': 'BPSK', 'fc_range': (2.0e6, 4.0e6), 'sr_range': (200e3, 400e3)},
            {'gen': 'FM', 'fc_range': (-6.0e6, -3.5e6), 'dev_range': (150e3, 300e3)},
            {'gen': 'Chirp', 'sweep_center': (-8.0e6, -6.5e6), 'sweep_span': (2.0e6, 3.5e6)},
        ],
    },
    'B_qpsk_fm_chirp': {
        'desc': 'QPSK + FM + Chirp',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (1.5e6, 3.5e6), 'sr_range': (200e3, 450e3)},
            {'gen': 'FM', 'fc_range': (-5.5e6, -3.0e6), 'dev_range': (150e3, 300e3)},
            {'gen': 'Chirp', 'sweep_center': (6.0e6, 8.0e6), 'sweep_span': (2.0e6, 3.5e6)},
        ],
    },
    'C_bpsk_fm': {
        'desc': 'BPSK + FM (2 signals)',
        'signals': [
            {'gen': 'BPSK', 'fc_range': (2.0e6, 4.5e6), 'sr_range': (250e3, 500e3)},
            {'gen': 'FM', 'fc_range': (-6.0e6, -3.0e6), 'dev_range': (150e3, 350e3)},
        ],
    },
    'D_qpsk_am_chirp': {
        'desc': 'QPSK + AM + Chirp',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (-4.0e6, -2.0e6), 'sr_range': (200e3, 400e3)},
            {'gen': 'AM', 'fc_range': (3.0e6, 5.5e6), 'depth_range': (0.4, 0.8)},
            {'gen': 'Chirp', 'sweep_center': (7.0e6, 8.5e6), 'sweep_span': (2.0e6, 3.0e6)},
        ],
    },
    'E_bpsk_fm_chirp_v2': {
        'desc': 'BPSK + FM + Chirp (alt placement)',
        'signals': [
            {'gen': 'BPSK', 'fc_range': (-4.0e6, -2.0e6), 'sr_range': (200e3, 400e3)},
            {'gen': 'FM', 'fc_range': (3.5e6, 6.0e6), 'dev_range': (150e3, 300e3)},
            {'gen': 'Chirp', 'sweep_center': (-8.0e6, -6.0e6), 'sweep_span': (2.0e6, 3.5e6)},
        ],
    },
    'F_qpsk_fm': {
        'desc': 'QPSK + FM (2 signals)',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (-5.0e6, -2.5e6), 'sr_range': (200e3, 450e3)},
            {'gen': 'FM', 'fc_range': (3.0e6, 6.0e6), 'dev_range': (150e3, 350e3)},
        ],
    },
    'G_8psk_fm_chirp': {
        'desc': '8PSK + FM + Chirp',
        'signals': [
            {'gen': '8PSK', 'fc_range': (1.5e6, 3.5e6), 'sr_range': (200e3, 400e3)},
            {'gen': 'FM', 'fc_range': (-6.0e6, -3.5e6), 'dev_range': (150e3, 300e3)},
            {'gen': 'Chirp', 'sweep_center': (6.0e6, 8.0e6), 'sweep_span': (2.0e6, 3.5e6)},
        ],
    },
    'H_bpsk_am_chirp': {
        'desc': 'BPSK + AM + Chirp',
        'signals': [
            {'gen': 'BPSK', 'fc_range': (-4.0e6, -2.0e6), 'sr_range': (250e3, 450e3)},
            {'gen': 'AM', 'fc_range': (3.0e6, 5.5e6), 'depth_range': (0.3, 0.8)},
            {'gen': 'Chirp', 'sweep_center': (-8.0e6, -6.0e6), 'sweep_span': (2.0e6, 3.5e6)},
        ],
    },
}

# Signal type -> human-readable category label
_TYPE_CATEGORY = {
    'BPSK': 'Digital Modulation', 'QPSK': 'Digital Modulation', '8PSK': 'Digital Modulation',
    '16QAM': 'Digital Modulation', '64QAM': 'Digital Modulation',
    'FM': 'Analog FM', 'AM-DSB': 'Analog AM',
    'Chirp (LFM)': 'Swept Frequency',
}

# The generated PSK waveforms use linear raised-cosine pulse shaping, so their
# complex-envelope magnitude varies even though ideal unshaped PSK is constant
# envelope.
_CONST_ENVELOPE = {'FM', 'Chirp (LFM)'}


# ============================================================
# Signal param builder
# ============================================================

def _make_params(rng, spec):
    gen = spec['gen']
    p = round(rng.uniform(-28, -36), 1)
    if gen in ('BPSK', 'QPSK', '8PSK'):
        sr = rng.choice(np.arange(spec['sr_range'][0], spec['sr_range'][1] + 1, 50e3))
        ro = rng.choice([0.25, 0.3, 0.35])
        return gen, {'fc': rng.uniform(*spec['fc_range']), 'sym_rate': sr,
                     'rolloff': ro, 'power_dbm': p}
    if gen == 'FM':
        return gen, {'fc': rng.uniform(*spec['fc_range']),
                     'deviation': rng.uniform(*spec['dev_range']),
                     'mod_freq': rng.uniform(8e3, 20e3),
                     'n_harmonics': rng.randint(1, 3), 'power_dbm': p}
    if gen == 'AM':
        return gen, {'fc': rng.uniform(*spec['fc_range']),
                     'mod_depth': round(rng.uniform(*spec['depth_range']), 2),
                     'mod_freq': rng.uniform(5e3, 20e3), 'power_dbm': p}
    if gen == 'Chirp':
        sc = rng.uniform(*spec['sweep_center'])
        sp = rng.uniform(*spec['sweep_span'])
        return gen, {'sweep_start': sc - sp / 2, 'sweep_end': sc + sp / 2, 'power_dbm': p}
    return gen, {'power_dbm': p}


# ============================================================
# Question generators (L1 -- basic observation & arithmetic)
# ============================================================

def qt_detection_frequency(signals, rng):
    """Q1: Signal detection & frequency measurement (20pts)."""
    n_sig = len(signals)

    # (a) signal count
    count = n_sig

    # (b) center frequencies
    freqs = []
    for s in signals:
        if 'Chirp' in s['type']:
            fc = (s['sweep_start_MHz'] + s['sweep_end_MHz']) / 2
        else:
            fc = s['center_frequency_MHz']
        freqs.append(round(fc, 2))

    # (c) strongest / weakest
    powers = [s['power_dBm'] for s in signals]
    idx_max = int(np.argmax(powers))
    idx_min = int(np.argmin(powers))

    # (d) power difference
    diff_db = round(powers[idx_max] - powers[idx_min], 1)

    freq_list_str = ', '.join(f"{f:+.2f} MHz" for f in freqs)

    q = (f"Observe the spectrum of this frequency band and answer the following questions:\n"
         f"    (a) How many distinct independent signals are present in this band? (4 pts)\n"
         f"    (b) Estimate the center frequency (MHz) of each signal, listed from lowest to highest. (8 pts)\n"
         f"    (c) Which signal has the strongest power? Which has the weakest? (Identify by center frequency) (4 pts)\n"
         f"    (d) What is the power difference in dB between the strongest and weakest signals? (4 pts)")

    gt = {
        'signal_count': count,
        'center_frequencies_MHz': sorted(freqs),
        'strongest_signal_MHz': freqs[idx_max],
        'strongest_power_dBm': powers[idx_max],
        'weakest_signal_MHz': freqs[idx_min],
        'weakest_power_dBm': powers[idx_min],
        # Tied powers make the identity non-unique; every tied signal is an
        # accepted answer (audit finding 6.4).
        'strongest_accepted_MHz': [f for f, p in zip(freqs, powers)
                                   if p == powers[idx_max]],
        'weakest_accepted_MHz': [f for f, p in zip(freqs, powers)
                                 if p == powers[idx_min]],
        'power_difference_dB': diff_db,
    }
    rubric = {
        'points': 20,
        'count': {'pts': 4, 'note': 'exact match'},
        'frequencies': {'pts': 8, 'tol': '+-0.5MHz per signal'},
        'strongest_weakest': {'pts': 4, 'note': 'identify correct signals'},
        'power_diff': {'pts': 4, 'tol': '+-2dB'},
    }
    return q, gt, rubric


def qt_power_db(signals, noise_dbm, fs, rng):
    """Q2: Power measurement & dB conversion (20pts)."""
    powers_dbm = [s['power_dBm'] for s in signals]
    powers_mw = [round(10 ** (p / 10), 4) for p in powers_dbm]

    # (c) total power in dBm
    total_mw = sum(powers_mw)
    total_dbm = round(10 * np.log10(total_mw), 1)

    # (d) SNR per signal
    # noise power in signal bandwidth
    npsd = noise_dbm - 10 * np.log10(fs)  # dBm/Hz
    snrs = []
    for s in signals:
        bw = s['bandwidth_MHz'] * 1e6
        noise_in_band = npsd + 10 * np.log10(bw)
        snr = round(s['power_dBm'] - noise_in_band, 1)
        snrs.append(snr)

    power_str = ', '.join(f"{p} dBm" for p in powers_dbm)

    q = (f"Power measurement and dB conversion:\n"
         f"    (a) Estimate the power (dBm) of each signal. (6 pts)\n"
         f"    (b) Convert the power of each signal from dBm to mW.\n"
         f"        Hint: P(mW) = 10^(P(dBm)/10) (4 pts)\n"
         f"    (c) Calculate the total power of all signals (dBm).\n"
         f"        Hint: P_total = 10·log₁₀(ΣPᵢ(mW)) (4 pts)\n"
         f"    (d) Given a noise floor of {noise_dbm} dBm/{fs/1e6:.0f}MHz,\n"
         f"        calculate the SNR (dB) of each signal within its respective bandwidth.\n"
         f"        Hint: N_sig = N₀(dBm/Hz) + 10·log₁₀(BW_Hz) (6 pts)")

    gt = {
        'powers_dBm': powers_dbm,
        'powers_mW': powers_mw,
        'total_power_mW': round(total_mw, 4),
        'total_power_dBm': total_dbm,
        'noise_psd_dBm_Hz': round(npsd, 1),
        'SNR_per_signal_dB': snrs,
    }
    rubric = {
        'points': 20,
        'power_est': {'pts': 6, 'tol': '+-3dB per signal'},
        'dbm_to_mw': {'pts': 4, 'note': 'formula application'},
        'total_power': {'pts': 4, 'tol': '+-1dB'},
        'snr': {'pts': 6, 'tol': '+-3dB per signal'},
    }
    return q, gt, rubric


def qt_sampling(signals, fs, N, rng):
    """Q3: Sampling parameters & frequency resolution (20pts)."""
    duration_ms = round(N / fs * 1e3, 4)
    ts_ns = round(1 / fs * 1e9, 2)
    delta_f = fs / N
    delta_f_hz = round(delta_f, 2)

    # (d) to halve delta_f
    n_needed = 2 * N

    q = (f"Sampling parameters and frequency resolution:\n"
         f"    (a) Given sampling rate fs = {fs/1e6:.0f} MHz and number of samples N = {N},\n"
         f"        calculate the recording duration T = N/fs (ms). (4 pts)\n"
         f"    (b) Calculate the sampling interval Ts = 1/fs (ns). (4 pts)\n"
         f"    (c) Calculate the FFT frequency resolution Δf = fs/N (Hz). (4 pts)\n"
         f"    (d) To halve the frequency resolution Δf, how many samples are needed?\n"
         f"        Should the sampling rate fs or the number of samples N be changed? Explain your reasoning. (8 pts)")

    gt = {
        'duration_ms': duration_ms,
        'Ts_ns': ts_ns,
        'delta_f_Hz': delta_f_hz,
        'N_for_half_delta_f': n_needed,
        'change_parameter': 'N (increase the number of samples)',
        'explanation': 'Increasing N halves Δf=fs/N; changing fs would also alter the observable frequency range',
    }
    rubric = {
        'points': 20,
        'duration': {'pts': 4, 'note': 'exact calculation'},
        'Ts': {'pts': 4, 'note': 'exact calculation'},
        'delta_f': {'pts': 4, 'note': 'exact calculation'},
        'halving': {'pts': 8, 'note': '4pts for N=2N, 4pts for reasoning'},
    }
    return q, gt, rubric


def qt_noise(signals, noise_dbm, fs, N, rng):
    """Q4: Noise floor estimation (20pts)."""
    npsd = noise_dbm - 10 * np.log10(fs)  # dBm/Hz
    noise_1mhz = round(npsd + 10 * np.log10(1e6), 1)  # dBm in 1MHz BW

    q = (f"Noise floor estimation:\n"
         f"    (a) Observe the signal-free regions of the spectrum and estimate the noise power spectral density (dBm/Hz).\n"
         f"        Hint: Noise floor (dBm/{fs/1e6:.0f}MHz) = {noise_dbm} dBm,\n"
         f"        N₀(dBm/Hz) = Noise floor (dBm) - 10·log₁₀(fs) (6 pts)\n"
         f"    (b) Calculate the noise power within a 1 MHz bandwidth (dBm).\n"
         f"        Hint: P_noise = N₀(dBm/Hz) + 10·log₁₀(1×10⁶) (5 pts)\n"
         f"    (c) Determine whether this is white noise (i.e., whether the PSD is approximately uniform across all frequencies). (5 pts)\n"
         f"    (d) If all signals were removed, would the noise floor change? Why or why not? (4 pts)")

    gt = {
        'noise_psd_dBm_Hz': round(npsd, 1),
        'noise_1MHz_dBm': noise_1mhz,
        'is_white_noise': True,
        'white_noise_reason': 'The PSD of AWGN noise is uniformly distributed across all frequencies',
        'noise_changes_without_signals': False,
        'reason': 'The noise is additive and independent of the signals; removing signals does not affect the noise floor',
    }
    rubric = {
        'points': 20,
        'psd': {'pts': 6, 'tol': '+-2dB'},
        'noise_1mhz': {'pts': 5, 'tol': '+-2dB'},
        'white_noise': {'pts': 5, 'note': 'correct judgment + reasoning'},
        'removal': {'pts': 4, 'note': 'correct answer + reasoning'},
    }
    return q, gt, rubric


def _bw_category(bw_khz):
    if bw_khz < 100:
        return 'Narrowband (<100kHz)'
    if bw_khz < 1000:
        return 'Midband (100kHz-1MHz)'
    return 'Wideband (>1MHz)'


def _measure_3db_bw_khz(waveform, fs):
    """Oracle 3 dB bandwidth of a clean per-signal waveform: outermost
    half-power crossings of its Welch PSD (handles double-humped FM spectra).
    Floored at one resolution bin (a pure carrier's width is not resolvable)."""
    nperseg = 4096
    freqs, psd = welch(waveform, fs=fs, nperseg=nperseg,
                       return_onesided=False)
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    above = freqs[psd >= psd.max() / 2]
    width_hz = max(float(above.max() - above.min()), fs / nperseg)
    return round(width_hz / 1e3, 1)


def qt_classification(signals, sig_waveforms, fs, rng):
    """Q5: Signal feature classification (20pts)."""
    # (a) narrowband / midband / wideband. The question says "classify by
    # bandwidth" without pinning the definition, so the category is accepted
    # under BOTH the occupied-bandwidth and the measured 3 dB definitions
    # (audit finding 6.2).
    classifications = []
    bw_3db_list = []
    categories_accepted = []
    for s, waveform in zip(signals, sig_waveforms):
        bw_hz = s['bandwidth_MHz'] * 1e6
        cat = _bw_category(bw_hz / 1e3)
        bw_3db = _measure_3db_bw_khz(waveform, fs)
        bw_3db_list.append(bw_3db)
        accepted = [cat]
        cat_3db = _bw_category(bw_3db)
        if cat_3db not in accepted:
            accepted.append(cat_3db)
        categories_accepted.append(accepted)
        classifications.append({'signal': s['type'],
                                'center_MHz': s.get('center_frequency_MHz',
                                                    (s.get('sweep_start_MHz', 0) +
                                                     s.get('sweep_end_MHz', 0)) / 2),
                                'bandwidth_kHz': round(bw_hz / 1e3, 1),
                                'category': cat})

    # (b) the question asks for the 3 dB bandwidth: ground truth is the
    # oracle-measured value; the occupied bandwidth stays as the upper end of
    # the accepted interval (audit finding 6.2).
    bw_list = [c['bandwidth_kHz'] for c in classifications]

    # (c) constant envelope
    const_env = []
    for s in signals:
        is_const = s['type'] in _CONST_ENVELOPE
        const_env.append({'signal': s['type'], 'constant_envelope': is_const})

    # (d) signal type identification
    type_labels = []
    for s in signals:
        t = s['type']
        if t in ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'):
            label = 'Digital Modulation'
        elif t == 'FM':
            label = 'Frequency Modulation (FM)'
        elif t == 'AM-DSB':
            label = 'Amplitude Modulation (AM)'
        elif 'Chirp' in t:
            label = 'Swept Frequency (Chirp)'
        else:
            label = 'Unknown'
        type_labels.append({'signal': t, 'label': label})

    q = (f"Signal feature classification:\n"
         f"    (a) Classify each signal by bandwidth: Narrowband (<100kHz), Midband (100kHz-1MHz), Wideband (>1MHz). (5 pts)\n"
         f"    (b) Estimate the approximate 3 dB bandwidth (kHz) of each signal. (6 pts)\n"
         f"    (c) Which signals are constant-envelope signals?\n"
         f"        Hint: FM and Chirp signals are constant-envelope; PSK modulation depends on implementation. (5 pts)\n"
         f"    (d) Identify the type of each signal: CW / FM / AM / Digital Modulation / Chirp. (4 pts)")

    gt = {
        'bandwidth_classifications': classifications,
        'bandwidths_kHz': bw_list,
        'bandwidths_3dB_kHz': bw_3db_list,
        'bandwidth_categories_accepted': categories_accepted,
        'constant_envelope': const_env,
        'signal_types': type_labels,
    }
    rubric = {
        'points': 20,
        'bw_category': {'pts': 5, 'note': 'correct category per signal'},
        'bw_estimate': {'pts': 6, 'tol': '+-30% per signal'},
        'const_envelope': {'pts': 5, 'note': 'correct identification'},
        'type_id': {'pts': 4, 'note': 'correct type per signal'},
    }
    return q, gt, rubric


# ============================================================
# Hint builder
# ============================================================

def _build_hints(signals):
    """Build explicit hints for preamble: signal count, locations, categories."""
    lines = []
    for s in signals:
        t = s['type']
        cat = _TYPE_CATEGORY.get(t, 'Unknown')
        if 'Chirp' in t:
            lines.append(f"  - Approx. {s['sweep_start_MHz']:+.1f}~{s['sweep_end_MHz']:+.1f} MHz: "
                         f"{cat} signal (linear frequency modulation)")
        else:
            fc = s['center_frequency_MHz']
            lines.append(f"  - Approx. {fc:+.1f} MHz: {cat} signal")
    return '\n'.join(lines)


# ============================================================
# Problem generation
# ============================================================

def generate_one(seed, arch_name, arch, output_dir, fs=20e6, N=32768):
    rng = np.random.RandomState(seed)
    t = np.arange(N) / fs
    duration = N / fs

    # Generate individual signals (waveforms kept for the Q5 3dB-bw oracle)
    all_signals = []
    sig_waveforms = []
    mixed = np.zeros(N, dtype=complex)
    for spec in arch['signals']:
        gen_type, params = _make_params(rng, spec)
        sig, meta = SIGNAL_GENERATORS[gen_type](rng, fs, N, t, params)
        mixed += sig
        all_signals.append(meta)
        sig_waveforms.append(sig)

    # Add noise
    noise_dbm = round(rng.uniform(-50, -55), 0)
    noise_power = 10 ** (noise_dbm / 10) / 1000
    noise = np.sqrt(noise_power / 2) * (rng.randn(N) + 1j * rng.randn(N))
    received = mixed + noise

    sample_id = f"EMRB_L1_{seed:04d}"
    np.save(os.path.join(output_dir, f"{sample_id}.npy"), received.astype(np.complex64))

    # Generate all 5 questions
    r1 = qt_detection_frequency(all_signals, rng)
    r2 = qt_power_db(all_signals, noise_dbm, fs, rng)
    r3 = qt_sampling(all_signals, fs, N, rng)
    r4 = qt_noise(all_signals, noise_dbm, fs, N, rng)
    r5 = qt_classification(all_signals, sig_waveforms, fs, rng)

    questions = []
    q_texts = []
    for i, (label, result) in enumerate([
        ('Q1', r1), ('Q2', r2), ('Q3', r3), ('Q4', r4), ('Q5', r5)
    ]):
        if result is None:
            continue
        q_str, gt, rubric = result
        rubric['scoring'] = L1_SCORING
        questions.append({'id': label, 'question': q_str,
                          'ground_truth': gt, 'rubric': rubric})
        q_texts.append(f"{label}. {q_str}")

    # Build signal category hint text
    sig_categories = []
    for s in all_signals:
        cat = _TYPE_CATEGORY.get(s['type'], 'Unknown')
        sig_categories.append(cat)
    cat_str = '/'.join(dict.fromkeys(sig_categories))  # deduplicate, preserve order

    hints = _build_hints(all_signals)
    question_text = f"""You are an electromagnetic signal analysis expert. Below is I/Q signal data collected from an electromagnetic environment.

Signal file: {sample_id}.npy
Sampling rate: {fs / 1e6:.0f} MHz
Number of samples: {N}
Data format: complex64 (numpy)
Recording duration: {duration * 1e3:.4f} ms

This frequency band contains {len(all_signals)} non-overlapping independent signal sources (types include {cat_str}), distributed as follows based on a preliminary scan:
{hints}

Please analyze these signals and answer the following questions (20 pts each, 100 pts total):

""" + '\n\n'.join(q_texts) + """

Please provide complete calculation procedures and numerical results for each question."""

    metadata = {
        'sample_id': sample_id, 'level': 'L1',
        'archetype': arch_name, 'archetype_desc': arch['desc'],
        'total_points': sum(q['rubric']['points'] for q in questions),
        'num_questions': len(questions),
        'question': question_text, 'questions': questions,
        'generation_params': {
            'fs': fs, 'N': N, 'duration_ms': round(duration * 1e3, 4),
            'signals': all_signals, 'noise_floor_dBm': noise_dbm, 'seed': seed,
        },
    }

    with open(os.path.join(output_dir, f"{sample_id}.json"), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    return metadata


# ============================================================
# Batch
# ============================================================

def generate_batch(num=40, seed_start=5000, output_dir='data/L1'):
    os.makedirs(output_dir, exist_ok=True)
    arch_names = list(ARCHETYPES.keys())
    per = num // len(arch_names)
    rem = num % len(arch_names)

    manifest = {'total': num, 'problems': [], 'arch_counts': {}, 'level': 'L1'}
    idx = 0
    for ai, aname in enumerate(arch_names):
        n = per + (1 if ai < rem else 0)
        manifest['arch_counts'][aname] = n
        for i in range(n):
            seed = seed_start + idx
            print(f"[{idx + 1}/{num}] {aname} seed={seed}...", end=' ')
            try:
                meta = generate_one(seed, aname, ARCHETYPES[aname], output_dir)
                ns = len(meta['generation_params']['signals'])
                print(f"OK ({ns} sigs, {meta['num_questions']} Qs)")
                manifest['problems'].append({
                    'sample_id': meta['sample_id'], 'archetype': aname,
                    'num_signals': ns, 'total_points': meta['total_points'],
                })
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
            idx += 1

    with open(os.path.join(output_dir, 'batch_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generated {idx} L1 problems in {output_dir}/")
    print(f"Archetype distribution: {manifest['arch_counts']}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num', type=int, default=40)
    p.add_argument('--seed-start', type=int, default=5000)
    p.add_argument('--output', type=str, default='data/L1')
    args = p.parse_args()
    generate_batch(args.num, args.seed_start, args.output)
