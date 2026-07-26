"""
EMRB L3 Batch Generator (v2): Tests DIFFERENT concepts from L4/L5.
5 questions × 20pts = 100pts.  Target GPT: 70-80.
8 archetypes × 5 = 40 problems.

Q1: Bitrate and spectral efficiency      (bit rate, spectral efficiency)
Q2: Eb/N0 and BER performance            (energy per bit, BER feasibility)
Q3: PAPR and PA nonlinearity analysis    (peak-to-average, PA compression, IBO)  ← NEW
Q4: Dynamic range and ADC requirements   (dynamic range, quantization)
Q5: Digital down-conversion and signal isolation  (DDC, decimation, data reduction)
"""
import numpy as np
import json
import os

from generation.signal_library import SIGNAL_GENERATORS

# Required Eb/N0 (dB) for BER = 1e-5, theoretical AWGN
EBN0_REQ = {'BPSK': 9.6, 'QPSK': 9.6, '8PSK': 13.0, '16QAM': 12.6, '64QAM': 16.5}

# Marker consumed by evaluation/l3_verifier.py via rubric['scoring'];
# single-sourced so the stamped marker can never drift from the scorer.
from evaluation.l3_verifier import SCORER_VERSION as L3_SCORING

# PAPR values within this margin of the extremum are considered tied; every
# tied signal is an accepted best/worst-PA answer (audit finding 6.4).
PAPR_TIE_DB = 0.3

# ============================================================
# Archetypes: 3-4 signals, well-separated
# ============================================================
ARCHETYPES = {
    'A_qpsk_fm': {
        'desc': 'QPSK + FM + Chirp',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (200e3, 500e3)},
            {'gen': 'FM', 'fc_range': (-6.5e6, -3.5e6), 'dev_range': (150e3, 350e3)},
            {'gen': 'Chirp', 'sweep_center': (5.5e6, 7.5e6), 'sweep_span': (2.5e6, 4.5e6)},
        ],
    },
    'B_16qam_fm_bpsk': {
        'desc': '16QAM + FM + Chirp + BPSK',
        'signals': [
            {'gen': '16QAM', 'fc_range': (-3.0e6, -1.0e6), 'sr_range': (200e3, 400e3)},
            {'gen': 'FM', 'fc_range': (-8.0e6, -5.0e6), 'dev_range': (150e3, 350e3)},
            {'gen': 'Chirp', 'sweep_center': (5.5e6, 7.5e6), 'sweep_span': (2.5e6, 4.5e6)},
            {'gen': 'BPSK', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (300e3, 600e3)},
        ],
    },
    'C_8psk_am': {
        'desc': '8PSK + AM + Chirp',
        'signals': [
            {'gen': '8PSK', 'fc_range': (1.5e6, 3.5e6), 'sr_range': (150e3, 350e3)},
            {'gen': 'AM', 'fc_range': (-6.0e6, -3.0e6), 'depth_range': (0.3, 0.9)},
            {'gen': 'Chirp', 'sweep_center': (-7.5e6, -5.5e6), 'sweep_span': (2.5e6, 4.5e6)},
        ],
    },
    'D_bpsk_fm_qpsk': {
        'desc': 'BPSK + FM + Chirp + QPSK',
        'signals': [
            {'gen': 'BPSK', 'fc_range': (-3.0e6, -1.0e6), 'sr_range': (300e3, 600e3)},
            {'gen': 'FM', 'fc_range': (4.0e6, 7.0e6), 'dev_range': (200e3, 400e3)},
            {'gen': 'Chirp', 'sweep_center': (-7.0e6, -5.5e6), 'sweep_span': (2.5e6, 4e6)},
            {'gen': 'QPSK', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (200e3, 500e3)},
        ],
    },
    'E_64qam_fm': {
        'desc': '64QAM + FM + Chirp',
        'signals': [
            {'gen': '64QAM', 'fc_range': (-3.0e6, -1.0e6), 'sr_range': (150e3, 300e3)},
            {'gen': 'FM', 'fc_range': (3.5e6, 6.0e6), 'dev_range': (150e3, 350e3)},
            {'gen': 'Chirp', 'sweep_center': (-7.0e6, -5.0e6), 'sweep_span': (2.5e6, 4.5e6)},
        ],
    },
    'F_qpsk_am_8psk': {
        'desc': 'QPSK + AM + Chirp + 8PSK',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (-3.0e6, -1.0e6), 'sr_range': (200e3, 500e3)},
            {'gen': 'AM', 'fc_range': (5.0e6, 8.0e6), 'depth_range': (0.4, 0.8)},
            {'gen': 'Chirp', 'sweep_center': (-7.0e6, -5.0e6), 'sweep_span': (2.5e6, 4.5e6)},
            {'gen': '8PSK', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (150e3, 350e3)},
        ],
    },
    'G_16qam_fm': {
        'desc': '16QAM + FM + Chirp',
        'signals': [
            {'gen': '16QAM', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (200e3, 400e3)},
            {'gen': 'FM', 'fc_range': (-6.0e6, -3.5e6), 'dev_range': (200e3, 400e3)},
            {'gen': 'Chirp', 'sweep_center': (6.0e6, 8.0e6), 'sweep_span': (2.5e6, 4e6)},
        ],
    },
    'H_bpsk_fm_16qam': {
        'desc': 'BPSK + FM + Chirp + 16QAM',
        'signals': [
            {'gen': 'BPSK', 'fc_range': (-8.0e6, -5.5e6), 'sr_range': (300e3, 600e3)},
            {'gen': 'FM', 'fc_range': (-4.0e6, -1.5e6), 'dev_range': (150e3, 350e3)},
            {'gen': 'Chirp', 'sweep_center': (6.0e6, 8.0e6), 'sweep_span': (2.5e6, 4.5e6)},
            {'gen': '16QAM', 'fc_range': (1.5e6, 3.5e6), 'sr_range': (200e3, 400e3)},
        ],
    },
}


# ============================================================
# Signal param builder
# ============================================================

def _make_params(rng, spec):
    gen = spec['gen']
    p = round(rng.uniform(-28, -36), 1)
    if gen in ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'):
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
# Question generators
# ============================================================

def _get_primary_digital(signals):
    """Return the first digital signal (for Q1/Q2/Q5 target)."""
    for s in signals:
        if s['type'] in ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'):
            return s
    return None


def qt_bitrate(signals, rng):
    """Q1: Bit rate & spectral efficiency."""
    ds = _get_primary_digital(signals)
    if not ds:
        return None
    Rs = ds['symbol_rate_kHz'] * 1e3
    bps = ds['bits_per_symbol']
    Rb = Rs * bps
    BW = ds['bandwidth_MHz'] * 1e6
    eta = Rb / BW
    q = (f"For the digitally modulated signal at {ds['center_frequency_MHz']:+.1f} MHz:\n"
         f"    (a) Identify its modulation scheme (BPSK/QPSK/8PSK/16QAM/64QAM, etc.) and estimate the symbol rate Rs (ksps).\n"
         f"    (b) Calculate the raw bitrate Rb = Rs × log₂(M) (kbps).\n"
         f"    (c) Calculate the spectral efficiency η = Rb / BW (bps/Hz), where BW is the occupied bandwidth.\n"
         f"    (d) If the same symbol rate were used with 64QAM instead, what bitrate (kbps) could be achieved?")
    gt = {'type': ds['type'], 'M': ds['M'], 'symbol_rate_ksps': ds['symbol_rate_kHz'],
          'bits_per_symbol': bps, 'bit_rate_kbps': round(Rb / 1e3, 1),
          'bandwidth_MHz': ds['bandwidth_MHz'],
          'spectral_efficiency_bps_Hz': round(eta, 3),
          'bit_rate_64QAM_kbps': round(Rs * 6 / 1e3, 1)}
    rubric = {'points': 20,
              'mod_type': {'pts': 6}, 'sym_rate': {'pts': 4, 'tol': '±15%'},
              'bit_rate': {'pts': 3}, 'spectral_eff': {'pts': 4}, '64qam': {'pts': 3}}
    return q, gt, rubric


def qt_ebn0(signals, noise_dbm, fs, rng):
    """Q2: Eb/N0 & BER feasibility."""
    ds = _get_primary_digital(signals)
    if not ds:
        return None
    BW = ds['bandwidth_MHz'] * 1e6
    npsd = noise_dbm - 10 * np.log10(fs)
    nib = npsd + 10 * np.log10(BW)
    snr = ds['power_dBm'] - nib
    eta = ds['bits_per_symbol'] * ds['symbol_rate_kHz'] * 1e3 / BW
    ebn0 = snr - 10 * np.log10(eta)
    req = EBN0_REQ.get(ds['type'], 10)
    margin = ebn0 - req
    q = (f"Given that the noise floor is {noise_dbm} dBm/{fs / 1e6:.0f}MHz. "
         f"For the digital signal at {ds['center_frequency_MHz']:+.1f} MHz:\n"
         f"    (a) Calculate the noise power (dBm) within its occupied bandwidth and the signal-to-noise ratio SNR (dB).\n"
         f"    (b) Convert SNR to Eb/N0 using Eb/N0 (dB) = SNR (dB) - 10·log₁₀(η).\n"
         f"    (c) What is the theoretical minimum Eb/N0 (dB) required to achieve BER = 10⁻⁵ for this modulation scheme over an AWGN channel?\n"
         f"    (d) Does the current Eb/N0 meet the BER = 10⁻⁵ requirement? What is the margin (dB)?")
    gt = {'noise_in_band_dBm': round(nib, 1), 'SNR_dB': round(snr, 1),
          'EbN0_dB': round(ebn0, 1), 'EbN0_required_dB': req,
          'margin_dB': round(margin, 1), 'feasible': bool(margin > 0)}
    rubric = {'points': 20,
              'SNR': {'pts': 5, 'tol': '±3dB'}, 'EbN0': {'pts': 5},
              'required': {'pts': 5}, 'margin': {'pts': 5, 'tol': '±3dB'}}
    return q, gt, rubric


def qt_papr(signals, sig_waveforms, rng):
    """Q3: PAPR & PA nonlinearity analysis."""
    # Compute PAPR for each signal from actual waveform; tie sets and
    # best/worst selection use the unrounded values (rounding first can move
    # a signal across the PAPR_TIE_DB boundary)
    papr_list = []
    papr_raw = []
    for i, (s, wf) in enumerate(zip(signals, sig_waveforms)):
        pwr = np.abs(wf) ** 2
        mean_p = np.mean(pwr)
        raw = 10 * np.log10(np.max(pwr) / mean_p) if mean_p > 0 else 0.0
        papr_raw.append(raw)
        papr_db = round(raw, 1)
        papr_list.append({'signal_id': i + 1, 'type': s['type'],
                          'PAPR_dB': papr_db,
                          'constant_envelope': bool(papr_db < 1.0)})

    # P1dB of the PA
    P1dB = int(rng.choice([-18, -19, -20, -21, -22]))

    # Find the signal with highest PAPR
    best_raw = min(papr_raw)
    worst_raw = max(papr_raw)
    worst = papr_list[papr_raw.index(worst_raw)]
    best = papr_list[papr_raw.index(best_raw)]

    # For the primary digital signal
    ds = _get_primary_digital(signals)
    ds_idx = next(i for i, s in enumerate(signals) if s is ds)
    ds_papr = papr_list[ds_idx]['PAPR_dB']
    ds_peak = ds['power_dBm'] + ds_papr
    ibo = P1dB - ds['power_dBm']
    clipping = bool(ds_peak > P1dB)

    sig_names = ', '.join(f"{s['type']}({s['center_frequency_MHz']:+.1f}MHz)"
                          for s in signals)
    q = (f"The following signals are present in this band: {sig_names}.\n"
         f"    (a) Estimate the peak-to-average power ratio (PAPR) (dB) for each signal.\n"
         f"        Hint: Constant-envelope signals (e.g., FM, Chirp) have PAPR close to 0 dB.\n"
         f"    (b) If these signals were to pass through the same power amplifier (PA), which signal is best suited for a nonlinear PA? Which is least suited? Why?\n"
         f"    (c) Assume the PA's 1 dB compression point is P1dB = {P1dB} dBm.\n"
         f"        For the digital signal at {ds['center_frequency_MHz']:+.1f} MHz "
         f"(average power approximately {ds['power_dBm']} dBm),\n"
         f"        calculate whether its peak power exceeds P1dB. Will nonlinear distortion occur?\n"
         f"    (d) Calculate the input back-off (IBO = P1dB - P_avg) (dB) for this digital signal,\n"
         f"        and explain how the back-off amount affects PA efficiency.")
    gt = {'papr_per_signal': papr_list,
          'best_for_PA': {'signal': best['type'], 'PAPR_dB': best['PAPR_dB']},
          'worst_for_PA': {'signal': worst['type'], 'PAPR_dB': worst['PAPR_dB']},
          'best_for_PA_accepted': sorted({
              p['type'] for p, raw in zip(papr_list, papr_raw)
              if raw <= best_raw + PAPR_TIE_DB}),
          'worst_for_PA_accepted': sorted({
              p['type'] for p, raw in zip(papr_list, papr_raw)
              if raw >= worst_raw - PAPR_TIE_DB}),
          'P1dB_dBm': P1dB,
          'digital_signal_type': ds['type'],
          'digital_avg_power_dBm': ds['power_dBm'],
          'digital_PAPR_dB': ds_papr,
          'digital_peak_dBm': round(ds_peak, 1),
          'exceeds_P1dB': clipping,
          'IBO_dB': round(ibo, 1)}
    rubric = {'points': 20,
              'papr_estimation': {'pts': 5, 'tol': '±2dB',
                                  'note': 'FM/Chirp≈0dB, PSK/QAM>0'},
              'pa_suitability': {'pts': 5, 'note': 'constant envelope best suited, high-order QAM least suited'},
              'p1db_analysis': {'pts': 5, 'note': 'peak=avg+PAPR vs P1dB'},
              'ibo_efficiency': {'pts': 5, 'note': 'larger IBO means lower PA efficiency'}}
    return q, gt, rubric


def qt_adc(signals, noise_dbm, fs, rng):
    """Q4: Dynamic range & ADC requirements."""
    npsd = noise_dbm - 10 * np.log10(fs)
    powers = [s['power_dBm'] for s in signals]
    p_max, p_min = max(powers), min(powers)
    n_sigs = len(signals)
    peak_above = 10 * np.log10(n_sigs)
    eff_peak = p_max + peak_above + 3
    noise_1k = npsd + 10 * np.log10(1e3)
    dr = eff_peak - noise_1k
    enob = int(np.ceil((dr - 1.76) / 6.02))
    dr_10 = 6.02 * 10 + 1.76

    q = (f"Receiver dynamic range and ADC requirements:\n"
         f"    (a) Estimate the power (dBm) of the strongest and weakest signals in the band, and calculate the inter-signal dynamic range.\n"
         f"    (b) Considering the peak power from coherent addition of {n_sigs} signals (+10·log₁₀({n_sigs})), plus a 3 dB margin,\n"
         f"        estimate the effective peak power (dBm).\n"
         f"    (c) Using the noise floor at 1 kHz resolution bandwidth as a reference, calculate the total dynamic range (dB) from effective peak to noise floor.\n"
         f"    (d) ADC effective dynamic range = 6.02 × ENOB + 1.76 dB. How many bits are needed at minimum? Is 10 bits sufficient?")
    gt = {'strongest_dBm': p_max, 'weakest_dBm': p_min,
          'signal_DR_dB': round(p_max - p_min, 1),
          'effective_peak_dBm': round(eff_peak, 1),
          'noise_1kHz_dBm': round(noise_1k, 1),
          'total_DR_dB': round(dr, 1), 'min_ENOB': enob,
          'DR_10bit_dB': round(dr_10, 1),
          'ten_bit_ok': bool(dr_10 > dr)}
    rubric = {'points': 20,
              'powers': {'pts': 4, 'tol': '±3dB'}, 'peak': {'pts': 5},
              'DR': {'pts': 5, 'tol': '±5dB'}, 'ENOB': {'pts': 6}}
    return q, gt, rubric


def qt_ddc(signals, fs, rng):
    """Q5: Digital down-conversion & decimation."""
    ds = _get_primary_digital(signals)
    if not ds:
        return None
    fc = ds['center_frequency_MHz'] * 1e6
    BW = ds['bandwidth_MHz'] * 1e6
    lpf = BW / 2 * 1.2
    min_rate = BW * 1.2
    decim = int(fs / min_rate)
    new_rate = fs / decim

    q = (f"If only the digital signal at {ds['center_frequency_MHz']:+.1f} MHz needs to be received:\n"
         f"    (a) How many MHz must the signal be frequency-shifted to baseband by digital down-conversion (DDC)?\n"
         f"    (b) What should the low-pass filter cutoff frequency be set to (kHz)? (Cover the signal bandwidth with ~20% margin.)\n"
         f"    (c) After filtering, what is the maximum decimation factor? What is the new sampling rate (kHz)?\n"
         f"    (d) Compared to the original {fs / 1e6:.0f} MHz sampling rate, by what factor is the data volume reduced? What is the significance?")
    gt = {'freq_shift_MHz': -ds['center_frequency_MHz'],
          'LPF_cutoff_kHz': round(lpf / 1e3, 1),
          'decimation': decim, 'new_rate_kHz': round(new_rate / 1e3, 1),
          'data_reduction': decim}
    rubric = {'points': 20,
              'shift': {'pts': 4}, 'lpf': {'pts': 5, 'tol': '±30%'},
              'decim': {'pts': 6, 'tol': '±factor_of_2'},
              'reduction': {'pts': 5}}
    return q, gt, rubric


# ============================================================
# Problem generation
# ============================================================

def _build_hints(signals):
    lines = []
    for s in signals:
        t = s['type']
        fc = s['center_frequency_MHz']
        if 'Chirp' in t:
            lines.append(f"  - Approx. {s['sweep_start_MHz']:+.1f}~{s['sweep_end_MHz']:+.1f} MHz: "
                         f"linear frequency modulated (chirp) signal")
        elif t in ('FM',):
            lines.append(f"  - Approx. {fc:+.1f} MHz: analog FM signal")
        elif t == 'AM-DSB':
            lines.append(f"  - Approx. {fc:+.1f} MHz: analog AM signal")
        else:
            lines.append(f"  - Approx. {fc:+.1f} MHz: digitally modulated signal")
    return '\n'.join(lines)


def generate_one(seed, arch_name, arch, output_dir, fs=20e6, N=32768):
    rng = np.random.RandomState(seed)
    t = np.arange(N) / fs
    duration = N / fs

    # Generate individual signals + keep waveforms for PAPR
    all_signals = []
    sig_waveforms = []
    mixed = np.zeros(N, dtype=complex)
    for spec in arch['signals']:
        gen_type, params = _make_params(rng, spec)
        sig, meta = SIGNAL_GENERATORS[gen_type](rng, fs, N, t, params)
        mixed += sig
        all_signals.append(meta)
        sig_waveforms.append(sig)

    noise_dbm = round(rng.uniform(-48, -53), 0)
    noise_power = 10 ** (noise_dbm / 10) / 1000
    noise = np.sqrt(noise_power / 2) * (rng.randn(N) + 1j * rng.randn(N))
    received = mixed + noise

    sample_id = f"EMRB_L3_{seed:04d}"
    np.save(os.path.join(output_dir, f"{sample_id}.npy"), received.astype(np.complex64))

    # Generate questions
    questions = []
    q_texts = []

    r1 = qt_bitrate(all_signals, rng)
    r2 = qt_ebn0(all_signals, noise_dbm, fs, rng)
    r3 = qt_papr(all_signals, sig_waveforms, rng)
    r4 = qt_adc(all_signals, noise_dbm, fs, rng)
    r5 = qt_ddc(all_signals, fs, rng)

    for i, (label, result) in enumerate([
        ('Q1', r1), ('Q2', r2), ('Q3', r3), ('Q4', r4), ('Q5', r5)
    ]):
        if result is None:
            continue
        q_str, gt, rubric = result
        rubric['scoring'] = L3_SCORING
        questions.append({'id': label, 'question': q_str,
                          'ground_truth': gt, 'rubric': rubric})
        q_texts.append(f"{label}. {q_str}")

    hints = _build_hints(all_signals)
    question_text = f"""You are an electromagnetic signal analysis expert. Below is I/Q signal data captured from an electromagnetic environment.

Signal file: {sample_id}.npy
Sampling rate: {fs / 1e6:.0f} MHz
Number of samples: {N}
Data format: complex64 (numpy)
Recording duration: {duration * 1e3:.4f} ms

There are {len(all_signals)} non-overlapping independent signal sources in this band. Based on a preliminary scan, they are distributed as follows:
{hints}

Please analyze these signals and answer the following questions:

""" + '\n\n'.join(q_texts) + """

Please provide the complete calculation process and numerical results for each question."""

    metadata = {
        'sample_id': sample_id, 'level': 'L3',
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

def generate_batch(num=40, seed_start=4000, output_dir='data/L3'):
    os.makedirs(output_dir, exist_ok=True)
    arch_names = list(ARCHETYPES.keys())
    per = num // len(arch_names)
    rem = num % len(arch_names)

    manifest = {'total': num, 'problems': [], 'arch_counts': {}, 'level': 'L3'}
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
    print(f"Generated {idx} L3 problems in {output_dir}/")
    print(f"Archetype distribution: {manifest['arch_counts']}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num', type=int, default=40)
    p.add_argument('--seed-start', type=int, default=4000)
    p.add_argument('--output', type=str, default='data/L3')
    args = p.parse_args()
    generate_batch(args.num, args.seed_start, args.output)
