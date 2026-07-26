"""
EMRB L5 Batch Generator: 8 archetypes × 5 = 40 complex problems.
Each problem: 3 big questions (34+33+33=100 pts).

Every scenario guarantees:
  - An overlap pair (for Q1 interference analysis)
  - A chirp overlapping a comm signal (for Q2 radar-comm)
  - ≥3 digital signals (for Q3 capacity)
  - ≥1 burst signal (tests detection thoroughness)
"""
import numpy as np
import json
import os

from generation.signal_library import SIGNAL_GENERATORS, apply_burst

# Public task constants used by the deterministic verifier. The prompt exposes
# only the physical assumptions and engineering constraints needed to define
# the task; derivations and verification formulas remain internal.
SCHEMA_VERSION = 'emrb-l5-verifiable-v5'
MASK_TARGET_RETENTION = 0.80
MASK_STOPBAND_ATTENUATION_DB = 40.0
PACKING_GUARD_MHZ = 0.05
Q2_SYMBOL_COUNT = 32
Q2_SYMBOL_ALIGNMENT_LAG = 16  # covers the +/-50 us Q2c entry-time tolerance
MODULATION_OPTIONS = (
    ('BPSK', 1, 8.0),
    ('QPSK', 2, 11.0),
    ('8PSK', 3, 14.0),
    ('16QAM', 4, 17.0),
    ('64QAM', 6, 23.0),
)
MODULATION_MARGIN_DB = 2.0
OFDM_GUARD_MHZ = 0.10
OFDM_OOB_ATTENUATION_DB = 40.0
OFDM_SUBCARRIER_COUNTS = (64, 128, 256, 512)
OFDM_SPACINGS_KHZ = (7.8125, 15.625, 31.25)
OFDM_CP_RATIOS = (0.125, 0.25)

# ============================================================
# Archetypes: each defines structure for one scenario family
# ============================================================
ARCHETYPES = {
    'A_psk_qam': {
        'desc': 'QPSK+16QAM overlap, chirp+BPSK, FM, burst QPSK',
        'overlap': {'t1': 'QPSK', 't2': '16QAM', 'fc': (1.2e6, 2.5e6),
                    'sr1': (400e3, 600e3), 'sr2': (300e3, 500e3)},
        'chirp': {'center': (5.5e6, 7.0e6), 'span': (3e6, 5e6)},
        'victim': {'gen': 'BPSK', 'sr': (250e3, 500e3)},
        'extras': [{'gen': 'FM', 'fc': (-7.5e6, -4.5e6), 'dev': (150e3, 400e3)}],
        'burst': {'gen': 'QPSK', 'fc': (-9.5e6, -7.0e6), 'sr': (300e3, 500e3)},
    },
    'B_8psk_qam': {
        'desc': '8PSK+16QAM overlap, chirp+QPSK, FM, burst BPSK',
        'overlap': {'t1': '8PSK', 't2': '16QAM', 'fc': (-3.5e6, -1.5e6),
                    'sr1': (200e3, 400e3), 'sr2': (250e3, 450e3)},
        'chirp': {'center': (5.5e6, 7.5e6), 'span': (3e6, 5e6)},
        'victim': {'gen': 'QPSK', 'sr': (250e3, 450e3)},
        'extras': [{'gen': 'FM', 'fc': (-8.5e6, -5.5e6), 'dev': (150e3, 350e3)}],
        'burst': {'gen': 'BPSK', 'fc': (0.5e6, 2.5e6), 'sr': (400e3, 600e3)},
    },
    'C_psk_psk_neg_chirp': {
        'desc': 'QPSK+8PSK overlap, negative chirp+BPSK, AM, burst QPSK',
        'overlap': {'t1': 'QPSK', 't2': '8PSK', 'fc': (1.5e6, 3.0e6),
                    'sr1': (350e3, 550e3), 'sr2': (200e3, 400e3)},
        'chirp': {'center': (-6.5e6, -5.0e6), 'span': (3e6, 5e6)},
        'victim': {'gen': 'BPSK', 'sr': (250e3, 450e3)},
        'extras': [{'gen': 'AM', 'fc': (5.5e6, 8.0e6), 'depth': (0.3, 0.9)}],
        'burst': {'gen': 'QPSK', 'fc': (-3.5e6, -1.5e6), 'sr': (300e3, 500e3)},
    },
    'D_bpsk_qpsk': {
        'desc': 'BPSK+QPSK overlap, chirp+16QAM, FM, burst 8PSK',
        'overlap': {'t1': 'BPSK', 't2': 'QPSK', 'fc': (1.0e6, 2.5e6),
                    'sr1': (400e3, 600e3), 'sr2': (300e3, 500e3)},
        'chirp': {'center': (6.0e6, 7.5e6), 'span': (3e6, 4.5e6)},
        'victim': {'gen': '16QAM', 'sr': (200e3, 400e3)},
        'extras': [{'gen': 'FM', 'fc': (-7.5e6, -4.5e6), 'dev': (200e3, 400e3)}],
        'burst': {'gen': '8PSK', 'fc': (-9.5e6, -7.0e6), 'sr': (200e3, 400e3)},
    },
    'E_high_qam': {
        'desc': '16QAM+64QAM overlap, chirp+BPSK, FM, burst QPSK',
        'overlap': {'t1': '16QAM', 't2': '64QAM', 'fc': (-3.0e6, -1.0e6),
                    'sr1': (250e3, 450e3), 'sr2': (200e3, 400e3)},
        'chirp': {'center': (5.5e6, 7.0e6), 'span': (3e6, 5e6)},
        'victim': {'gen': 'BPSK', 'sr': (300e3, 500e3)},
        'extras': [{'gen': 'FM', 'fc': (-8.5e6, -5.5e6), 'dev': (150e3, 350e3)}],
        'burst': {'gen': 'QPSK', 'fc': (1.0e6, 3.0e6), 'sr': (300e3, 500e3)},
    },
    'F_dense': {
        'desc': 'QPSK+16QAM overlap, neg chirp+8PSK, FM, AM, burst BPSK (7 signals)',
        'overlap': {'t1': 'QPSK', 't2': '16QAM', 'fc': (1.5e6, 3.0e6),
                    'sr1': (350e3, 550e3), 'sr2': (250e3, 450e3)},
        'chirp': {'center': (-6.5e6, -5.0e6), 'span': (3e6, 5e6)},
        'victim': {'gen': '8PSK', 'sr': (200e3, 350e3)},
        'extras': [
            {'gen': 'FM', 'fc': (5.0e6, 7.5e6), 'dev': (200e3, 400e3)},
            {'gen': 'AM', 'fc': (7.5e6, 9.0e6), 'depth': (0.4, 0.8)},
        ],
        'burst': {'gen': 'BPSK', 'fc': (-3.5e6, -1.5e6), 'sr': (400e3, 600e3)},
    },
    'G_fsk_mix': {
        'desc': '8PSK+BPSK overlap, neg chirp+QPSK, FM, 4FSK, burst 16QAM (7 signals)',
        'overlap': {'t1': '8PSK', 't2': 'BPSK', 'fc': (2.0e6, 3.5e6),
                    'sr1': (200e3, 400e3), 'sr2': (350e3, 550e3)},
        'chirp': {'center': (-6.5e6, -5.0e6), 'span': (2.5e6, 4.5e6)},
        'victim': {'gen': 'QPSK', 'sr': (250e3, 400e3)},
        'extras': [
            {'gen': 'FM', 'fc': (6.0e6, 8.5e6), 'dev': (150e3, 350e3)},
            {'gen': '4FSK', 'fc': (-2.0e6, -0.5e6), 'sr': (100e3, 200e3), 'dev': (100e3, 200e3)},
        ],
        'burst': {'gen': '16QAM', 'fc': (4.5e6, 6.0e6), 'sr': (200e3, 350e3)},
    },
    'H_qpsk_64qam': {
        'desc': 'QPSK+64QAM overlap, chirp+BPSK, FM, 2FSK, burst QPSK (7 signals)',
        'overlap': {'t1': 'QPSK', 't2': '64QAM', 'fc': (-3.0e6, -1.0e6),
                    'sr1': (350e3, 550e3), 'sr2': (200e3, 400e3)},
        'chirp': {'center': (5.5e6, 7.5e6), 'span': (3e6, 5e6)},
        'victim': {'gen': 'BPSK', 'sr': (300e3, 500e3)},
        'extras': [
            {'gen': 'FM', 'fc': (-8.0e6, -5.5e6), 'dev': (200e3, 400e3)},
            {'gen': '2FSK', 'fc': (1.0e6, 3.0e6), 'sr': (150e3, 300e3), 'dev': (100e3, 250e3)},
        ],
        'burst': {'gen': 'QPSK', 'fc': (-9.5e6, -7.0e6), 'sr': (300e3, 500e3)},
    },
}


# ============================================================
# Signal placement helpers
# ============================================================

def _rand_sr(rng, sr_range):
    return rng.choice(np.arange(sr_range[0], sr_range[1] + 1, 50e3))


def _place_overlap_pair(rng, spec):
    """Place two signals with controlled partial overlap (0.10-0.25 MHz)."""
    fc_center = rng.uniform(*spec['fc'])
    sr1 = _rand_sr(rng, spec['sr1'])
    sr2 = _rand_sr(rng, spec['sr2'])
    ro1 = rng.choice([0.25, 0.3, 0.35])
    ro2 = rng.choice([0.25, 0.3, 0.35])
    bw1 = sr1 * (1 + ro1)
    bw2 = sr2 * (1 + ro2)
    desired_ov = rng.uniform(0.10e6, 0.25e6)
    sep = (bw1 / 2 + bw2 / 2) - desired_ov
    fc1 = fc_center - sep / 2
    fc2 = fc_center + sep / 2
    p1 = round(rng.uniform(-28, -32), 1)
    p2 = round(rng.uniform(-33, -38), 1)
    sigs = [
        (spec['t1'], {'fc': fc1, 'sym_rate': sr1, 'rolloff': ro1, 'power_dbm': p1}),
        (spec['t2'], {'fc': fc2, 'sym_rate': sr2, 'rolloff': ro2, 'power_dbm': p2}),
    ]
    return sigs


def _place_chirp_and_victim(rng, chirp_spec, victim_spec):
    """Place chirp + comm signal within sweep range."""
    sc = rng.uniform(*chirp_spec['center'])
    span = rng.uniform(*chirp_spec['span'])
    sw_start = sc - span / 2
    sw_end = sc + span / 2
    cp = round(rng.uniform(-27, -32), 1)

    vsr = _rand_sr(rng, victim_spec['sr'])
    vro = rng.choice([0.25, 0.3, 0.35])
    vbw = vsr * (1 + vro)
    margin = vbw / 2 + 0.2e6
    vfc = rng.uniform(sw_start + margin, sw_end - margin)
    vp = round(rng.uniform(-34, -40), 1)

    sigs = [
        ('Chirp', {'sweep_start': sw_start, 'sweep_end': sw_end, 'power_dbm': cp}),
        (victim_spec['gen'], {
            'fc': vfc, 'sym_rate': vsr, 'rolloff': vro,
            'power_dbm': vp, '_capture_symbols': True,
        }),
    ]
    return sigs


def _place_extra(rng, spec):
    """Place one extra signal (FM, AM, FSK)."""
    gen = spec['gen']
    p = round(rng.uniform(-30, -36), 1)
    if gen == 'FM':
        return (gen, {'fc': rng.uniform(*spec['fc']),
                      'deviation': rng.uniform(*spec['dev']),
                      'mod_freq': rng.uniform(8e3, 20e3),
                      'n_harmonics': rng.randint(1, 4), 'power_dbm': p})
    elif gen == 'AM':
        return (gen, {'fc': rng.uniform(*spec['fc']),
                      'mod_depth': round(rng.uniform(*spec['depth']), 2),
                      'mod_freq': rng.uniform(5e3, 25e3), 'power_dbm': p})
    elif gen in ('2FSK', '4FSK'):
        return (gen, {'fc': rng.uniform(*spec['fc']),
                      'sym_rate': rng.uniform(*spec['sr']),
                      'freq_deviation': rng.uniform(*spec['dev']),
                      'power_dbm': p})
    return None


def _place_burst(rng, spec):
    """Place a burst signal."""
    gen = spec['gen']
    sr = _rand_sr(rng, spec['sr'])
    ro = rng.choice([0.25, 0.3, 0.35])
    fc = rng.uniform(*spec['fc'])
    p = round(rng.uniform(-38, -44), 1)
    start = round(rng.uniform(0.15, 0.35), 2)
    end = round(rng.uniform(start + 0.2, min(start + 0.45, 0.85)), 2)
    return (gen, {'fc': fc, 'sym_rate': sr, 'rolloff': ro, 'power_dbm': p}), start, end


# ============================================================
# Ground truth computation
# ============================================================

def infer_bits_per_symbol(signal):
    """Return bits/symbol for a digital modulation metadata record."""
    bits = signal.get('bits_per_symbol')
    if bits is not None:
        return int(bits)

    modulation = signal.get('type', '').replace(' (burst)', '')
    M = signal.get('M')
    if M and any(family in modulation for family in ('PSK', 'QAM', 'FSK')):
        bits = np.log2(M)
        if np.isclose(bits, round(bits)):
            return int(round(bits))
    return None


def occupied_intervals_mhz(all_signals, fs):
    """Return the merged spectral support inside the observed Nyquist band."""
    band_lo = -fs / 2 / 1e6
    band_hi = fs / 2 / 1e6
    intervals = []
    for signal in all_signals:
        if 'sweep_start_MHz' in signal and 'sweep_end_MHz' in signal:
            lo = min(signal['sweep_start_MHz'], signal['sweep_end_MHz'])
            hi = max(signal['sweep_start_MHz'], signal['sweep_end_MHz'])
        else:
            center = signal['center_frequency_MHz']
            half_bw = signal.get('bandwidth_MHz', 0) / 2
            lo, hi = center - half_bw, center + half_bw
        lo, hi = max(lo, band_lo), min(hi, band_hi)
        if hi > lo:
            intervals.append((lo, hi))

    intervals.sort()
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def signal_interval_mhz(signal):
    """Return one signal's occupied interval in MHz."""
    if 'sweep_start_MHz' in signal and 'sweep_end_MHz' in signal:
        return (
            min(signal['sweep_start_MHz'], signal['sweep_end_MHz']),
            max(signal['sweep_start_MHz'], signal['sweep_end_MHz']),
        )
    center = signal['center_frequency_MHz']
    half_bw = signal.get('bandwidth_MHz', 0) / 2
    return center - half_bw, center + half_bw


def free_intervals_mhz(all_signals, fs):
    """Return unoccupied intervals inside the observed Nyquist band."""
    occupied = occupied_intervals_mhz(all_signals, fs)
    band_lo, band_hi = -fs / 2 / 1e6, fs / 2 / 1e6
    free = []
    cursor = band_lo
    for lo, hi in occupied:
        if lo > cursor:
            free.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < band_hi:
        free.append((cursor, band_hi))
    return free


def interval_overlap_mhz(first, second):
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def evaluate_extraction_mask(target, interferer, passband):
    """Evaluate the public rectangular-mask model used by Q1(c)."""
    target_interval = signal_interval_mhz(target)
    interferer_interval = signal_interval_mhz(interferer)
    target_bw = target_interval[1] - target_interval[0]
    interferer_bw = interferer_interval[1] - interferer_interval[0]
    passband = (min(passband), max(passband))
    target_inband = interval_overlap_mhz(target_interval, passband) / target_bw
    interferer_inband = interval_overlap_mhz(interferer_interval, passband) / interferer_bw
    stopband_gain = 10 ** (-MASK_STOPBAND_ATTENUATION_DB / 10)
    target_fraction = target_inband + (1 - target_inband) * stopband_gain
    interferer_fraction = interferer_inband + (1 - interferer_inband) * stopband_gain
    sir_before = target['power_dBm'] - interferer['power_dBm']
    sir_after = (
        target['power_dBm'] + 10 * np.log10(target_fraction)
        - interferer['power_dBm'] - 10 * np.log10(interferer_fraction)
    )
    return {
        'target_inband_fraction': target_inband,
        'interferer_inband_fraction': interferer_inband,
        'SIR_before_dB': sir_before,
        'SIR_after_dB': sir_after,
        'improvement_dB': sir_after - sir_before,
    }


def optimize_extraction_mask(target, interferer, grid_size=321):
    """Find the best feasible contiguous passband under the public mask model."""
    target_lo, target_hi = signal_interval_mhz(target)
    target_bw = target_hi - target_lo
    grid = np.linspace(target_lo, target_hi, grid_size)
    best = None
    for lo_index, lo in enumerate(grid[:-1]):
        minimum_hi = lo + MASK_TARGET_RETENTION * target_bw
        first_hi = int(np.searchsorted(grid, minimum_hi, side='left'))
        for hi in grid[max(lo_index + 1, first_hi):]:
            metrics = evaluate_extraction_mask(target, interferer, (lo, hi))
            if metrics['target_inband_fraction'] + 1e-9 < MASK_TARGET_RETENTION:
                continue
            candidate = {
                'passband_MHz': [float(lo), float(hi)],
                **metrics,
            }
            if best is None or (
                candidate['SIR_after_dB'], -(hi - lo)
            ) > (
                best['SIR_after_dB'],
                -(best['passband_MHz'][1] - best['passband_MHz'][0]),
            ):
                best = candidate
    return best


def pack_additional_channels(all_signals, fs, channel_bw_mhz):
    """Construct a maximum left-packed placement for Q1(d)."""
    centers = []
    for lo, hi in free_intervals_mhz(all_signals, fs):
        gap_width = hi - lo
        count = int(np.floor(
            (gap_width - PACKING_GUARD_MHZ + 1e-12)
            / (channel_bw_mhz + PACKING_GUARD_MHZ)
        ))
        count = max(0, count)
        first_center = lo + PACKING_GUARD_MHZ + channel_bw_mhz / 2
        centers.extend(
            first_center + index * (channel_bw_mhz + PACKING_GUARD_MHZ)
            for index in range(count)
        )
    return centers


def _interval_overlap(first, second):
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _signal_interval(signal):
    half_bandwidth = signal['bandwidth_MHz'] / 2
    return (
        signal['center_frequency_MHz'] - half_bandwidth,
        signal['center_frequency_MHz'] + half_bandwidth,
    )


def find_chirp_victim(all_signals):
    """Return the digital signal with greatest overlap with the chirp sweep."""
    chirp_index = next(
        index for index, signal in enumerate(all_signals)
        if 'Chirp' in signal.get('type', '')
    )
    chirp = all_signals[chirp_index]
    chirp_interval = tuple(sorted([
        chirp['sweep_start_MHz'], chirp['sweep_end_MHz']
    ]))
    candidates = []
    for index, signal in enumerate(all_signals):
        if index == chirp_index or infer_bits_per_symbol(signal) is None:
            continue
        overlap = _interval_overlap(chirp_interval, _signal_interval(signal))
        candidates.append((overlap, -index, index))
    if not candidates or max(candidates)[0] <= 0:
        return chirp_index, None, 0.0
    overlap, _, victim_index = max(candidates)
    return chirp_index, victim_index, overlap


def compute_chirp_crossing(chirp, victim, fs, N):
    """Build time-localized, symbol-level truth for the chirp crossing."""
    duration_s = N / fs
    sweep_start = chirp['sweep_start_MHz']
    sweep_end = chirp['sweep_end_MHz']
    sweep_rate = (sweep_end - sweep_start) / duration_s
    victim_lo, victim_hi = _signal_interval(victim)
    crossings = sorted([
        (victim_lo - sweep_start) / sweep_rate,
        (victim_hi - sweep_start) / sweep_rate,
    ])
    entry_s = max(0.0, crossings[0])
    exit_s = min(duration_s, crossings[1])
    if exit_s <= entry_s:
        raise RuntimeError('chirp does not cross the selected victim')

    source_symbols = victim.get('_source_symbols_iq')
    samples_per_symbol = victim.get('_samples_per_symbol')
    if not source_symbols or not samples_per_symbol:
        raise RuntimeError('Q2 victim is missing hidden source symbols')

    anchor = int(np.ceil(entry_s * fs / samples_per_symbol - 1e-12))
    windows = []
    for lag in range(-Q2_SYMBOL_ALIGNMENT_LAG, Q2_SYMBOL_ALIGNMENT_LAG + 1):
        start = anchor + lag
        stop = start + Q2_SYMBOL_COUNT
        if start >= 0 and stop <= len(source_symbols):
            windows.append(source_symbols[start:stop])
    if len(windows) != 2 * Q2_SYMBOL_ALIGNMENT_LAG + 1:
        raise RuntimeError('Q2 symbol recovery window is outside the recording')

    constellation = np.asarray(source_symbols, dtype=float)
    constellation = constellation[:, 0] + 1j * constellation[:, 1]
    unique = np.unique(np.round(constellation, decimals=10))
    distances = np.abs(unique[:, None] - unique[None, :])
    minimum_distance = np.min(distances[distances > 1e-9])
    return {
        'entry_time_ms': entry_s * 1e3,
        'exit_time_ms': exit_s * 1e3,
        'duration_ms': (exit_s - entry_s) * 1e3,
        'symbol_anchor_index': anchor,
        'symbol_count': Q2_SYMBOL_COUNT,
        'modulation_order': int(victim['M']),
        'minimum_constellation_distance': float(minimum_distance),
        'symbol_windows_iq': windows,
        'symbols_iq': windows[Q2_SYMBOL_ALIGNMENT_LAG],
    }


def optimize_ofdm_design(all_signals, noise_dbm, fs):
    """Exhaustively solve the finite OFDM design space used by Q3(d)."""
    gaps = free_intervals_mhz(all_signals, fs)
    if not gaps:
        return {}
    gap_lo, gap_hi = max(gaps, key=lambda item: item[1] - item[0])
    leakage_limit_dbm = noise_dbm + 10 * np.log10(1e6 / fs)
    maximum_power_dbm = leakage_limit_dbm + OFDM_OOB_ATTENUATION_DB
    candidates = []
    for n_sc in OFDM_SUBCARRIER_COUNTS:
        for spacing_khz in OFDM_SPACINGS_KHZ:
            bandwidth_mhz = n_sc * spacing_khz / 1000
            if bandwidth_mhz + 2 * OFDM_GUARD_MHZ > gap_hi - gap_lo + 1e-12:
                continue
            noise_inband_dbm = noise_dbm + 10 * np.log10(
                bandwidth_mhz * 1e6 / fs
            )
            for cp_ratio in OFDM_CP_RATIOS:
                for modulation, bits, threshold_db in MODULATION_OPTIONS:
                    minimum_power_dbm = (
                        noise_inband_dbm + threshold_db + MODULATION_MARGIN_DB
                    )
                    if minimum_power_dbm > maximum_power_dbm + 1e-12:
                        continue
                    rate_mbps = (
                        n_sc * bits * spacing_khz * 1e3
                        / (1 + cp_ratio) / 1e6
                    )
                    candidates.append({
                        'center_MHz': (gap_lo + gap_hi) / 2,
                        'n_sc': n_sc,
                        'sc_spacing_kHz': spacing_khz,
                        'cp_ratio': cp_ratio,
                        'mod': modulation,
                        'bits': bits,
                        'bw_MHz': bandwidth_mhz,
                        'rate_Mbps': rate_mbps,
                        'power_dBm': minimum_power_dbm,
                        'SNR_dB': threshold_db + MODULATION_MARGIN_DB,
                        'adjacent_leakage_dBm': (
                            minimum_power_dbm - OFDM_OOB_ATTENUATION_DB
                        ),
                    })
    if not candidates:
        return {
            'gap_MHz': [gap_lo, gap_hi],
            'gap_bw_MHz': gap_hi - gap_lo,
            'leakage_limit_dBm': leakage_limit_dbm,
            'design': None,
        }
    best = max(
        candidates,
        key=lambda item: (
            item['rate_Mbps'], -item['power_dBm'], -item['bw_MHz']
        ),
    )
    return {
        'gap_MHz': [gap_lo, gap_hi],
        'gap_bw_MHz': gap_hi - gap_lo,
        'guard_MHz': OFDM_GUARD_MHZ,
        'oob_attenuation_dB': OFDM_OOB_ATTENUATION_DB,
        'leakage_limit_dBm': leakage_limit_dbm,
        'maximum_power_dBm': maximum_power_dbm,
        'design': best,
    }


def extract_question_sections(question_text):
    """Extract exact Q1-Q3 text for section-level metadata."""
    markers = {
        'Q1': '===== Question 1:',
        'Q2': '===== Question 2:',
        'Q3': '===== Question 3:',
    }
    starts = {qid: question_text.index(marker) for qid, marker in markers.items()}
    end_marker = 'Briefly provide the signal evidence'
    ends = {
        'Q1': starts['Q2'],
        'Q2': starts['Q3'],
        'Q3': question_text.index(end_marker),
    }
    return {qid: question_text[starts[qid]:ends[qid]].strip()
            for qid in markers}

def compute_ground_truth(all_signals, noise_dbm, fs, N):
    """Compute all Q1-Q3 ground truth from signal metadata."""
    npsd = noise_dbm - 10 * np.log10(fs)  # dBm/Hz
    gt = {}

    # ---- Q1(a): signal list ----
    gt['Q1a'] = [
        {k: s[k] for k in ['type', 'center_frequency_MHz', 'bandwidth_MHz', 'power_dBm']
         if k in s}
        | ({'duty_cycle': s['duty_cycle']} if 'duty_cycle' in s else {})
        | {'signal_id': i + 1}
        for i, s in enumerate(all_signals)
    ]

    # ---- Q1(b): overlap pair (exclude chirp) ----
    best_ov, pair = 0, None
    for i in range(len(all_signals)):
        for j in range(i + 1, len(all_signals)):
            si, sj = all_signals[i], all_signals[j]
            if 'Chirp' in si.get('type', '') or 'Chirp' in sj.get('type', ''):
                continue
            ov = max(0, (si.get('bandwidth_MHz', 0) / 2 + sj.get('bandwidth_MHz', 0) / 2)
                     - abs(si['center_frequency_MHz'] - sj['center_frequency_MHz']))
            if ov > best_ov:
                best_ov, pair = ov, (i, j)

    si_idx = wi_idx = None
    sir_w = None
    if pair:
        a, b = all_signals[pair[0]], all_signals[pair[1]]
        if a['power_dBm'] >= b['power_dBm']:
            si_idx, wi_idx = pair
        else:
            si_idx, wi_idx = pair[1], pair[0]
        sir_w = all_signals[wi_idx]['power_dBm'] - all_signals[si_idx]['power_dBm']

    gt['Q1b'] = {
        'pair_ids': [pair[0] + 1, pair[1] + 1] if pair else None,
        'pair_types': [all_signals[pair[0]]['type'], all_signals[pair[1]]['type']] if pair else None,
        'target_id': wi_idx + 1 if wi_idx is not None else None,
        'interferer_id': si_idx + 1 if si_idx is not None else None,
        'overlap_MHz': round(best_ov, 3),
        'SIR_weak_dB': round(sir_w, 1) if sir_w is not None else None,
    }

    # ---- Q1(c): constrained spectral extraction mask ----
    if pair and wi_idx is not None:
        target = all_signals[wi_idx]
        interferer = all_signals[si_idx]
        optimum = optimize_extraction_mask(target, interferer)
        gt['Q1c'] = {
            'target_id': wi_idx + 1,
            'interferer_id': si_idx + 1,
            'minimum_target_retention': MASK_TARGET_RETENTION,
            'stopband_attenuation_dB': MASK_STOPBAND_ATTENUATION_DB,
            'optimal_passband_MHz': [
                round(value, 4) for value in optimum['passband_MHz']
            ],
            'target_inband_fraction': round(
                optimum['target_inband_fraction'], 4
            ),
            'interferer_inband_fraction': round(
                optimum['interferer_inband_fraction'], 4
            ),
            'SIR_before_dB': round(optimum['SIR_before_dB'], 2),
            'optimal_SIR_after_dB': round(optimum['SIR_after_dB'], 2),
            'optimal_improvement_dB': round(optimum['improvement_dB'], 2),
        }
    else:
        gt['Q1c'] = {}

    # ---- Q1(d): spectral occupancy and guarded channel placement ----
    occupied = occupied_intervals_mhz(all_signals, fs)
    total_bw = sum(hi - lo for lo, hi in occupied)
    gt['Q1d'] = {
        'total_bw_MHz': round(total_bw, 2),
        'available_MHz': fs / 1e6,
        'occupancy_pct': round(total_bw / (fs / 1e6) * 100, 1),
    }
    if pair and wi_idx is not None:
        channel_bw = all_signals[wi_idx]['bandwidth_MHz']
        packed_centers = pack_additional_channels(all_signals, fs, channel_bw)
        gt['Q1d']['packing'] = {
            'channel_bandwidth_MHz': round(channel_bw, 4),
            'guard_MHz': PACKING_GUARD_MHZ,
            'required_count': len(all_signals),
            'maximum_count': len(packed_centers),
            'reference_centers_MHz': [
                round(center, 4) for center in packed_centers[:len(all_signals)]
            ],
        }

    # ---- Q2: radar-communication waveform recovery ----
    chirp_index, victim_index, overlap_mhz = find_chirp_victim(all_signals)
    chirp = all_signals[chirp_index]
    victim = all_signals[victim_index]
    gt['Q2a'] = {k: chirp[k] for k in [
        'sweep_start_MHz', 'sweep_end_MHz', 'bandwidth_MHz',
        'chirp_rate_MHz_per_ms', 'time_bandwidth_product',
        'processing_gain_dB',
    ]}

    victim_bandwidth = victim['bandwidth_MHz']
    gt['Q2b'] = {
        'victim_id': victim_index + 1,
        'victim_type': victim['type'],
        'center_MHz': round(victim['center_frequency_MHz'], 6),
        'bandwidth_MHz': round(victim_bandwidth, 6),
        'symbol_rate_kHz': round(victim['symbol_rate_kHz'], 3),
        'rolloff': round(victim['rolloff'], 3),
        'power_dBm': round(victim['power_dBm'], 2),
        'overlap_MHz': round(overlap_mhz, 6),
        'overlap_fraction': round(overlap_mhz / victim_bandwidth, 6),
    }
    crossing = compute_chirp_crossing(chirp, victim, fs, N)
    gt['Q2c'] = {
        'entry_time_ms': round(crossing['entry_time_ms'], 6),
        'exit_time_ms': round(crossing['exit_time_ms'], 6),
        'duration_ms': round(crossing['duration_ms'], 6),
    }
    gt['Q2d'] = {
        'victim_id': victim_index + 1,
        'symbol_anchor_index': crossing['symbol_anchor_index'],
        'symbol_count': crossing['symbol_count'],
        'modulation_order': crossing['modulation_order'],
        'minimum_constellation_distance': round(
            crossing['minimum_constellation_distance'], 10
        ),
        'symbol_windows_iq': crossing['symbol_windows_iq'],
        'symbols_iq': crossing['symbols_iq'],
    }

    # ---- Q3(a-d): digital capacity analysis ----
    digital = [(i, s, infer_bits_per_symbol(s))
               for i, s in enumerate(all_signals)
               if infer_bits_per_symbol(s) is not None]

    q3a = []
    for idx, s, bits_per_symbol in digital:
        bw = s['bandwidth_MHz'] * 1e6
        nib = npsd + 10 * np.log10(bw)
        snr = s['power_dBm'] - nib
        snr_l = 10 ** (snr / 10)
        cap = bw * np.log2(1 + snr_l)
        sr = s.get('symbol_rate_kHz', 0) * 1e3
        se = sr * bits_per_symbol / bw if bw > 0 else 0
        sh = np.log2(1 + snr_l)
        q3a.append({
            'id': idx + 1, 'type': s['type'], 'bw_MHz': s['bandwidth_MHz'],
            'SNR_dB': round(snr, 1), 'cap_Mbps': round(cap / 1e6, 2),
            'se_actual': round(se, 2), 'se_shannon': round(sh, 2),
            'gap': round(sh - se, 2),
        })
    gt['Q3a'] = q3a

    # Q3(b)
    if q3a:
        worst = max(q3a, key=lambda x: x['gap'])
        cur_snr = worst['SNR_dB']
        old_b = next(bits for i, s, bits in digital if i + 1 == worst['id'])
        rec = ('BPSK', 1)
        for m, b, required_snr in reversed(MODULATION_OPTIONS):
            if cur_snr >= required_snr + MODULATION_MARGIN_DB:
                rec = (m, b)
                break
        gt['Q3b'] = {
            'worst_id': worst['id'], 'worst_type': worst['type'],
            'cur_bits': old_b, 'rec_mod': rec[0], 'rec_bits': rec[1],
            'improvement_pct': round((rec[1] - old_b) / old_b * 100, 1) if old_b else 0,
        }
    else:
        gt['Q3b'] = {}

    # Q3(c) water-filling
    if len(digital) >= 2:
        ch = []
        for idx, s, _ in digital:
            bw = s['bandwidth_MHz'] * 1e6
            n_mw = 10 ** ((npsd + 10 * np.log10(bw)) / 10)
            p_mw = 10 ** (s['power_dBm'] / 10)
            ch.append({'idx': idx, 'type': s['type'], 'bw': bw,
                       'n_mw': n_mw, 'p_mw': p_mw})
        Ptot = sum(c['p_mw'] for c in ch)
        nc = len(ch)
        bandwidths = np.array([c['bw'] for c in ch])
        noise_psd = np.array([c['n_mw'] / c['bw'] for c in ch])
        active = np.ones(nc, dtype=bool)
        mu = 0.0
        for _ in range(nc):
            mu = ((Ptot + np.sum(bandwidths[active] * noise_psd[active]))
                  / np.sum(bandwidths[active]))
            next_active = active & (mu > noise_psd)
            if np.array_equal(next_active, active):
                break
            active = next_active
        wf = bandwidths * np.maximum(mu - noise_psd, 0)
        cb_tot = sum(c['bw'] * np.log2(1 + c['p_mw'] / c['n_mw']) for c in ch)
        ca_tot = sum(ch[i]['bw'] * np.log2(1 + wf[i] / ch[i]['n_mw'])
                     for i in range(nc) if wf[i] > 0)
        alloc = []
        for ci, c in enumerate(ch):
            alloc.append({
                'id': c['idx'] + 1, 'type': c['type'],
                'old_dBm': round(10 * np.log10(c['p_mw']), 1),
                'new_dBm': round(10 * np.log10(wf[ci]), 1) if wf[ci] > 0 else None,
            })
        gt['Q3c'] = {
            'total_power_dBm': round(10 * np.log10(Ptot), 1),
            'water_level_dBm_per_Hz': round(10 * np.log10(mu), 1) if mu > 0 else None,
            'allocation': alloc,
            'cap_before_Mbps': round(cb_tot / 1e6, 2),
            'cap_after_Mbps': round(ca_tot / 1e6, 2),
            'improvement_pct': round((ca_tot - cb_tot) / cb_tot * 100, 1) if cb_tot > 0 else 0,
        }
    else:
        gt['Q3c'] = {}

    # Q3(d) constrained OFDM design in the largest gap
    ofdm = optimize_ofdm_design(all_signals, noise_dbm, fs)
    gt['Q3d'] = {
        key: (
            {sub_key: round(sub_value, 6) if isinstance(sub_value, float) else sub_value
             for sub_key, sub_value in value.items()}
            if isinstance(value, dict) else
            [round(item, 6) for item in value]
            if isinstance(value, list) else
            round(value, 6) if isinstance(value, float) else value
        )
        for key, value in ofdm.items()
    }

    return gt


# ============================================================
# Question text template
# ============================================================

def build_question(sample_id, fs, N, noise_dbm):
    duration = N / fs
    return f"""You are an electromagnetic signal analysis expert. Below is I/Q signal data captured in a complex electromagnetic environment.

Signal file: {sample_id}.npy
Sampling rate: {fs / 1e6:.0f} MHz
Number of samples: {N}
Data format: complex64 (numpy)
Amplitude calibration: samples are in sqrt(watt), so mean(abs(x)^2) gives average power in watts.
Recording duration: {duration * 1e3:.4f} ms

Multiple independent signal sources exist within this frequency band, potentially including digitally modulated, analog modulated, radar, and burst signals.
Spectral overlap between signals may be present. Please perform a comprehensive spectrum analysis on your own, then answer the following three comprehensive questions.

===== Question 1: Comprehensive Spectrum Situational Awareness and Interference Mitigation (34 pts) =====

(a) Identify all independent signal sources in the band (including possible low-power or burst signals), and provide for each signal:
    center frequency (MHz), occupied bandwidth (MHz), modulation type, estimated power (dBm).
    For a burst signal, report its power during the active interval rather than its average over the full recording. Do not miss any signals. (12 pts)

(b) Find the pair of non-chirp signals with the largest occupied-band overlap.
    Calculate their overlap bandwidth (MHz) and SIR (dB), using the weaker signal as the target. If a burst is involved, use its active-interval power. Reuse the signal IDs assigned in (a). (8 pts)

(c) Design one contiguous spectral extraction passband for the weaker signal in (b) that maximizes post-filter SIR while retaining at least {MASK_TARGET_RETENTION * 100:.0f}% of its occupied bandwidth. For a reproducible comparison, assume an ideal rectangular mask with 0 dB passband gain and {MASK_STOPBAND_ATTENUATION_DB:.0f} dB stopband attenuation, and approximate each signal as having uniform power spectral density within its occupied bandwidth. Report the passband edges, retained target fraction, post-filter SIR, and SIR improvement. Alternative passbands are accepted according to their verified performance. (8 pts)

(d) Calculate the union spectral occupancy of the entire +/-{fs / 2 / 1e6:.0f} MHz band, counting overlapping frequencies only once. Then assess whether the band can support one additional channel for every source identified in (a). Use the target bandwidth from (b) for each new channel and maintain a {PACKING_GUARD_MHZ:.2f} MHz guard from existing signals, other new channels, and the band edges. Report the number that can be placed and one feasible list of center frequencies. (6 pts)

===== Question 2: Radar-Communication Coexistence and Waveform Recovery (33 pts) =====

(a) Locate the linear frequency modulated (LFM/Chirp) signal in the band and estimate its:
    sweep start and stop frequencies (MHz), sweep bandwidth (MHz), sweep rate (MHz/ms), and time-bandwidth product (TBP). (7 pts)

(b) Among the digitally modulated signals, identify the one with the largest occupied-band overlap with the chirp sweep. Separate the chirp from this communication waveform, then report its signal ID, center frequency, occupied bandwidth, modulation type, symbol rate, pulse-shaping rolloff, and power. Reuse the signal ID assigned in Question 1(a). (12 pts)

(c) Determine when the chirp instantaneous frequency crosses the occupied band of the communication signal identified in (b). Report the entry time, exit time, and crossing duration in milliseconds from the start of the recording. (4 pts)

(d) Recover the first {Q2_SYMBOL_COUNT} consecutive communication symbols whose nominal symbol intervals begin at or after the entry time in (c). Report the symbols in transmission order as normalized [I, Q] pairs. A common complex scale, phase rotation, and complex conjugation are accepted during verification. (10 pts)

===== Question 3: Multi-Link Capacity Optimization and New Link Deployment (33 pts) =====

(a) Identify all digitally modulated signals in the band, and for each signal calculate:
    - In-band SNR (dB)
    - Shannon channel capacity (Mbps)
    - Actual spectral efficiency (bps/Hz) of the current modulation scheme vs. Shannon limit spectral efficiency
    The noise floor is {noise_dbm} dBm/{fs / 1e6:.0f}MHz. For a burst link, evaluate SNR, capacity, and spectral efficiency during its active interval without multiplying by duty cycle. (9 pts)

(b) Which digital signal has the spectral efficiency furthest from the Shannon limit?
    Recommend the highest-rate modem profile supported by its current SNR with a {MODULATION_MARGIN_DB:.0f} dB implementation margin. The available profiles and required SNRs are BPSK 8 dB, QPSK 11 dB, 8PSK 14 dB, 16QAM 17 dB, and 64QAM 23 dB. Report the recommended modulation and throughput increase relative to the current bits per symbol. (8 pts)

(c) Assume all digital signals share a fixed total power budget (= sum of current powers of all digital signals).
    For a burst link, use its active-interval power in this budget. Apply bandwidth-aware water filling to maximize aggregate Shannon capacity. Report the water level, the new power allocation for each signal, capacities before and after allocation, and the total improvement percentage. (9 pts)

(d) In the largest contiguous unoccupied spectral gap within the band, design an OFDM communication link:
    maximize its net rate using Nsc in {list(OFDM_SUBCARRIER_COUNTS)}, subcarrier spacing in {list(OFDM_SPACINGS_KHZ)} kHz, CP ratio in {list(OFDM_CP_RATIOS)}, and a modem profile from (b). Maintain a {OFDM_GUARD_MHZ:.2f} MHz guard at both gap edges, use a {MODULATION_MARGIN_DB:.0f} dB implementation margin, and assume {OFDM_OOB_ATTENUATION_DB:.0f} dB out-of-band attenuation. Adjacent-channel leakage must not exceed the measured noise power in 1 MHz, and the link must provide at least 2 Mbps. Report the gap, link placement, selected parameters, received power, occupied bandwidth, net rate, and leakage. Alternative designs are accepted according to verified feasibility and rate. (7 pts)

Briefly provide the signal evidence and calculations supporting each answer before the final block. Only the machine-readable fields in the final block are scored. Signal IDs are arbitrary strings, but the same IDs must be reused across all sub-questions. Replace every null below with a number, string, or list as appropriate, and output valid JSON without comments.

===ANSWERS===
{{
  "schema_version": "{SCHEMA_VERSION}",
  "Q1a": {{"signals": [{{"id": "S1", "center_MHz": null, "bandwidth_MHz": null, "modulation": null, "power_dBm": null}}]}},
  "Q1b": {{"pair_ids": [null, null], "target_id": null, "overlap_MHz": null, "sir_dB": null}},
  "Q1c": {{"target_id": null, "passband_MHz": [null, null], "target_retained_fraction": null, "post_sir_dB": null, "improvement_dB": null}},
  "Q1d": {{"total_occupied_MHz": null, "occupancy_pct": null, "additional_channel_count": null, "additional_centers_MHz": [null]}},
  "Q2a": {{"signal_id": null, "sweep_start_MHz": null, "sweep_end_MHz": null, "bandwidth_MHz": null, "chirp_rate_MHz_per_ms": null, "tbp": null}},
  "Q2b": {{"victim_id": null, "center_MHz": null, "bandwidth_MHz": null, "modulation": null, "symbol_rate_kHz": null, "rolloff": null, "power_dBm": null}},
  "Q2c": {{"entry_time_ms": null, "exit_time_ms": null, "duration_ms": null}},
  "Q2d": {{"victim_id": null, "symbols_iq": [[null, null]]}},
  "Q3a": {{"links": [{{"id": null, "snr_dB": null, "capacity_Mbps": null, "actual_efficiency_bps_per_Hz": null, "shannon_efficiency_bps_per_Hz": null}}]}},
  "Q3b": {{"worst_id": null, "recommended_modulation": null, "bits_per_symbol": null, "throughput_improvement_pct": null}},
  "Q3c": {{"water_level_dBm_per_Hz": null, "allocations": [{{"id": null, "power_dBm": null}}], "capacity_before_Mbps": null, "capacity_after_Mbps": null, "improvement_pct": null}},
  "Q3d": {{"gap_MHz": [null, null], "center_MHz": null, "n_subcarriers": null, "spacing_kHz": null, "cp_ratio": null, "modulation": null, "power_dBm": null, "occupied_bandwidth_MHz": null, "rate_Mbps": null, "adjacent_leakage_dBm": null}}
}}
===END===
"""


def build_rubric(all_signals):
    """Build the fixed 34/33/33 deterministic L5 rubric."""
    n_digital = sum(
        infer_bits_per_symbol(signal) is not None for signal in all_signals
    )
    return [
        {'id': 'Q1', 'title': 'Comprehensive Spectrum Situational Awareness and Interference Mitigation', 'rubric': {
            'points': 34,
            'Q1a': {'pts': 12, 'n_signals': len(all_signals)},
            'Q1b': {'pts': 8, 'tol_overlap': '±0.05MHz', 'tol_SIR': '±3dB'},
            'Q1c': {'pts': 8, 'method': 'deterministic_spectral_mask'},
            'Q1d': {'pts': 6, 'method': 'deterministic_guarded_packing'},
        }},
        {'id': 'Q2', 'title': 'Radar-Communication Coexistence and Waveform Recovery', 'rubric': {
            'points': 33,
            'Q2a': {'pts': 7, 'method': 'measured_chirp_characterization'},
            'Q2b': {'pts': 12, 'method': 'overlapped_waveform_recovery'},
            'Q2c': {'pts': 4, 'method': 'time_frequency_crossing'},
            'Q2d': {'pts': 10, 'method': 'hidden_symbol_sequence_verification'},
        }},
        {'id': 'Q3', 'title': 'Multi-Link Capacity Optimization and New Link Deployment', 'rubric': {
            'points': 33,
            'Q3a': {'pts': 9, 'n_digital': n_digital},
            'Q3b': {'pts': 8},
            'Q3c': {'pts': 9, 'tol': '±10%'},
            'Q3d': {'pts': 7, 'method': 'deterministic_ofdm_verification'},
        }},
    ]


# ============================================================
# Problem generation
# ============================================================

def generate_one_problem(seed, arch_name, arch, output_dir, fs=20e6, N=65536):
    rng = np.random.RandomState(seed)
    t = np.arange(N) / fs
    duration = N / fs

    # Collect signal specs
    sig_specs = []
    # 1. Overlap pair
    sig_specs.extend(_place_overlap_pair(rng, arch['overlap']))

    # 2. Chirp + victim
    sig_specs.extend(_place_chirp_and_victim(rng, arch['chirp'], arch['victim']))

    # 3. Extras
    for ext in arch['extras']:
        s = _place_extra(rng, ext)
        if s:
            sig_specs.append(s)

    # 4. Burst (remember index for burst processing)
    burst_spec, b_start, b_end = _place_burst(rng, arch['burst'])
    burst_idx = len(sig_specs)
    sig_specs.append(burst_spec)

    # Generate signals
    all_signals = []
    mixed = np.zeros(N, dtype=complex)
    for i, (gen_type, params) in enumerate(sig_specs):
        gen_func = SIGNAL_GENERATORS[gen_type]
        sig, meta = gen_func(rng, fs, N, t, params)

        if i == burst_idx:
            sig, bm = apply_burst(sig, rng, N,
                                  {'start_frac': b_start, 'end_frac': b_end})
            meta.update(bm)
            meta['type'] = meta['type'] + ' (burst)'

        mixed += sig
        all_signals.append(meta)

    # Noise
    noise_dbm = round(rng.uniform(-48, -53), 0)
    noise_power = 10 ** (noise_dbm / 10) / 1000
    noise = np.sqrt(noise_power / 2) * (rng.randn(N) + 1j * rng.randn(N))
    received = mixed + noise

    # Save signal
    sample_id = f"EMRB_L5_{seed:04d}"
    np.save(os.path.join(output_dir, f"{sample_id}.npy"), received.astype(np.complex64))

    # Ground truth
    gt = compute_ground_truth(all_signals, noise_dbm, fs, N)

    # Question text
    q_text = build_question(sample_id, fs, N, noise_dbm)
    question_sections = extract_question_sections(q_text)

    rubric = build_rubric(all_signals)

    # Merge rubric + ground truth
    questions = []
    for r in rubric:
        qid = r['id']
        questions.append({**r, 'question': question_sections[qid], 'ground_truth': {
            k: v for k, v in gt.items() if k.startswith(qid)
        }})

    public_signals = [
        {key: value for key, value in signal.items() if not key.startswith('_')}
        for signal in all_signals
    ]
    metadata = {
        'sample_id': sample_id, 'level': 'L5',
        'answer_schema_version': SCHEMA_VERSION,
        'archetype': arch_name, 'archetype_desc': arch['desc'],
        'total_points': 100, 'num_questions': 3,
        'question': q_text, 'questions': questions,
        'generation_params': {
            'fs': fs, 'N': N, 'duration_ms': round(duration * 1e3, 4),
            'signals': public_signals,
            'noise_floor_dBm': noise_dbm,
            'seed': seed,
        },
        'verification': {
            'scoring': 'deterministic',
            'mask_target_retention': MASK_TARGET_RETENTION,
            'mask_stopband_attenuation_dB': MASK_STOPBAND_ATTENUATION_DB,
            'packing_guard_MHz': PACKING_GUARD_MHZ,
            'q2_symbol_count': Q2_SYMBOL_COUNT,
            'q2_symbol_alignment_lag': Q2_SYMBOL_ALIGNMENT_LAG,
            'modulation_margin_dB': MODULATION_MARGIN_DB,
            'ofdm_guard_MHz': OFDM_GUARD_MHZ,
            'ofdm_oob_attenuation_dB': OFDM_OOB_ATTENUATION_DB,
        },
    }

    with open(os.path.join(output_dir, f"{sample_id}.json"), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    return metadata


# ============================================================
# Batch orchestrator
# ============================================================

def generate_batch(num=40, seed_start=2000, output_dir='data/L5'):
    os.makedirs(output_dir, exist_ok=True)
    arch_names = list(ARCHETYPES.keys())
    per = num // len(arch_names)
    rem = num % len(arch_names)

    manifest = {
        'total': num, 'problems': [],
        'arch_counts': {}, 'level': 'L5',
    }

    idx = 0
    for ai, aname in enumerate(arch_names):
        n = per + (1 if ai < rem else 0)
        manifest['arch_counts'][aname] = n
        for i in range(n):
            seed = seed_start + idx
            print(f"[{idx + 1}/{num}] {aname} seed={seed}...", end=' ')
            try:
                meta = generate_one_problem(seed, aname, ARCHETYPES[aname], output_dir)
                ns = len(meta['generation_params']['signals'])
                print(f"OK ({ns} signals)")
                manifest['problems'].append({
                    'sample_id': meta['sample_id'],
                    'archetype': aname,
                    'num_signals': ns,
                    'total_points': 100,
                })
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
            idx += 1

    with open(os.path.join(output_dir, 'batch_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generated {idx} L5 problems in {output_dir}/")
    print(f"Archetype distribution: {manifest['arch_counts']}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num', type=int, default=40)
    p.add_argument('--seed-start', type=int, default=2000)
    p.add_argument('--output', type=str, default='data/L5')
    args = p.parse_args()
    generate_batch(args.num, args.seed_start, args.output)
