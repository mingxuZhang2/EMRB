"""
EMRB L2 Batch Generator: Tests signal processing METHODOLOGY.
5 questions × 20pts = 100pts.  Target GPT: 80-90.
8 archetypes × 5 = 40 problems.

Q1: Spectral analysis parameters & windowing (FFT resolution, windowing, Welch)
Q2: Bandwidth definition comparison          (3dB, 99%, null-to-null bandwidth)
Q3: Autocorrelation & periodicity analysis   (autocorrelation, periodicity detection)
Q4: Time-frequency analysis (STFT/Spectrogram) (chirp sweep rate, burst timing, uncertainty)
Q5: Signal energy & power relationships      (energy, Eb, total power, dBm vs mW)
"""
import numpy as np
import json
import os

from scipy.signal import find_peaks

from generation.signal_library import SIGNAL_GENERATORS, apply_burst

# Q3 (autocorrelation) answer-format marker and oracle constants. The oracle
# accepts a scenario only if the received signal's |R(tau)| comb is unambiguous
# under these thresholds; otherwise the whole sample is regenerated with a new
# rng seed. The deterministic scorer in evaluation/l2_verifier.py keys on
# AUTOCORR_SCORING via the question rubric.
AUTOCORR_SCHEMA = 'emrb-l2-autocorr-v1'
# single-sourced so the stamped marker can never drift from the scorer
from evaluation.l2_verifier import SCORER_VERSION as AUTOCORR_SCORING
# Delays below 20 us are dominated by pulse-shaping tails and circular chirp
# correlation sidelobes; the comb period 1/f_mod is always >= 50 us.
ACORR_MIN_LAG_US = 20.0
ACORR_HARMONIC_TOL = 0.03
ACORR_COMPETITOR_EXCLUSION = 0.08
ACORR_REQUIRED_HARMONICS = (1, 2, 3)
ACORR_MIN_COMB_PROMINENCE = 0.02
ACORR_COMPETITOR_MARGIN = 0.5
ACORR_COMPETITOR_HEIGHT_FRACTION = 0.5
ACORR_FILTER_BW_FACTOR = 1.25
ACORR_FILTER_GUARD_MHZ = 0.05
ACORR_FILTERED_MIN_PROMINENCE = 0.05
ACORR_FILTERED_PERSIST_RATIO = 2.0

# ============================================================
# Archetypes: 3-4 signals per scenario
# ============================================================
ARCHETYPES = {
    'A_qpsk_fm_chirp': {
        'q3': ('FM', 'digital'),
        'desc': 'QPSK + FM + Chirp',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (200e3, 500e3)},
            {'gen': 'FM', 'fc_range': (-6.5e6, -3.5e6), 'dev_range': (150e3, 350e3)},
            {'gen': 'Chirp', 'sweep_center': (5.5e6, 7.5e6), 'sweep_span': (2.5e6, 4.5e6)},
        ],
        'burst': None,
    },
    'B_16qam_fm_chirp_burst': {
        'q3': ('FM', 'source'),
        'desc': '16QAM + FM + Chirp + burst_BPSK',
        'signals': [
            {'gen': '16QAM', 'fc_range': (-3.0e6, -1.0e6), 'sr_range': (200e3, 400e3)},
            {'gen': 'FM', 'fc_range': (-8.0e6, -5.0e6), 'dev_range': (150e3, 350e3)},
            {'gen': 'Chirp', 'sweep_center': (5.5e6, 7.5e6), 'sweep_span': (2.5e6, 4.5e6)},
            {'gen': 'BPSK', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (300e3, 600e3),
             'burst': True},
        ],
        'burst': {'start_range': (0.15, 0.35), 'end_range': (0.65, 0.85)},
    },
    'C_bpsk_am_chirp': {
        'q3': ('AM', 'digital'),
        'desc': 'BPSK + AM(comb source) + Chirp',
        'signals': [
            {'gen': 'BPSK', 'fc_range': (1.5e6, 3.5e6), 'sr_range': (200e3, 500e3),
             'power_range': (-32, -38)},
            {'gen': 'AM', 'fc_range': (-6.0e6, -3.0e6), 'depth_range': (0.8, 0.95),
             'power_range': (-22, -26)},
            {'gen': 'Chirp', 'sweep_center': (-7.5e6, -5.5e6), 'sweep_span': (2.5e6, 4.5e6),
             'power_range': (-32, -38)},
        ],
        'burst': None,
    },
    'D_8psk_fm_chirp_burst': {
        'q3': ('FM', 'digital'),
        'desc': '8PSK + FM + Chirp + burst_QPSK',
        'signals': [
            {'gen': '8PSK', 'fc_range': (-3.0e6, -1.0e6), 'sr_range': (150e3, 350e3)},
            {'gen': 'FM', 'fc_range': (4.0e6, 7.0e6), 'dev_range': (200e3, 400e3)},
            {'gen': 'Chirp', 'sweep_center': (-7.0e6, -5.5e6), 'sweep_span': (2.5e6, 4e6)},
            {'gen': 'QPSK', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (200e3, 500e3),
             'burst': True},
        ],
        'burst': {'start_range': (0.10, 0.30), 'end_range': (0.60, 0.80)},
    },
    'E_qpsk_fm_chirp2': {
        'q3': ('FM', 'source'),
        'desc': 'QPSK + FM + Chirp (different placement)',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (-3.5e6, -1.5e6), 'sr_range': (200e3, 500e3)},
            {'gen': 'FM', 'fc_range': (4.0e6, 7.0e6), 'dev_range': (150e3, 350e3)},
            {'gen': 'Chirp', 'sweep_center': (-7.5e6, -5.5e6), 'sweep_span': (2.5e6, 4.5e6)},
        ],
        'burst': None,
    },
    'F_16qam_am_chirp_burst': {
        'q3': ('AM', 'source'),
        'desc': '16QAM + AM(comb source, filtered band) + Chirp + burst_BPSK',
        'signals': [
            {'gen': '16QAM', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (200e3, 400e3),
             'power_range': (-32, -38)},
            {'gen': 'AM', 'fc_range': (-6.0e6, -3.5e6), 'depth_range': (0.8, 0.95),
             'power_range': (-22, -26)},
            {'gen': 'Chirp', 'sweep_center': (5.5e6, 7.5e6), 'sweep_span': (2.5e6, 4.5e6),
             'power_range': (-32, -38)},
            {'gen': 'BPSK', 'fc_range': (-8.0e6, -5.5e6), 'sr_range': (300e3, 600e3),
             'power_range': (-32, -38), 'burst': True},
        ],
        'burst': {'start_range': (0.20, 0.40), 'end_range': (0.70, 0.90)},
    },
    'G_bpsk_fm_chirp2': {
        'q3': ('FM', 'digital'),
        'desc': 'BPSK + FM + Chirp',
        'signals': [
            {'gen': 'BPSK', 'fc_range': (-3.0e6, -1.0e6), 'sr_range': (300e3, 600e3)},
            {'gen': 'FM', 'fc_range': (-7.0e6, -4.5e6), 'dev_range': (200e3, 400e3)},
            {'gen': 'Chirp', 'sweep_center': (5.5e6, 7.5e6), 'sweep_span': (2.5e6, 4e6)},
        ],
        'burst': None,
    },
    'H_qpsk_fm_chirp_burst': {
        'q3': ('FM', 'digital'),
        'desc': 'QPSK + FM + Chirp + burst_8PSK',
        'signals': [
            {'gen': 'QPSK', 'fc_range': (1.0e6, 3.0e6), 'sr_range': (200e3, 500e3)},
            {'gen': 'FM', 'fc_range': (-4.5e6, -2.0e6), 'dev_range': (150e3, 350e3)},
            {'gen': 'Chirp', 'sweep_center': (5.5e6, 7.5e6), 'sweep_span': (2.5e6, 4.5e6)},
            {'gen': '8PSK', 'fc_range': (-8.0e6, -5.5e6), 'sr_range': (150e3, 350e3),
             'burst': True},
        ],
        'burst': {'start_range': (0.15, 0.35), 'end_range': (0.65, 0.85)},
    },
}


# ============================================================
# Signal param builder
# ============================================================

def _make_params(rng, spec):
    gen = spec['gen']
    p = round(rng.uniform(*spec.get('power_range', (-28, -36))), 1)
    if gen in ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'):
        sr = rng.choice(np.arange(spec['sr_range'][0], spec['sr_range'][1] + 1, 50e3))
        ro = rng.choice([0.25, 0.3, 0.35])
        return gen, {'fc': rng.uniform(*spec['fc_range']), 'sym_rate': sr,
                     'rolloff': ro, 'power_dbm': p}
    if gen == 'FM':
        # Single-tone modulating waveform: Q3's autocorrelation comb requires
        # a strictly T_mod-periodic message. (Multi-harmonic FM draws
        # non-integer harmonic ratios, which breaks exact periodicity.)
        # The tone completes an integer number of cycles over the record so
        # the circular (FFT) autocorrelation's wraparound comb lands ON the
        # k*T_mod grid instead of creating off-grid competitors.
        return gen, {'fc': rng.uniform(*spec['fc_range']),
                     'deviation': rng.uniform(*spec['dev_range']),
                     'mod_freq': rng.randint(14, 33) * 20e6 / 32768,
                     'n_harmonics': 1, 'power_dbm': p}
    if gen == 'AM':
        # integer cycles over the record, same reason as FM above
        return gen, {'fc': rng.uniform(*spec['fc_range']),
                     'mod_depth': round(rng.uniform(*spec['depth_range']), 2),
                     'mod_freq': rng.randint(9, 33) * 20e6 / 32768,
                     'power_dbm': p}
    if gen == 'Chirp':
        sc = rng.uniform(*spec['sweep_center'])
        sp = rng.uniform(*spec['sweep_span'])
        return gen, {'sweep_start': sc - sp / 2, 'sweep_end': sc + sp / 2, 'power_dbm': p}
    return gen, {'power_dbm': p}


# ============================================================
# Helper: find primary digital signal
# ============================================================

def _get_primary_digital(signals):
    """Return the first digital signal."""
    for s in signals:
        if s['type'] in ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'):
            return s
    return None


def _get_chirp(signals):
    """Return the chirp signal."""
    for s in signals:
        if 'Chirp' in s['type']:
            return s
    return None


def _get_burst_signal(signals):
    """Return the burst signal if any."""
    for s in signals:
        if s.get('is_burst', False):
            return s
    return None


# ============================================================
# Question generators
# ============================================================

def qt_spectral_analysis(signals, fs, N, rng):
    """Q1: Spectral analysis parameters & windowing (20pts)."""
    delta_f = fs / N

    # Find two closest signals by center frequency for part (c)
    freqs = sorted([s['center_frequency_MHz'] for s in signals])
    min_sep = float('inf')
    for i in range(len(freqs) - 1):
        sep = freqs[i + 1] - freqs[i]
        if sep > 0:
            min_sep = min(min_sep, sep)
    # Use a plausible close separation (between two signals)
    # Pick two signals that are closest
    delta_f_sep_Hz = min_sep * 1e6
    T_min = 1.0 / delta_f_sep_Hz

    # Welch: K=8 segments, 50% overlap
    K = 8
    # With 50% overlap, need (K+1)/2 * segment_length = N for K segments
    # Actually: with 50% overlap, K segments of length L: total = L + (K-1)*L/2 = L*(K+1)/2
    # So L = 2*N/(K+1)
    L = int(2 * N / (K + 1))
    # Alternative: simpler — L = N // (K//2 + 1) for 50% overlap
    # Standard: K segments with 50% overlap from N points: L = 2*N/(K+1)
    # More standard: with 50% overlap, L such that we get K segments:
    # number of segments = floor((N - L) / (L/2)) + 1 = K
    # => (N - L) / (L/2) + 1 = K => N - L = (K-1)*L/2 => N = L + (K-1)*L/2 = L*(K+1)/2
    # => L = 2*N / (K+1) = 2*32768/9 ≈ 7282
    L_exact = 2 * N / (K + 1)
    L = int(round(L_exact))
    variance_reduction_dB = round(10 * np.log10(K), 1)

    q = (f"Spectral analysis parameters & windowing:\n"
         f"    (a) Given sampling rate fs = {fs / 1e6:.0f} MHz, number of samples N = {N}.\n"
         f"        Compute the current FFT frequency resolution Δf = fs/N (Hz). (4 pts)\n"
         f"    (b) Rectangular window vs Hamming window comparison:\n"
         f"        - How does the main lobe width change? (Rectangular: 1×Δf, Hamming: 1.81×Δf)\n"
         f"        - How does the sidelobe level change? (Rectangular: -13dB, Hamming: -42dB)\n"
         f"        For multi-signal analysis in this frequency band, which window function is more suitable? Explain your reasoning. (6 pts)\n"
         f"    (c) Given that the frequency separation between the two closest signals in this band is approximately {min_sep:.2f} MHz,\n"
         f"        what is the minimum observation time needed to resolve these two signals in the frequency domain? (4 pts)\n"
         f"    (d) Using the Welch method (50% overlap, {K} segments) for power spectral density estimation:\n"
         f"        - How many samples are needed per segment?\n"
         f"        - By how many dB is the spectral estimation variance reduced? (6 pts)")

    gt = {
        'delta_f_Hz': round(delta_f, 2),
        'delta_f_formula': 'fs/N',
        'rect_mainlobe': '1×Δf',
        'rect_sidelobe_dB': -13,
        'hamming_mainlobe': '1.81×Δf',
        'hamming_sidelobe_dB': -42,
        'recommended_window': 'Hamming',
        'window_reason': 'In multi-signal scenarios, the Hamming window has lower sidelobes (-42dB vs -13dB), reducing spectral leakage from strong signals into weak signals',
        'min_freq_sep_MHz': round(min_sep, 2),
        'min_observation_time_us': round(T_min * 1e6, 2),
        'min_observation_formula': 'T_min = 1/Δf_sep',
        'welch_K': K,
        'welch_segment_length': L,
        'welch_segment_formula': 'L = 2N/(K+1)',
        'variance_reduction_dB': variance_reduction_dB,
        'variance_reduction_formula': '10·log₁₀(K)',
    }

    rubric = {
        'points': 20,
        'scoring': AUTOCORR_SCORING,
        'delta_f': {'pts': 4, 'tol': 'exact (fs/N)'},
        'windowing': {'pts': 6,
                      'note': 'Hamming better than rectangular (sidelobes 29dB lower); main lobe slightly wider but acceptable; '
                              'must explain spectral leakage concept'},
        'min_obs_time': {'pts': 4, 'tol': '±20%',
                         'note': 'T_min = 1/Δf_sep'},
        'welch': {'pts': 6, 'tol': '±20% on segment length',
                  'note': 'variance reduction ≈ 9dB (10log10(8))'},
    }
    return q, gt, rubric


def qt_bandwidth_definitions(signals, fs, rng):
    """Q2: Bandwidth definition comparison (20pts)."""
    ds = _get_primary_digital(signals)
    if not ds:
        return None
    Rs = ds['symbol_rate_kHz'] * 1e3
    alpha = ds['rolloff']
    mod_type = ds['type']

    # For raised-cosine pulse shaped signal:
    # 3dB BW ≈ Rs (the -3dB points span approximately Rs)
    bw_3db = Rs
    # 99% power BW ≈ Rs × (1 + alpha)
    bw_99 = Rs * (1 + alpha)
    # Null-to-null BW = 2 × Rs for rectangular pulse, Rs × (1 + alpha) for raised cosine
    # For raised cosine: the spectrum goes to zero at f = ±Rs(1+α)/2 from center
    # So null-to-null (double-sided) = Rs × (1 + alpha)
    # But actually for sinc-based (no rolloff), null-to-null = 2×Rs
    # For raised cosine with rolloff α: null-to-null = Rs×(1+α) (same as 99%)
    bw_null = Rs * (1 + alpha)

    q = (f"For the {mod_type} digitally modulated signal at {ds['center_frequency_MHz']:+.1f} MHz "
         f"(symbol rate Rs ≈ {ds['symbol_rate_kHz']:.0f} ksps, roll-off factor α = {alpha}):\n"
         f"    (a) Estimate its 3dB bandwidth (the frequency span at which power drops by half) (Hz). (5 pts)\n"
         f"    (b) Estimate its 99% power bandwidth (the bandwidth containing 99% of the signal energy) (Hz). (5 pts)\n"
         f"    (c) Estimate its null-to-null bandwidth (Hz). (5 pts)\n"
         f"    (d) Why do these three bandwidth definitions yield different values?\n"
         f"        For frequency planning (allocating frequency bands to signals), which definition is most appropriate? Why? (5 pts)")

    gt = {
        'signal_type': mod_type,
        'symbol_rate_kHz': ds['symbol_rate_kHz'],
        'rolloff': alpha,
        'bw_3dB_kHz': round(bw_3db / 1e3, 1),
        'bw_3dB_formula': '≈ Rs',
        'bw_99_kHz': round(bw_99 / 1e3, 1),
        'bw_99_formula': '≈ Rs × (1 + α)',
        'bw_null_kHz': round(bw_null / 1e3, 1),
        'bw_null_formula': '≈ Rs × (1 + α) for raised cosine',
        'best_for_planning': '99% power bandwidth',
        'reason': '99% power bandwidth is most suitable for frequency planning because it captures nearly all signal energy '
                  'while accounting for spectral spreading in the roll-off region, making it the practical standard for channel spacing design',
    }

    rubric = {
        'points': 20,
        'scoring': AUTOCORR_SCORING,
        'bw_3dB': {'pts': 5, 'tol': '±30%',
                   'note': '≈Rs, accept 0.5Rs~1.5Rs'},
        'bw_99': {'pts': 5, 'tol': '±25%',
                  'note': '≈Rs(1+α)'},
        'bw_null': {'pts': 5, 'tol': '±25%',
                    'note': '≈Rs(1+α) for raised cosine'},
        'explanation': {'pts': 5,
                        'note': 'Must explain the physical meaning differences among definitions; '
                                '99% BW or occupied BW is most suitable for frequency planning'},
    }
    return q, gt, rubric


def _autocorr_magnitude(x):
    """Normalized |R(tau)| via the FFT method stated verbatim in the question."""
    spectrum = np.fft.fft(x)
    acorr = np.fft.ifft(np.abs(spectrum) ** 2)
    magnitude = np.abs(acorr)
    return magnitude / magnitude[0]


def _acorr_window(fs, N):
    lo = int(np.ceil(ACORR_MIN_LAG_US * 1e-6 * fs))
    hi = N // 2
    return lo, hi


def _occupied_interval_mhz(signal):
    if 'Chirp' in signal['type']:
        return float(signal['sweep_start_MHz']), float(signal['sweep_end_MHz'])
    center = float(signal['center_frequency_MHz'])
    half_bw = float(signal.get('bandwidth_MHz', 0.0)) / 2
    return center - half_bw, center + half_bw


def _grid_masks(delays_us, t_mod_us, exclusion=ACORR_COMPETITOR_EXCLUSION):
    """Classify peak delays against the k*T_mod harmonic grid.

    Peaks between the on-grid and competitor zones are skirt sidelobes of a
    comb peak and count as neither. A broad-humped comb (tone AM) uses a
    wider exclusion so slope wiggles on the hump skirts are not treated as
    competitors — only trough-region structure is.
    """
    harmonic_of = np.round(delays_us / t_mod_us)
    offsets = np.abs(delays_us - harmonic_of * t_mod_us)
    on_grid = (harmonic_of >= 1) & (offsets <= ACORR_HARMONIC_TOL * t_mod_us)
    competitor = offsets > exclusion * t_mod_us
    return harmonic_of, on_grid, competitor


def _comb_analysis(received, f_mod_hz, fs, N,
                   competitor_exclusion=ACORR_COMPETITOR_EXCLUSION):
    """Locate the T_mod peak comb in |R(tau)| and verify it is unambiguous.

    Comb peaks decay with delay, so membership is decided per harmonic-grid
    window rather than by a global prominence cut. Returns comb measurements,
    or None when any oracle check fails (missing low harmonics, competing
    off-grid peaks, weak comb, or global maximum away from the grid).
    """
    t_mod_us = 1e6 / f_mod_hz
    if t_mod_us <= ACORR_MIN_LAG_US * 1.5:
        return None
    magnitude = _autocorr_magnitude(received)
    lo, hi = _acorr_window(fs, N)
    segment = magnitude[lo:hi]
    window_hi_us = hi / fs * 1e6
    if window_hi_us // t_mod_us < len(ACORR_REQUIRED_HARMONICS):
        return None
    peak_idx, props = find_peaks(segment, prominence=1e-4)
    if len(peak_idx) == 0:
        return None
    prominences = props['prominences']
    delays_us = (peak_idx + lo) / fs * 1e6

    harmonic_of, on_grid, competitor = _grid_masks(delays_us, t_mod_us,
                                                   competitor_exclusion)

    # Best peak inside each harmonic window.
    comb = {}
    for i in np.flatnonzero(on_grid):
        k = int(harmonic_of[i])
        if k not in comb or prominences[i] > comb[k][1]:
            comb[k] = (delays_us[i], float(prominences[i]),
                       float(magnitude[peak_idx[i] + lo]))

    if any(k not in comb for k in ACORR_REQUIRED_HARMONICS):
        return None
    p_ref = min(comb[k][1] for k in ACORR_REQUIRED_HARMONICS)
    if p_ref < ACORR_MIN_COMB_PROMINENCE:
        return None

    # Ambiguity margin: every off-grid peak must stay well below the comb,
    # both in prominence (for prominence-based peak pickers) and in absolute
    # height (an AM carrier pedestal can lift off-grid wiggles toward comb
    # height, which would trap solvers that threshold at half the maximum).
    if np.any(competitor):
        if prominences[competitor].max() >= ACORR_COMPETITOR_MARGIN * p_ref:
            return None
        # height comparison is pedestal-referenced: an AM carrier lifts the
        # whole |R| floor, so absolute heights would reject every AM scene
        pedestal = float(np.median(segment))
        comb_above = min(comb[k][2] for k in ACORR_REQUIRED_HARMONICS) - pedestal
        heights_above = segment[peak_idx] - pedestal
        if heights_above[competitor].max() >= \
                ACORR_COMPETITOR_HEIGHT_FRACTION * max(comb_above, 1e-9):
            return None

    # The global maximum of the search window must itself sit on the grid.
    argmax_delay_us = (int(np.argmax(segment)) + lo) / fs * 1e6
    k = round(argmax_delay_us / t_mod_us)
    if k < 1 or abs(argmax_delay_us - k * t_mod_us) > ACORR_HARMONIC_TOL * t_mod_us:
        return None

    spacing = float(np.median([comb[k][0] / k for k in sorted(comb)]))
    return {
        'comb_spacing_us': spacing,
        'first_peak_us': comb[1][0],
        'first_peak_R': comb[1][2],
        'max_R': float(segment.max()),
        'n_comb_peaks': len(comb),
        'min_comb_prominence': p_ref,
    }


def _digital_filter_band_mhz(ds, signals):
    """Filter band isolating the primary digital signal, or None if not isolable."""
    rs_mhz = float(ds['symbol_rate_kHz']) / 1e3
    width = ACORR_FILTER_BW_FACTOR * (1 + float(ds['rolloff'])) * rs_mhz
    center = float(ds['center_frequency_MHz'])
    band = (center - width / 2, center + width / 2)
    for s in signals:
        if s is ds:
            continue
        s_lo, s_hi = _occupied_interval_mhz(s)
        if s_hi > band[0] - ACORR_FILTER_GUARD_MHZ and s_lo < band[1] + ACORR_FILTER_GUARD_MHZ:
            return None
    return band


def _source_filter_band_mhz(src, signals):
    """Filter band isolating the comb source itself, or None if not isolable."""
    lo, hi = _occupied_interval_mhz(src)
    center = (lo + hi) / 2
    width = ACORR_FILTER_BW_FACTOR * max(hi - lo, 0.05)
    band = (center - width / 2, center + width / 2)
    for s in signals:
        if s is src:
            continue
        s_lo, s_hi = _occupied_interval_mhz(s)
        if s_hi > band[0] - ACORR_FILTER_GUARD_MHZ and s_lo < band[1] + ACORR_FILTER_GUARD_MHZ:
            return None
    return band


def _filtered_comb_state(received, band_mhz, f_mod_hz, fs, N):
    """(on_grid_max, off_grid_max) peak prominences of |R(tau)| after the
    bandpass filter, classified against the k*T_mod harmonic grid."""
    freqs_mhz = np.fft.fftfreq(N, 1 / fs) / 1e6
    band_mask = (freqs_mhz >= band_mhz[0]) & (freqs_mhz <= band_mhz[1])
    if not np.any(band_mask):
        return None
    filtered = np.fft.ifft(np.fft.fft(received) * band_mask)
    magnitude = _autocorr_magnitude(filtered)
    lo, hi = _acorr_window(fs, N)
    peak_idx, props = find_peaks(magnitude[lo:hi], prominence=1e-5)
    if len(peak_idx) == 0:
        return 0.0, 0.0
    prominences = props['prominences']
    delays_us = (peak_idx + lo) / fs * 1e6
    t_mod_us = 1e6 / f_mod_hz
    _, on_grid, competitor = _grid_masks(delays_us, t_mod_us)
    on_grid_max = float(prominences[on_grid].max()) if np.any(on_grid) else 0.0
    off_grid_max = float(prominences[competitor].max()) if np.any(competitor) else 0.0
    return on_grid_max, off_grid_max


def qt_autocorrelation(signals, received, fs, N, q3_mode=('FM', 'digital')):
    """Q3: Autocorrelation & periodicity analysis (20pts, deterministic).

    Every reported quantity is defined by an explicit estimator and verified
    against the actual waveform (oracle). Returns None when the scenario does
    not admit an unambiguous answer, which triggers sample regeneration.

    q3_mode = (source_type, filter_target) varies the two categorical answers
    across archetypes (remediation log §8.2.2): the comb source is a
    single-tone FM or a tone-modulated AM, and part (d)'s bandpass filter
    keeps either the primary digital signal (comb vanishes) or the comb
    source itself (comb persists).
    """
    source_type, filter_target = q3_mode
    src_meta_type = 'FM' if source_type == 'FM' else 'AM-DSB'
    sources = [s for s in signals if s['type'] == src_meta_type]
    if len(sources) != 1:
        return None
    # a single-tone FM would out-comb an AM source: never mix them
    if source_type == 'AM' and any(s['type'] == 'FM' for s in signals):
        return None
    src = sources[0]
    f_mod_khz = float(src['modulating_frequency_kHz'])

    ds = _get_primary_digital(signals)
    if ds is None:
        return None
    if filter_target == 'digital':
        keep, keep_desc_type = ds, 'digitally modulated'
        band = _digital_filter_band_mhz(ds, signals)
    else:
        keep, keep_desc_type = src, source_type
        band = _source_filter_band_mhz(src, signals)
    if band is None:
        return None

    exclusion = (0.25 if source_type == 'AM'
                 else ACORR_COMPETITOR_EXCLUSION)
    comb = _comb_analysis(received, f_mod_khz * 1e3, fs, N,
                          competitor_exclusion=exclusion)
    if comb is None:
        return None
    state = _filtered_comb_state(received, band, f_mod_khz * 1e3, fs, N)
    if state is None:
        return None
    on_grid_max, off_grid_max = state
    persists_gt = filter_target == 'source'
    if persists_gt:
        # the surviving comb must be unambiguous with margin
        if (on_grid_max < 2 * ACORR_FILTERED_MIN_PROMINENCE
                or on_grid_max < 2 * ACORR_FILTERED_PERSIST_RATIO
                * max(off_grid_max, 1e-9)):
            return None
    else:
        # the comb must vanish: what remains on the grid is correlation
        # noise, so it must show no preference for the grid over the
        # off-grid background (scale-free), or be negligible outright
        if not (on_grid_max <= max(off_grid_max, 1e-9)
                or on_grid_max < ACORR_FILTERED_MIN_PROMINENCE):
            return None
    residual = on_grid_max

    _, hi = _acorr_window(fs, N)
    window_hi_us = hi / fs * 1e6

    q = (f"Autocorrelation & periodicity analysis (answer format version: {AUTOCORR_SCHEMA}):\n"
         f"    (a) Compute the autocorrelation of the full received signal x with the FFT method:\n"
         f"        R(τ) = IFFT(|FFT(x)|²), normalized so that R(0) = 1, and work with the magnitude |R(τ)|.\n"
         f"        Over the delay range {ACORR_MIN_LAG_US:.0f} μs ≤ τ < {window_hi_us:.1f} μs (smaller delays are dominated by\n"
         f"        pulse-shaping and chirp correlation structure and are excluded), report the maximum value\n"
         f"        of |R(τ)| (a number between 0 and 1). (4 pts)\n"
         f"    (b) In the same delay range, |R(τ)| exhibits a set of strong, equally spaced peaks (a periodic comb).\n"
         f"        Report the spacing Δτ between consecutive comb peaks, in μs. (6 pts)\n"
         f"    (c) This periodicity is produced by exactly one signal in the band. Which signal is it (modulation\n"
         f"        type), and what is the corresponding modulating frequency f_mod = 1/Δτ, in kHz? (3+3 pts)\n"
         f"    (d) If the received signal is bandpass filtered so that only the {keep_desc_type} signal near\n"
         f"        {keep['center_frequency_MHz']:+.1f} MHz remains, does the peak comb with spacing Δτ persist in |R(τ)|?\n"
         f"        Answer true or false and briefly explain why. (4 pts)\n"
         f"    In the final answer, return this question as one single-line JSON object:\n"
         f'    {{"max_R_magnitude": <number>, "comb_spacing_us": <number>, "source_signal": "<modulation type>",\n'
         f'    "modulating_freq_kHz": <number>, "comb_persists_after_filtering": <true|false>,\n'
         f'    "explanation": "<1-2 sentences>"}}')

    gt = {
        'schema': AUTOCORR_SCHEMA,
        'max_R_magnitude': round(comb['max_R'], 4),
        'comb_spacing_us': round(comb['comb_spacing_us'], 2),
        'comb_first_peak_us': round(comb['first_peak_us'], 2),
        'comb_first_peak_R': round(comb['first_peak_R'], 4),
        'n_comb_peaks': comb['n_comb_peaks'],
        'source_signal': source_type,
        'source_signal_type': src['type'],
        'modulating_freq_kHz': round(f_mod_khz, 3),
        'theory_period_us': round(1e3 / f_mod_khz, 2),
        'comb_persists_after_filtering': persists_gt,
        'filter_target': filter_target,
        'filter_band_MHz': [round(band[0], 3), round(band[1], 3)],
        'filtered_comb_residual_prominence': round(residual, 5),
        'search_window_us': [ACORR_MIN_LAG_US, round(window_hi_us, 1)],
    }

    rubric = {
        'points': 20,
        'scoring': AUTOCORR_SCORING,
        'max_R_magnitude': {'pts': 4, 'tol': 'abs ±0.08 full, ±0.16 half'},
        'comb_spacing': {'pts': 6, 'tol': '±5% full, ±15% half'},
        'source_signal': {'pts': 3, 'note': 'modulation family must match (FM vs AM)'},
        'modulating_freq': {'pts': 3, 'tol': '±5% full, ±15% half'},
        'filtered': {'pts': 4,
                     'note': 'JSON boolean, oracle-verified per scenario. '
                             'The prose explanation is not separately scored.'},
    }
    return q, gt, rubric


def qt_stft_spectrogram(signals, fs, N, rng):
    """Q4: Time-frequency analysis (STFT/Spectrogram) (20pts)."""
    duration_ms = N / fs * 1e3

    # Chirp sweep rate
    chirp = _get_chirp(signals)
    if chirp:
        sweep_bw_MHz = chirp['bandwidth_MHz']
        chirp_rate = chirp['chirp_rate_MHz_per_ms']
    else:
        sweep_bw_MHz = 0
        chirp_rate = 0

    # Burst timing
    burst = _get_burst_signal(signals)
    has_burst = burst is not None
    if has_burst:
        burst_info = burst.get('burst_info', {})
        burst_window = burst_info.get('active_window', 'N/A')
        # Parse start/end from active_window "XX%-YY%"
        parts = burst_window.replace('%', '').split('-')
        if len(parts) == 2:
            burst_start_ms = round(float(parts[0]) / 100 * duration_ms, 3)
            burst_end_ms = round(float(parts[1]) / 100 * duration_ms, 3)
        else:
            burst_start_ms = 0
            burst_end_ms = duration_ms
    else:
        burst_start_ms = None
        burst_end_ms = None

    # Build question text
    burst_part = ""
    if has_burst:
        burst_part = (f"    (c) Is there a burst signal in the spectrogram? If so, estimate its start and end times (ms). (6 pts)\n")
    else:
        burst_part = (f"    (c) Is there a burst signal in the spectrogram? If not, explain how you determined this.\n"
                      f"        If all signals are continuous, describe their characteristics along the time axis. (6 pts)\n")

    q = (f"Time-frequency analysis (STFT/Spectrogram):\n"
         f"    (a) Generate a spectrogram using STFT (Short-Time Fourier Transform) and describe the main visible features.\n"
         f"        (Hint: focus on the morphological differences of different signals on the time-frequency plane) (4 pts)\n"
         f"    (b) Estimate the sweep rate (MHz/ms) of the linear frequency modulated (Chirp) signal from the spectrogram.\n"
         f"        Sweep rate = bandwidth (MHz) / duration (ms). (6 pts)\n"
         f"{burst_part}"
         f"    (d) Relationship between STFT window length and time-frequency resolution: the Heisenberg uncertainty principle requires Δt × Δf ≥ 1.\n"
         f"        Is it possible to simultaneously achieve a time resolution of 0.1ms and a frequency resolution of 1kHz?\n"
         f"        Explain why. (4 pts)")

    # Uncertainty: Δt=0.1ms, Δf=1kHz → Δt×Δf = 0.1e-3 × 1e3 = 0.1 < 1 → impossible
    gt = {
        'spectrogram_features': [
            'Digital modulated signal: appears as a wideband horizontal stripe at a fixed frequency',
            'FM/AM signal: appears as a narrowband continuous signal at a fixed frequency',
            'Chirp signal: appears as a diagonal line (frequency varies linearly with time)',
        ],
        'chirp_sweep_rate_MHz_per_ms': chirp_rate if chirp else 'N/A',
        'chirp_bandwidth_MHz': sweep_bw_MHz if chirp else 'N/A',
        'has_burst': has_burst,
        'burst_start_ms': burst_start_ms,
        'burst_end_ms': burst_end_ms,
        'uncertainty_possible': False,
        'uncertainty_product': 0.1,
        'uncertainty_explanation': (
            'Δt×Δf = 0.1ms × 1kHz = 0.1 < 1, which violates the Heisenberg uncertainty principle, '
            'so it is impossible to achieve both resolutions simultaneously. A trade-off is required: '
            'shortening the window improves time resolution but degrades frequency resolution, and vice versa.'
        ),
    }

    if has_burst:
        gt['burst_signal_type'] = burst['type']

    rubric = {
        'points': 20,
        'scoring': AUTOCORR_SCORING,
        'features': {'pts': 4, 'note': 'Correctly describe time-frequency characteristics of the 3 signal types'},
        'chirp_rate': {'pts': 6, 'tol': '±25%',
                       'note': f'sweep rate ≈ {chirp_rate} MHz/ms'},
        'burst_timing': {'pts': 6, 'tol': '±0.1ms' if has_burst else 'N/A',
                         'note': 'correct identification of burst presence/absence'},
        'uncertainty': {'pts': 4,
                        'note': 'Δt×Δf=0.1 < 1, impossible'},
    }
    return q, gt, rubric


def qt_energy_power(signals, sig_waveforms, noise_dbm, fs, N, rng):
    """Q5: Signal energy & power relationships (20pts)."""
    duration = N / fs

    # Per-signal energy
    energy_list = []
    for s, waveform in zip(signals, sig_waveforms):
        # Use record-average power from the final waveform. This is identical
        # to the configured power for continuous signals and includes the
        # actual gate and ramp loss for burst signals.
        p_w = float(np.mean(np.abs(waveform) ** 2))
        p_mw = p_w * 1000
        p_dbm = 10 * np.log10(p_mw) if p_mw > 0 else -999
        E_j = p_w * duration
        E_dbj = 10 * np.log10(E_j) if E_j > 0 else -999
        entry = {
            'type': s['type'],
            'power_dBm': round(p_dbm, 1),
            'power_mW': round(p_mw, 6),
            'energy_J': f"{E_j:.2e}",
            'energy_dBJ': round(E_dbj, 1),
        }
        energy_list.append(entry)

    # Digital signal Eb
    ds = _get_primary_digital(signals)
    if ds:
        ds_idx = next(i for i, s in enumerate(signals) if s is ds)
        p_w = float(np.mean(np.abs(sig_waveforms[ds_idx]) ** 2))
        Es = p_w * duration
        Rs = ds['symbol_rate_kHz'] * 1e3
        bps = ds['bits_per_symbol']
        total_bits = int(Rs * bps * duration)
        Eb = Es / total_bits if total_bits > 0 else 0
        Eb_dBJ = round(10 * np.log10(Eb), 1) if Eb > 0 else -999
    else:
        total_bits = 0
        Eb = 0
        Eb_dBJ = -999

    # Total received power (all signals + noise)
    total_mw = sum(float(np.mean(np.abs(waveform) ** 2)) * 1000
                   for waveform in sig_waveforms)
    noise_mw = 10 ** (noise_dbm / 10)
    total_with_noise_mw = total_mw + noise_mw
    total_dbm = round(10 * np.log10(total_with_noise_mw), 1)

    q = (f"Signal energy & power relationships:\n"
         f"    (a) Compute the energy of each signal Es = Ps × T (where Ps is the signal power (W), T is the duration (s)).\n"
         f"        List each signal's power (mW) and energy (J). (5 pts)\n"
         f"    (b) For the primary digitally modulated signal ({ds['center_frequency_MHz']:+.1f} MHz),\n"
         f"        compute the energy per bit Eb = Es / total number of bits.\n"
         f"        Total bits = Rs × log₂(M) × T. (5 pts)\n"
         f"    (c) Compute the total received power (all signals + noise) (dBm).\n"
         f"        The noise floor power is approximately {noise_dbm} dBm. (5 pts)\n"
         f"    (d) Why is the total power of multiple independent signals computed by summing power in mW rather than in dBm?\n"
         f"        Derive the formula and verify with the data in this problem. (5 pts)")

    gt = {
        'duration_s': duration,
        'energy_per_signal': energy_list,
        'digital_signal': {
            'type': ds['type'] if ds else 'N/A',
            'total_bits': total_bits,
            'Eb_J': f"{Eb:.2e}",
            'Eb_dBJ': Eb_dBJ,
        },
        'total_signal_power_mW': round(total_mw, 6),
        'noise_power_mW': round(noise_mw, 6),
        'total_received_power_mW': round(total_with_noise_mw, 6),
        'total_received_power_dBm': total_dbm,
        'power_addition_explanation': (
            'dBm is a logarithmic unit, and logarithmic functions do not satisfy linear superposition: '
            '10log₁₀(P₁+P₂) ≠ 10log₁₀(P₁) + 10log₁₀(P₂). '
            'Each dBm value must first be converted to mW (P_mW = 10^(P_dBm/10)), '
            'summed in the linear domain, then converted back to dBm: P_total_dBm = 10log₁₀(ΣP_i_mW).'
        ),
    }

    rubric = {
        'points': 20,
        'scoring': AUTOCORR_SCORING,
        'energy': {'pts': 5, 'tol': '±1 order of magnitude',
                   'note': 'correct mW→W conversion and E=P×T'},
        'Eb': {'pts': 5, 'tol': '±3dB',
               'note': 'correct total bits count and Eb=Es/Nbits'},
        'total_power': {'pts': 5, 'tol': '±2dB',
                        'note': 'sum in mW then convert to dBm'},
        'explanation': {'pts': 5,
                        'note': 'Must explain that values in the logarithmic domain cannot be directly summed; provide the correct formula'},
    }
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
        elif t == 'FM':
            lines.append(f"  - Approx. {fc:+.1f} MHz: analog modulated signal")
        elif t == 'AM-DSB':
            lines.append(f"  - Approx. {fc:+.1f} MHz: analog modulated signal")
        else:
            bst = " (burst mode)" if s.get('is_burst', False) else ""
            lines.append(f"  - Approx. {fc:+.1f} MHz: digitally modulated signal{bst}")
    return '\n'.join(lines)


def generate_one(slot_seed, arch_name, arch, output_dir, fs=20e6, N=32768,
                 max_attempts=50):
    """Generate one problem for a fixed sample id, reseeding until the Q3
    autocorrelation oracle accepts the scenario."""
    for attempt in range(max_attempts):
        rng_seed = slot_seed + 100000 * attempt
        meta = _generate_attempt(slot_seed, rng_seed, attempt, arch_name, arch,
                                 output_dir, fs, N)
        if meta is not None:
            return meta
    raise RuntimeError(
        f"EMRB_L2_{slot_seed:04d} ({arch_name}): no oracle-valid Q3 scenario "
        f"after {max_attempts} attempts"
    )


def _generate_attempt(slot_seed, rng_seed, attempt, arch_name, arch, output_dir,
                      fs, N):
    rng = np.random.RandomState(rng_seed)
    t = np.arange(N) / fs
    duration = N / fs

    # Generate individual signals + keep waveforms
    all_signals = []
    sig_waveforms = []
    mixed = np.zeros(N, dtype=complex)

    burst_params = arch.get('burst', None)

    for spec in arch['signals']:
        gen_type, params = _make_params(rng, spec)
        sig, meta = SIGNAL_GENERATORS[gen_type](rng, fs, N, t, params)

        # Apply burst if specified
        is_burst = spec.get('burst', False)
        if is_burst and burst_params:
            start_frac = round(rng.uniform(*burst_params['start_range']), 2)
            end_frac = round(rng.uniform(*burst_params['end_range']), 2)
            sig, burst_info = apply_burst(sig, rng, N,
                                          {'start_frac': start_frac,
                                           'end_frac': end_frac})
            meta['is_burst'] = True
            meta['burst_info'] = burst_info
        else:
            meta['is_burst'] = False

        mixed += sig
        all_signals.append(meta)
        sig_waveforms.append(sig)

    noise_dbm = round(rng.uniform(-48, -53), 0)
    noise_power = 10 ** (noise_dbm / 10) / 1000
    noise = np.sqrt(noise_power / 2) * (rng.randn(N) + 1j * rng.randn(N))
    received = mixed + noise

    sample_id = f"EMRB_L2_{slot_seed:04d}"

    # Generate questions. Q3 runs an oracle over the actual waveform and may
    # reject the scenario — in that case abort before writing any files.
    questions = []
    q_texts = []

    r3 = qt_autocorrelation(all_signals, received, fs, N,
                            q3_mode=arch.get('q3', ('FM', 'digital')))
    if r3 is None:
        return None

    r1 = qt_spectral_analysis(all_signals, fs, N, rng)
    r2 = qt_bandwidth_definitions(all_signals, fs, rng)
    r4 = qt_stft_spectrogram(all_signals, fs, N, rng)
    r5 = qt_energy_power(all_signals, sig_waveforms, noise_dbm, fs, N, rng)

    np.save(os.path.join(output_dir, f"{sample_id}.npy"), received.astype(np.complex64))

    for i, (label, result) in enumerate([
        ('Q1', r1), ('Q2', r2), ('Q3', r3), ('Q4', r4), ('Q5', r5)
    ]):
        if result is None:
            continue
        q_str, gt, rubric = result
        questions.append({'id': label, 'question': q_str,
                          'ground_truth': gt, 'rubric': rubric})
        q_texts.append(f"{label}. {q_str}")

    hints = _build_hints(all_signals)
    question_text = f"""You are an electromagnetic signal analysis expert. Below is I/Q signal data collected from an electromagnetic environment.

Signal file: {sample_id}.npy
Sampling rate: {fs / 1e6:.0f} MHz
Number of samples: {N}
Data format: complex64 (numpy)
Recording duration: {duration * 1e3:.4f} ms

There are {len(all_signals)} signals present in this frequency band. Preliminary spectral scan results are as follows:
{hints}

Please analyze these signals and answer the following questions:

""" + '\n\n'.join(q_texts) + """

For each question, provide the complete calculation process and numerical results."""

    metadata = {
        'sample_id': sample_id, 'level': 'L2',
        'archetype': arch_name, 'archetype_desc': arch['desc'],
        'total_points': sum(q['rubric']['points'] for q in questions),
        'num_questions': len(questions),
        'question': question_text, 'questions': questions,
        'generation_params': {
            'fs': fs, 'N': N, 'duration_ms': round(duration * 1e3, 4),
            'signals': all_signals, 'noise_floor_dBm': noise_dbm,
            'seed': rng_seed, 'slot_seed': slot_seed, 'q3_attempt': attempt,
        },
    }

    with open(os.path.join(output_dir, f"{sample_id}.json"), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    return metadata


# ============================================================
# Batch
# ============================================================

def generate_batch(num=40, seed_start=6000, output_dir='data/L2'):
    os.makedirs(output_dir, exist_ok=True)
    arch_names = list(ARCHETYPES.keys())
    per = num // len(arch_names)
    rem = num % len(arch_names)

    manifest = {'total': num, 'problems': [], 'arch_counts': {}, 'level': 'L2'}
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
                tries = meta['generation_params']['q3_attempt'] + 1
                print(f"OK ({ns} sigs, {meta['num_questions']} Qs, "
                      f"{tries} attempt{'s' if tries > 1 else ''})")
                manifest['problems'].append({
                    'sample_id': meta['sample_id'], 'archetype': aname,
                    'num_signals': ns, 'total_points': meta['total_points'],
                })
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
            idx += 1

    manifest['total'] = len(manifest['problems'])
    with open(os.path.join(output_dir, 'batch_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generated {idx} L2 problems in {output_dir}/")
    print(f"Archetype distribution: {manifest['arch_counts']}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num', type=int, default=40)
    p.add_argument('--seed-start', type=int, default=6000)
    p.add_argument('--output', type=str, default='data/L2')
    args = p.parse_args()
    generate_batch(args.num, args.seed_start, args.output)
