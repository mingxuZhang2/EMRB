"""
EMRB Signal Library: Modular signal generators for L4 problems.
Each generator: (rng, fs, N, t, params) -> (complex_signal, metadata)
"""
import numpy as np


def raised_cosine_filter(n_taps, sps, rolloff=0.35):
    t = np.arange(n_taps) - (n_taps - 1) / 2
    t = t / sps
    h = np.sinc(t)
    d = 1 - (2 * rolloff * t) ** 2
    d[np.abs(d) < 1e-10] = 1e-10
    h *= np.cos(np.pi * rolloff * t) / d
    return h / np.sum(h)


def _set_power(sig, power_dbm):
    p = 10 ** (power_dbm / 10) / 1000
    current = np.mean(np.abs(sig) ** 2)
    if current > 0:
        sig = sig / np.sqrt(current) * np.sqrt(p)
    return sig


def generate_psk(rng, fs, N, t, params):
    """Generate M-PSK signal (BPSK, QPSK, 8PSK)."""
    M = params['M']
    fc = params['fc']
    sym_rate = params['sym_rate']
    rolloff = params.get('rolloff', 0.3)
    power_dbm = params['power_dbm']

    sps = int(fs / sym_rate)
    n_sym = N // sps + 10
    symbols = rng.randint(0, M, n_sym)
    bb = np.zeros(N, dtype=complex)
    for i, s in enumerate(symbols):
        idx = i * sps
        if idx < N:
            bb[idx] = np.exp(1j * 2 * np.pi * s / M)
    rc = raised_cosine_filter(8 * sps + 1, sps, rolloff)
    bb = np.convolve(bb, rc, mode='same')
    sig = bb * np.exp(1j * 2 * np.pi * fc * t)
    sig = _set_power(sig, power_dbm)

    mod_name = {2: 'BPSK', 4: 'QPSK', 8: '8PSK'}[M]
    bw = sym_rate * (1 + rolloff)
    meta = {
        'type': mod_name, 'center_frequency_MHz': fc / 1e6,
        'symbol_rate_kHz': sym_rate / 1e3, 'rolloff': rolloff,
        'M': M, 'bits_per_symbol': int(np.log2(M)),
        'power_dBm': power_dbm, 'bandwidth_MHz': round(bw / 1e6, 3),
    }
    if params.get('_capture_symbols'):
        valid_symbols = symbols[:(N - 1) // sps + 1]
        points = np.exp(1j * 2 * np.pi * valid_symbols / M)
        meta['_source_symbols_iq'] = [
            [float(point.real), float(point.imag)] for point in points
        ]
        meta['_samples_per_symbol'] = sps
    return sig, meta


def generate_qam(rng, fs, N, t, params):
    """Generate M-QAM signal (16QAM, 64QAM)."""
    M = params['M']
    fc = params['fc']
    sym_rate = params['sym_rate']
    rolloff = params.get('rolloff', 0.3)
    power_dbm = params['power_dbm']

    k = int(np.sqrt(M))
    levels = np.arange(-(k - 1), k, 2)
    sps = int(fs / sym_rate)
    n_sym = N // sps + 10
    bb = np.zeros(N, dtype=complex)
    source_symbols = []
    for i in range(n_sym):
        idx = i * sps
        if idx < N:
            point = rng.choice(levels) + 1j * rng.choice(levels)
            bb[idx] = point
            source_symbols.append(point)
    constellation_scale = np.sqrt(np.mean(np.abs(levels) ** 2) * 2)
    bb /= constellation_scale
    rc = raised_cosine_filter(8 * sps + 1, sps, rolloff)
    bb = np.convolve(bb, rc, mode='same')
    sig = bb * np.exp(1j * 2 * np.pi * fc * t)
    sig = _set_power(sig, power_dbm)

    bw = sym_rate * (1 + rolloff)
    meta = {
        'type': f'{M}QAM', 'center_frequency_MHz': fc / 1e6,
        'symbol_rate_kHz': sym_rate / 1e3, 'rolloff': rolloff,
        'M': M, 'bits_per_symbol': int(np.log2(M)),
        'power_dBm': power_dbm, 'bandwidth_MHz': round(bw / 1e6, 3),
    }
    if params.get('_capture_symbols'):
        points = np.asarray(source_symbols) / constellation_scale
        meta['_source_symbols_iq'] = [
            [float(point.real), float(point.imag)] for point in points
        ]
        meta['_samples_per_symbol'] = sps
    return sig, meta


def generate_fm(rng, fs, N, t, params):
    """Generate FM signal with optional harmonics."""
    fc = params['fc']
    deviation = params['deviation']
    mod_freq = params['mod_freq']
    power_dbm = params['power_dbm']
    n_harmonics = params.get('n_harmonics', 2)

    msg = np.sin(2 * np.pi * mod_freq * t)
    max_harmonic_freq = mod_freq
    for h in range(2, n_harmonics + 1):
        ratio = rng.uniform(0.2, 0.7)
        freq_mult = rng.uniform(2.0, 6.0)
        msg += ratio * np.sin(2 * np.pi * freq_mult * mod_freq * t)
        max_harmonic_freq = max(max_harmonic_freq, freq_mult * mod_freq)
    msg /= np.max(np.abs(msg))

    phase = 2 * np.pi * fc * t + 2 * np.pi * deviation * np.cumsum(msg) / fs
    sig = np.exp(1j * phase)
    sig = _set_power(sig, power_dbm)

    mod_index = deviation / max_harmonic_freq
    carson_bw = 2 * (deviation + max_harmonic_freq)
    meta = {
        'type': 'FM', 'center_frequency_MHz': fc / 1e6,
        'frequency_deviation_kHz': deviation / 1e3,
        'modulating_frequency_kHz': mod_freq / 1e3,
        'max_modulating_freq_kHz': round(max_harmonic_freq / 1e3, 1),
        'modulation_index': round(mod_index, 2),
        'carson_bandwidth_kHz': round(carson_bw / 1e3, 1),
        'power_dBm': power_dbm,
        'bandwidth_MHz': round(carson_bw / 1e6, 3),
    }
    return sig, meta


def generate_am(rng, fs, N, t, params):
    """Generate AM-DSB signal."""
    fc = params['fc']
    mod_depth = params['mod_depth']
    mod_freq = params['mod_freq']
    power_dbm = params['power_dbm']

    msg = np.sin(2 * np.pi * mod_freq * t)
    sig = (1 + mod_depth * msg) * np.exp(1j * 2 * np.pi * fc * t)
    sig = _set_power(sig, power_dbm)

    efficiency = mod_depth ** 2 / (2 + mod_depth ** 2)
    bw = 2 * mod_freq
    meta = {
        'type': 'AM-DSB', 'center_frequency_MHz': fc / 1e6,
        'modulation_depth': mod_depth, 'modulating_frequency_kHz': mod_freq / 1e3,
        'efficiency': round(efficiency, 3),
        'power_dBm': power_dbm, 'bandwidth_MHz': round(bw / 1e6, 3),
    }
    return sig, meta


def generate_chirp(rng, fs, N, t, params):
    """Generate LFM chirp signal."""
    sweep_start = params['sweep_start']
    sweep_end = params['sweep_end']
    power_dbm = params['power_dbm']

    duration = N / fs
    chirp_bw = sweep_end - sweep_start
    chirp_rate = chirp_bw / duration
    phase = 2 * np.pi * (sweep_start * t + 0.5 * chirp_rate * t ** 2)
    sig = np.exp(1j * phase)
    sig = _set_power(sig, power_dbm)

    fc = (sweep_start + sweep_end) / 2
    tbp = chirp_bw * duration
    meta = {
        'type': 'Chirp (LFM)', 'center_frequency_MHz': fc / 1e6,
        'sweep_start_MHz': sweep_start / 1e6, 'sweep_end_MHz': sweep_end / 1e6,
        'bandwidth_MHz': chirp_bw / 1e6,
        'chirp_rate_MHz_per_ms': round(chirp_rate / 1e9, 2),
        'time_bandwidth_product': round(tbp, 1),
        'processing_gain_dB': round(10 * np.log10(tbp), 1),
        'range_resolution_m': round(3e8 / (2 * chirp_bw), 2),
        'power_dBm': power_dbm,
    }
    return sig, meta


def generate_ofdm(rng, fs, N, t, params):
    """Generate OFDM signal."""
    fc = params['fc']
    n_sc = params['n_subcarriers']
    sc_spacing = params['subcarrier_spacing']
    cp_ratio = params.get('cp_ratio', 0.25)
    power_dbm = params['power_dbm']

    sym_len = int(fs / sc_spacing)
    cp_len = int(sym_len * cp_ratio)
    total_sym_len = sym_len + cp_len
    n_ofdm_symbols = N // total_sym_len

    # subcarriers sit in bins -n_sc//2 .. n_sc/2-1 so the occupied band is
    # centered on fc as the metadata claims (ifft(data, n=sym_len) would put
    # them in bins 0..n_sc-1, shifting the whole band up by bw/2); for even
    # n_sc the bin set is half a subcarrier low, compensated at the mixer
    if n_sc >= sym_len:
        raise ValueError('OFDM occupied band exceeds the sample rate')
    sc_bins = (np.arange(n_sc) - n_sc // 2) % sym_len
    mixer_offset = sc_spacing / 2 if n_sc % 2 == 0 else 0.0
    sig = np.zeros(N, dtype=complex)
    for s_idx in range(n_ofdm_symbols):
        data = (rng.choice([-1, 1], n_sc) + 1j * rng.choice([-1, 1], n_sc)) / np.sqrt(2)
        freq_bins = np.zeros(sym_len, dtype=complex)
        freq_bins[sc_bins] = data
        ofdm_sym = np.fft.ifft(freq_bins)
        with_cp = np.concatenate([ofdm_sym[-cp_len:], ofdm_sym])
        start = s_idx * total_sym_len
        end = min(start + len(with_cp), N)
        sig[start:end] = with_cp[:end - start]

    sig = sig * np.exp(1j * 2 * np.pi * (fc + mixer_offset) * t)
    sig = _set_power(sig, power_dbm)

    occ_bw = n_sc * sc_spacing
    sym_duration = 1 / sc_spacing
    cp_duration = sym_duration * cp_ratio
    meta = {
        'type': 'OFDM', 'center_frequency_MHz': fc / 1e6,
        'n_subcarriers': n_sc, 'subcarrier_spacing_kHz': sc_spacing / 1e3,
        'cp_ratio': cp_ratio, 'cp_duration_us': round(cp_duration * 1e6, 2),
        'symbol_duration_us': round(sym_duration * 1e6, 2),
        'occupied_bandwidth_MHz': round(occ_bw / 1e6, 3),
        'power_dBm': power_dbm, 'bandwidth_MHz': round(occ_bw / 1e6, 3),
    }
    return sig, meta


def generate_fsk(rng, fs, N, t, params):
    """Generate M-FSK signal."""
    M = params.get('M', 2)
    fc = params['fc']
    sym_rate = params['sym_rate']
    freq_dev = params['freq_deviation']
    power_dbm = params['power_dbm']

    sps = int(fs / sym_rate)
    n_sym = N // sps + 10
    symbols = rng.randint(0, M, n_sym)
    freq_offsets = np.linspace(-(M - 1) / 2, (M - 1) / 2, M) * freq_dev

    inst_freq = np.zeros(N)
    for i, s in enumerate(symbols):
        start = i * sps
        end = min(start + sps, N)
        inst_freq[start:end] = freq_offsets[s]

    phase = 2 * np.pi * fc * t + 2 * np.pi * np.cumsum(inst_freq) / fs
    sig = np.exp(1j * phase)
    sig = _set_power(sig, power_dbm)

    mod_index = freq_dev / sym_rate
    bw = M * freq_dev + 2 * sym_rate  # rough estimate
    meta = {
        'type': f'{M}FSK', 'center_frequency_MHz': fc / 1e6,
        'symbol_rate_kHz': sym_rate / 1e3, 'freq_deviation_kHz': freq_dev / 1e3,
        'modulation_index': round(mod_index, 2),
        'M': M, 'bits_per_symbol': int(np.log2(M)), 'power_dBm': power_dbm,
        'bandwidth_MHz': round(bw / 1e6, 3),
    }
    return sig, meta


def apply_burst(sig, rng, N, params):
    """Apply burst windowing to any signal."""
    start_frac = params['start_frac']
    end_frac = params['end_frac']
    start_idx = int(start_frac * N)
    end_idx = int(end_frac * N)
    ramp = int(0.01 * N)

    window = np.zeros(N)
    window[start_idx:end_idx] = 1.0
    if ramp > 0:
        window[start_idx:start_idx + ramp] = np.linspace(0, 1, ramp)
        window[end_idx - ramp:end_idx] = np.linspace(1, 0, ramp)

    # Re-normalize power during active period
    active = window > 0.5
    power_before = np.mean(np.abs(sig[active]) ** 2) if np.any(active) else 1
    sig_burst = sig * window
    duty_cycle = end_frac - start_frac

    return sig_burst, {
        'active_window': f'{start_frac * 100:.0f}%-{end_frac * 100:.0f}%',
        'duty_cycle': round(duty_cycle, 2),
        'snr_loss_dB': round(10 * np.log10(duty_cycle), 1),
    }


# Registry
SIGNAL_GENERATORS = {
    'BPSK': lambda rng, fs, N, t, p: generate_psk(rng, fs, N, t, {**p, 'M': 2}),
    'QPSK': lambda rng, fs, N, t, p: generate_psk(rng, fs, N, t, {**p, 'M': 4}),
    '8PSK': lambda rng, fs, N, t, p: generate_psk(rng, fs, N, t, {**p, 'M': 8}),
    '16QAM': lambda rng, fs, N, t, p: generate_qam(rng, fs, N, t, {**p, 'M': 16}),
    '64QAM': lambda rng, fs, N, t, p: generate_qam(rng, fs, N, t, {**p, 'M': 64}),
    'FM': generate_fm,
    'AM': generate_am,
    'Chirp': generate_chirp,
    'OFDM': generate_ofdm,
    '2FSK': lambda rng, fs, N, t, p: generate_fsk(rng, fs, N, t, {**p, 'M': 2}),
    '4FSK': lambda rng, fs, N, t, p: generate_fsk(rng, fs, N, t, {**p, 'M': 4}),
}
