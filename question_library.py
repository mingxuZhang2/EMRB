"""
EMRB Question Library: Question generators for L4 problems.
Each returns (question_text, ground_truth, rubric).
"""
import numpy as np

# Marker for the deterministic scorer of the generic (non-repaired) L4
# question instances (evaluation/l4_generic_verifier.py). Question texts are
# unchanged, so stored responses stay --score-only re-evaluable.
# Single-sourced so the stamped marker can never drift from the scorer.
from evaluation.l4_generic_verifier import SCORER_VERSION as L4_GENERIC_SCORING


def _signal_center_mhz(signal):
    """Return the center frequency represented by a signal's metadata."""
    if 'center_frequency_MHz' in signal:
        return float(signal['center_frequency_MHz'])
    return (float(signal['sweep_start_MHz']) + float(signal['sweep_end_MHz'])) / 2


def _signal_interval_mhz(signal):
    """Return the occupied-frequency interval represented by the metadata."""
    if 'Chirp' in signal.get('type', ''):
        return float(signal['sweep_start_MHz']), float(signal['sweep_end_MHz'])

    center = _signal_center_mhz(signal)
    bandwidth = float(signal.get('bandwidth_MHz', 0.0))
    return center - bandwidth / 2, center + bandwidth / 2


def _merged_occupied_intervals(signals, band_lo, band_hi):
    """Merge all generated-signal intervals after clipping to the observed band."""
    intervals = []
    for signal in signals:
        lo, hi = _signal_interval_mhz(signal)
        lo, hi = max(float(band_lo), lo), min(float(band_hi), hi)
        if hi > lo:
            intervals.append((lo, hi))

    intervals.sort()
    merged = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return [(lo, hi) for lo, hi in merged]


def qt_symbol_rate_mod_order(signals, rng):
    """QT01: Estimate symbol rate + modulation order for 1-2 digital signals."""
    digital = [s for s in signals if s['type'] in ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM')]
    if len(digital) < 1:
        return None
    targets = digital[:2] if len(digital) >= 2 else digital[:1]

    if len(targets) == 2:
        f1, f2 = targets[0]['center_frequency_MHz'], targets[1]['center_frequency_MHz']
        lo, hi = min(f1, f2) - 1, max(f1, f2) + 1
        q = (f"There are two digitally modulated signals in the {lo:+.1f}~{hi:+.1f} MHz region. "
             f"Please estimate their symbol rates (ksym/s) and modulation orders respectively, and explain how you distinguished between the two signals.")
        gt = {f"signal_{i+1}": {"type": s['type'], "symbol_rate_ksps": s['symbol_rate_kHz'],
              "M": s['M'], "center_frequency_MHz": s['center_frequency_MHz']}
              for i, s in enumerate(targets)}
        rubric = {"points": 20, "scoring": L4_GENERIC_SCORING}
        for i, s in enumerate(targets):
            rubric[f"symbol_rate_{i+1}"] = {"pts": 4, "tolerance": "±15%", "answer": s['symbol_rate_kHz']}
            mod_scoring = {s['type']: 4, 'PSK': 2, 'QAM': 2, 'digital': 1}
            rubric[f"mod_order_{i+1}"] = {"pts": 4, "scoring": mod_scoring, "answer": s['type']}
        rubric["reasoning"] = {"pts": 4, "note": "Whether the analysis method is reasonable (spectral nulls, cyclostationary, etc.)"}
    else:
        s = targets[0]
        q = (f"There is a digitally modulated signal near {s['center_frequency_MHz']:+.1f} MHz. "
             f"Please estimate its symbol rate (ksym/s) and modulation order.")
        gt = {"symbol_rate_ksps": s['symbol_rate_kHz'], "type": s['type'], "M": s['M'],
              "center_frequency_MHz": s['center_frequency_MHz']}
        # 'reasoning' was never asked in this variant; its 4 pts move onto the
        # two requested outputs
        rubric = {"points": 20, "scoring": L4_GENERIC_SCORING,
                  "symbol_rate": {"pts": 10, "tolerance": "±15%", "answer": s['symbol_rate_kHz']},
                  "mod_order": {"pts": 10, "answer": s['type']}}
    return q, gt, rubric


def qt_fm_params(signals, rng):
    """QT02: FM parameter estimation."""
    fm_sigs = [s for s in signals if s['type'] == 'FM']
    if not fm_sigs:
        return None
    s = fm_sigs[0]
    q = (f"For the FM signal near {s['center_frequency_MHz']:+.1f} MHz, please estimate: "
         f"(a) frequency deviation (kHz), (b) maximum modulating frequency (kHz), "
         f"(c) modulation index beta, (d) verify whether the observed bandwidth is consistent with Carson's rule.")
    # For multitone FM the scalar "modulation index" is convention-dependent:
    # accept beta w.r.t. the highest modulating tone (the stored GT, Carson's
    # deviation ratio) and w.r.t. the fundamental. Carson bandwidth is NOT
    # convention-dependent — it is defined with the highest modulating tone.
    deviation = float(s['frequency_deviation_kHz'])
    f_max = float(s['max_modulating_freq_kHz'])
    f_fund = float(s['modulating_frequency_kHz'])
    gt = {
        "frequency_deviation_kHz": s['frequency_deviation_kHz'],
        "max_modulating_freq_kHz": s['max_modulating_freq_kHz'],
        "modulation_index": s['modulation_index'],
        "modulation_index_accepted": sorted({float(s['modulation_index']),
                                             round(deviation / f_max, 2),
                                             round(deviation / f_fund, 2)}),
        "carson_bandwidth_kHz": s['carson_bandwidth_kHz'],
        "carson_bandwidth_accepted_kHz": sorted({
            float(s['carson_bandwidth_kHz']),
            round(2 * (deviation + f_max), 1)}),
    }
    # 'reasoning' was never a sub-question; its 3 pts move onto (a)-(d)
    rubric = {"points": 20, "scoring": L4_GENERIC_SCORING,
              "deviation": {"pts": 6, "tolerance": "±20%"},
              "mod_freq": {"pts": 5, "tolerance": "±30%"},
              "mod_index": {"pts": 4, "tolerance": "±30%"},
              "carson_value": {"pts": 3, "tolerance": "±20%"},
              "carson_verdict": {"pts": 2, "note": "observed bandwidth is consistent with Carson by construction"}}
    return q, gt, rubric


def qt_chirp_radar(signals, rng):
    """QT03: Chirp TBP, processing gain, range resolution."""
    chirps = [s for s in signals if s['type'] == 'Chirp (LFM)']
    if not chirps:
        return None
    s = chirps[0]
    q = (f"There is a linear frequency modulated (Chirp) signal in the +{s['sweep_start_MHz']:.0f}~+{s['sweep_end_MHz']:.0f} MHz range. "
         f"Please calculate: (a) time-bandwidth product (TBP), (b) processing gain after matched filtering (dB), "
         f"(c) if this chirp is used for radar ranging, what is the corresponding range resolution in meters?")
    gt = {
        "TBP": s['time_bandwidth_product'],
        "processing_gain_dB": s['processing_gain_dB'],
        "range_resolution_m": s['range_resolution_m'],
    }
    # 'formulas' was never a sub-question; its 3 pts move onto (a)-(c)
    rubric = {"points": 20, "scoring": L4_GENERIC_SCORING,
              "TBP": {"pts": 7, "tolerance": "±10%"},
              "processing_gain": {"pts": 7, "tolerance": "±1dB"},
              "range_resolution": {"pts": 6, "tolerance": "±10%"}}
    return q, gt, rubric


def qt_burst_analysis(signals, rng):
    """QT04: Burst signal modulation, duty cycle, SNR loss."""
    burst = [s for s in signals if 'duty_cycle' in s]
    if not burst:
        return None
    s = burst[0]
    q = (f"There is a burst signal near {s['center_frequency_MHz']:+.1f} MHz. "
         f"Please answer: (a) What is the modulation type of this signal? "
         f"(b) What is its duty cycle over the entire recording? "
         f"(c) If detecting this signal using a full-band FFT, how much will the FFT output SNR decrease (in dB) compared to the SNR during the signal's active period? Please derive the result.")
    base_type = s['type'].replace(' (burst)', '')
    gt = {
        "modulation": base_type,
        "duty_cycle": s['duty_cycle'],
        "snr_loss_dB": s['snr_loss_dB'],
    }
    scoring_map = {base_type: 5, 'PSK': 3, 'digital': 1}
    rubric = {"points": 20, "scoring": L4_GENERIC_SCORING,
              "modulation": {"pts": 5, "scoring": scoring_map},
              "duty_cycle": {"pts": 5, "tolerance": "±0.05"},
              "snr_loss_value": {"pts": 6, "tolerance": "±1dB",
                                 "note": "SNR_loss=10log10(duty_cycle); sign-agnostic"},
              "snr_loss_derivation": {"pts": 4, "note": "derivation must invoke 10log10 of the duty cycle"}}
    return q, gt, rubric


def qt_spectral_gap_link_budget(signals, noise_dbm, fs, rng):
    """QT05: Preserve valid legacy questions and replace blocked-gap cases."""
    digital = sorted(
        [
            s for s in signals
            if 'bandwidth_MHz' in s
            and s['type'] not in ('FM', 'AM-DSB', 'Chirp (LFM)')
        ],
        key=lambda x: x['center_frequency_MHz'],
    )
    if len(digital) < 2:
        return None

    legacy_gaps = []
    for s1, s2 in zip(digital, digital[1:]):
        upper1 = s1['center_frequency_MHz'] + s1['bandwidth_MHz'] / 2
        lower2 = s2['center_frequency_MHz'] - s2['bandwidth_MHz'] / 2
        gap = lower2 - upper1
        if gap > 0:
            legacy_gaps.append((gap, upper1, lower2))
    if not legacy_gaps:
        return None

    legacy_gap_mhz, legacy_lo, legacy_hi = max(legacy_gaps, key=lambda x: x[0])
    rolloff = rng.choice([0.2, 0.25, 0.3])
    mod_type = rng.choice(['16QAM', '64QAM'])
    snr_req = {'16QAM': 17, '64QAM': 23}[mod_type]
    bits_per_sym = {'16QAM': 4, '64QAM': 6}[mod_type]

    blockers = []
    for signal in signals:
        signal_lo, signal_hi = _signal_interval_mhz(signal)
        overlap = min(legacy_hi, signal_hi) - max(legacy_lo, signal_lo)
        if overlap > 1e-9:
            blockers.append(signal)

    if not blockers:
        sym_rate = legacy_gap_mhz * 1e6 / (1 + rolloff)
        data_rate = sym_rate * bits_per_sym
        noise_psd = noise_dbm - 10 * np.log10(fs)
        noise_in_band = noise_psd + 10 * np.log10(legacy_gap_mhz * 1e6)
        req_power = noise_in_band + snr_req

        q = (f"Analyze the spectral environment comprehensively: if deploying a new {mod_type} communication link "
             f"in the {legacy_lo:+.2f}~{legacy_hi:+.2f} MHz spectral gap (rolloff={rolloff}), "
             f"(a) what is the maximum available contiguous bandwidth in MHz? "
             f"(b) what is the maximum achievable data rate in kbps? "
             f"(c) under the current noise environment (noise floor approx. {noise_dbm} dBm/{fs/1e6:.0f}MHz), "
             f"what minimum received signal power in the link bandwidth is required to meet the minimum SNR for {mod_type} ({snr_req} dB)?")
        gt = {
            "available_gap_MHz": round(legacy_gap_mhz, 3),
            "symbol_rate_ksps": round(sym_rate / 1e3, 1),
            "data_rate_kbps": round(data_rate / 1e3, 1),
            "required_power_dBm": round(req_power, 1),
        }
        rubric = {"points": 20, "scoring": L4_GENERIC_SCORING,
                  "bandwidth": {"pts": 6, "tolerance": "±0.05MHz"},
                  "data_rate": {"pts": 7, "tolerance": "±20%"},
                  "power": {"pts": 7, "tolerance": "±3dB"}}
        return q, gt, rubric

    band_lo = -float(fs) / 2 / 1e6
    band_hi = float(fs) / 2 / 1e6
    occupied = _merged_occupied_intervals(signals, band_lo, band_hi)
    gaps = []
    cursor = band_lo
    for lo, hi in occupied:
        if lo > cursor:
            gaps.append((lo - cursor, cursor, lo))
        cursor = max(cursor, hi)
    if cursor < band_hi:
        gaps.append((band_hi - cursor, cursor, band_hi))

    guard_mhz = 0.05
    feasible = [gap for gap in gaps if gap[0] > 2 * guard_mhz]
    if not feasible:
        return None

    raw_gap_mhz, raw_lo, raw_hi = max(feasible, key=lambda item: (item[0], -item[1]))
    usable_lo = raw_lo + guard_mhz
    usable_hi = raw_hi - guard_mhz
    usable_gap_mhz = usable_hi - usable_lo
    sym_rate = usable_gap_mhz * 1e6 / (1 + rolloff)
    data_rate = sym_rate * bits_per_sym
    noise_psd = noise_dbm - 10 * np.log10(fs)
    noise_in_band = noise_psd + 10 * np.log10(usable_gap_mhz * 1e6)
    req_power = noise_in_band + snr_req

    q = (f"Analyze the complete {band_lo:+.0f} to {band_hi:+.0f} MHz observed band and identify the largest "
         f"contiguous interval not occupied by any generated signal. Reserve a 0.05 MHz guard band at each "
         f"edge of that interval. (a) Report the raw gap boundaries and bandwidth, followed by the usable "
         f"boundaries and bandwidth after applying the guards. (b) A new {mod_type} link with rolloff={rolloff} "
         f"uses the entire usable interval. Calculate its maximum symbol rate and raw data rate. "
         f"(c) The measured noise floor is {noise_dbm} dBm across {fs/1e6:.0f} MHz. Calculate the minimum "
         f"received signal power required in the new link bandwidth to meet an SNR of {snr_req} dB. "
         "In the final answer, return this question as one JSON object with fields "
         "raw_gap_bounds_MHz, raw_gap_MHz, usable_gap_bounds_MHz, usable_gap_MHz, "
         "symbol_rate_ksps, data_rate_kbps, noise_in_link_band_dBm, and "
         "minimum_received_power_dBm.")

    gt = {
        "raw_gap_bounds_MHz": [round(raw_lo, 3), round(raw_hi, 3)],
        "raw_gap_MHz": round(raw_gap_mhz, 3),
        "guard_each_side_MHz": guard_mhz,
        "usable_gap_bounds_MHz": [round(usable_lo, 3), round(usable_hi, 3)],
        "available_gap_MHz": round(usable_gap_mhz, 3),
        "modulation": mod_type,
        "rolloff": rolloff,
        "symbol_rate_ksps": round(sym_rate / 1e3, 1),
        "data_rate_kbps": round(data_rate / 1e3, 1),
        "noise_in_link_band_dBm": round(noise_in_band, 1),
        "minimum_received_power_dBm": round(req_power, 1),
    }
    rubric = {"points": 20, "scoring": "l4-deterministic-v1",
              "gap_identification": {"pts": 8, "tolerance": "±0.05MHz per boundary"},
              "link_rate": {"pts": 6, "tolerance": "±15%"},
              "received_power": {"pts": 6, "tolerance": "±2dB"}}
    return q, gt, rubric


def qt_ofdm_params(signals, rng):
    """QT06: OFDM parameter estimation."""
    ofdm = [s for s in signals if s['type'] == 'OFDM']
    if not ofdm:
        return None
    s = ofdm[0]
    q = (f"There is an OFDM signal near {s['center_frequency_MHz']:+.1f} MHz. "
         f"Please estimate: (a) subcarrier spacing (kHz), (b) CP duration (us), "
         f"(c) effective occupied bandwidth (MHz), (d) the useful OFDM symbol duration excluding the CP (us).")
    gt = {
        "subcarrier_spacing_kHz": s['subcarrier_spacing_kHz'],
        "cp_duration_us": s['cp_duration_us'],
        "occupied_bandwidth_MHz": s['occupied_bandwidth_MHz'],
        "symbol_duration_us": s['symbol_duration_us'],
    }
    rubric = {"points": 20, "scoring": L4_GENERIC_SCORING,
              "sc_spacing": {"pts": 6, "tolerance": "±20%"},
              "cp_duration": {"pts": 5, "tolerance": "±30%"},
              "occupied_bw": {"pts": 5, "tolerance": "±20%"},
              "sym_duration": {"pts": 4, "tolerance": "±20%"}}
    return q, gt, rubric


def qt_am_params(signals, rng):
    """QT08: AM parameter estimation."""
    am = [s for s in signals if s['type'] == 'AM-DSB']
    if not am:
        return None
    s = am[0]
    q = (f"There is an AM signal near {s['center_frequency_MHz']:+.1f} MHz. "
         f"Please estimate: (a) modulation depth, (b) modulating signal frequency (kHz), "
         f"(c) transmission efficiency eta (ratio of signal power to total power), (d) occupied bandwidth (kHz).")
    gt = {
        "modulation_depth": s['modulation_depth'],
        "modulating_freq_kHz": s['modulating_frequency_kHz'],
        "efficiency": s['efficiency'],
        "bandwidth_kHz": round(s['bandwidth_MHz'] * 1e3, 1),
    }
    rubric = {"points": 20, "scoring": L4_GENERIC_SCORING,
              "mod_depth": {"pts": 5, "tolerance": "±15%"},
              "mod_freq": {"pts": 5, "tolerance": "±20%"},
              "efficiency": {"pts": 5, "tolerance": "±30%", "note": "η=m²/(2+m²)"},
              "bandwidth": {"pts": 5, "tolerance": "±20%"}}
    return q, gt, rubric


def qt_interference_sir(signals, rng):
    """QT07: Analyze spectral separation and isolation of the critical pair."""
    if len(signals) < 2:
        return None

    # Retain the original applicability rule so repairing QT07 does not alter
    # which problem slots receive a fallback question.
    ordered = sorted(signals, key=lambda signal: signal['center_frequency_MHz'])
    closest_center_separation = min(
        abs(right['center_frequency_MHz'] - left['center_frequency_MHz'])
        for left, right in zip(ordered, ordered[1:])
    )
    if closest_center_separation > 3:
        return None

    candidates = []
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            s1, s2 = signals[i], signals[j]
            lo1, hi1 = _signal_interval_mhz(s1)
            lo2, hi2 = _signal_interval_mhz(s2)
            overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
            guard_gap = max(0.0, max(lo1, lo2) - min(hi1, hi2))
            candidates.append({
                'i': i, 'j': j, 's1': s1, 's2': s2,
                'lo1': lo1, 'hi1': hi1, 'lo2': lo2, 'hi2': hi2,
                'overlap': overlap, 'guard_gap': guard_gap,
            })

    overlapping = [item for item in candidates if item['overlap'] > 0]
    if overlapping:
        pair = max(overlapping, key=lambda item: (item['overlap'], -item['i'], -item['j']))
    else:
        pair = min(candidates, key=lambda item: (item['guard_gap'], item['i'], item['j']))

    s1, s2 = pair['s1'], pair['s2']
    if float(s1['power_dBm']) <= float(s2['power_dBm']):
        target, other = s1, s2
        target_id, other_id = pair['i'] + 1, pair['j'] + 1
        target_interval = (pair['lo1'], pair['hi1'])
    else:
        target, other = s2, s1
        target_id, other_id = pair['j'] + 1, pair['i'] + 1
        target_interval = (pair['lo2'], pair['hi2'])

    center_sep = abs(_signal_center_mhz(s1) - _signal_center_mhz(s2))
    overlap = pair['overlap']
    guard_gap = pair['guard_gap']
    target_bw = target_interval[1] - target_interval[0]
    clean_target_bw = max(0.0, target_bw - overlap)
    overlap_fraction = overlap / target_bw * 100 if target_bw > 0 else 0.0
    power_ratio = float(target['power_dBm']) - float(other['power_dBm'])
    relation = 'overlapping' if overlap > 0 else 'separated'
    full_isolation = overlap == 0

    q = ("Identify the most spectrally critical pair of signals in the capture. If any occupied-bandwidth "
         "intervals overlap, select the pair with the largest overlap; otherwise select the pair with the "
         "smallest guard gap. (a) Report both signal types, center frequencies, occupied-frequency intervals, "
         "and their center-frequency separation. (b) Determine whether the intervals overlap. Report the "
         "overlap bandwidth if they overlap, or the guard-gap bandwidth if they are separated. "
         "(c) Treat the weaker signal as the target and calculate its target-to-other received-power ratio in dB. "
         "(d) Determine whether frequency-selective filtering alone can preserve the target's full occupied "
         "bandwidth while rejecting the other signal. If the pair is separated, report the available transition "
         "bandwidth. If it overlaps, report the target bandwidth outside the overlap and the percentage of the "
         "target bandwidth that is overlapped. In the final answer, return this question as one JSON object "
         "with fields pair, center_separation_MHz, spectral_relation, overlap_MHz, guard_gap_MHz, target_type, "
         "target_to_other_power_ratio_dB, full_band_isolation_possible, available_transition_bandwidth_MHz, "
         "nonoverlapped_target_bandwidth_MHz, and overlapped_target_fraction_pct. Each entry in pair must "
         "contain type, center_MHz, and occupied_interval_MHz.")

    gt = {
        "pair": [
            {"signal_id": pair['i'] + 1, "type": s1['type'],
             "center_frequency_MHz": round(_signal_center_mhz(s1), 3),
             "occupied_interval_MHz": [round(pair['lo1'], 3), round(pair['hi1'], 3)]},
            {"signal_id": pair['j'] + 1, "type": s2['type'],
             "center_frequency_MHz": round(_signal_center_mhz(s2), 3),
             "occupied_interval_MHz": [round(pair['lo2'], 3), round(pair['hi2'], 3)]},
        ],
        "center_separation_MHz": round(center_sep, 3),
        "spectral_relation": relation,
        "overlap_MHz": round(overlap, 3),
        "guard_gap_MHz": round(guard_gap, 3),
        "target_signal_id": target_id,
        "target_type": target['type'],
        "other_signal_id": other_id,
        "target_to_other_power_ratio_dB": round(power_ratio, 1),
        "full_band_isolation_possible": full_isolation,
        "available_transition_bandwidth_MHz": round(guard_gap, 3) if full_isolation else 0.0,
        "nonoverlapped_target_bandwidth_MHz": round(clean_target_bw, 3),
        "overlapped_target_fraction_pct": round(overlap_fraction, 1),
    }
    rubric = {"points": 20, "scoring": "l4-deterministic-v1",
              "pair_and_separation": {"pts": 5, "tolerance": "±0.1MHz"},
              "spectral_relation": {"pts": 5, "tolerance": "±0.05MHz"},
              "target_power_ratio": {"pts": 4, "tolerance": "±2dB"},
              "isolation_geometry": {"pts": 6, "tolerance": "±10%"}}
    return q, gt, rubric


def qt_channel_capacity(signals, noise_dbm, fs, rng):
    """QT10: Shannon channel capacity estimation."""
    # Preserve the original L4 candidate set after FSK metadata gained an
    # explicit bits_per_symbol field for L5 scoring.
    digital = [
        s for s in signals
        if s.get('bits_per_symbol') and 'FSK' not in s.get('type', '')
    ]
    if not digital:
        return None
    s = rng.choice(digital) if len(digital) > 1 else digital[0]

    bw = s['bandwidth_MHz'] * 1e6
    noise_psd = noise_dbm - 10 * np.log10(fs)
    noise_in_band = noise_psd + 10 * np.log10(bw)
    snr_db = s['power_dBm'] - noise_in_band
    snr_linear = 10 ** (snr_db / 10)
    capacity = bw * np.log2(1 + snr_linear)

    q = (f"For the {s['type']} signal at {s['center_frequency_MHz']:+.1f} MHz, "
         f"given a noise floor of {noise_dbm} dBm/{fs/1e6:.0f}MHz: "
         f"(a) estimate the SNR (dB) of this signal within its occupied bandwidth. "
         f"(b) According to the Shannon formula, what is the theoretical channel capacity upper bound in Mbps? "
         f"(c) How far is the current modulation scheme's spectral efficiency ({s['bits_per_symbol']} bits/symbol) from the Shannon limit?")

    actual_rate = s['symbol_rate_kHz'] * 1e3 * s['bits_per_symbol'] if 'symbol_rate_kHz' in s else 0
    spectral_eff_actual = s['bits_per_symbol'] * (s.get('symbol_rate_kHz', 0) * 1e3) / bw if bw > 0 else 0
    shannon_eff = np.log2(1 + snr_linear)
    gap = shannon_eff - spectral_eff_actual

    gt = {
        "snr_dB": round(snr_db, 1),
        "shannon_capacity_Mbps": round(capacity / 1e6, 2),
        "spectral_efficiency_gap": round(gap, 2),
    }

    # An answer must be scored against ONE self-consistent convention, never a
    # per-field best-of (an inconsistent mix must not earn full credit).
    # Within a convention the gap accepts two readings, because the prompt
    # itself calls bits_per_symbol the "spectral efficiency" while the GT uses
    # the true efficiency Rs*b/BW.
    def _convention(convention_snr_db):
        eff = np.log2(1 + 10 ** (convention_snr_db / 10))
        return {
            "snr_dB": round(convention_snr_db, 1),
            "shannon_capacity_Mbps": round(bw * eff / 1e6, 2),
            "spectral_efficiency_gap": sorted({
                round(eff - spectral_eff_actual, 2),
                round(eff - s['bits_per_symbol'], 2)}),
        }

    conventions = [_convention(snr_db)]
    # power_dBm is the burst's ACTIVE-period power; a model measuring average
    # power over the full recording legitimately gets snr + 10log10(duty)
    if 'duty_cycle' in s and 0 < float(s['duty_cycle']) < 1:
        conventions.append(
            _convention(snr_db + 10 * np.log10(float(s['duty_cycle']))))
    gt["accepted_conventions"] = conventions
    rubric = {"points": 20, "scoring": L4_GENERIC_SCORING,
              "snr": {"pts": 6, "tolerance": "±3dB"},
              "capacity": {"pts": 8, "tolerance": "±20%"},
              "gap_analysis": {"pts": 6, "note": "Correctly computes the gap between current efficiency and the Shannon limit"}}
    return q, gt, rubric


# Registry: question type -> generator function
QUESTION_GENERATORS = {
    'QT01': qt_symbol_rate_mod_order,
    'QT02': qt_fm_params,
    'QT03': qt_chirp_radar,
    'QT04': qt_burst_analysis,
    'QT05': qt_spectral_gap_link_budget,  # needs noise_dbm, fs
    'QT06': qt_ofdm_params,
    'QT07': qt_interference_sir,
    'QT08': qt_am_params,
    'QT10': qt_channel_capacity,  # needs noise_dbm, fs
}
