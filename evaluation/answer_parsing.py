"""Shared parsing utilities for the deterministic L1/L3 verifiers.

These verifiers score the existing labeled-line answer format
(``Q1a: <prose with values and units>``) without any prompt change, so old
model responses stay score-only re-evaluable. The audit's binding rules are
implemented here: every value is read from the sub-answer it was requested in,
units are converted to the criterion's canonical unit, per-signal values are
bound by signal identity (type token or center-frequency anchor) or by
position — never by a global number bag — and booleans use negation-resolved
polarity that rejects contradictory answers.
"""
import re

# --- text normalization -----------------------------------------------------

_SUPERSCRIPTS = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
_SUBSCRIPTS = {ord(c): None for c in '₀₁₂₃₄₅₆₇₈₉'}


def normalize_text(text):
    """Fold unicode notation into plain ASCII scientific notation."""
    text = str(text or '').translate(_SUPERSCRIPTS).translate(_SUBSCRIPTS)
    # unicode minus / full-width minus -> ASCII hyphen (en-dash is a range)
    text = text.replace('−', '-').replace('－', '-')
    # digit-group separators: thin/no-break spaces, "65 536", "65,536"
    text = re.sub(r'(?<=\d)[   ](?=\d)', '', text)
    text = re.sub(r'(?<=\d)[ ,](?=\d{3}(?:\D|$))', '', text)
    # A×10^B, A×10B (from superscripts), A*10^B, A·10^B -> AeB
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*[×x*·]\s*10\s*\^?\s*\(?([+-]?\d+)\)?',
        r'\1e\2', text)
    # standalone 10^B -> 1eB
    text = re.sub(r'(?<![\d.])10\s*\^\s*\(?([+-]?\d+)\)?', r'1e\1', text)
    return text


# --- quantity extraction ----------------------------------------------------

_UNIT_RE = (
    r'(dBm/Hz|dBW/Hz|dB/Hz|GHz/s|MHz/ms|MHz/us|MHz/s|kHz/ms|kHz/us|kHz/s|'
    r'GHz|MHz|kHz|Hz|dBm|dBJ|dBc|dB|mW|uW|µW|W|'
    r'Msym/s|ksym/s|sym/s|Msps|ksps|sps|Mbaud|kbaud|baud|Mbps|kbps|'
    r'bps/Hz|bits/s/Hz|bit/s/Hz|b/s/Hz|bps|ms|us|μs|ns|s|km|cm|m|'
    r'dBJ|pJ|nJ|uJ|µJ|mJ|J|bits?|bit|samples?|%)'
)
_NUM_RE = r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
# unit boundary is a lookahead, not \b: '%' has no word boundary before ')'
_QTY_RE = re.compile(rf'({_NUM_RE})\s*(?:{_UNIT_RE}(?![0-9A-Za-z]))?')

UNIT_FAMILIES = {
    'freq': {'Hz': 1.0, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9},
    'time': {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'μs': 1e-6, 'ns': 1e-9},
    'power_lin': {'W': 1e3, 'mW': 1.0, 'uW': 1e-3, 'µW': 1e-3},  # canonical mW
    # dB (ratio) and dBm (absolute power) are distinct dimensions and must
    # not be interchangeable (remediation log §8.1.5); dBJ/dBc-labeled values
    # belong to neither family and are dropped from both
    'db': {'dB': 1.0},
    'dbm': {'dBm': 1.0},
    'dbj': {'dBJ': 1.0},
    # spectral density is its own dimension: a dBm-labeled total power must
    # not satisfy a dBm/Hz criterion or vice versa (remediation log §10.5);
    # dBW/Hz and dB/Hz are tokenized so they cannot leak into dBm/dB, but
    # convert additively and therefore belong to no multiplicative family
    'psd': {'dBm/Hz': 1.0},
    'sweeprate': {'MHz/ms': 1.0, 'GHz/s': 1.0, 'MHz/us': 1e3, 'kHz/us': 1.0,
                  'kHz/ms': 1e-3, 'MHz/s': 1e-3, 'kHz/s': 1e-6},
    'energy': {'J': 1.0, 'mJ': 1e-3, 'uJ': 1e-6, 'µJ': 1e-6, 'nJ': 1e-9,
               'pJ': 1e-12},
    'symrate': {'sps': 1.0, 'ksps': 1e3, 'Msps': 1e6,
                'sym/s': 1.0, 'ksym/s': 1e3, 'Msym/s': 1e6,
                'baud': 1.0, 'kbaud': 1e3, 'Mbaud': 1e6},
    'bitrate': {'bps': 1.0, 'kbps': 1e3, 'Mbps': 1e6},
    'eff': {'bps/Hz': 1.0, 'bits/s/Hz': 1.0, 'bit/s/Hz': 1.0, 'b/s/Hz': 1.0},
    'length': {'m': 1.0, 'km': 1e3, 'cm': 1e-2},
    'ratio': {'%': 0.01},  # unitless values are already fractions
    'count': {'bits': 1.0, 'bit': 1.0, 'samples': 1.0, 'sample': 1.0},
    'none': {},
}


def extract_quantities(text):
    """Return [(value, unit_or_None, char_position)] from normalized text."""
    out = []
    for m in _QTY_RE.finditer(normalize_text(text)):
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        out.append((value, m.group(2), m.start()))
    return out


def quantities_in_family(text, family, target_unit, explicit_only=False):
    """Values convertible to ``target_unit``; unitless values are assumed to
    already be in the target unit (models frequently omit the requested unit).
    ``explicit_only`` keeps only values written with a family unit."""
    table = UNIT_FAMILIES[family]
    scale_to_target = table.get(target_unit, 1.0)
    values = []
    for value, unit, pos in extract_quantities(text):
        if unit is None:
            if not explicit_only:
                values.append((value, pos))
        elif unit in table:
            values.append((value * table[unit] / scale_to_target, pos))
    return values


# Explicit answer revisions must take precedence over superseded scratch
# values.  This is opt-in because L2/L4 scorer versions have not changed;
# L1/L3 enable it through local wrappers in their verifier modules.
_FINAL_VALUE_RE = re.compile(
    r'(?i)(?:\bfinal\s+(?:answer|result|value|estimate|correction)\b|'
    r'\bcorrect(?:ed)?\s+(?:answer|result|value)\b|'
    r'\brevised\s+(?:answer|result|value|estimate)\b|'
    r'\bcorrection\s*(?=[:=])|\bactually\s*(?=[:,])|'
    r'最终(?:答案|结果|数值|估计)?|最后(?:答案|结果|数值)|'
    r'更正(?:为|后)?|修正(?:为|后)?|应为|改为|正确(?:答案|结果|数值)(?:是|为)?)')

_REJECTED_BEFORE_RE = re.compile(
    r'(?i)(?:\bnot|\binstead\s+of|\brather\s+than|\bwrong|\bincorrect|'
    r'\binvalid|\bdiscard(?:ed)?|\bignore(?:d)?|\bretract(?:ed)?|'
    r'\bsuperseded|不是|并非|而非|错误(?:的)?|不正确(?:的)?|无效(?:的)?|'
    r'舍弃|忽略|撤回)\s*(?:(?:the\s+)?(?:answer|result|value|estimate)|'
    r'答案|结果|数值|估计)?\s*(?:is|was|=|:|为|是)?\s*$')

_REJECTED_AFTER_RE = re.compile(
    r'(?i)^\s*(?:is|was|were|are|has\s+been|被)?\s*'
    r'(?:(?:explicitly|previously|later)\s+)?'
    r'(?:wrong|incorrect|invalid|retracted|superseded|discarded|'
    r'not\s+(?:the\s+)?(?:answer|result|correct)|错误|不正确|无效|'
    r'已?撤回|已?作废|已?舍弃)')


def asserted_quantities_in_family(text, family, target_unit,
                                  explicit_only=False):
    """Return numeric candidates that remain asserted by the answer.

    When an answer explicitly introduces a final or corrected value, only
    candidates in the last such revision are eligible.  Numeric mentions
    explicitly described as wrong, retracted, or superseded are removed.
    Ordinary derivations remain untouched, including answers that compare a
    measured value with a threshold in the same sub-answer.
    """
    norm = normalize_text(text)
    candidates = quantities_in_family(
        norm, family, target_unit, explicit_only=explicit_only)
    eligible = []
    for value, pos in candidates:
        match = _QTY_RE.match(norm, pos)
        end = match.end() if match else pos
        before = norm[max(0, pos - 80):pos]
        after = norm[end:end + 80]
        if _REJECTED_BEFORE_RE.search(before):
            continue
        if _REJECTED_AFTER_RE.search(after):
            continue
        eligible.append((value, pos))

    markers = list(_FINAL_VALUE_RE.finditer(norm))
    for marker in reversed(markers):
        revised = [item for item in eligible if item[1] >= marker.end()]
        if revised:
            return revised
    return eligible


# --- scalar scoring ---------------------------------------------------------

def score_scalar(text, reference, family, target_unit,
                 full, half, mode='rel', candidate_policy='best'):
    """Score one requested scalar inside its own sub-answer.

    mode: 'rel' (fractional), 'abs' (same unit), 'exact_int', 'factor'
    (multiplicative window, e.g. factor-of-2), or 'interval' (reference is a
    (lo, hi) pair of equally-acceptable definitions, tolerances expand it).
    Returns (ratio, reason).
    """
    if reference is None:
        return 0.0, 'no reference value'
    if candidate_policy == 'best':
        candidates = quantities_in_family(text, family, target_unit)
    elif candidate_policy == 'asserted':
        candidates = asserted_quantities_in_family(text, family, target_unit)
    else:
        raise ValueError(f'unknown candidate policy: {candidate_policy}')
    if not candidates:
        return 0.0, f'no {target_unit or "numeric"} value found'
    best = None
    for value, _ in candidates:
        if mode == 'rel':
            err = abs(value - reference) / max(abs(reference), 1e-12)
        elif mode == 'abs':
            err = abs(value - reference)
        elif mode == 'exact_int':
            # the answer itself must be an integer; 2.51 is not "3"
            exact = (abs(value - round(value)) <= 1e-6
                     and round(value) == round(reference))
            err = 0.0 if exact else float('inf')
        elif mode == 'factor':
            if value <= 0 or reference <= 0:
                err = float('inf')
            else:
                ratio = max(value / reference, reference / value)
                err = ratio - 1.0
        elif mode == 'interval':
            lo, hi = min(reference), max(reference)
            if value < lo:
                err = (lo - value) / max(abs(lo), 1e-12)
            elif value > hi:
                err = (value - hi) / max(abs(hi), 1e-12)
            else:
                err = 0.0
        else:
            raise ValueError(mode)
        if best is None or err < best[0]:
            best = (err, value)
    err, value = best
    if mode == 'interval':
        reference = f'[{min(reference):g}, {max(reference):g}]'
        if err <= full + 1e-9:
            return 1.0, f'found={value:.6g} in {reference}'
        if half is not None and err <= half + 1e-9:
            return 0.5, f'found={value:.6g} near {reference} (partial)'
        return 0.0, f'best={value:.6g} outside {reference}'
    if err <= full + 1e-9:
        return 1.0, f'found={value:.6g} ref={reference:.6g}'
    if half is not None and err <= half + 1e-9:
        return 0.5, f'found={value:.6g} ref={reference:.6g} (partial)'
    return 0.0, f'best={value:.6g} ref={reference:.6g} outside tolerance'


def score_scalar_any(text, references, family, target_unit, full, half,
                     mode='rel', candidate_policy='best'):
    """Score against an accepted set of references; best one wins."""
    best = (0.0, 'no accepted value matched')
    for reference in references:
        ratio, reason = score_scalar(text, reference, family, target_unit,
                                     full, half, mode, candidate_policy)
        if ratio > best[0]:
            best = (ratio, reason)
    return best


# --- per-signal binding -----------------------------------------------------

TYPE_TOKENS = {
    'BPSK': ('bpsk',),
    'QPSK': ('qpsk',),
    '8PSK': ('8psk', '8-psk'),
    '16QAM': ('16qam', '16-qam'),
    '64QAM': ('64qam', '64-qam'),
    'FM': ('fm', '调频'),
    'AM-DSB': ('am-dsb', 'am', '调幅'),
    'Chirp (LFM)': ('chirp', 'lfm', 'swept', '扫频', '线性调频'),
    'OFDM': ('ofdm',),
    '2FSK': ('2fsk', '2-fsk', 'fsk'),
    '4FSK': ('4fsk', '4-fsk'),
}

_DIGITAL_TYPES = ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM')

DIGITAL_TOKENS = ('digital', '数字调制', '数字')

# '线性调频' is a chirp, not an FM mention
_TOKEN_PATTERN_OVERRIDES = {'调频': r'(?<!线性)调频'}


def _token_positions(text, tokens):
    positions = []
    for token in tokens:
        pattern = _TOKEN_PATTERN_OVERRIDES.get(
            token, rf'(?<![0-9a-z]){re.escape(token)}(?![0-9a-z])')
        for m in re.finditer(pattern, text, re.IGNORECASE):
            positions.append(m.start())
    return positions


def freq_anchor_positions(text, signal):
    """Positions of frequency mentions identifying the signal: its center
    frequency, or anywhere inside the sweep range for a chirp (±0.5 MHz)."""
    if 'sweep_start_MHz' in signal:
        edges = (float(signal['sweep_start_MHz']),
                 float(signal['sweep_end_MHz']))
        lo, hi = min(edges), max(edges)
    else:
        lo = hi = float(signal['center_frequency_MHz'])
    return [pos for value, unit, pos in extract_quantities(text)
            if unit == 'MHz' and lo - 0.5 <= value <= hi + 0.5]


def signal_anchor(text, signal, signals, center_mhz=None):
    """Earliest identity anchor of a signal: its type token (digital signals
    also answer to 'digital' when unique) or its frequency in MHz."""
    positions = _token_positions(text, TYPE_TOKENS.get(signal['type'], ()))
    if (signal['type'] in _DIGITAL_TYPES
            and sum(s['type'] in _DIGITAL_TYPES for s in signals) == 1):
        positions += _token_positions(text, DIGITAL_TOKENS)
    if center_mhz is not None:
        positions += freq_anchor_positions(text, signal)
    return min(positions) if positions else None


def signal_center_mhz(signal):
    if 'center_frequency_MHz' in signal:
        return float(signal['center_frequency_MHz'])
    return (float(signal['sweep_start_MHz'])
            + float(signal['sweep_end_MHz'])) / 2


def score_per_signal(text, signals, references, family, target_unit,
                     full, half, mode='rel', use_freq_anchor=True,
                     candidate_policy='best'):
    """Score one requested value per signal (audit rule: identity binding
    first, positional binding otherwise; never a global bag).

    Returns (mean_ratio, reason). ``references[i]`` may be a scalar or a list
    of accepted scalars for signal i.
    """
    n = len(signals)
    anchors = [
        signal_anchor(text, s, signals,
                      signal_center_mhz(s) if use_freq_anchor else None)
        for s in signals
    ]
    ratios = []
    details = []
    if all(a is not None for a in anchors) and len(set(anchors)) == n:
        order = sorted(range(n), key=lambda i: anchors[i])
        bounds = [anchors[i] for i in order] + [len(normalize_text(text))]
        for rank, i in enumerate(order):
            segment = normalize_text(text)[bounds[rank]:bounds[rank + 1]]
            refs = references[i] if isinstance(references[i], (list, tuple)) \
                else [references[i]]
            ratio, _ = score_scalar_any(segment, refs, family, target_unit,
                                        full, half, mode, candidate_policy)
            ratios.append(ratio)
            details.append(f"{signals[i]['type']}:{ratio:g}")
        return sum(ratios) / n, 'anchored ' + ' '.join(details)
    # positional: k-th in-family quantity belongs to the k-th signal; values
    # written with an explicit unit are preferred when enough exist, so a
    # bare "M = 4" cannot occupy a symbol-rate slot
    values = quantities_in_family(text, family, target_unit,
                                  explicit_only=True)
    if len(values) < n:
        values = quantities_in_family(text, family, target_unit)
    for i in range(n):
        refs = references[i] if isinstance(references[i], (list, tuple)) \
            else [references[i]]
        if i >= len(values):
            ratios.append(0.0)
            continue
        value = values[i][0]
        best = 0.0
        for reference in refs:
            snippet = f'{value:.10g}'
            ratio, _ = score_scalar(snippet, reference, 'none', None,
                                    full, half, mode, candidate_policy)
            best = max(best, ratio)
        ratios.append(best)
    return sum(ratios) / n, 'positional ' + ' '.join(f'{r:g}' for r in ratios)


# --- polarity (booleans in prose) --------------------------------------------

_NEGATORS = ('not', 'no', 'none', 'non', "n't", 'never', 'without', 'cannot',
             "won't", 'wouldn', 'doesn', 'isn', '不', '不会', '没有', '未',
             '无', '否', '非')


_POST_NEGATORS = ('unlikely', 'improbable', 'not', "n't", 'never', '不')

# clause boundary that does not split decimal numbers; colons are label
# separators ("AM signal: no"), not boundaries
_CLAUSE_RE = re.compile(r'(?<!\d)\.(?!\d)|[;!?\n。；，]')


def _marker_hits(text, markers):
    hits = []
    lowered = str(text or '').lower()
    for marker in markers:
        for m in re.finditer(rf'(?<![0-9a-z]){re.escape(marker.lower())}',
                             lowered):
            # negation windows stop at clause boundaries so "No distortion;
            # the requirement is met" does not negate "met"
            before = _CLAUSE_RE.split(
                lowered[max(0, m.start() - 24):m.start()])[-1]
            after = _CLAUSE_RE.split(lowered[m.end():m.end() + 24])[0]
            negated = any(
                re.search(rf'(?<![0-9a-z]){re.escape(neg)}(?![0-9a-z])'
                          if neg.isascii() else re.escape(neg), before)
                for neg in _NEGATORS)
            negated = negated or any(
                re.search(rf'(?<![0-9a-z]){re.escape(neg)}(?![0-9a-z])', after)
                for neg in _POST_NEGATORS)
            hits.append(not negated)
    return hits


def polarity(text, positive_markers, negative_markers):
    """Resolve a yes/no answer. Returns True/False, or None when absent or
    contradictory. A leading yes/no token dominates mixed prose, but a prose
    verdict that unanimously contradicts it ("Yes, the requirement is not
    met") makes the answer contradictory."""
    stripped = str(text or '').strip().lower()
    lead = re.match(r'^(yes|no|true|false)\b|^(是|否)', stripped)
    body = stripped
    lead_value = None
    lead_strong = False
    if lead:
        lead_value = (lead.group(1) or lead.group(2)) in ('yes', 'true', '是')
        rest = stripped[lead.end():].lstrip()
        # "No," is a verdict; "No distortion occurs" is a determiner
        lead_strong = not rest or not rest[0].isalnum()
        # a verdict token votes separately, so it must not negate the prose
        # ("No, ... met" is a contradiction); a determiner stays in place
        body = rest if lead_strong else stripped
    votes = []
    for hit in _marker_hits(body, positive_markers):
        votes.append(hit)          # negated positive marker -> negative vote
    for hit in _marker_hits(body, negative_markers):
        votes.append(not hit)      # negated negative marker -> positive vote
    unanimous = votes[0] if votes and (all(votes) or not any(votes)) else None
    if lead_strong:
        if unanimous is not None and unanimous != lead_value:
            return None
        return lead_value
    if unanimous is not None:
        return unanimous
    return lead_value


_INDEX_RE = re.compile(
    r'(?:signal|sig|信号)\s*#?(\d)|(?<![0-9a-z])s(\d)(?![0-9a-z])',
    re.IGNORECASE)


def index_reference(chunk, n):
    """0-based signal index from a 'Signal 2' / 'S2' / '信号2' reference."""
    m = _INDEX_RE.search(chunk)
    if m:
        idx = int(m.group(1) or m.group(2)) - 1
        if 0 <= idx < n:
            return idx
    return None


# a hedged mention ("digital may be constant-envelope if PSK-like") is a
# non-answer: it neither claims nor rejects
_HEDGE_PREFIXES = ('may', 'might', 'likely', 'possibl', 'perhaps', 'depend',
                   'uncertain', 'unclear', '可能', '取决', '不确定')


def _prefix_positions(clause, prefixes):
    positions = []
    for prefix in prefixes:
        pattern = (rf'(?<![0-9a-z]){re.escape(prefix)}'
                   if prefix.isascii() else re.escape(prefix))
        positions += [m.start()
                      for m in re.finditer(pattern, clause, re.IGNORECASE)]
    return positions


def _word_positions(clause, words):
    positions = []
    for word in words:
        pattern = (rf'(?<![0-9a-z]){re.escape(word)}(?![0-9a-z])'
                   if word.isascii() else re.escape(word))
        positions += [m.start()
                      for m in re.finditer(pattern, clause, re.IGNORECASE)]
    return positions


def claimed_indices(text, token_sets, signals=None, extra_negator_patterns=()):
    """Which of the ``token_sets`` an enumeration-style answer asserts.

    Mentions (type tokens; 'Signal k'/'S k'/'信号k' references, which models
    number by ascending frequency per Q1's sorted listing; and, when
    ``signals`` is given, identifying frequencies) are collected per clause.
    Each negator or hedge word attaches to its nearest mention in the clause:
    a negated mention rejects its index outright (negative evidence wins), a
    hedged one simply does not claim it."""
    n = len(token_sets)
    norm = normalize_text(text)
    ascending = (sorted(range(n),
                        key=lambda i: signal_center_mhz(signals[i]))
                 if signals else list(range(n)))
    claimed, rejected = set(), set()
    for clause in _CLAUSE_RE.split(norm):
        mentions = []
        for i, tokens in enumerate(token_sets):
            mentions += [(pos, i) for pos in _token_positions(clause, tokens)]
        for m in _INDEX_RE.finditer(clause):
            idx = int(m.group(1) or m.group(2)) - 1
            if 0 <= idx < n:
                mentions.append((m.start(), ascending[idx]))
        if signals:
            for i, signal in enumerate(signals):
                mentions += [(pos, i)
                             for pos in freq_anchor_positions(clause, signal)]
        if not mentions:
            continue
        mentions.sort()
        negs = _word_positions(clause, _NEGATORS)
        for pattern in extra_negator_patterns:   # raw regex, caller-supplied
            negs += [m.start()
                     for m in re.finditer(pattern, clause, re.IGNORECASE)]
        hedges = _prefix_positions(clause, _HEDGE_PREFIXES)
        hedged = set()
        for pos in negs:
            rejected.add(min(mentions, key=lambda m: abs(m[0] - pos))[1])
        for pos in hedges:
            hedged.add(min(mentions, key=lambda m: abs(m[0] - pos))[1])
        claimed.update(i for _, i in mentions if i not in hedged)
    return claimed - rejected


def has_token(text, tokens):
    """Word-boundary presence of any token (case-insensitive)."""
    return bool(_token_positions(str(text or ''), tokens))


def asserted_token(text, tokens):
    """True when at least one occurrence of the tokens is ASSERTED — i.e.
    not negated within its clause. Qualitative credit must require the
    stated relation, not its vocabulary: "does not reduce leakage" mentions
    leakage but asserts nothing (remediation log §12.4)."""
    return any(_marker_hits(str(text or ''), tokens))


def token_negated(text, tokens):
    """True when every occurrence of the tokens is negated."""
    hits = _marker_hits(str(text or ''), tokens)
    return bool(hits) and not any(hits)
