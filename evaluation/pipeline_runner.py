"""Structured 3-stage pipeline runner for EMRB evaluation.

Stage 1 (Localization): Fixed code template identifies unresolved regions of interest
Stage 2 (Hypothesis & Analysis): LLM records provisional hypotheses, then solves tasks
Stage 3 (Constraint Verification): Deterministic checks guide one repair pass
"""
import json
import re
import time
import httpx
from openai import OpenAI

from .config import DEFAULT_MODEL, MAX_TURNS, MODEL_PROVIDERS, get_client_config
from . import executor as executor_module
from .executor import (
    execute_python,
    extract_signal_filename,
    isolated_signal_workspace,
)
from .runner import (
    MODEL_NAME_MAP,
    PROVIDERS_NEEDING_STREAM,
    _answer_block_complete,
    _apply_force_answer_controls,
    _cache_bust_wire_messages,
    _collect_stream,
)

PIPELINE_PROTOCOL_VERSION = 'reconpilot-finalization-dsml-fallback-20260723'
PIPELINE_REASONING_MODELS = frozenset({
    'gpt-5',
    'gpt-5.4',
    'gpt-5.5',
    'deepseek-reasoner',
    'deepseek-v4-pro',
    'deepseek-v4-flash',
    'o1',
    'o3',
    'o4-mini',
    'gemini-3.1-pro-preview',
    'gemini-3.5-flash',
})
# Models known to stall (empty/truncated final turn) once the tool-call
# history gets long; matches runner.py's `extended_finalization` set.
# deepseek-v4-flash added 2026-07-23: on the direct DeepSeek API it keeps
# compulsively re-issuing DSML pseudo-tool-calls in the force-answer loop
# no matter how many attempts it's given; a fresh context built only from
# the saved measurements (no "I keep wanting to call tools" momentum) is
# what actually breaks the loop.
CLEAN_FALLBACK_MODELS = frozenset({
    'gemini-3.5-flash',
    'glm-5.2',
    'mimo-v2.5-pro',
    'deepseek-v4-flash',
    'deepseek-v4-pro',
})
MAX_PIPELINE_CODE_CALLS = 10
# L5 free-form runs average ~10 code calls (up to 15) to resolve multi-signal
# scenes; the flat 10-call cap measurably starved L5 pipeline exploration
# relative to free-form (audit, 2026-07-23). Raise it close to the turn
# budget so turns — not this cap — are the binding constraint, matching
# free-form's behavior.
MAX_PIPELINE_CODE_CALLS_L5 = 14
# The force-answer loop is only reached once code_calls is already at (or
# one below) the cap above, so satisfying a lingering DSML/inline-tool
# pseudo-call there needs its own small headroom rather than reusing the
# same cap that just triggered "stop and answer" (audit, 2026-07-23).
FINALIZATION_CODE_CALL_ALLOWANCE = 3
INITIAL_INVENTORY_CODE_CALLS = 2
SOURCE_AUDIT_CODE_CALLS = 2
STAGE3_RESERVED_TURNS = 3
RECON_TIMEOUT_S = 90

PIPELINE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute Python code locally. numpy/scipy/matplotlib available. "
                       "Signal .npy files are in the current directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    },
}]


def pipeline_execution_profile(model):
    """Resolved, credential-free route used by the ReconPilot runner."""
    provider = MODEL_PROVIDERS.get(model, ('deepseek',))[0]
    _, base_url = get_client_config(model)
    api_model = MODEL_NAME_MAP.get(model, model)
    reasoning = model in PIPELINE_REASONING_MODELS
    return {
        'requested_model': model,
        'provider': provider,
        'base_url': base_url,
        'transport': 'openai-compatible-pipeline',
        'api_model': api_model,
        'reasoning_model': reasoning,
        'api_timeout_s': 900 if reasoning else 300,
        'generation_max_tokens': 40000 if reasoning else 4096,
        'temperature': 0.0,
        'stream': provider in PROVIDERS_NEEDING_STREAM,
        'clean_headers': False,
        'bypass_environment_proxy': False,
        'required_tool_cache_bust': False,
        'tool_choice': 'auto',
        'parallel_tool_calls': False,
        **_finalization_controls(model, provider),
    }


def _finalization_controls(model, provider):
    """Text-only-turn controls for reasoning models that emit DSML markup
    instead of proper tool calls when tool_choice is restricted."""
    nonthinking_finalization = model in {
        'deepseek-v4-pro', 'deepseek-v4-flash'}
    reasoning_knob_supported = provider != 'deepseek'
    return {
        'nonthinking_finalization': nonthinking_finalization,
        'force_answer_tool_choice': (
            'no-tools' if nonthinking_finalization else 'none'),
        'force_answer_reasoning_effort': (
            'none' if nonthinking_finalization and reasoning_knob_supported
            else None),
        'force_answer_thinking': (
            False if nonthinking_finalization and reasoning_knob_supported
            else None),
    }


def _apply_finalization_tool_policy(create_kwargs, profile):
    """Text-only-turn tool handling for the force-answer/verify/clean-
    fallback requests. Most models just need the tool schema kept visible
    but unusable (``tool_choice="none"``); nonthinking_finalization models
    (deepseek-v4-*, glm-5.2, mimo-v2.5-pro) keep faking a tool call as
    markup text even with that flag, so they instead need `tools` omitted
    entirely and the provider's thinking/tool-bias template flag disabled
    via ``_apply_force_answer_controls`` (audit, 2026-07-23)."""
    if profile['force_answer_tool_choice'] == 'none':
        create_kwargs['tools'] = PIPELINE_TOOLS
        create_kwargs['tool_choice'] = 'none'
    else:
        _apply_force_answer_controls(create_kwargs, profile)


def _stage2_tool_choice(profile, is_l5, inventory_complete,
                        source_revision_complete, checkpoint_due, code_calls,
                        inventory_checkpoint_at, source_audit_checkpoint_at):
    """Force measurements in both evidence-gathering phases."""
    collecting_inventory = (
        not inventory_complete and code_calls < inventory_checkpoint_at)
    collecting_source_audit = (
        inventory_complete and not source_revision_complete
        and source_audit_checkpoint_at is not None
        and code_calls < source_audit_checkpoint_at)
    if (is_l5 and not checkpoint_due
            and (collecting_inventory or collecting_source_audit)):
        return {
            'type': 'function',
            'function': {'name': 'execute_python'},
        }
    return profile['tool_choice']


RECON_CODE_TEMPLATE = '''\
import numpy as np
from scipy import signal as sig
from scipy.ndimage import binary_closing, label, uniform_filter1d

data = np.load('{npy_file}')
N = len(data)
fs = {fs}

# The fixed stage localizes occupied regions only. It deliberately does not
# infer source count, modulation, or chirp parameters from a mixed signal.
nperseg = min(4096, N // 4)
freqs, psd = sig.welch(
    data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
    return_onesided=False, scaling='density')
freqs = np.fft.fftshift(freqs)
psd = np.fft.fftshift(psd)
psd_dbm = 10 * np.log10(np.maximum(psd, 1e-20) * 1000)
noise_floor = float(np.median(psd_dbm))
threshold = noise_floor + 6.0
smoothed = uniform_filter1d(psd_dbm, size=21)
occupied = binary_closing(smoothed > threshold, structure=np.ones(9))
components, component_count = label(occupied)

regions = []
bin_width = float(abs(freqs[1] - freqs[0]))
for component_id in range(1, component_count + 1):
    indices = np.flatnonzero(components == component_id)
    if len(indices) < 3:
        continue
    left, right = int(indices[0]), int(indices[-1])
    local_peak = float(np.max(psd_dbm[left:right + 1]))
    if local_peak < noise_floor + 10.0:
        continue
    power_w = float(np.sum(psd[left:right + 1]) * bin_width)
    regions.append((left, right, power_w, local_peak))

f_stft, t_stft, z_stft = sig.stft(
    data, fs=fs, nperseg=min(1024, N // 8), noverlap=min(768, N // 8 - 1),
    return_onesided=False, boundary=None)
order = np.argsort(f_stft)
f_stft = f_stft[order]
stft_power = np.abs(z_stft[order, :]) ** 2

print("=" * 64)
print("UNRESOLVED SPECTRAL REGION MAP")
print("=" * 64)
print(f"Samples: {{N}}, sampling rate: {{fs/1e6:.1f}} MHz, "
      f"duration: {{N/fs*1e3:.4f}} ms")
print(f"Noise floor: {{noise_floor:.1f}} dBm/Hz; "
      f"full-band power: {{10*np.log10(max(np.mean(np.abs(data)**2), 1e-20)*1000):.1f}} dBm")
print("Each ROI may contain one source, multiple overlapping sources, a burst, "
      "or a swept signal. ROI count is not source count.")
print()

for roi_index, (left, right, power_w, local_peak) in enumerate(regions, 1):
    low_hz, high_hz = float(freqs[left]), float(freqs[right])
    roi_mask = (f_stft >= low_hz) & (f_stft <= high_hz)
    energy = np.sum(stft_power[roi_mask, :], axis=0)
    energy_db = 10 * np.log10(np.maximum(energy, 1e-20))
    variation_db = float(np.percentile(energy_db, 90) - np.percentile(energy_db, 10))
    roi_freqs = f_stft[roi_mask]
    roi_power = stft_power[roi_mask, :]
    if roi_power.size:
        bin_power_db = 10 * np.log10(np.maximum(roi_power, 1e-20))
        bin_variation = (np.percentile(bin_power_db, 90, axis=1)
                         - np.percentile(bin_power_db, 10, axis=1))
        local_variation_db = float(np.percentile(bin_variation, 95))
        dominant = roi_freqs[np.argmax(roi_power, axis=0)]
        dominant_span = float(np.percentile(dominant, 95) - np.percentile(dominant, 5))
    else:
        local_variation_db = 0.0
        dominant_span = 0.0
    width_hz = high_hz - low_hz
    flags = []
    if width_hz > 1.0e6:
        flags.append("broad")
    if variation_db > 6.0:
        flags.append("time-varying")
    if local_variation_db > 8.0:
        flags.append("locally-time-varying")
    if dominant_span > max(0.35e6, 0.45 * width_hz):
        flags.append("moving-dominant-frequency")
    if not flags:
        flags.append("unresolved-stationary")
    print(f"ROI R{{roi_index}}: {{low_hz/1e6:+.3f}} to {{high_hz/1e6:+.3f}} MHz, "
          f"width={{width_hz/1e3:.1f}} kHz, integrated power="
          f"{{10*np.log10(max(power_w, 1e-20)*1000):.1f}} dBm")
    print(f"  observations: peak={{local_peak:.1f}} dBm/Hz, "
          f"time variation={{variation_db:.1f}} dB, "
          f"local-bin variation={{local_variation_db:.1f}} dB, "
          f"dominant-frequency span={{dominant_span/1e3:.1f}} kHz, "
          f"flags={{','.join(flags)}}")

print()
print("Required follow-up: audit the source count in every ROI. For a broad or "
      "moving ROI, inspect the time-frequency ridge and the residual after "
      "masking that ridge. For a time-varying or locally-time-varying ROI, "
      "compare active and inactive spectra at each frequency bin; stable total "
      "ROI power does not rule out a burst hidden under a continuous source. "
      "Do not assign source IDs from this map alone.")
print("=" * 64)
print("END REGION MAP")
print("=" * 64)
'''


def _reconnaissance_failed(output):
    """Return whether the deterministic reconnaissance execution failed.

    ReconPilot relies on this report as the context for every subsequent
    analysis turn.  Continuing after a traceback would silently turn the
    method comparison into an evaluation of a broken front end.
    """
    return ("Traceback (most recent call last)" in output
            or output.startswith("[ERROR]"))


STAGE2_SYSTEM = """\
You are an expert in electromagnetic signal analysis. Analyze the raw I/Q data
and answer the requested questions.

## Context
You have received an UNRESOLVED SPECTRAL REGION MAP. It only localizes energy.
An ROI is not a signal and the number of ROIs is not the number of sources.
Never copy ROI boundaries or centers into the answer without source-level
measurements from the raw data.

## Tools
You can call the execute_python tool to run Python code locally.
Available libraries: numpy, scipy, matplotlib, sklearn, etc.
Signal files (.npy) are in the current working directory — load with np.load('filename.npy').
Each tool invocation starts a fresh Python process. Every code block must
reload the signal file and redefine any variables it uses; variables from a
previous tool call are not retained.

## Strategy
1. Use the region map only to prioritize measurements. Analyze the questions
   directly from the raw data and revise or discard any signal hypothesis when
   later evidence contradicts it.
2. When source discovery is required, test whether each ROI contains one or
   multiple signals. Inspect residuals after isolating dominant components and
   compare time-frequency behavior for continuous, burst, and swept signals.
   Individual FFT bins or local maxima are not independent-source evidence.
3. Write targeted analysis code for each sub-question:
   - Bandwidth: use correct definition (3dB, 99%, null-to-null) as asked
   - Autocorrelation: use FFT method R=IFFT(|FFT(x)|²), check specific lag ranges
   - PSD/Energy: pay attention to units (dBm/Hz vs dBm/bin vs total dBm), apply correct normalization
   - Bitrate: use autocorrelation peak or eye diagram, NOT FFT main lobe
   - PAPR: compute on time-domain signal, peak_power / mean_power
   - Classification: check spectral shape, bandwidth, periodicity patterns
4. Assign source IDs only after the measurements support the final scene, then
   reuse those IDs consistently across dependent answers.
5. Verify numerical results and combine related measurements in each code call.

## Output Format
After completing your analysis, end your response with:

===ANSWERS===
Q1a: <value> <unit>
Q1b: <value> <unit>
...
===END===

Include answers for ALL sub-questions."""

STAGE3_JSON_ADDENDUM = """

IMPORTANT: every question that requested a JSON object must keep exactly \
the same single-line JSON format with all required fields. Do NOT convert \
JSON answers into labeled lines; reproduce the corrected JSON in full."""

STAGE2_USER_TEMPLATE = """\
## Unresolved Spectral Region Map (automated)
```
{recon_result}
```

## Questions
{question_text}

The map is only a localization aid. Answer the questions directly from the raw
data. Do not produce a preliminary source inventory, and do not infer source
count from the number of ROIs."""

L5_INVENTORY_REQUEST = """\
Before answering Q1, Q2, or Q3, inspect the raw data and record provisional
source hypotheses. Every ROI in the map must appear exactly once in
``roi_coverage``. Preserve uncertainty when the current measurements do not
distinguish one source from multiple overlapping sources.

Use this exact machine-readable format:
===SOURCE_INVENTORY===
{"signals":[{"id":"S1","roi_id":"R1","center_MHz":0.0,
"bandwidth_MHz":0.0,"behavior":"continuous|burst|chirp",
"modulation_hypothesis":"unknown","evidence":["measurement"]}],
"roi_coverage":[{"roi_id":"R1","source_ids":["S1"],
"tests":["method applied"],"source_count_status":"resolved|uncertain",
"source_count_evidence":["current evidence"],
"next_test":"measurement that would distinguish remaining hypotheses"}],
"unresolved":["specific remaining uncertainty"]}
===END_INVENTORY===

Do not output ===ANSWERS=== yet. Use code first where necessary. The inventory
must be based on source-separation evidence, not merely copied from ROI peaks.
This is a hypothesis checkpoint, not a final scene model. Do not invent an
extra source merely to satisfy a later question, and do not mark source count
as resolved when the evidence supports competing explanations."""

L5_SOURCE_AUDIT_REQUEST = """\
The provisional hypotheses expose where source count is still uncertain, but
recording uncertainty is useful only if the targeted stage tests it. Use the
next two code calls only for a scene-revision audit. Do not spend these calls on
SIR, capacity, channel packing, OFDM design, or symbol recovery.

In one combined call, audit every stationary or burst ROI. Compare one-source
and multi-source explanations using occupied-band edges that persist across
smoothing scales and modulation-consistent evidence. Search for hidden bursts
with per-frequency-bin temporal contrasts or subband-specific change points;
do not define active and inactive frames from total ROI power, because a weak
burst can be hidden under a continuous source with stable aggregate power.

In the other call, audit every moving or broad ROI. Fit and mask each measured
time-frequency ridge with a width supported by the data, then inspect the
residual for stationary or burst components. A collection of raw PSD peaks is
not evidence of independent sources. After both calls, pause for the requested
scene-revision checkpoint before performing downstream calculations."""

L5_SOURCE_REVISION_REQUEST = """\
Pause tool use and revise the scene hypotheses from the two audit results. Every
ROI must appear exactly once in ``roi_reviews``. Use this exact format:
===SOURCE_REVISION===
{"signals":[{"id":"S1","roi_id":"R1","center_MHz":0.0,
"bandwidth_MHz":0.0,"behavior":"continuous|burst|chirp",
"modulation_hypothesis":"unknown","evidence":["audit observation"]}],
"roi_reviews":[{"roi_id":"R1","source_ids":["S1"],
"executed_tests":["test actually run"],"observations":["measured result"],
"source_count_decision":"one|multiple|uncertain",
"revision":"kept|split|merged|relabelled"}],
"unresolved":["remaining uncertainty"]}
===END_SOURCE_REVISION===

Report the evidence even when the decision remains uncertain. This checkpoint
does not require an overlapping non-chirp pair, a chirp victim, or any expected
number of sources. Do not output ===ANSWERS=== yet."""

L5_SOLVE_REQUEST = """\
The evidence-audited scene revision remains a hypothesis, not ground truth.
Later task measurements may still add, merge, remove, or relabel
sources; the IDs in final Q1a become authoritative.

Complete Q1a before deriving any dependent result. Then calculate occupied-band
overlap for every final non-chirp pair and select the global maximum for Q1b.
Use that same final source set for Q1d and Q3. For Q2, use the measured chirp
ridge and its residual to identify the communication victim before computing
victim parameters, crossing times, or symbols. Resolve the remaining numerical
tasks with targeted code, preserve cross-question consistency, and end with the
exact requested ===ANSWERS=== JSON block."""


def _is_l5_question(question_text):
    return bool(re.search(r'emrb-l5-verifiable-v\d+', question_text))


def _expected_roi_ids(recon_result):
    return set(re.findall(r'(?m)^ROI (R\d+):', recon_result or ''))


def _extract_inventory(content):
    blocks = re.findall(
        r'===SOURCE_INVENTORY===\s*(.*?)\s*===END_INVENTORY===',
        content or '', flags=re.DOTALL)
    if not blocks:
        raise ValueError('missing SOURCE_INVENTORY block')
    try:
        payload = json.loads(blocks[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid inventory JSON: {exc}') from exc
    if not isinstance(payload, dict):
        raise ValueError('inventory must be a JSON object')
    return payload


def _inventory_issues(content, expected_rois):
    """Schema and ROI-coverage checks without using hidden source count."""
    try:
        payload = _extract_inventory(content)
    except ValueError as exc:
        return [str(exc)]
    signals = payload.get('signals')
    coverage = payload.get('roi_coverage')
    unresolved = payload.get('unresolved')
    issues = []
    if not isinstance(signals, list) or not signals:
        issues.append('signals must be a non-empty list')
        signals = []
    if not isinstance(coverage, list):
        issues.append('roi_coverage must be a list')
        coverage = []
    if not isinstance(unresolved, list):
        issues.append('unresolved must be a list')

    signal_ids = []
    signal_by_id = {}
    for index, source in enumerate(signals):
        if not isinstance(source, dict):
            issues.append(f'signals[{index}] must be an object')
            continue
        source_id = source.get('id')
        roi_id = source.get('roi_id')
        if not isinstance(source_id, str) or not source_id.strip():
            issues.append(f'signals[{index}] has no id')
        else:
            signal_ids.append(source_id)
            signal_by_id[source_id] = source
        if roi_id not in expected_rois:
            issues.append(f'{source_id or index} references unknown ROI {roi_id}')
        for key in ('center_MHz', 'bandwidth_MHz'):
            value = source.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                issues.append(f'{source_id or index}.{key} must be numeric')
        evidence = source.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            issues.append(f'{source_id or index} has no source-level evidence')
    if len(signal_ids) != len(set(signal_ids)):
        issues.append('source IDs must be unique')

    covered = []
    known_ids = set(signal_ids)
    for index, region in enumerate(coverage):
        if not isinstance(region, dict):
            issues.append(f'roi_coverage[{index}] must be an object')
            continue
        roi_id = region.get('roi_id')
        covered.append(roi_id)
        if roi_id not in expected_rois:
            issues.append(f'coverage references unknown ROI {roi_id}')
        ids = region.get('source_ids')
        if not isinstance(ids, list) or not ids:
            issues.append(f'{roi_id} has no source_ids')
        elif not set(ids).issubset(known_ids):
            issues.append(f'{roi_id} references an undefined source ID')
        elif any(signal_by_id[source_id].get('roi_id') != roi_id
                 for source_id in ids):
            issues.append(f'{roi_id} includes a source assigned to another ROI')
        tests = region.get('tests')
        if not isinstance(tests, list) or not tests:
            issues.append(f'{roi_id} has no decomposition test')
        if region.get('source_count_status') not in {'resolved', 'uncertain'}:
            issues.append(f'{roi_id} has invalid source_count_status')
        count_evidence = region.get('source_count_evidence')
        if not isinstance(count_evidence, list) or not count_evidence:
            issues.append(f'{roi_id} has no source-count evidence')
        next_test = region.get('next_test')
        if not isinstance(next_test, str) or not next_test.strip():
            issues.append(f'{roi_id} has no next_test for hypothesis revision')
    if set(covered) != expected_rois or len(covered) != len(set(covered)):
        issues.append('roi_coverage must contain every ROI exactly once')
    return issues


def _extract_source_revision(content):
    blocks = re.findall(
        r'===SOURCE_REVISION===\s*(.*?)\s*===END_SOURCE_REVISION===',
        content or '', flags=re.DOTALL)
    if not blocks:
        raise ValueError('missing SOURCE_REVISION block')
    try:
        payload = json.loads(blocks[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid source revision JSON: {exc}') from exc
    if not isinstance(payload, dict):
        raise ValueError('source revision must be a JSON object')
    return payload


def _source_revision_issues(content, expected_rois):
    """Require an evidence audit for each ROI without prescribing its result."""
    try:
        payload = _extract_source_revision(content)
    except ValueError as exc:
        return [str(exc)]
    signals = payload.get('signals')
    reviews = payload.get('roi_reviews')
    unresolved = payload.get('unresolved')
    issues = []
    if not isinstance(signals, list) or not signals:
        issues.append('signals must be a non-empty list')
        signals = []
    if not isinstance(reviews, list):
        issues.append('roi_reviews must be a list')
        reviews = []
    if not isinstance(unresolved, list):
        issues.append('unresolved must be a list')

    signal_ids = []
    signal_by_id = {}
    for index, source in enumerate(signals):
        if not isinstance(source, dict):
            issues.append(f'signals[{index}] must be an object')
            continue
        source_id = source.get('id')
        roi_id = source.get('roi_id')
        if not isinstance(source_id, str) or not source_id.strip():
            issues.append(f'signals[{index}] has no id')
        else:
            signal_ids.append(source_id)
            signal_by_id[source_id] = source
        if roi_id not in expected_rois:
            issues.append(f'{source_id or index} references unknown ROI {roi_id}')
        for key in ('center_MHz', 'bandwidth_MHz'):
            value = source.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                issues.append(f'{source_id or index}.{key} must be numeric')
        evidence = source.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            issues.append(f'{source_id or index} has no audit evidence')
    if len(signal_ids) != len(set(signal_ids)):
        issues.append('source IDs must be unique')

    reviewed = []
    known_ids = set(signal_ids)
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            issues.append(f'roi_reviews[{index}] must be an object')
            continue
        roi_id = review.get('roi_id')
        reviewed.append(roi_id)
        if roi_id not in expected_rois:
            issues.append(f'review references unknown ROI {roi_id}')
        ids = review.get('source_ids')
        if not isinstance(ids, list) or not ids:
            issues.append(f'{roi_id} has no source_ids')
        elif not set(ids).issubset(known_ids):
            issues.append(f'{roi_id} references an undefined source ID')
        elif any(signal_by_id[source_id].get('roi_id') != roi_id
                 for source_id in ids):
            issues.append(f'{roi_id} includes a source assigned to another ROI')
        tests = review.get('executed_tests')
        if not isinstance(tests, list) or not tests:
            issues.append(f'{roi_id} has no executed source-count test')
        observations = review.get('observations')
        if not isinstance(observations, list) or not observations:
            issues.append(f'{roi_id} has no measured audit observation')
        if review.get('source_count_decision') not in {
                'one', 'multiple', 'uncertain'}:
            issues.append(f'{roi_id} has invalid source_count_decision')
        revision = review.get('revision')
        if revision not in {'kept', 'split', 'merged', 'relabelled'}:
            issues.append(f'{roi_id} has invalid revision')
    if set(reviewed) != expected_rois or len(reviewed) != len(set(reviewed)):
        issues.append('roi_reviews must contain every ROI exactly once')
    return issues


def _l5_consistency_issues(content):
    """Check answer dependencies that follow from the model's own inventory."""
    from evaluation.l5_verifier import extract_answer_json
    try:
        payload = extract_answer_json(content)
    except ValueError as exc:
        return [str(exc)]

    issues = []
    q1a = payload.get('Q1a', {})
    sources = q1a.get('signals', []) if isinstance(q1a, dict) else []
    source_by_id = {
        item.get('id'): item for item in sources
        if isinstance(item, dict) and isinstance(item.get('id'), str)
    }
    if len(source_by_id) != len(sources):
        issues.append('Q1a source IDs must be present and unique')

    def require_source(section, key):
        value = section.get(key) if isinstance(section, dict) else None
        if value not in source_by_id:
            issues.append(f'{key}={value!r} is not defined in Q1a')
        return value

    q1b = payload.get('Q1b', {})
    pair_ids = q1b.get('pair_ids', []) if isinstance(q1b, dict) else []
    if not isinstance(pair_ids, list) or len(pair_ids) != 2:
        issues.append('Q1b pair_ids must contain two source IDs')
    else:
        for source_id in pair_ids:
            if source_id not in source_by_id:
                issues.append(f'Q1b pair source {source_id!r} is not in Q1a')

    def answer_is_chirp(source):
        description = str(source.get('modulation', '')).lower()
        return 'chirp' in description or 'lfm' in description

    overlap_by_pair = {}
    non_chirp_sources = [
        source for source in sources
        if isinstance(source, dict) and not answer_is_chirp(source)
    ]
    for index, first in enumerate(non_chirp_sources):
        for second in non_chirp_sources[index + 1:]:
            values = (
                first.get('center_MHz'), first.get('bandwidth_MHz'),
                second.get('center_MHz'), second.get('bandwidth_MHz'),
            )
            if not all(isinstance(value, (int, float))
                       and not isinstance(value, bool) for value in values):
                continue
            first_low = values[0] - values[1] / 2
            first_high = values[0] + values[1] / 2
            second_low = values[2] - values[3] / 2
            second_high = values[2] + values[3] / 2
            overlap_by_pair[frozenset((first.get('id'), second.get('id')))] = max(
                0.0, min(first_high, second_high) - max(first_low, second_low))
    selected_pair = (frozenset(pair_ids)
                     if isinstance(pair_ids, list) and len(pair_ids) == 2
                     else frozenset())
    if overlap_by_pair:
        max_overlap = max(overlap_by_pair.values())
        maximizing_pairs = {
            pair for pair, overlap in overlap_by_pair.items()
            if abs(overlap - max_overlap) <= 0.02
        }
        if selected_pair not in maximizing_pairs:
            issues.append(
                'Q1b pair is not the largest non-chirp overlap in Q1a')
        reported_overlap = (q1b.get('overlap_MHz')
                            if isinstance(q1b, dict) else None)
        computed_overlap = overlap_by_pair.get(selected_pair)
        if (computed_overlap is not None
                and isinstance(reported_overlap, (int, float))
                and not isinstance(reported_overlap, bool)
                and abs(float(reported_overlap) - computed_overlap)
                > max(0.02, 0.1 * computed_overlap)):
            issues.append(
                'Q1b overlap_MHz is inconsistent with the selected source '
                'intervals in Q1a')
    q1_target = require_source(q1b, 'target_id')
    q1c = payload.get('Q1c', {})
    q1c_target = require_source(q1c, 'target_id')
    if q1_target and q1c_target and q1_target != q1c_target:
        issues.append('Q1c must filter the Q1b target signal')

    q2a = payload.get('Q2a', {})
    chirp_id = require_source(q2a, 'signal_id')
    q2b = payload.get('Q2b', {})
    victim_id = require_source(q2b, 'victim_id')
    q2d = payload.get('Q2d', {})
    if isinstance(q2d, dict) and q2d.get('victim_id') != victim_id:
        issues.append('Q2d victim_id must match Q2b victim_id')
    if chirp_id and victim_id and chirp_id == victim_id:
        issues.append('Q2 chirp and communication victim must be different sources')

    def number(section, key):
        value = section.get(key) if isinstance(section, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    start = number(q2a, 'sweep_start_MHz')
    end = number(q2a, 'sweep_end_MHz')
    chirp_bw = number(q2a, 'bandwidth_MHz')
    chirp_rate = number(q2a, 'chirp_rate_MHz_per_ms')
    tbp = number(q2a, 'tbp')
    victim_center = number(q2b, 'center_MHz')
    victim_bw = number(q2b, 'bandwidth_MHz')
    if None not in (start, end, chirp_bw):
        sweep_low, sweep_high = sorted((start, end))
        measured_bw = sweep_high - sweep_low
        if measured_bw > 0 and abs(chirp_bw - measured_bw) > max(0.2, 0.15 * measured_bw):
            issues.append('Q2a bandwidth is inconsistent with sweep endpoints')
        if None not in (victim_center, victim_bw):
            victim_low = victim_center - victim_bw / 2
            victim_high = victim_center + victim_bw / 2
            if min(sweep_high, victim_high) <= max(sweep_low, victim_low):
                issues.append('Q2b victim does not overlap the reported chirp sweep')
    if None not in (chirp_bw, chirp_rate, tbp) and chirp_rate > 0:
        expected_tbp = chirp_bw * chirp_bw / chirp_rate * 1000.0
        if abs(tbp - expected_tbp) > max(10.0, 0.2 * expected_tbp):
            issues.append(
                'Q2a tbp is inconsistent with bandwidth and chirp rate; '
                'convert MHz times ms to the dimensionless Hz times s value')

    q2c = payload.get('Q2c', {})
    entry = number(q2c, 'entry_time_ms')
    exit_time = number(q2c, 'exit_time_ms')
    duration = number(q2c, 'duration_ms')
    if None not in (entry, exit_time, duration):
        if exit_time < entry:
            issues.append('Q2c exit_time_ms precedes entry_time_ms')
        expected_duration = exit_time - entry
        if abs(duration - expected_duration) > max(0.02, 0.1 * abs(expected_duration)):
            issues.append('Q2c duration_ms is inconsistent with entry and exit times')
        if duration <= 0:
            issues.append('Q2c crossing duration must be positive for an overlapping victim')

    passband = q1c.get('passband_MHz') if isinstance(q1c, dict) else None
    target = source_by_id.get(q1c_target)
    if (isinstance(passband, list) and len(passband) == 2
            and isinstance(target, dict)):
        target_center = target.get('center_MHz')
        if (isinstance(target_center, (int, float))
                and not min(passband) <= target_center <= max(passband)):
            issues.append('Q1c passband does not contain its target center')
    q1d = payload.get('Q1d', {})
    channel_count = number(q1d, 'additional_channel_count')
    channel_centers = (q1d.get('additional_centers_MHz')
                       if isinstance(q1d, dict) else None)
    if (channel_count is not None and isinstance(channel_centers, list)
            and int(channel_count) != len(channel_centers)):
        issues.append(
            'Q1d additional_channel_count does not match the number of '
            'reported centers')
    return issues

STAGE3_VERIFY = """\
Review your answers above. For each numerical answer:
1. Check units are correct and consistent
2. Verify the value is physically reasonable (e.g., bandwidth > 0, SNR makes sense for the signal quality)
3. Cross-check: if you computed bandwidth two ways, do they agree?

If any answer seems wrong, correct it. Then output your FINAL verified answers:

===ANSWERS===
Q1a: ...
...
===END==="""


def protocol_fingerprint(model, max_turns):
    """Model-aware identity of the current ReconPilot protocol."""
    import hashlib
    from .runner import _force_answer_prompts
    profile = pipeline_execution_profile(model)
    blob = json.dumps({
        'version': PIPELINE_PROTOCOL_VERSION,
        'model_profile': profile,
        'recon_code_template': RECON_CODE_TEMPLATE,
        'stage2_system': STAGE2_SYSTEM,
        'stage2_user_template': STAGE2_USER_TEMPLATE,
        'stage3_verify': STAGE3_VERIFY,
        'stage3_json_addendum': STAGE3_JSON_ADDENDUM,
        'stage2_tools': PIPELINE_TOOLS,
        'force_answer_prompts': [
            _force_answer_prompts(marker)
            for marker in ('emrb-l4-repaired-v1', 'emrb-l2-autocorr-v1',
                           'emrb-l5-verifiable-v5', '')],
        'max_turns': max_turns,
        'max_pipeline_code_calls': MAX_PIPELINE_CODE_CALLS,
        'max_pipeline_code_calls_l5': MAX_PIPELINE_CODE_CALLS_L5,
        'clean_fallback_models': sorted(CLEAN_FALLBACK_MODELS),
        'stage3_reserved_turns': STAGE3_RESERVED_TURNS,
        'recon_timeout_s': RECON_TIMEOUT_S,
        'executor': {
            'code_timeout_s': executor_module.CODE_TIMEOUT,
            'max_output_len': executor_module.MAX_OUTPUT_LEN,
            'preamble': executor_module.PREAMBLE,
            'isolation': 'bubblewrap-unshare-all-v1',
        },
        'answer_selection': 'complete-typed-schema-and-full-coverage-v1',
        'l5_constraint_validation': 'cross-section-dependencies-v1',
        'stage2_policy': 'direct-question-guided-analysis-v1',
        'finalization_tool_choice': 'none-with-visible-schema-v1',
        'retry_policy': {'max_retries': 10, 'backoff_cap_s': 120},
    }, sort_keys=True, ensure_ascii=False)
    return {
        'mode': 'pipeline',
        'version': PIPELINE_PROTOCOL_VERSION,
        'model': model,
        'provider': profile['provider'],
        'api_model': profile['api_model'],
        'transport': profile['transport'],
        'base_url': profile['base_url'],
        'max_turns': max_turns,
        'fingerprint': hashlib.md5(blob.encode()).hexdigest(),
    }


def _question_requires_json(question_text):
    """True when any question in the problem demands a JSON-object answer."""
    return bool('emrb-l2-autocorr-v1' in question_text
                or 'emrb-l4-repaired-v1' in question_text
                or re.search(r'emrb-l5-verifiable-v\d+', question_text))


_QUESTION_HEADER_RE = re.compile(r'(?m)^\s*(Q\d+)\s*[.:：]')
_SUBQUESTION_HEADER_RE = re.compile(r'\(([a-d])\)\s*', re.IGNORECASE)
# the two shapes our generators use to state the required JSON fields:
# an inline brace template ({"key": <value>, ...}) or the prose sentence
# "return this question as one JSON object with fields a, b, and c."
_TEMPLATE_KEY_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:')
_FIELDS_SENTENCE_RE = re.compile(
    r'as one JSON object with fields\s+(.+?)\.(?:\s|$)', re.DOTALL)


def _section_required_keys(section):
    """Top-level answer keys the question section demands, or an empty set
    when no template can be located."""
    keys = set(_TEMPLATE_KEY_RE.findall(section))
    if keys:
        return keys
    m = _FIELDS_SENTENCE_RE.search(section)
    if not m:
        return set()
    for token in re.split(r',\s*', m.group(1)):
        token = re.sub(r'^and\s+', '', token.strip())
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', token):
            keys.add(token)
    return keys


def _json_question_requirements(question_text):
    """[(question_id, required_key_set)] for every question that demands a
    JSON answer. The L2 marker sits inside its question section; the
    repaired-L4 marker is a problem-level instruction, and the JSON-bearing
    questions are the ones whose section states its field template."""
    headers = list(_QUESTION_HEADER_RE.finditer(question_text))
    has_repaired = 'emrb-l4-repaired-v1' in question_text
    requirements = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) \
            else len(question_text)
        section = question_text[header.start():end]
        keys = _section_required_keys(section)
        if 'emrb-l2-autocorr-v1' in section or (has_repaired and keys):
            requirements.append((header.group(1), keys))
    return requirements


def _required_answer_labels(question_text):
    """Labels that a complete mixed-format answer block must contain.

    L2 uses one line per lettered sub-question except for its single Q3 JSON
    object. Repaired L4 explicitly requests one line per top-level question.
    """
    headers = list(_QUESTION_HEADER_RE.finditer(question_text))
    json_ids = {qid for qid, _ in _json_question_requirements(question_text)}
    repaired_l4 = 'emrb-l4-repaired-v1' in question_text
    required = set()
    for index, header in enumerate(headers):
        qid = header.group(1)
        end = headers[index + 1].start() if index + 1 < len(headers) \
            else len(question_text)
        section = question_text[header.start():end]
        if repaired_l4 or qid in json_ids:
            required.add(qid)
            continue
        letters = {
            match.group(1).lower()
            for match in _SUBQUESTION_HEADER_RE.finditer(section)
        }
        if letters:
            required.update(f'{qid}{letter}' for letter in letters)
        else:
            required.add(qid)
    return required


def _answer_coverage_ok(question_text, content):
    from evaluation.auto_scorer import parse_answer_block
    answers = parse_answer_block(content)
    headers = list(_QUESTION_HEADER_RE.finditer(question_text))
    json_ids = {qid for qid, _ in _json_question_requirements(question_text)}
    repaired_l4 = 'emrb-l4-repaired-v1' in question_text

    def present(label):
        return label in answers and bool(str(answers[label]).strip())

    if not headers:
        return False
    for index, header in enumerate(headers):
        qid = header.group(1)
        end = headers[index + 1].start() if index + 1 < len(headers) \
            else len(question_text)
        section = question_text[header.start():end]
        if qid in json_ids:
            if not present(qid):
                return False
            continue
        letters = {
            match.group(1).lower()
            for match in _SUBQUESTION_HEADER_RE.finditer(section)
        }
        if repaired_l4 and present(qid):
            continue
        required = ({f'{qid}{letter}' for letter in letters}
                    if letters else {qid})
        if repaired_l4 and not letters:
            if not any(
                    key.startswith(qid) and key != qid and present(key)
                    for key in answers):
                return False
        elif not all(present(label) for label in required):
            return False
    return True


def _answer_coverage_profile(question_text, content):
    """Comparable per-question coverage for successive answer rewrites."""
    from evaluation.auto_scorer import parse_answer_block
    answers = parse_answer_block(content)
    profile = {}
    for qid in {
            match.group(1)
            for match in _QUESTION_HEADER_RE.finditer(question_text)}:
        if qid in answers and str(answers[qid]).strip():
            # The repaired-L4 prompt explicitly defines one top-level line as
            # a complete question answer, so it dominates split sub-lines.
            profile[qid] = 1000
        else:
            profile[qid] = sum(
                1 for key, value in answers.items()
                if key.startswith(qid) and key != qid
                and str(value).strip()
            )
    return profile


def _structured_answer_ok(question_text, content):
    """Validate a candidate answer block with the same parsers the
    deterministic verifiers use AND against the answer schema's required
    fields — a brace in prose ("verified set {S1, S2}"), malformed JSON, or
    a syntactically valid but field-incomplete object ("Q3: {}") must not
    replace an earlier complete answer (§10.2, §12.1)."""
    l5_marker = re.search(r'emrb-l5-verifiable-v\d+', question_text)
    if l5_marker:
        from evaluation.l5_verifier import (REQUIRED_ANSWER_SECTIONS,
                                            extract_answer_json,
                                            validate_answer_structure)
        try:
            payload = extract_answer_json(content)
        except ValueError:
            return False
        return (payload.get('schema_version') == l5_marker.group(0)
                and all(section in payload
                        for section in REQUIRED_ANSWER_SECTIONS)
                and validate_answer_structure(payload))
    from evaluation.l4_verifier import (
        _answer_json,
        validate_repaired_payload_structure,
    )
    requirements = _json_question_requirements(question_text)
    if not requirements:
        # marker present but no per-question section found: require at least
        # one parseable JSON-object answer line anywhere in the block
        from evaluation.auto_scorer import parse_answer_block
        answers = parse_answer_block(content)
        for qid in answers:
            try:
                payload = _answer_json(content, qid)
            except ValueError:
                continue
            if (validate_repaired_payload_structure(payload)
                    and _answer_coverage_ok(question_text, content)):
                return True
        return False
    for qid, required_keys in requirements:
        try:
            payload = _answer_json(content, qid)
        except ValueError:
            return False
        if not required_keys.issubset(payload.keys()):
            return False
        if 'emrb-l2-autocorr-v1' in question_text:
            from evaluation.l2_verifier import validate_q3_payload_structure
            if not validate_q3_payload_structure(payload):
                return False
        elif not validate_repaired_payload_structure(payload, required_keys):
            return False
    return _answer_coverage_ok(question_text, content)


def _select_final_response(messages, question_text):
    """Newest assistant answer block; a schema-bearing task keeps the newest
    candidate whose JSON answers actually parse, so a structured answer is
    never lost to a generic verification rewrite (§8.1.4, §10.2)."""
    requires_json = _question_requires_json(question_text)
    fallback = ""
    plain_answer = ""
    complete_plain_answer = ""
    structured = ""
    structured_profile = None
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else ""
        if content and (isinstance(m, dict) and m.get("role") == "assistant"):
            fallback = content
            if "===ANSWERS===" in content:
                plain_answer = content
                if not requires_json:
                    if _answer_block_complete(question_text, content):
                        complete_plain_answer = content
                    continue
                if _structured_answer_ok(question_text, content):
                    profile = _answer_coverage_profile(
                        question_text, content)
                    if (structured_profile is None
                            or all(
                                profile.get(qid, 0) >= previous
                                for qid, previous
                                in structured_profile.items())):
                        structured = content
                        structured_profile = profile
    if requires_json and structured:
        return structured
    return complete_plain_answer or plain_answer or fallback


def _extract_dsml_code(content):
    if not content or 'DSML' not in content:
        return None
    m = re.search(
        r'DSML.*?parameter\s+name="code"[^>]*>\s*\n?(.*)',
        content, re.DOTALL,
    )
    if not m:
        return None
    code = m.group(1)
    # The capture runs to the end of the message, so it swallows the
    # trailing </...DSML...parameter>/</...invoke>/</...tool_calls> closing
    # tags too; left in, they're a guaranteed SyntaxError on execution (the
    # full-width '｜' isn't valid Python). No real code starts a line with
    # '</', so truncate at the first one.
    closing_tag = re.search(r'(?m)^\s*</', code)
    if closing_tag:
        code = code[:closing_tag.start()]
    return code.strip()


def _extract_default_api_tool_code(content):
    """Extract anti-route Gemini's text-encoded Python tool invocation.

    Some OpenAI-compatible anti routes return a successful assistant message
    containing ``call:default_api:execute_python{code:...}`` instead of a
    structured ``tool_calls`` object.  Treating that text as a final answer
    skips the requested measurement and corrupts the score.
    """
    if not content or 'call:default_api:execute_python' not in content:
        return None
    match = re.search(
        r'call:default_api:execute_python\s*\{\s*code:(.*)\}\s*$',
        content, re.DOTALL,
    )
    if not match:
        return None
    code = match.group(1).strip()
    # Some compatible OpenAI routes serialize the final newline as ``\\n``.
    if code.endswith("\\n"):
        code = code[:-2]
    return code


def _run_pipeline(question_text, sample_dir, model=None, max_turns=None,
                   verbose=True, skip_stage3=False):
    """Run one EMRB problem through the 3-stage structured pipeline.

    Stage 1: Fixed localization code → unresolved regions of interest
    Stage 2: LLM source decomposition → task-specific analysis
    Stage 3: Cross-section constraint checks → one repair pass

    ``skip_stage3`` is an ablation hook (paper §P3-2): it keeps the answer
    -completion safety net (force-answer / clean-fallback) that any runnable
    result needs, but skips the constraint-verification and repair turn, so
    the returned answer is whatever Stage 2 produced.

    Returns dict with response, messages, turns, code_calls, elapsed_s, stage_info.
    """
    model = model or DEFAULT_MODEL
    max_turns = max_turns or MAX_TURNS

    api_key, base_url = get_client_config(model)
    profile = pipeline_execution_profile(model)
    api_model = profile['api_model']
    api_timeout = profile['api_timeout_s']
    client_kwargs = dict(api_key=api_key, base_url=base_url, timeout=api_timeout)
    if profile['clean_headers']:
        client_kwargs["default_headers"] = {"User-Agent": "python-httpx/0.27"}
    if profile['bypass_environment_proxy']:
        client_kwargs["http_client"] = httpx.Client(
            trust_env=False, timeout=api_timeout
        )
    client = OpenAI(**client_kwargs)
    gen_max_tokens = profile['generation_max_tokens']
    cache_bust_spaces = 1 if profile['required_tool_cache_bust'] else 0

    t0 = time.time()
    code_calls = 0
    turn = 0
    stage_info = {}
    is_l5 = _is_l5_question(question_text)
    max_code_calls = MAX_PIPELINE_CODE_CALLS_L5 if is_l5 else MAX_PIPELINE_CODE_CALLS

    # ================================================================
    # STAGE 1: Reconnaissance (fixed code template, no LLM)
    # ================================================================
    if verbose:
        print("  [Stage 1] Running fixed reconnaissance...")

    # Extract fs and npy filename from question text
    fs_match = re.search(r'(?:采样率|Sampling rate)[：:]\s*([\d.]+)\s*MHz', question_text)

    fs = float(fs_match.group(1)) * 1e6 if fs_match else 20e6
    npy_file = extract_signal_filename(question_text)

    recon_code = RECON_CODE_TEMPLATE.format(npy_file=npy_file, fs=int(fs))
    recon_result = execute_python(
        recon_code, sample_dir, timeout=RECON_TIMEOUT_S)
    code_calls += 1

    if _reconnaissance_failed(recon_result):
        raise RuntimeError(
            "Localization failed; refusing to continue with an invalid "
            "region map."
        )

    if verbose:
        lines = recon_result.strip().split('\n')
        print(f"    Recon done: {len(lines)} lines of output")

    stage_info['recon_output'] = recon_result
    stage_info['recon_time'] = round(time.time() - t0, 1)
    expected_rois = _expected_roi_ids(recon_result)
    stage_info['roi_ids'] = sorted(expected_rois)

    # ================================================================
    # STAGE 2: Targeted Analysis (LLM with recon context)
    # ================================================================
    if verbose:
        print("  [Stage 2] LLM targeted analysis...")

    stage2_user = STAGE2_USER_TEMPLATE.format(
        recon_result=recon_result, question_text=question_text)

    messages = [
        {"role": "system", "content": STAGE2_SYSTEM},
        {"role": "user", "content": stage2_user},
    ]
    stage2_turns = 0
    analysis_max = max(0, max_turns - STAGE3_RESERVED_TURNS)
    # Stage 2 analyzes the questions directly. Source inventories and their
    # checkpoints are intentionally disabled because they anchor later work to
    # an error-prone intermediate decomposition.
    inventory_complete = True
    source_revision_complete = True
    inventory_checkpoint_at = 1 + INITIAL_INVENTORY_CODE_CALLS
    source_audit_checkpoint_at = None
    inventory_checkpoint_announced_at = None
    source_revision_checkpoint_announced_at = None
    source_audit_start_code_calls = None
    limit_prompted = False

    while turn < analysis_max:
        inventory_checkpoint_due = bool(
            is_l5 and not inventory_complete
            and code_calls >= inventory_checkpoint_at)
        source_revision_checkpoint_due = bool(
            is_l5 and inventory_complete and not source_revision_complete
            and source_audit_checkpoint_at is not None
            and code_calls >= source_audit_checkpoint_at)
        checkpoint_due = (
            inventory_checkpoint_due or source_revision_checkpoint_due)
        if (inventory_checkpoint_due
                and inventory_checkpoint_announced_at
                != inventory_checkpoint_at):
            messages.append({
                "role": "user",
                "content": (
                    "Pause tool analysis and produce the complete "
                    "provisional SOURCE_INVENTORY checkpoint now. Preserve "
                    "uncertainty supported by the evidence already collected; "
                    "do not request code in this turn."
                ),
            })
            inventory_checkpoint_announced_at = inventory_checkpoint_at
        elif (source_revision_checkpoint_due
              and source_revision_checkpoint_announced_at
              != source_audit_checkpoint_at):
            messages.append({
                "role": "user",
                "content": L5_SOURCE_REVISION_REQUEST,
            })
            source_revision_checkpoint_announced_at = (
                source_audit_checkpoint_at)

        # After enough code executions, nudge model to wrap up
        if code_calls >= max_code_calls and not limit_prompted:
            if is_l5 and not inventory_complete:
                limit_message = (
                    "The code budget is exhausted. Produce the complete "
                    "provisional SOURCE_INVENTORY checkpoint now from the "
                    "evidence already collected; do not request another tool "
                    "call."
                )
            elif is_l5 and not source_revision_complete:
                limit_message = (
                    "The code budget is exhausted. Produce the complete "
                    "SOURCE_REVISION checkpoint from the audit evidence now; "
                    "do not request another tool call."
                )
            else:
                limit_message = (
                    "The code budget is exhausted. Compile the complete "
                    "final answer now from the evidence already collected.\n"
                    "Format:\n===ANSWERS===\nQ1a: ...\n...\n===END==="
                )
            messages.append({
                "role": "user",
                "content": limit_message,
            })
            limit_prompted = True

        turn += 1
        stage2_turns += 1
        if verbose:
            print(f"  [Stage 2, turn {stage2_turns}] calling {model}...", end=" ", flush=True)

        retries = 0
        try:
            while True:
                try:
                    wire_messages = (
                        _cache_bust_wire_messages(messages, cache_bust_spaces)
                        if profile['required_tool_cache_bust'] else messages)
                    create_kwargs = dict(
                        model=api_model,
                        messages=wire_messages,
                        max_tokens=gen_max_tokens,
                        temperature=profile['temperature'],
                    )
                    if (not checkpoint_due
                            and code_calls < max_code_calls):
                        tool_choice = _stage2_tool_choice(
                            profile, is_l5, inventory_complete,
                            source_revision_complete, checkpoint_due,
                            code_calls, inventory_checkpoint_at,
                            source_audit_checkpoint_at)
                        create_kwargs.update(
                            tools=PIPELINE_TOOLS,
                            tool_choice=tool_choice,
                            parallel_tool_calls=profile['parallel_tool_calls'],
                        )
                    if profile['stream']:
                        create_kwargs["stream"] = True
                        resp = _collect_stream(client.chat.completions.create(**create_kwargs))
                    else:
                        resp = client.chat.completions.create(**create_kwargs)
                    break
                except Exception as e:
                    retries += 1
                    wait = min(10 * (2 ** min(retries - 1, 4)), 120)
                    # Parallel evaluation previously hid this path because
                    # ``verbose`` is false with multiple workers.  A failed
                    # request must remain visible in the run log.
                    print(
                        f"  [Stage 2] API error: {e}, retry #{retries} "
                        f"in {wait}s...", flush=True)
                    if retries > 10:
                        raise RuntimeError(f"Stage 2 API failed after {retries} retries: {e}") from e
                    if profile['required_tool_cache_bust']:
                        cache_bust_spaces += 1
                    time.sleep(wait)
        except Exception as e:
            messages.append({
                "role": "assistant",
                "content": f"Pipeline stopped because the model API failed: {e}",
            })
            if verbose:
                print("failed")
            break

        choice = resp.choices[0]
        msg = choice.message

        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        reasoning = getattr(msg, 'reasoning_content', None)
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if msg.tool_calls:
            for tool_index, tc in enumerate(msg.tool_calls):
                if checkpoint_due:
                    checkpoint_name = (
                        'SOURCE_INVENTORY' if inventory_checkpoint_due
                        else 'SOURCE_REVISION')
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            f"[SKIPPED] {checkpoint_name} is due. Output the "
                            "checkpoint from existing evidence now."
                        ),
                    })
                    continue
                if tool_index > 0:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            "[SKIPPED] ReconPilot executes one analysis call "
                            "per turn. Incorporate the first result, then "
                            "request one targeted follow-up if needed."
                        ),
                    })
                    continue
                if tc.function.name == "execute_python":
                    if code_calls >= max_code_calls:
                        # Hard stop: refuse to run more code
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "[BLOCKED] Code execution limit reached. Output your answers now.",
                        })
                        continue
                    try:
                        args = json.loads(tc.function.arguments)
                        code = args["code"]
                    except (json.JSONDecodeError, KeyError):
                        code = tc.function.arguments
                    if verbose:
                        first_line = code.strip().split('\n')[0][:60]
                        print(f"exec: {first_line}...")
                    result = execute_python(code, sample_dir)
                    code_calls += 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool: {tc.function.name}",
                    })
        else:
            dsml_code = _extract_dsml_code(msg.content)
            default_api_code = (
                _extract_default_api_tool_code(msg.content)
                if not dsml_code else None
            )
            inline_code = dsml_code or default_api_code
            if inline_code:
                if checkpoint_due:
                    checkpoint_name = (
                        'SOURCE_INVENTORY' if inventory_checkpoint_due
                        else 'SOURCE_REVISION')
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The {checkpoint_name} checkpoint is due. Do not "
                            "run more code in this turn; output the complete "
                            f"{checkpoint_name} block now."
                        ),
                    })
                elif code_calls >= max_code_calls:
                    messages.append({
                        "role": "user",
                        "content": "[BLOCKED] Code execution limit reached. Output your answers now.",
                    })
                else:
                    if verbose:
                        first_line = inline_code.strip().split('\n')[0][:60]
                        tag = 'dsml' if dsml_code else 'default_api'
                        print(f"exec({tag}): {first_line}...")
                    result = execute_python(inline_code, sample_dir)
                    code_calls += 1
                    messages.append({
                        "role": "user",
                        "content": f"Code execution result:\n{result}",
                    })
                continue
            if not (msg.content or '').strip():
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was empty. Continue the "
                        "analysis or provide the complete final answer; do "
                        "not return an empty response."
                    ),
                })
                continue
            if is_l5 and not inventory_complete:
                schema_errors = _inventory_issues(
                    msg.content, expected_rois)
                if schema_errors:
                    inventory_checkpoint_at = code_calls
                    inventory_checkpoint_announced_at = None
                    messages.append({
                        "role": "user",
                        "content": (
                            "The provisional hypothesis checkpoint has schema "
                            "or coverage errors:\n"
                            + "\n".join(f"- {item}" for item in schema_errors)
                            + "\nCorrect the SOURCE_INVENTORY block from the "
                              "evidence already collected. Preserve genuine "
                              "source-count uncertainty; do not run code in "
                              "this correction turn."
                        ),
                    })
                    continue
                inventory_complete = True
                stage_info['source_inventory'] = _extract_inventory(
                    msg.content)
                stage_info['inventory_turn'] = stage2_turns
                source_audit_start_code_calls = code_calls
                source_audit_checkpoint_at = (
                    code_calls + SOURCE_AUDIT_CODE_CALLS)
                messages.append({
                    "role": "user",
                    "content": L5_SOURCE_AUDIT_REQUEST,
                })
                if verbose:
                    print("hypotheses recorded")
                continue
            if is_l5 and not source_revision_complete:
                if not source_revision_checkpoint_due:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Continue the required scene-revision audit with "
                            "execute_python. Complete both audit calls before "
                            "writing SOURCE_REVISION."
                        ),
                    })
                    continue
                revision_errors = _source_revision_issues(
                    msg.content, expected_rois)
                if revision_errors:
                    source_revision_checkpoint_announced_at = (
                        source_audit_checkpoint_at)
                    messages.append({
                        "role": "user",
                        "content": (
                            "The SOURCE_REVISION checkpoint has schema or "
                            "coverage errors:\n"
                            + "\n".join(
                                f"- {item}" for item in revision_errors)
                            + "\nCorrect the SOURCE_REVISION block from the "
                              "audit evidence already collected. A source "
                              "count may remain uncertain; do not run code in "
                              "this correction turn."
                        ),
                    })
                    continue
                source_revision_complete = True
                stage_info['source_revision'] = _extract_source_revision(
                    msg.content)
                stage_info['source_revision_turn'] = stage2_turns
                stage_info['source_audit_code_calls'] = (
                    code_calls - source_audit_start_code_calls)
                messages.append({
                    "role": "user",
                    "content": L5_SOLVE_REQUEST,
                })
                if verbose:
                    print("scene revised")
                continue
            if is_l5 and not _answer_block_complete(
                    question_text, msg.content):
                messages.append({
                    "role": "user",
                    "content": (
                        "Continue the question-guided analysis from the raw "
                        "data. Revise signal hypotheses whenever measurements "
                        "require it, then output the complete requested "
                        "===ANSWERS=== JSON block."
                    ),
                })
                continue
            if verbose:
                print("done (no tool call)")
            break

        if choice.finish_reason == "stop" and not msg.tool_calls:
            if verbose:
                print("done (stop)")
            break

    stage_info['stage2_turns'] = stage2_turns
    stage_info['stage2_time'] = round(time.time() - t0 - stage_info['recon_time'], 1)

    messages, turn, code_calls, stage3_info = _run_stage3(
        question_text, sample_dir, model, api_model, client, gen_max_tokens,
        profile, messages, turn, code_calls, max_code_calls, is_l5, verbose,
        skip_stage3=skip_stage3)
    stage_info.update(stage3_info)

    elapsed = time.time() - t0

    final = _select_final_response(messages, question_text)
    if is_l5:
        stage_info['consistency_issues_after'] = _l5_consistency_issues(final)

    return {
        "response": final,
        "messages": messages,
        "turns": turn,
        "code_calls": code_calls,
        "elapsed_s": round(elapsed, 1),
        "pipeline": True,
        "stage_info": stage_info,
    }


def _run_stage3(question_text, sample_dir, model, api_model, client,
                 gen_max_tokens, profile, messages, turn, code_calls,
                 max_code_calls, is_l5, verbose, skip_stage3=False):
    """Stage 3: answer-completion safety net plus optional constraint
    verification/repair pass. Extracted so the canonical pipeline and the
    no-stage2 ablation (paper §P3-2) take the identical finalization path.

    Returns (messages, turn, code_calls, stage_info) where stage_info holds
    only the keys this stage contributes.
    """
    stage_info = {}
    stage3_t0 = time.time()
    if verbose:
        print("  [Stage 3] Answer completion..."
              if skip_stage3 else "  [Stage 3] Verification pass...")

    last_content = ""
    for m in reversed(messages):
        c = m.get("content", "") if isinstance(m, dict) else ""
        if c and (isinstance(m, dict) and m.get("role") == "assistant"):
            last_content = c
            break

    if not _answer_block_complete(question_text, last_content):
        # Force answer first — schema-aware, same prompts as the free-form
        # runner (remediation log §8.1.4)
        from evaluation.runner import _force_answer_prompts
        messages.append({
            "role": "user",
            "content": _force_answer_prompts(question_text)[0],
        })
        for attempt in range(4):
            try:
                turn += 1
                create_kwargs = dict(
                    model=api_model, messages=messages,
                    max_tokens=gen_max_tokens,
                    temperature=profile['temperature'],
                )
                _apply_finalization_tool_policy(create_kwargs, profile)
                if profile['stream']:
                    create_kwargs["stream"] = True
                    resp = _collect_stream(client.chat.completions.create(**create_kwargs))
                else:
                    resp = client.chat.completions.create(**create_kwargs)
                msg = resp.choices[0].message
                last_content = msg.content or ""
                messages.append({"role": "assistant", "content": last_content})
                if _answer_block_complete(question_text, last_content):
                    break
                # Some models (confirmed: deepseek-v4-flash on the direct
                # DeepSeek API, where the reasoning-effort knob above can't
                # be disabled) keep wanting "one more measurement" even
                # with tools omitted, and encode it as a DSML/inline-tool
                # pseudo-call instead of a real answer. Running it — same
                # as Stage 2 already does — satisfies that impulse instead
                # of discarding real analysis text as a wasted attempt.
                # We only reach the force-answer loop once code_calls is
                # already at (or one below) max_code_calls, so gating this
                # on the same cap would just recreate the deadlock; a small
                # separate allowance lets the model actually close out.
                inline_code = (_extract_dsml_code(last_content)
                               or _extract_default_api_tool_code(last_content))
                if inline_code and code_calls < max_code_calls + FINALIZATION_CODE_CALL_ALLOWANCE:
                    result = execute_python(inline_code, sample_dir)
                    code_calls += 1
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Code execution result:\n{result}\n\nStop "
                            "calling tools now. Output the complete final "
                            "answer block."
                        ),
                    })
                    continue
                if attempt < 3:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was empty or incomplete. "
                            "Output the complete final answer block now."
                        ),
                    })
            except Exception:
                break

    # Some models (gemini-3.5-flash, glm-5.2, mimo-v2.5-pro) can get stuck
    # emitting empty or truncated turns once a long tool-call history has
    # accumulated, and repeating the nudge in the same conversation does not
    # recover them (confirmed by the ReconPilot/free-form audit, 2026-07-23).
    # The free-form runner already carries a fix for this — a fresh,
    # isolated prompt built only from the saved tool outputs — ported here
    # unchanged so the pipeline stops losing these runs to empty finals.
    if (not _answer_block_complete(question_text, last_content)
            and model in CLEAN_FALLBACK_MODELS):
        from evaluation.runner import _clean_final_answer_prompt, _force_answer_prompts
        for attempt in range(2):
            clean_prompt = _clean_final_answer_prompt(
                question_text, messages, _force_answer_prompts(question_text)[0])
            try:
                turn += 1
                clean_kwargs = dict(
                    model=api_model,
                    messages=[
                        {"role": "system", "content": (
                            "Complete all requested final answers from "
                            "saved measurements. Obey the exact output "
                            "template and return no analysis narrative.")},
                        {"role": "user", "content": clean_prompt},
                    ],
                    max_tokens=gen_max_tokens,
                    temperature=profile['temperature'],
                )
                _apply_finalization_tool_policy(clean_kwargs, profile)
                if profile['stream']:
                    clean_kwargs["stream"] = True
                    resp = _collect_stream(client.chat.completions.create(**clean_kwargs))
                else:
                    resp = client.chat.completions.create(**clean_kwargs)
                clean_content = resp.choices[0].message.content or ""
                messages.append({"role": "user", "content": clean_prompt})
                messages.append({"role": "assistant", "content": clean_content})
                if _answer_block_complete(question_text, clean_content):
                    last_content = clean_content
                    break
            except Exception:
                break

    # An earlier assistant turn may contain the complete answer even when the
    # latest forced response is truncated or prose-only.
    selected_before_verify = _select_final_response(messages, question_text)
    if _answer_block_complete(question_text, selected_before_verify):
        last_content = selected_before_verify

    # Verification turn. L5 gets deterministic cross-section findings and
    # one targeted measurement opportunity before the final no-tool rewrite.
    if not skip_stage3 and _answer_block_complete(question_text, last_content):
        verify_prompt = STAGE3_VERIFY
        if _question_requires_json(question_text):
            verify_prompt += STAGE3_JSON_ADDENDUM
        consistency_issues = (
            _l5_consistency_issues(last_content) if is_l5 else [])
        if is_l5:
            stage_info['consistency_issues_before'] = consistency_issues
            if consistency_issues:
                verify_prompt += (
                    "\n\nThe deterministic dependency audit found these "
                    "contradictions:\n"
                    + "\n".join(f"- {item}" for item in consistency_issues)
                    + "\nUse at most one targeted code call if a measurement "
                      "is needed, then correct every affected field."
                )
            else:
                verify_prompt += (
                    "\n\nThe deterministic dependency audit found no "
                    "cross-section contradiction. Preserve the source IDs "
                    "and only correct values supported by your measurements."
                )
        messages.append({"role": "user", "content": verify_prompt})
        try:
            turn += 1
            create_kwargs = dict(
                model=api_model, messages=messages,
                max_tokens=gen_max_tokens,
                temperature=profile['temperature'],
            )
            allow_repair_tool = bool(
                is_l5 and consistency_issues
                and code_calls < max_code_calls)
            if allow_repair_tool:
                create_kwargs.update(
                    tools=PIPELINE_TOOLS,
                    tool_choice=profile['tool_choice'],
                    parallel_tool_calls=profile['parallel_tool_calls'],
                )
            else:
                _apply_finalization_tool_policy(create_kwargs, profile)
            if profile['stream']:
                create_kwargs["stream"] = True
                resp = _collect_stream(client.chat.completions.create(**create_kwargs))
            else:
                resp = client.chat.completions.create(**create_kwargs)
            verify_msg = resp.choices[0].message
            verify_assistant = {
                "role": "assistant", "content": verify_msg.content or ""}
            if verify_msg.tool_calls:
                verify_assistant["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in verify_msg.tool_calls
                ]
            messages.append(verify_assistant)

            if verify_msg.tool_calls:
                for tool_index, tc in enumerate(verify_msg.tool_calls):
                    if tool_index > 0:
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": "[SKIPPED] Only one verification call is allowed.",
                        })
                        continue
                    try:
                        args = json.loads(tc.function.arguments)
                        code = args['code']
                    except (json.JSONDecodeError, KeyError):
                        code = tc.function.arguments
                    result = execute_python(code, sample_dir)
                    code_calls += 1
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": result,
                    })
                messages.append({
                    "role": "user",
                    "content": (
                        "Use the verification result to repair the audited "
                        "fields. Output the complete exact ===ANSWERS=== JSON "
                        "block now. Do not call another tool."
                    ),
                })
                turn += 1
                final_kwargs = dict(
                    model=api_model, messages=messages,
                    max_tokens=gen_max_tokens,
                    temperature=profile['temperature'],
                )
                _apply_finalization_tool_policy(final_kwargs, profile)
                if profile['stream']:
                    final_kwargs['stream'] = True
                    final_resp = _collect_stream(
                        client.chat.completions.create(**final_kwargs))
                else:
                    final_resp = client.chat.completions.create(**final_kwargs)
                final_msg = final_resp.choices[0].message
                messages.append({
                    "role": "assistant", "content": final_msg.content or ""})
        except Exception:
            pass

    stage_info['stage3_time'] = round(time.time() - stage3_t0, 1)
    return messages, turn, code_calls, stage_info


def run_pipeline(question_text, sample_dir, model=None, max_turns=None, verbose=True):
    """Run the pipeline in a workspace that exposes only the requested signal."""
    with isolated_signal_workspace(question_text, sample_dir) as workspace:
        return _run_pipeline(question_text, workspace, model, max_turns, verbose)


ABLATION_MODES = ('no-stage3', 'recon-only', 'no-stage2', 'no-stage1')


def _run_recon_only_ablation(question_text, sample_dir, model, max_turns, verbose):
    """Ablation for paper §P3-2: Stage 1 fixed reconnaissance, then an
    unmodified free-form agent loop over the same turn budget as the
    free-form baseline (no Stage 2 targeted-analysis framing or checkpoints,
    no Stage 3 verification). Isolates whether the recon injection alone
    accounts for ReconPilot's gain, as distinct from Stage 2's structuring.
    """
    from .runner import _is_anthropic_provider, _run_anthropic, _run_openai
    model = model or DEFAULT_MODEL
    max_turns = max_turns or MAX_TURNS

    with isolated_signal_workspace(question_text, sample_dir) as workspace:
        fs_match = re.search(
            r'(?:采样率|Sampling rate)[：:]\s*([\d.]+)\s*MHz', question_text)
        fs = float(fs_match.group(1)) * 1e6 if fs_match else 20e6
        npy_file = extract_signal_filename(question_text)
        recon_code = RECON_CODE_TEMPLATE.format(npy_file=npy_file, fs=int(fs))
        recon_result = execute_python(recon_code, workspace, timeout=RECON_TIMEOUT_S)
        if _reconnaissance_failed(recon_result):
            raise RuntimeError(
                "Localization failed; refusing to continue with an invalid "
                "region map."
            )
        augmented_question = (
            "## Automated Reconnaissance (localization only; an ROI is not "
            "a signal and ROI count is not source count)\n```\n"
            + recon_result + "\n```\n\n" + question_text
        )
        if _is_anthropic_provider(model):
            result = _run_anthropic(
                augmented_question, workspace, model, max_turns, verbose)
        else:
            result = _run_openai(
                augmented_question, workspace, model, max_turns, verbose)
        result['pipeline'] = False
        result['stage_info'] = {
            'recon_output': recon_result, 'ablation_mode': 'recon-only'}
        return result


def _run_no_stage2_ablation(question_text, sample_dir, model, max_turns, verbose):
    """Ablation for paper §P3-2: Stage 1 fixed reconnaissance and Stage 3
    verification/repair are both kept exactly as in the canonical pipeline;
    only Stage 2's structured targeted-analysis framing (source inventory,
    checkpoints, tool-choice nudges) is replaced by an unmodified free-form
    agent loop over the recon-augmented question, given the same turn
    budget Stage 2 would have had (``max_turns`` minus the turns reserved
    for Stage 3). ``no-stage3`` and ``recon-only`` together only bound
    Stage 2's contribution by subtraction; this isolates it directly.
    """
    from .runner import _is_anthropic_provider, _run_anthropic, _run_openai
    model = model or DEFAULT_MODEL
    max_turns = max_turns or MAX_TURNS
    is_l5 = _is_l5_question(question_text)
    max_code_calls = MAX_PIPELINE_CODE_CALLS_L5 if is_l5 else MAX_PIPELINE_CODE_CALLS
    analysis_max = max(0, max_turns - STAGE3_RESERVED_TURNS)

    with isolated_signal_workspace(question_text, sample_dir) as workspace:
        t0 = time.time()
        fs_match = re.search(
            r'(?:采样率|Sampling rate)[：:]\s*([\d.]+)\s*MHz', question_text)
        fs = float(fs_match.group(1)) * 1e6 if fs_match else 20e6
        npy_file = extract_signal_filename(question_text)
        recon_code = RECON_CODE_TEMPLATE.format(npy_file=npy_file, fs=int(fs))
        recon_result = execute_python(recon_code, workspace, timeout=RECON_TIMEOUT_S)
        if _reconnaissance_failed(recon_result):
            raise RuntimeError(
                "Localization failed; refusing to continue with an invalid "
                "region map."
            )
        stage_info = {
            'recon_output': recon_result,
            'recon_time': round(time.time() - t0, 1),
            'ablation_mode': 'no-stage2',
        }
        augmented_question = (
            "## Automated Reconnaissance (localization only; an ROI is not "
            "a signal and ROI count is not source count)\n```\n"
            + recon_result + "\n```\n\n" + question_text
        )

        stage2_t0 = time.time()
        if _is_anthropic_provider(model):
            free_form = _run_anthropic(
                augmented_question, workspace, model, analysis_max, verbose)
        else:
            free_form = _run_openai(
                augmented_question, workspace, model, analysis_max, verbose)
        stage_info['stage2_turns'] = free_form['turns']
        stage_info['stage2_time'] = round(time.time() - stage2_t0, 1)

        api_key, base_url = get_client_config(model)
        profile = pipeline_execution_profile(model)
        api_model = profile['api_model']
        api_timeout = profile['api_timeout_s']
        client_kwargs = dict(api_key=api_key, base_url=base_url, timeout=api_timeout)
        if profile['clean_headers']:
            client_kwargs["default_headers"] = {"User-Agent": "python-httpx/0.27"}
        if profile['bypass_environment_proxy']:
            client_kwargs["http_client"] = httpx.Client(
                trust_env=False, timeout=api_timeout
            )
        client = OpenAI(**client_kwargs)
        gen_max_tokens = profile['generation_max_tokens']

        messages, turn, code_calls, stage3_info = _run_stage3(
            question_text, workspace, model, api_model, client, gen_max_tokens,
            profile, free_form['messages'], free_form['turns'],
            free_form['code_calls'], max_code_calls, is_l5, verbose,
            skip_stage3=False)
        stage_info.update(stage3_info)

        elapsed = time.time() - t0
        final = _select_final_response(messages, question_text)
        if is_l5:
            stage_info['consistency_issues_after'] = _l5_consistency_issues(final)

        return {
            "response": final,
            "messages": messages,
            "turns": turn,
            "code_calls": code_calls,
            "elapsed_s": round(elapsed, 1),
            "pipeline": False,
            "stage_info": stage_info,
        }


def _run_no_stage1_ablation(question_text, sample_dir, model, max_turns, verbose):
    """Ablation: drop Stage 1 only, keep Stage 2 and Stage 3 as reported.

    This is the ``no-stage2`` reported configuration with the reconnaissance
    region map removed: an unmodified free-form agent loop over the original
    question text, given the same Stage 2 turn budget (``max_turns`` minus the
    turns reserved for Stage 3), followed by the identical Stage 3
    verification and repair path. It is the missing cell of the Stage 1 x
    Stage 3 grid: comparing it against the reported configuration isolates
    Stage 1's contribution without also removing Stage 3, which
    ``recon-only`` versus free form cannot do.
    """
    from .runner import _is_anthropic_provider, _run_anthropic, _run_openai
    model = model or DEFAULT_MODEL
    max_turns = max_turns or MAX_TURNS
    is_l5 = _is_l5_question(question_text)
    max_code_calls = MAX_PIPELINE_CODE_CALLS_L5 if is_l5 else MAX_PIPELINE_CODE_CALLS
    analysis_max = max(0, max_turns - STAGE3_RESERVED_TURNS)

    with isolated_signal_workspace(question_text, sample_dir) as workspace:
        t0 = time.time()
        stage_info = {'ablation_mode': 'no-stage1', 'recon_output': None}

        stage2_t0 = time.time()
        if _is_anthropic_provider(model):
            free_form = _run_anthropic(
                question_text, workspace, model, analysis_max, verbose)
        else:
            free_form = _run_openai(
                question_text, workspace, model, analysis_max, verbose)
        stage_info['stage2_turns'] = free_form['turns']
        stage_info['stage2_time'] = round(time.time() - stage2_t0, 1)

        api_key, base_url = get_client_config(model)
        profile = pipeline_execution_profile(model)
        api_model = profile['api_model']
        api_timeout = profile['api_timeout_s']
        client_kwargs = dict(api_key=api_key, base_url=base_url, timeout=api_timeout)
        if profile['clean_headers']:
            client_kwargs["default_headers"] = {"User-Agent": "python-httpx/0.27"}
        if profile['bypass_environment_proxy']:
            client_kwargs["http_client"] = httpx.Client(
                trust_env=False, timeout=api_timeout
            )
        client = OpenAI(**client_kwargs)
        gen_max_tokens = profile['generation_max_tokens']

        messages, turn, code_calls, stage3_info = _run_stage3(
            question_text, workspace, model, api_model, client, gen_max_tokens,
            profile, free_form['messages'], free_form['turns'],
            free_form['code_calls'], max_code_calls, is_l5, verbose,
            skip_stage3=False)
        stage_info.update(stage3_info)

        elapsed = time.time() - t0
        final = _select_final_response(messages, question_text)
        if is_l5:
            stage_info['consistency_issues_after'] = _l5_consistency_issues(final)

        return {
            "response": final,
            "messages": messages,
            "turns": turn,
            "code_calls": code_calls,
            "elapsed_s": round(elapsed, 1),
            "pipeline": False,
            "stage_info": stage_info,
        }


def run_pipeline_ablation(question_text, sample_dir, model=None,
                           max_turns=None, verbose=True, mode='no-stage3'):
    """Dispatch to one of the three ReconPilot ablations (paper §P3-2).

    ``no-stage3`` runs the real Stage 1 + Stage 2 and drops only the Stage 3
    verification/repair turn. ``recon-only`` drops both Stage 2's structured
    framing and Stage 3, leaving just the recon injection over a free-form
    loop. ``no-stage2`` keeps Stage 1 and Stage 3 but replaces Stage 2 with
    an unmodified free-form loop, isolating Stage 2's structuring alone.
    ``no-stage1`` is that same configuration with the reconnaissance map
    removed, isolating Stage 1's contribution.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"unknown ablation mode: {mode!r}")
    if mode == 'no-stage3':
        with isolated_signal_workspace(question_text, sample_dir) as workspace:
            return _run_pipeline(
                question_text, workspace, model, max_turns, verbose,
                skip_stage3=True)
    if mode == 'no-stage2':
        return _run_no_stage2_ablation(
            question_text, sample_dir, model, max_turns, verbose)
    if mode == 'no-stage1':
        return _run_no_stage1_ablation(
            question_text, sample_dir, model, max_turns, verbose)
    return _run_recon_only_ablation(
        question_text, sample_dir, model, max_turns, verbose)
