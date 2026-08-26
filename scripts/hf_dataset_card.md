---
license: cc-by-4.0
language:
- en
task_categories:
- question-answering
tags:
- electromagnetic
- signal-processing
- iq-data
- agentic-benchmark
- code-execution
- reasoning
size_categories:
- n<1K
pretty_name: 'EMRB: Electromagnetic Reasoning Benchmark'
configs:
- config_name: default
  data_files:
  - split: L1
    path: parquet/L1-*.parquet
  - split: L2
    path: parquet/L2-*.parquet
  - split: L3
    path: parquet/L3-*.parquet
  - split: L4
    path: parquet/L4-*.parquet
  - split: L5
    path: parquet/L5-*.parquet
---

# EMRB: A Multi-Level Benchmark for Evaluating LLM Reasoning over Raw Electromagnetic Signals

[Paper (arXiv:2608.24086)](https://arxiv.org/abs/2608.24086) | [Code](https://github.com/mingxuZhang2/EMRB)

EMRB evaluates language models on raw I/Q signal analysis. Unlike benchmarks built on
preprocessed features or structured tables, each problem provides only the complex baseband
capture: the quantities a question refers to (center frequencies, symbol rates, modulation
order, noise floor, occupied bandwidth) exist nowhere in the prompt and must be recovered by
writing and executing code. A model reads the question, writes Python, runs it in a sandbox,
iterates, and emits a structured answer block that deterministic verifiers score.

{{TOTAL}} problems ({{PER_LEVEL}}), 920 sub-questions, 11 signal types, and 27 question types
with zero overlap across levels.

| Split | Theme | Sub-Qs | Topics |
|:-----:|-------|:------:|--------|
| **L1** | Basic Measurement | 5 | Signal detection, power, sampling params, noise floor, classification |
| **L2** | Signal Processing | 5 | FFT/windowing, bandwidth definitions, autocorrelation, STFT, energy/PSD |
| **L3** | Communication Theory | 5 | Bitrate/symbol rate, Eb/N0 and BER, PAPR, ADC quantization, DDC/mixing |
| **L4** | Multi-Signal Analysis | 5 | Symbol rate, FM, chirp/radar, burst, link budget, OFDM, interference, AM, Shannon |
| **L5** | System Design | 3 | Spectrum survey, radar coexistence, OFDM system design |

Across 14 evaluated LLMs, overall scores range from 24.1% to 78.9% with no ceiling effect. The
cross-model mean falls from 84.9% on L1 to 21.2% on L5.

## Layout

The same 200 problems are published in two representations.

```
parquet/L{1..5}-00000-of-00001.parquet   tabular view: drives load_dataset and the viewer
raw/L{1..5}/EMRB_*.npy, EMRB_*.json      canonical files, what the evaluation harness reads
```

Parquet has no complex type, so each capture is stored as two `float32` sequence columns,
`iq_real` and `iq_imag`. The `raw/` tree is the format of record and is byte-identical to
`data/` in the GitHub repository.

## Loading

Tabular access, one row per problem:

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("mingxuzhang/EMRB")          # splits L1 ... L5
row = ds["L4"][0]

iq = np.asarray(row["iq_real"], dtype=np.float32) + 1j * np.asarray(row["iq_imag"], dtype=np.float32)
print(row["sample_id"], iq.shape, row["sampling_rate_hz"])
print(row["prompt"])
```

Canonical files, for running the released harness unchanged:

```python
from huggingface_hub import snapshot_download

path = snapshot_download("mingxuzhang/EMRB", repo_type="dataset", allow_patterns="raw/*")
# path/raw/L3/EMRB_L3_4000.npy  ->  complex64, 32768 samples at 20 MHz
```

## Fields

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | e.g. `EMRB_L3_4000` |
| `level` | string | `L1` ... `L5` |
| `archetype` | string | Signal-mixture archetype (8 per level) |
| `archetype_desc` | string | Human-readable archetype, e.g. `BPSK + FM + Chirp` |
| `num_questions` | int32 | 5 for L1-L4, 3 for L5 |
| `total_points` | int32 | 100 per problem |
| `prompt` | string | Full problem statement handed to the model |
| `questions_json` | string | JSON list of sub-questions, each with `ground_truth` and `rubric` |
| `metadata_json` | string | The complete original JSON, lossless (includes `generation_params` with true signal parameters, and L5 `verification` rules) |
| `sampling_rate_hz` | float64 | 20 MHz for all problems |
| `num_samples` | int32 | 32768 (L1-L4) or 65536 (L5) |
| `npy_file` | string | Path of the canonical capture inside this repo |
| `iq_real`, `iq_imag` | sequence[float32] | Real and imaginary parts of the `complex64` capture |

Ground truth, scoring rubrics, and the generator parameters that produced each capture all
ship with the data, so scores are reproducible without contacting the authors. Anyone
evaluating a model should keep `questions_json` and `metadata_json` out of the model's context.

## Scoring

Scoring is fully deterministic, with no LLM judge anywhere in the reported numbers. Per-level
verifiers apply entity-bound matching (an answer is validated against the specific signal and
sub-question it refers to), quantity-specific tolerances (for example ±0.5 MHz on center
frequency, ±2 dB on power), functional acceptance rules for L4/L5 design answers, and
prerequisite-gated scoring at L5. The verifiers, the agent loop, and the ReconPilot pipeline
are in the [GitHub repository](https://github.com/mingxuZhang2/EMRB).

## Generation

All captures are synthesized from a seeded signal library (8 archetypes × 5 seeds per level),
which is why ground truth is exact rather than annotated. Regenerating the data is
deterministic: `python generate.py --level L1`. Synthesis also means the benchmark carries no
real-world capture and no personal or transmitter-identifying information.

## Licensing

Data CC BY 4.0. Code in the GitHub repository MIT.

## Citation

```bibtex
@misc{zhang2026emrbmultilevelbenchmarkevaluating,
      title={EMRB: A Multi-Level Benchmark for Evaluating LLM Reasoning over Raw Electromagnetic Signals},
      author={Mingxu Zhang and Ying Sun and Yuhan Li and Yang Ji and Dazhong Shen and Ke Zhang and Shan Huang},
      year={2026},
      eprint={2608.24086},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.24086},
}
```
