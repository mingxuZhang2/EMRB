"""Regression tests for the shared answer parser and unit machinery
(evaluation/auto_scorer.parse_answer_block, evaluation/answer_parsing)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.answer_parsing import (
    quantities_in_family,
    score_scalar,
)
from evaluation.auto_scorer import parse_answer_block


def _block(body):
    return f'===ANSWERS===\n{body}\n===END==='


def test_parser_plain_labels():
    answers = parse_answer_block(_block('Q1a: 3 signals\nQ1b: -4.7 MHz\n'
                                        'Q2: whole question'))
    assert answers['Q1a'] == '3 signals'
    assert answers['Q1b'] == '-4.7 MHz'
    assert answers['Q2'] == 'whole question'


def test_parser_decorated_labels():
    answers = parse_answer_block(_block(
        '**Q1a**: bold\n'
        '**Q1b:** bold colon inside\n'
        'Q2a (Signal @+1.16 MHz): parenthesized\n'
        'Q2B: uppercase letter\n'
        'Q3a： full-width colon\n'
        '## Q3b: heading label'))
    assert answers['Q1a'] == 'bold'
    assert answers['Q1b'] == 'bold colon inside'
    assert answers['Q2a'] == 'parenthesized'
    assert answers['Q2b'] == 'uppercase letter'
    assert answers['Q3a'] == 'full-width colon'
    assert answers['Q3b'] == 'heading label'


def test_parser_multiline_continuation_and_json():
    answers = parse_answer_block(_block(
        'Q1: first line\nsecond line\nQ3: {"comb_spacing_us": 96.9}'))
    assert answers['Q1'] == 'first line\nsecond line'
    assert answers['Q3'] == '{"comb_spacing_us": 96.9}'


def test_parser_duplicate_labels_keep_last():
    answers = parse_answer_block(_block('Q1a: first\nQ1a: revised'))
    assert answers['Q1a'] == 'revised'


def test_parser_uses_last_answer_block():
    response = (_block('Q1a: <value>') + '\nintermediate prose\n'
                + _block('Q1a: 3 signals\nQ1b: -4.7 MHz'))
    answers = parse_answer_block(response)
    assert answers == {'Q1a': '3 signals', 'Q1b': '-4.7 MHz'}


def test_parser_accepts_bulleted_labels():
    answers = parse_answer_block(_block(
        '- Q1a: 3 signals\n'
        '- **Q1b:** -4.7 MHz'))
    assert answers['Q1a'] == '3 signals'
    assert answers['Q1b'] == '-4.7 MHz'


def test_parser_accepts_question_headings_with_lettered_parts():
    answers = parse_answer_block(_block(
        '**Q1. Spectrum observation**\n'
        '**(a)** 3 signals\n'
        '**(b) Center frequencies:** -4.7 MHz'))
    assert answers['Q1a'] == '3 signals'
    assert answers['Q1b'] == 'Center frequencies:** -4.7 MHz'


def test_parser_merges_per_signal_suffixes():
    answers = parse_answer_block(_block(
        'Q3a_BPSK: 4.0 dB\n'
        'Q3a_FM: 0.2 dB\n'
        'Q3b: FM is best suited'))
    assert answers['Q3a'] == 'BPSK: 4.0 dB\nFM: 0.2 dB'
    assert answers['Q3b'] == 'FM is best suited'


def test_parser_does_not_split_on_prose_q_references():
    answers = parse_answer_block(_block(
        'Q1a: uses the result of Q2 (see below): still Q1a text'))
    assert list(answers) == ['Q1a']


def test_symbol_rate_unit_spellings():
    for text in ('350 ksym/s', '350 ksps', '0.35 Msym/s', '350 kbaud',
                 '350000 baud', '350000 sym/s', '350'):
        ratio, _ = score_scalar(text, 350, 'symrate', 'ksps', 0.01, None)
        assert ratio == 1.0, text


def test_length_units():
    for text in ('47 m', '0.047 km', '4700 cm'):
        ratio, _ = score_scalar(text, 47, 'length', 'm', 0.01, None)
        assert ratio == 1.0, text
    # unit suffixes of other families must not leak into 'length'
    assert quantities_in_family('5 ms and 3 Mbps', 'length', 'm') == []


def test_spectral_efficiency_unit_spellings():
    for text in ('8.35 bps/Hz', '8.35 bits/s/Hz', '8.35 bit/s/Hz',
                 '8.35 b/s/Hz', '8.35'):
        ratio, _ = score_scalar(text, 8.35, 'eff', 'bps/Hz', 0.01, None)
        assert ratio == 1.0, text


def test_percent_and_fraction_ratios():
    for text in ('duty cycle = 35%', 'duty cycle = 0.35',
                 'duty cycle ≈ 35 %'):
        ratio, _ = score_scalar(text, 0.35, 'ratio', None, 0.01, None,
                                mode='abs')
        assert ratio == 1.0, text
    ratio, _ = score_scalar('30%', 0.35, 'ratio', None, 0.01, None,
                            mode='abs')
    assert ratio == 0.0


def test_explicit_only_quantities():
    text = 'M = 4, rate = 100 ksym/s'
    explicit = quantities_in_family(text, 'symrate', 'ksps',
                                    explicit_only=True)
    assert [value for value, _ in explicit] == [100.0]
    loose = quantities_in_family(text, 'symrate', 'ksps')
    assert [value for value, _ in loose] == [4.0, 100.0]


def test_wrong_unit_magnitude_is_not_credited():
    # 15.625 MHz is not 15.625 kHz even though the digits match
    ratio, _ = score_scalar('15.625 MHz', 15.625, 'freq', 'kHz', 0.20, 0.40)
    assert ratio == 0.0
