"""Rule-based auto-scorer for EMRB evaluation.

Scores objective items (numerical with tolerance, categorical modulation, boolean).
Returns unscored subjective points for LLM-judge to handle.
"""
import re
from collections import OrderedDict

MODULATION_TYPES = {
    'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM',
    'FM', 'AM-DSB', 'AM', 'OFDM', '2FSK', '4FSK',
    'Chirp (LFM)', 'Chirp',
}

MOD_FAMILIES = {
    'BPSK': 'PSK', 'QPSK': 'PSK', '8PSK': 'PSK',
    '16QAM': 'QAM', '64QAM': 'QAM',
}

SUBJECTIVE_KEYS = {
    'reasoning', 'method', 'note', 'design', 'formulas',
    'filter_design', 'gap_analysis', 'coexistence',
    'pa_suitability', 'ibo_efficiency', 'reduction',
}


def parse_answer_block(response):
    answers = OrderedDict()
    if '===ANSWERS===' not in response:
        return answers
    # Models sometimes quote the format template before emitting their real
    # answer.  The final answer block is authoritative.
    block = response.rsplit('===ANSWERS===', 1)[1]
    if '===END===' in block:
        block = block.split('===END===', 1)[0]
    current_key = None
    current_lines = []
    current_question = None
    merge_duplicate = False

    def flush():
        nonlocal current_key, current_lines, merge_duplicate
        if current_key is None:
            return
        value = '\n'.join(current_lines).strip()
        if merge_duplicate and current_key in answers:
            answers[current_key] = '\n'.join(
                part for part in (answers[current_key], value) if part
            )
        else:
            # An undecorated repeated label is a revision; keep the latest.
            answers[current_key] = value
        current_key = None
        current_lines = []
        merge_duplicate = False

    for line in block.strip().split('\n'):
        stripped = line.strip()

        # Markdown section followed by lettered parts, for example
        # "**Q1. Spectrum observation**" and "**(a)** 3 signals".
        heading = re.match(
            r'[*_#>+\-•\s]*(Q\d+)\s*[.．、]\s*.+', stripped,
            re.IGNORECASE,
        )
        if heading:
            flush()
            current_question = heading.group(1).upper()
            continue

        # Tolerate bullets, bold labels, uppercase letters, parenthesized
        # decorations, and per-signal suffixes such as Q3a_BPSK.
        m = re.match(
            r'[*_#>+\-•\s]*(Q\d+)([a-zA-Z])?'
            r'(?P<suffix>_[^:：\s]{1,40})?[*_]*\s*'
            r'(?:[(（][^)）]{0,80}[)）])?\s*[*_]*\s*[:：][*_\s]*(.*)',
            stripped,
        )
        if m:
            flush()
            current_question = m.group(1).upper()
            current_key = current_question + (m.group(2) or '').lower()
            suffix = (m.group('suffix') or '').lstrip('_')
            value = m.group(4).strip()
            if suffix:
                value = f'{suffix}: {value}' if value else f'{suffix}:'
                merge_duplicate = True
            current_lines = [value] if value else []
            continue

        lettered = re.match(
            r'[*_#>+\-•\s]*[(（]\s*([a-zA-Z])\s*[)）][*_\s]*'
            r'(?:[:：.]\s*)?(.*)',
            stripped,
        )
        if lettered and current_question:
            flush()
            current_key = current_question + lettered.group(1).lower()
            value = lettered.group(2).strip()
            current_lines = [value] if value else []
        elif current_key is not None:
            current_lines.append(stripped)
    flush()
    return answers


def extract_numbers(text):
    return [float(x) for x in re.findall(
        r'(?<![A-Za-z])[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', text)]


def score_number(model_val, gt_val, tol_pct=20):
    if abs(gt_val) < 1e-12:
        return 1.0 if abs(model_val) < 0.01 else 0.0
    err = abs(model_val - gt_val) / abs(gt_val) * 100
    if err <= tol_pct:
        return 1.0
    if err <= tol_pct * 2:
        return 0.5
    return 0.0


def _is_subjective(key):
    return any(sk in key.lower() for sk in SUBJECTIVE_KEYS)


def _parse_tolerance(rubric_item):
    """Extract tolerance percentage from rubric item."""
    for k in ('tol', 'tolerance', 'tol_pct'):
        v = rubric_item.get(k, '')
        if isinstance(v, (int, float)):
            return v
        m = re.search(r'(\d+)', str(v))
        if m:
            return int(m.group(1))
    return 20  # default


def flatten_gt(ground_truth):
    """Flatten GT into (key, value, type) tuples for auto-scoring."""
    items = []
    for key, val in ground_truth.items():
        if isinstance(val, bool):
            items.append((key, val, 'bool'))
        elif isinstance(val, (int, float)):
            items.append((key, float(val), 'number'))
        elif isinstance(val, str):
            clean = val.replace(' (burst)', '')
            if clean in MODULATION_TYPES:
                items.append((key, clean, 'modulation'))
        elif isinstance(val, list):
            for idx, elem in enumerate(val):
                fk = f"{key}[{idx}]"
                if isinstance(elem, (int, float)):
                    items.append((fk, float(elem), 'number'))
                elif isinstance(elem, str) and elem.replace(' (burst)', '') in MODULATION_TYPES:
                    items.append((fk, elem.replace(' (burst)', ''), 'modulation'))
                elif isinstance(elem, dict):
                    for k2, v2 in elem.items():
                        fk2 = f"{key}[{idx}].{k2}"
                        if isinstance(v2, (int, float)):
                            items.append((fk2, float(v2), 'number'))
                        elif isinstance(v2, str) and v2.replace(' (burst)', '') in MODULATION_TYPES:
                            items.append((fk2, v2.replace(' (burst)', ''), 'modulation'))
        elif isinstance(val, dict):
            for k2, v2 in val.items():
                fk = f"{key}.{k2}"
                if isinstance(v2, bool):
                    items.append((fk, v2, 'bool'))
                elif isinstance(v2, (int, float)):
                    items.append((fk, float(v2), 'number'))
                elif isinstance(v2, str) and v2.replace(' (burst)', '') in MODULATION_TYPES:
                    items.append((fk, v2.replace(' (burst)', ''), 'modulation'))
                elif isinstance(v2, list):
                    for idx, elem in enumerate(v2):
                        fk2 = f"{fk}[{idx}]"
                        if isinstance(elem, (int, float)):
                            items.append((fk2, float(elem), 'number'))
    return items


def auto_score_question(response, ground_truth, rubric, question_id='Q1'):
    """Auto-score one question. Returns dict with:
      - total_score: points earned (objective only)
      - objective_max: max points for objective items
      - subjective_max: remaining points needing LLM-judge
      - sub_scores: per-item details
    Or None if can't auto-score at all.
    """
    gt_items = flatten_gt(ground_truth)
    if not gt_items:
        return None

    rubric_total = rubric.get('points', 20)

    # Scope answers to this question only
    all_answers = parse_answer_block(response)
    q_num = question_id.replace('Q', '')
    scoped = {k: v for k, v in all_answers.items() if k.startswith(f'Q{q_num}')}
    answer_text = ' '.join(scoped.values()) if scoped else ""
    if not answer_text:
        return None

    # Classify rubric sub-items as objective vs subjective
    objective_pts = 0
    subjective_pts = 0
    rubric_pts_map = {}  # key -> pts
    for rk, rv in rubric.items():
        if not isinstance(rv, dict) or 'pts' not in rv:
            continue
        if _is_subjective(rk):
            subjective_pts += rv['pts']
        else:
            objective_pts += rv['pts']
            rubric_pts_map[rk] = rv

    # If no explicit sub-items in rubric, treat all as objective
    if objective_pts == 0 and subjective_pts == 0:
        objective_pts = rubric_total
        subjective_pts = 0

    # Distribute objective points across GT items.
    # Group GT items by matching rubric key, then split that key's pts evenly.
    item_rubric_key = []  # which rubric key each GT item belongs to
    for key, val, typ in gt_items:
        matched_rk = None
        for rk in rubric_pts_map:
            if rk.lower() in key.lower() or key.lower() in rk.lower():
                matched_rk = rk
                break
        item_rubric_key.append(matched_rk)

    # Count how many GT items share each rubric key
    from collections import Counter
    rk_counts = Counter(rk for rk in item_rubric_key if rk is not None)

    # Assign per-item points = rubric_pts / count_of_items_in_group
    item_pts_list = []
    for rk in item_rubric_key:
        if rk is not None:
            item_pts_list.append(rubric_pts_map[rk]['pts'] / rk_counts[rk])
        else:
            item_pts_list.append(None)

    # Fill unmatched items with equal share of remaining objective points
    assigned = sum(p for p in item_pts_list if p is not None)
    remaining = max(0, objective_pts - assigned)
    n_unassigned = sum(1 for p in item_pts_list if p is None)
    fill = remaining / max(n_unassigned, 1)
    item_pts_list = [p if p is not None else fill for p in item_pts_list]

    # Score each GT item
    all_numbers = extract_numbers(answer_text)
    used = set()
    scored = 0.0
    details = []

    for (key, gt_val, typ), pts in zip(gt_items, item_pts_list):
        earned = 0.0
        reason = ""

        if typ == 'number':
            best_idx, best_err = None, float('inf')
            for i, n in enumerate(all_numbers):
                if i in used:
                    continue
                err = abs(n - gt_val) / max(abs(gt_val), 1e-12) * 100
                if err < best_err:
                    best_idx, best_err = i, err
            if best_idx is not None and best_err < 50:
                used.add(best_idx)
                frac = score_number(all_numbers[best_idx], gt_val)
                earned = pts * frac
                reason = (f"found={all_numbers[best_idx]:.4g} gt={gt_val:.4g} "
                          f"err={best_err:.1f}%")
            else:
                reason = f"gt={gt_val:.4g} not found"

        elif typ == 'modulation':
            pat = r'\b' + re.escape(gt_val) + r'\b'
            if re.search(pat, answer_text, re.IGNORECASE):
                earned = pts
                reason = f"gt={gt_val} found"
            else:
                gt_fam = MOD_FAMILIES.get(gt_val, '')
                found_fam = False
                if gt_fam:
                    for mod in MODULATION_TYPES:
                        if MOD_FAMILIES.get(mod) == gt_fam and mod != gt_val:
                            if re.search(r'\b' + re.escape(mod) + r'\b',
                                         answer_text, re.IGNORECASE):
                                earned = pts * 0.5
                                reason = f"gt={gt_val}, family match {mod}"
                                found_fam = True
                                break
                if not found_fam:
                    reason = f"gt={gt_val} not found"

        elif typ == 'bool':
            tl = answer_text.lower()
            pos = any(w in tl for w in
                      ['yes', 'true', '是', '满足', '足够', 'sufficient', '可以'])
            neg = any(w in tl for w in
                      ['no', 'false', '否', '不满足', '不够', 'insufficient',
                       '不足', '不可以', 'not enough'])
            if (gt_val and pos) or (not gt_val and neg):
                earned = pts
                reason = "correct"
            else:
                reason = f"expected {'yes' if gt_val else 'no'}"

        scored += earned
        details.append({"key": key, "gt": gt_val, "score": round(earned, 1),
                        "max": round(pts, 1), "reason": reason})

    obj_max = min(round(sum(p for p in item_pts_list), 1), rubric_total)
    scored = min(scored, obj_max)

    return {
        "total_score": round(scored, 1),
        "objective_max": obj_max,
        "subjective_max": round(rubric_total - obj_max, 1),
        "rubric_total": rubric_total,
        "sub_scores": details,
        "method": "auto",
    }
