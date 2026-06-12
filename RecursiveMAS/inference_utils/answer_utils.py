# inference_utils.py

import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - keeps answer-checker self-tests lightweight.
    class _TorchStub:
        Tensor = Any

    torch = _TorchStub()


_GSM8K_KEYS = {"gsm8k", "openai/gsm8k"}
_MATH500_KEYS = {"math500", "math-500", "huggingfaceh4/math-500"}
_MEDQA_KEYS = {
    "medqa",
    "local/medqa",
    "dataset/medqa.json",
    "./dataset/medqa.json",
}
_GPQA_KEYS = {
    "gpqa",
    "gpqa_diamond",
    "idavidrein/gpqa",
    "idavidrein/gpqa:gpqa_diamond",
    "idavidrein/gpqa_diamond",
}
MATH500_CHECKER_VERSION = "math500_strict_v1"

def _dataset_key(name: str) -> str:
    return name.strip().lower()


def _ensure_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def truncate_text_chars(text: str, max_chars: int) -> str:
    text = _ensure_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _is_gsm8k_dataset(name: str) -> bool:
    return _dataset_key(name) in _GSM8K_KEYS


def _is_math500_dataset(name: str) -> bool:
    return _dataset_key(name) in _MATH500_KEYS


def is_medqa_dataset(name: str) -> bool:
    return _dataset_key(name) in _MEDQA_KEYS


def is_gpqa_dataset(name: str) -> bool:
    return _dataset_key(name) in _GPQA_KEYS


def is_choice_dataset(name: str) -> bool:
    key = _dataset_key(name)
    return key in _MEDQA_KEYS or key in _GPQA_KEYS



def ensure_choice_instruction(question: str) -> str:
    question = _ensure_text(question).rstrip()
    if "choose the correct option" in question.lower():
        return question
    return (
        f"{question}\n\n"
        "Choose the correct option (A/B/C/D)."
    )


def strip_choice_instruction_lines(question: str) -> str:
    """Remove extra choice-output instructions, keep stem + A/B/C/D options."""
    text = _ensure_text(question)
    if not text:
        return text

    choose_pat = re.compile(r"^\s*Choose\s+the\s+correct\s+option\s*\(A/B/C/D\)\.?\s*$", re.IGNORECASE)
    final_choice_pat = re.compile(r"^\s*Final\s*Choice\s*:\s*.*$", re.IGNORECASE)

    kept_lines = []
    for line in text.splitlines():
        if choose_pat.match(line):
            continue
        if final_choice_pat.match(line):
            continue
        kept_lines.append(line)

    # Collapse overly long blank runs introduced by line removal.
    out_lines = []
    prev_blank = False
    for line in kept_lines:
        is_blank = (line.strip() == "")
        if is_blank and prev_blank:
            continue
        out_lines.append(line)
        prev_blank = is_blank
    return "\n".join(out_lines).strip()


def is_gemma_model_name(model_name_or_path: str) -> bool:
    return "gemma" in _dataset_key(_ensure_text(model_name_or_path))


def soften_planner_format_instruction(prompt: str) -> str:
    """Replace rigid Step-1 template instruction with a softer instruction."""
    text = _ensure_text(prompt)
    if not text:
        return text

    replacement = (
        "Provide a clear step-by-step plan (within 3-5 steps) to solve the question. "
        "Do not calculate the final answer."
    )

    text = text.replace(
        "Your response should be in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ...",
        replacement,
    )
    text = text.replace(
        "Output only a concise plan in the format:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ...",
        replacement,
    )
    return text


def _normalize_option_text(text: str) -> str:
    text = _ensure_text(text).strip()
    text = re.sub(r"^\s*[A-Da-d]\s*[\.\):\-]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _extract_choice_core(text: str) -> Optional[str]:
    text = _ensure_text(text).strip()
    if not text:
        return None

    # Prefer explicit "choice/answer/option" mentions.
    keyword_match = re.search(
        r"(?:final\s*(?:choice|answer)|correct\s*(?:choice|option|answer)|choice|option|answer)\s*[:\-]?\s*[\(\[]?\s*([A-Da-d])\b",
        text,
        flags=re.IGNORECASE,
    )
    if keyword_match:
        return keyword_match.group(1).upper()

    # Accept short direct forms like "A", "(B)", "C: ...".
    direct_match = re.match(r"^\s*[\(\[]?\s*([A-Da-d])(?:\s*[\)\]]|\s*[:\.\-]|$)", text)
    if direct_match:
        return direct_match.group(1).upper()

    return None


def extract_choice_answer(text: str, default: Optional[str] = None) -> Optional[str]:
    text = _ensure_text(text)
    candidates = []

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        candidates.append(boxed)

    final_lines = re.findall(
        r"Final\s*(?:Choice|Answer)\s*:\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if final_lines:
        candidates.append(final_lines[-1])

    candidates.append(text.strip())

    for candidate in candidates:
        letter = _extract_choice_core(candidate)
        if letter is not None:
            return letter

    if default is None:
        return None

    fallback = _extract_choice_core(default)
    return fallback if fallback is not None else "A"


def medqa_gold_to_choice(sample: dict, default_choice: str = "A") -> str:
    answer_raw = _ensure_text(sample.get("answer", ""))
    answer_choice = extract_choice_answer(answer_raw, default=None)
    if answer_choice is not None:
        return answer_choice

    options = sample.get("options")
    if isinstance(options, list):
        norm_answer = _normalize_option_text(answer_raw)
        for opt in options:
            opt_str = _ensure_text(opt)
            label = _extract_choice_core(opt_str)
            if label is None:
                continue
            norm_opt = _normalize_option_text(opt_str)
            if norm_answer and (norm_answer == norm_opt or norm_answer in norm_opt or norm_opt in norm_answer):
                return label

    return extract_choice_answer(default_choice, default="A")


def _scan_balanced_braced_content(text: str, open_brace_idx: int) -> Tuple[str, bool]:
    depth = 0
    start = open_brace_idx + 1
    for idx in range(open_brace_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx], True
            if depth < 0:
                return text[start:idx], False
    return text[start:], False


def _boxed_answer_candidates(text: str) -> List[Tuple[str, bool]]:
    text = _ensure_text(text)
    candidates: List[Tuple[int, str, bool]] = []
    for match in re.finditer(r"\\(?:boxed|fbox)\{", text):
        open_brace_idx = match.end() - 1
        content, complete = _scan_balanced_braced_content(text, open_brace_idx)
        candidates.append((match.start(), content.strip(), complete))
    return [(content, complete) for _, content, complete in sorted(candidates, key=lambda item: item[0])]


def has_incomplete_boxed_answer(text: str) -> bool:
    candidates = _boxed_answer_candidates(text)
    return bool(candidates and not candidates[-1][1])


def extract_boxed_answer(text: str) -> Optional[str]:
    candidates = _boxed_answer_candidates(text)
    if not candidates:
        return None
    complete = [content for content, ok in candidates if ok and content.strip()]
    if complete:
        return complete[-1].strip()
    incomplete = [content for content, ok in candidates if not ok and content.strip()]
    return incomplete[-1].strip() if incomplete else None


def extract_pred_answer(text: str) -> Optional[str]:
    text = _ensure_text(text)
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    final_matches = re.findall(r"Final\s+Answer\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if final_matches:
        final_answer = final_matches[-1].strip()
        if final_answer:
            return final_answer

    fallback = text.strip()
    return fallback if fallback else None


def extract_gsm8k_gold_answer(text: str) -> str:
    text = _ensure_text(text)
    hash_match = re.search(r"####\s*(.+)$", text, flags=re.DOTALL)
    if hash_match:
        candidate = hash_match.group(1).strip()
        if candidate:
            return candidate
    return text.strip()


def extract_gold_answer(text: str, dataset_name: str) -> str:
    text = _ensure_text(text)
    if is_choice_dataset(dataset_name):
        choice = extract_choice_answer(text, default=None)
        return choice if choice is not None else "A"
    if _is_gsm8k_dataset(dataset_name):
        return extract_gsm8k_gold_answer(text)
    return text.strip()


def normalize_answer_string(text: str) -> str:
    text = _ensure_text(text)
    # Rule:
    # 1) Drop decimal point and everything on its right.
    # 2) Keep digits only.
    integer_part = text.split(".", 1)[0]
    digits = "".join(re.findall(r"\d", integer_part))
    if not digits:
        return ""
    normalized = digits.lstrip("0")
    return normalized if normalized else "0"


def normalize_raw_no_space(text: str) -> str:
    text = _ensure_text(text)
    return re.sub(r"\s+", "", text.strip())


def normalize_freeform_answer_string(text: str) -> str:
    text = _ensure_text(text).strip().lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_freeform_em_string(text: str) -> str:
    text = normalize_freeform_answer_string(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _replace_text_macros(text: str) -> str:
    return re.sub(r"\\text\{([^{}]*)\}", r"\1", text)


def normalize_latex_text_string(text: str) -> str:
    text = _ensure_text(text)
    normalized = text.strip()
    normalized = _replace_text_macros(normalized)
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\,", "").replace("\\;", "")
    normalized = normalized.replace("\\!", "").replace("\\:", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace("\"", "").replace("'", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _replace_simple_fractions(text: str) -> str:
    def repl(match: re.Match) -> str:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator == 0:
            return match.group(0)
        return str(numerator / denominator)

    return re.sub(r"\\frac\{\s*(-?\d+)\s*\}\{\s*(-?\d+)\s*\}", repl, text)


def normalize_int_from_first_number(text: str) -> str:
    text = _ensure_text(text)
    normalized = text.strip()
    normalized = _replace_text_macros(normalized)
    normalized = _replace_simple_fractions(normalized)
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("\\circ", "")
    normalized = normalized.replace("^\\circ", "")

    number_match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not number_match:
        return ""

    value_str = number_match.group(0)
    int_part = value_str.split(".", 1)[0]
    if int_part in {"", "-", "+"}:
        return ""
    try:
        return str(int(int_part))
    except (ValueError, OverflowError):
        return ""


def _strip_matching_outer(text: str, left: str, right: str) -> Optional[str]:
    text = _ensure_text(text).strip()
    if text.startswith(left) and text.endswith(right):
        return text[len(left) : len(text) - len(right)].strip()
    return None


def _strip_outer_braces_once(text: str) -> Optional[str]:
    text = _ensure_text(text).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    content, complete = _scan_balanced_braced_content(text, 0)
    if complete and len(content) + 2 == len(text):
        return content.strip()
    return None


def strip_harmless_math_wrappers(text: str) -> str:
    out = _ensure_text(text).strip()
    for _ in range(12):
        previous = out
        boxed = extract_boxed_answer(out)
        if boxed is not None and re.fullmatch(r"\\(?:boxed|fbox)\{.*\}", out, flags=re.DOTALL):
            out = boxed.strip()

        for left, right in (("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$"), ("$", "$")):
            inner = _strip_matching_outer(out, left, right)
            if inner is not None:
                out = inner
                break

        braced = _strip_outer_braces_once(out)
        if braced is not None:
            out = braced

        if out == previous:
            break
    return out.strip()


def _latex_text_repl(match: re.Match) -> str:
    return match.group(1).strip()


def normalize_latex_strict(text: str) -> str:
    text = strip_harmless_math_wrappers(text)
    text = re.sub(r"\\text\{([^{}]*)\}", _latex_text_repl, text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\!", "").replace("\\:", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("−", "-")
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _brace_error(text: str) -> Optional[str]:
    depth = 0
    for char in _ensure_text(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return "unmatched_closing_brace"
    if depth != 0:
        return "unbalanced_braces"
    return None


def _parse_required_group(text: str, start_idx: int) -> Tuple[Optional[str], int, Optional[str]]:
    idx = start_idx
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text) or text[idx] != "{":
        return None, idx, "missing_braces"
    content, complete = _scan_balanced_braced_content(text, idx)
    if not complete:
        return content, len(text), "missing_closing_brace"
    if not content.strip():
        return content, idx + len(content) + 2, "empty_group"
    return content, idx + len(content) + 2, None


def is_malformed_math_answer(text: str) -> Tuple[bool, str]:
    text = _ensure_text(text).strip()
    if not text:
        return True, "empty"

    brace_reason = _brace_error(text)
    if brace_reason:
        return True, brace_reason

    if text.count("\\left") != text.count("\\right"):
        return True, "unbalanced_left_right"

    for match in re.finditer(r"\\(?:dfrac|tfrac|frac)", text):
        _, next_idx, reason = _parse_required_group(text, match.end())
        if reason:
            return True, "frac_missing_numerator_or_denominator"
        _, _, reason = _parse_required_group(text, next_idx)
        if reason:
            return True, "frac_missing_numerator_or_denominator"

    for match in re.finditer(r"\\sqrt", text):
        idx = match.end()
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx < len(text) and text[idx] == "[":
            closing = text.find("]", idx + 1)
            if closing < 0:
                return True, "sqrt_missing_closing_bracket"
            idx = closing + 1
        _, _, reason = _parse_required_group(text, idx)
        if reason:
            return True, "sqrt_missing_radicand"

    begins = re.findall(r"\\begin\{([^{}]+)\}", text)
    ends = re.findall(r"\\end\{([^{}]+)\}", text)
    for env in set(begins + ends):
        if begins.count(env) != ends.count(env):
            return True, f"begin_without_end:{env}"

    stripped = text.rstrip()
    if re.search(r"\\[A-Za-z]+$", stripped):
        return True, "unfinished_latex_command"
    if re.search(r"\\(?:begin|end)\{[^{}]*$", stripped):
        return True, "unfinished_environment"

    return False, ""


def _contains_matrix_like(text: str) -> bool:
    compact = normalize_latex_strict(text)
    return bool(
        re.search(r"\\begin\{(?:p|b|v|V)?matrix\}", compact)
        or re.search(r"\\end\{(?:p|b|v|V)?matrix\}", compact)
        or "\\\\" in compact
    )


def _contains_long_prose(text: str) -> bool:
    cleaned = strip_harmless_math_wrappers(text)
    return bool(re.search(r"\b(the|answer|real|target|but|and|not)\b", cleaned, flags=re.IGNORECASE))


def is_plain_integer_like_answer(text: str) -> bool:
    s = normalize_latex_strict(text)
    if not s or _contains_long_prose(text):
        return False
    if any(token in s for token in ("\\frac", "/", "\\sqrt", "\\begin", "\\end", "^2", "sqrt")):
        return False
    s = s.replace(",", "")
    s = s.replace("^\\circ", "").replace("\\circ", "").replace("^circ", "").replace("°", "")
    return bool(re.fullmatch(r"[+-]?\d+(?:\.0+)?", s))


def normalize_plain_integer_answer(text: str) -> str:
    if not is_plain_integer_like_answer(text):
        return ""
    s = normalize_latex_strict(text).replace(",", "")
    s = s.replace("^\\circ", "").replace("\\circ", "").replace("^circ", "").replace("°", "")
    if "." in s:
        s = s.split(".", 1)[0]
    try:
        return str(int(s))
    except ValueError:
        return ""


def parse_rational_answer(text: str) -> Optional[Fraction]:
    if _contains_matrix_like(text) or _contains_long_prose(text):
        return None
    s = normalize_latex_strict(text).replace(",", "")
    if not s:
        return None
    if re.search(r"[A-Za-z]", s.replace("\\frac", "")):
        return None
    if "\\sqrt" in s or "sqrt" in s:
        return None

    frac_match = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", s)
    if frac_match:
        sign = -1 if frac_match.group(1) == "-" else 1
        numerator = int(frac_match.group(2))
        denominator = int(frac_match.group(3))
        if denominator == 0:
            return None
        return Fraction(sign * numerator, denominator)

    slash_match = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", s)
    if slash_match:
        denominator = int(slash_match.group(2))
        if denominator == 0:
            return None
        return Fraction(int(slash_match.group(1)), denominator)

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return Fraction(s)

    return None


def _latex_to_sympyish(text: str) -> Optional[str]:
    if _contains_matrix_like(text) or _contains_long_prose(text):
        return None
    s = normalize_latex_strict(text)
    if not s:
        return None
    if re.search(r"[A-Za-z]", re.sub(r"\\(?:frac|sqrt|pi)", "", s)):
        return None

    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi")
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^([+-]?\d+)", r"**\1", s)
    s = re.sub(r"(\d)(sqrt|pi|\()", r"\1*\2", s)
    s = re.sub(r"(\))(\d|sqrt|pi|\()", r"\1*\2", s)
    if not re.fullmatch(r"[0-9+\-*/().,sqrtpi* ]+", s):
        return None
    return s


def _sympy_scalar_equal(gold: str, pred: str) -> Optional[bool]:
    gold_expr = _latex_to_sympyish(gold)
    pred_expr = _latex_to_sympyish(pred)
    if not gold_expr or not pred_expr:
        return None
    try:
        import sympy as sp  # type: ignore
    except Exception:
        return None
    try:
        lhs = sp.sympify(gold_expr)
        rhs = sp.sympify(pred_expr)
        return bool(sp.simplify(lhs - rhs) == 0)
    except Exception:
        return None


def _scalar_answers_equal(gold: str, pred: str) -> bool:
    if normalize_latex_strict(gold) == normalize_latex_strict(pred):
        return True
    gold_rat = parse_rational_answer(gold)
    pred_rat = parse_rational_answer(pred)
    if gold_rat is not None and pred_rat is not None:
        return gold_rat == pred_rat
    sympy_equal = _sympy_scalar_equal(gold, pred)
    if sympy_equal is not None:
        return sympy_equal
    if is_plain_integer_like_answer(gold) and is_plain_integer_like_answer(pred):
        return normalize_plain_integer_answer(gold) == normalize_plain_integer_answer(pred)
    return False


def _parse_matrix_entries(text: str) -> Optional[List[List[str]]]:
    s = strip_harmless_math_wrappers(text)
    env_match = re.search(
        r"\\begin\{(?P<env>(?:p|b|v|V)?matrix)\}(?P<body>.*)\\end\{(?P=env)\}",
        s,
        flags=re.DOTALL,
    )
    if env_match:
        body = env_match.group("body")
    else:
        body = s.strip()
        if body.startswith("[") and body.endswith("]"):
            body = body[1:-1]
        elif body.startswith("(") and body.endswith(")"):
            body = body[1:-1]
        else:
            return None

    rows = [row.strip() for row in re.split(r"\\\\", body) if row.strip()]
    if not rows:
        return None
    parsed_rows: List[List[str]] = []
    for row in rows:
        entries = [entry.strip() for entry in re.split(r"&|,", row) if entry.strip()]
        if not entries:
            return None
        parsed_rows.append(entries)
    return parsed_rows


def _matrix_answers_equal(gold: str, pred: str) -> bool:
    if normalize_latex_strict(gold) == normalize_latex_strict(pred):
        return True
    gold_rows = _parse_matrix_entries(gold)
    pred_rows = _parse_matrix_entries(pred)
    if gold_rows is None or pred_rows is None:
        return False
    if len(gold_rows) != len(pred_rows):
        return False
    for gold_row, pred_row in zip(gold_rows, pred_rows):
        if len(gold_row) != len(pred_row):
            return False
        for gold_entry, pred_entry in zip(gold_row, pred_row):
            if not _scalar_answers_equal(gold_entry, pred_entry):
                return False
    return True


def _normalize_short_text_answer(text: str) -> str:
    s = strip_harmless_math_wrappers(text)
    s = re.sub(r"\\text\{([^{}]*)\}", _latex_text_repl, s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    if not re.fullmatch(r"[a-z ]{1,32}", s):
        return ""
    if len(s.split()) > 3:
        return ""
    return s


def _math500_detail(
    gold_answer: str,
    pred_answer: Optional[str],
    correct: bool,
    gold_norm: str,
    pred_norm: str,
    judge_method: str,
    answer_invalid: bool = False,
    invalid_reason: str = "",
) -> Dict[str, Any]:
    return {
        "gold_answer_parsed": gold_answer,
        "pred_answer_parsed": pred_answer,
        "correct": bool(correct),
        "gold_norm": gold_norm,
        "pred_norm": pred_norm,
        "judge_method": judge_method,
        "answer_invalid": bool(answer_invalid),
        "invalid_reason": invalid_reason,
        "checker_version": MATH500_CHECKER_VERSION,
    }


def compare_math500_answers_detailed(gold_text: str, pred_text: str) -> Dict[str, Any]:
    gold_answer = extract_gold_answer(gold_text, "math500")
    pred_answer = extract_pred_answer(pred_text)
    if pred_answer is None:
        return _math500_detail(
            gold_answer,
            None,
            False,
            f"invalid_check:{normalize_latex_strict(gold_answer)}",
            "invalid:empty",
            "invalid",
            answer_invalid=True,
            invalid_reason="empty",
        )

    if has_incomplete_boxed_answer(pred_text):
        return _math500_detail(
            gold_answer,
            pred_answer,
            False,
            f"invalid_check:{normalize_latex_strict(gold_answer)}",
            "invalid:unbalanced_braces",
            "invalid",
            answer_invalid=True,
            invalid_reason="unbalanced_braces",
        )

    pred_invalid, pred_reason = is_malformed_math_answer(pred_answer)
    if pred_invalid:
        return _math500_detail(
            gold_answer,
            pred_answer,
            False,
            f"invalid_check:{normalize_latex_strict(gold_answer)}",
            f"invalid:{pred_reason}",
            "invalid",
            answer_invalid=True,
            invalid_reason=pred_reason,
        )

    gold_norm = normalize_latex_strict(gold_answer)
    pred_norm = normalize_latex_strict(pred_answer)
    if gold_norm and pred_norm and gold_norm == pred_norm:
        return _math500_detail(gold_answer, pred_answer, True, f"latex_exact:{gold_norm}", f"latex_exact:{pred_norm}", "latex_exact")

    gold_rat = parse_rational_answer(gold_answer)
    pred_rat = parse_rational_answer(pred_answer)
    if gold_rat is not None and pred_rat is not None:
        method = "rational"
        return _math500_detail(
            gold_answer,
            pred_answer,
            gold_rat == pred_rat,
            f"{method}:{gold_rat}",
            f"{method}:{pred_rat}",
            method,
        )

    sympy_equal = _sympy_scalar_equal(gold_answer, pred_answer)
    if sympy_equal is not None:
        return _math500_detail(
            gold_answer,
            pred_answer,
            bool(sympy_equal),
            f"sympy:{gold_norm}",
            f"sympy:{pred_norm}",
            "sympy",
        )

    if _contains_matrix_like(gold_answer) or _contains_matrix_like(pred_answer):
        matrix_equal = _matrix_answers_equal(gold_answer, pred_answer)
        return _math500_detail(
            gold_answer,
            pred_answer,
            matrix_equal,
            f"matrix:{gold_norm}",
            f"matrix:{pred_norm}",
            "matrix",
        )

    gold_text_norm = _normalize_short_text_answer(gold_answer)
    pred_text_norm = _normalize_short_text_answer(pred_answer)
    if gold_text_norm and pred_text_norm:
        return _math500_detail(
            gold_answer,
            pred_answer,
            gold_text_norm == pred_text_norm,
            f"text:{gold_text_norm}",
            f"text:{pred_text_norm}",
            "text",
        )

    if is_plain_integer_like_answer(gold_answer) and is_plain_integer_like_answer(pred_answer):
        gold_int = normalize_plain_integer_answer(gold_answer)
        pred_int = normalize_plain_integer_answer(pred_answer)
        return _math500_detail(
            gold_answer,
            pred_answer,
            gold_int == pred_int,
            f"integer:{gold_int}",
            f"integer:{pred_int}",
            "integer",
        )

    return _math500_detail(
        gold_answer,
        pred_answer,
        False,
        f"strict:{gold_norm}" if gold_norm else "",
        f"strict:{pred_norm}" if pred_norm else "",
        "no_match",
    )


def compare_math500_answers(gold_text: str, pred_text: str) -> Tuple[str, Optional[str], bool, str, str]:
    detail = compare_math500_answers_detailed(gold_text, pred_text)
    return (
        detail["gold_answer_parsed"],
        detail["pred_answer_parsed"],
        bool(detail["correct"]),
        detail["gold_norm"],
        detail["pred_norm"],
    )


def normalize_date_string(text: str) -> str:
    text = normalize_freeform_answer_string(text)
    if not text:
        return ""

    text = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text)
    looks_datey = bool(
        re.search(r"\b\d{4}\b", text)
        or re.search(
            r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\b",
            text,
        )
        or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
    )
    if not looks_datey:
        return ""

    try:
        from dateutil import parser as date_parser  # type: ignore
    except Exception:
        return ""

    try:
        parsed = date_parser.parse(text, fuzzy=False, default=None)
    except Exception:
        return ""

    has_year = bool(re.search(r"\b\d{4}\b", text))
    has_month = bool(
        re.search(
            r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\b",
            text,
        )
        or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
    )
    has_day = bool(
        re.search(r"\b\d{1,2}(st|nd|rd|th)?\b", text)
        and has_month
    )

    if has_year and has_month and has_day:
        return parsed.strftime("%Y-%m-%d")
    if has_year and has_month:
        return parsed.strftime("%Y-%m")
    if has_year:
        return parsed.strftime("%Y")
    return ""


def compare_answers(
    gold_text: str,
    pred_text: str,
    dataset_name: str = "openai/gsm8k",
) -> Tuple[str, Optional[str], bool, str, str]:
    detail = compare_answers_detailed(gold_text, pred_text, dataset_name=dataset_name)
    return (
        detail["gold_answer_parsed"],
        detail["pred_answer_parsed"],
        bool(detail["correct"]),
        detail["gold_norm"],
        detail["pred_norm"],
    )


def compare_answers_detailed(
    gold_text: str,
    pred_text: str,
    dataset_name: str = "openai/gsm8k",
) -> Dict[str, Any]:
    gold_answer = extract_gold_answer(gold_text, dataset_name)

    if is_choice_dataset(dataset_name):
        # Per request, default to choice A if parsing fails, then override when parsed.
        pred_answer = extract_choice_answer(pred_text, default="A")
        gold_choice = extract_choice_answer(gold_answer, default="A")
        pred_choice = extract_choice_answer(pred_answer, default="A")
        correct = bool(gold_choice == pred_choice)
        return {
            "gold_answer_parsed": gold_choice,
            "pred_answer_parsed": pred_choice,
            "correct": correct,
            "gold_norm": f"choice:{gold_choice.lower()}",
            "pred_norm": f"choice:{pred_choice.lower()}",
            "judge_method": "choice",
            "answer_invalid": False,
            "invalid_reason": "",
            "checker_version": "choice_v1",
        }

    if _is_math500_dataset(dataset_name):
        return compare_math500_answers_detailed(gold_text, pred_text)

    pred_answer = extract_pred_answer(pred_text)
    if pred_answer is None:
        return {
            "gold_answer_parsed": gold_answer,
            "pred_answer_parsed": None,
            "correct": False,
            "gold_norm": "",
            "pred_norm": "",
            "judge_method": "not_found",
            "answer_invalid": True,
            "invalid_reason": "empty",
            "checker_version": "freeform_v1",
        }

    strategies = [
        ("date", normalize_date_string),
        ("em", normalize_freeform_em_string),
        ("lower", normalize_freeform_answer_string),
    ]

    fallback_gold = ""
    fallback_pred = ""
    for strategy_name, strategy_fn in strategies:
        gold_norm = strategy_fn(gold_answer)
        pred_norm = strategy_fn(pred_answer)
        if gold_norm and pred_norm and not fallback_gold:
            fallback_gold = f"{strategy_name}:{gold_norm}"
            fallback_pred = f"{strategy_name}:{pred_norm}"
        if gold_norm and pred_norm and gold_norm == pred_norm:
            return {
                "gold_answer_parsed": gold_answer,
                "pred_answer_parsed": pred_answer,
                "correct": True,
                "gold_norm": f"{strategy_name}:{gold_norm}",
                "pred_norm": f"{strategy_name}:{pred_norm}",
                "judge_method": strategy_name,
                "answer_invalid": False,
                "invalid_reason": "",
                "checker_version": "freeform_v1",
            }

    return {
        "gold_answer_parsed": gold_answer,
        "pred_answer_parsed": pred_answer,
        "correct": False,
        "gold_norm": fallback_gold,
        "pred_norm": fallback_pred,
        "judge_method": "no_match",
        "answer_invalid": False,
        "invalid_reason": "",
        "checker_version": "freeform_v1",
    }



def format_latent_info(latent: torch.Tensor) -> str:
    steps = int(latent.size(0)) if latent.ndim >= 1 else 0
    hidden = int(latent.size(1)) if latent.ndim >= 2 else 0
    dtype = str(latent.dtype).replace("torch.", "")
    return f"<latent_embedding steps={steps} hidden={hidden} dtype={dtype}>"
