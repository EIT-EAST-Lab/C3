import re
from functools import partial
from typing import Optional
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"

def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer

# 其余内容保持原逻辑，仅去掉重复的 import 与零碎注释噪音
# --- 下面内容与原实现一致（为节省版面不改动核心逻辑） ---
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a); b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".": string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5": string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string

def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,))
    )

def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac").replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt").replace("π", "pi").replace("∞", "inf").replace("∪", "U").replace("·", "*").replace("×", "*")
    return expr.strip()

def _is_float(num: str) -> bool:
    try: float(num); return True
    except ValueError: return False

def _is_int(x: float) -> bool:
    try: return abs(x - int(round(x))) <= 1e-7
    except: return False

def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))

def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False

def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    return int(float(x))

def _inject_implicit_mixed_number(step: str):
    return re.compile("([0-9]) +([0-9])").sub("\\1+\\2", step)

def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr: break
        expr = next_expr
    return next_expr

def _normalize(expr: str) -> str:
    if expr is None: return None
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None: expr = m.group("text")
    expr = (expr.replace("\\%", "%").replace("\\$", "$")
                 .replace("$", "").replace("%", "")
                 .replace(" or ", " , ").replace(" and ", " , "))
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in ["degree","cm","centimeter","meter","mile","second","minute","hour","day","week","month","year","foot","feet","inch","yard"]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}": expr = expr[1:-1]
    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)): expr = str(int(round(float(expr))))
    if "\\" in expr:
        try: expr = _parse_latex(expr)
        except: pass
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr).replace(" ", "")
    expr = expr.replace("{", "").replace("}", "").lower()
    if _str_is_int(expr): expr = str(_str_to_int(expr))
    return expr

def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "").replace("frac", "")
    return len({x for x in expr if x.isalpha()})

def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2: return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr: return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None: return False
    return True

def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            simplified = sympy.simplify(_sympy_parse(expr))
            if abs(simplified) <= 0.04 * sympy.simplify(ground_truth_normalized):
                return True
    except:
        pass
    return False

def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0: return []
    if (len(expr) > 2 and expr[0] in TUPLE_CHARS and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems

def grade_answer(given_answer: str, ground_truth: str) -> bool:
    if given_answer is None: return False

    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized is None: return False
    if ground_truth_normalized == given_normalized: return True
    if len(given_normalized) == 0: return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        is_correct = True
        for gt_elem, gv_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(gt_elem) and _is_frac(gv_elem):
                is_correct = gt_elem == gv_elem
            elif _is_float(gt_elem) and _is_float(gv_elem):
                if _str_is_int(gt_elem):
                    try: is_correct = round(float(gv_elem)) == int(gt_elem)
                    except: is_correct = False
                else:
                    gt_elem_f = float(gt_elem); gv_elem_f = float(gv_elem)
                    eps = abs(gt_elem_f) * 0.04
                    is_correct = (gt_elem_f - eps) <= gv_elem_f <= (gt_elem_f + eps)
            else:
                is_correct = are_equal_under_sympy(gt_elem, gv_elem)
            if not is_correct: break

    return is_correct

def extract_answer(prediction, pattern=r"The final answer is \$([^$]*)\$"):
    match = re.findall(pattern, prediction)
    if match:
        answer = match[0]
    else:
        pattern2 = r"\$([^$]*)\$"
        match2 = re.findall(pattern2, prediction)
        if match2:
            answer = match2[0]
        else:
            if "answer is" in prediction:
                answer = prediction.split("answer is")[-1].strip()
                if len(answer) > 1 and answer[-1] == ".": answer = answer[:-1].strip()
                if len(answer) > 1 and answer[0] == ":": answer = answer[1:].strip()
            else:
                match3 = re.findall(r"-?[0-9]+\.?[0-9]*", prediction)
                answer = match3[-1] if match3 else ""
    return answer

def exact_match_score(prediction, ground_truth):
    prediction = extract_answer(prediction)
    return grade_answer(prediction, ground_truth)
