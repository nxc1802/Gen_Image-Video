"""
services/music/yue2/score.py
ABC notation parser, validator, chord extractor, and score diffing/editing engine for YuE2.
Supports tasks: symbolic_score (YUE-02, YUE-03, YUE-06), custom_score (YUE-05),
plan_reuse (YUE-10), score_edit (YUE-15), agentic_editing (YUE-16).
"""
import difflib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from factory_contracts.errors import InvalidRequestError

# Regex patterns for ABC headers
HEADER_LINE_REGEX = re.compile(r"^([A-Za-z]):\s*(.*)$")
METER_REGEX = re.compile(r"^(?:(?:[1-9]|1[0-6])/(?:[1248]|16)|C|C\||none)$", re.IGNORECASE)
KEY_REGEX = re.compile(
    r"^[A-G][b#]?(?:[ \t]*(?:m|min|maj|dor|mix|lyd|phr|loc|major|minor))?(?:\s+(?:clef=[a-z0-9]+|bass|treble|alto))?$",
    re.IGNORECASE,
)
TEMPO_REGEX = re.compile(
    r"^(?:(?:\d+/\d+=)?[1-9]\d*(?:\.\d+)?|\"[^\"]*\"\s*[1-9]\d*(?:\.\d+)?|[1-9]\d*)$",
    re.IGNORECASE,
)
NOTE_LENGTH_REGEX = re.compile(r"^1/(?:[1248]|16|32|64)$")
CHORD_REGEX = re.compile(
    r'"([A-G][b#]?(?:m|maj|min|dim|aug|sus[24]?|add|[0-9#b+-\^])*(?:/[A-G][b#]?)?)"'
)
NOTE_OR_REST_REGEX = re.compile(r"[A-Ga-gzx]")


def extract_headers(abc_text: str) -> Dict[str, str]:
    """Extracts all header fields (X, T, M, L, Q, K, etc.) from an ABC score."""
    headers: Dict[str, str] = {}
    for line in abc_text.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("%"):
            continue
        match = HEADER_LINE_REGEX.match(trimmed)
        if match:
            key, val = match.groups()
            headers[key.upper()] = val.strip()
        elif ":" in trimmed and len(trimmed.split(":")[0]) == 1:
            parts = trimmed.split(":", 1)
            headers[parts[0].upper()] = parts[1].strip()
    return headers


def extract_chords(abc_text: str) -> List[str]:
    """Extracts guitar/harmony chord symbols defined in quotes (e.g., 'C', 'Am', 'G7')."""
    chords: List[str] = []
    for line in abc_text.splitlines():
        line_code = line.split("%")[0].strip()
        if not line_code or HEADER_LINE_REGEX.match(line_code):
            continue
        matches = CHORD_REGEX.findall(line_code)
        chords.extend(matches)
    return chords


def _check_balanced_delimiters(text: str) -> None:
    """
    Validates balanced delimiters in the music body.
    Checks brackets [], quotes "", braces {}, and parentheses () for slurs.
    """
    lines = text.splitlines()
    for line_num, line in enumerate(lines, start=1):
        line_code = line.split("%")[0].strip()
        if not line_code or HEADER_LINE_REGEX.match(line_code):
            continue

        # Check quotes balance per line
        quote_count = line_code.count('"')
        if quote_count % 2 != 0:
            raise InvalidRequestError(
                f"Malformed ABC score: unclosed quotes on line {line_num}: {line.strip()}"
            )

    # Global bracket and brace balance in non-header lines
    bracket_stack: List[Tuple[str, int]] = []
    in_quote = False

    for line_num, line in enumerate(lines, start=1):
        line_code = line.split("%")[0].strip()
        if not line_code or HEADER_LINE_REGEX.match(line_code):
            continue

        i = 0
        while i < len(line_code):
            ch = line_code[i]
            if ch == '"':
                in_quote = not in_quote
                i += 1
                continue

            if in_quote:
                i += 1
                continue

            # Check bar delimiters like [| and |] which are valid ABC bar syntax
            if ch == "[" and i + 1 < len(line_code) and line_code[i + 1] in ("|", ":", "]"):
                # Barline like [| or [: or empty chord []
                if line_code[i + 1] == "]":
                    # empty chord []
                    i += 2
                    continue
                i += 2
                continue
            if ch == "|" and i + 1 < len(line_code) and line_code[i + 1] == "]":
                i += 2
                continue

            if ch == "[":
                bracket_stack.append(("[", line_num))
            elif ch == "]":
                if not bracket_stack or bracket_stack[-1][0] != "[":
                    raise InvalidRequestError(
                        f"Malformed ABC score: unmatched closing bracket ']' on line {line_num}"
                    )
                bracket_stack.pop()
            elif ch == "{":
                bracket_stack.append(("{", line_num))
            elif ch == "}":
                if not bracket_stack or bracket_stack[-1][0] != "{":
                    raise InvalidRequestError(
                        f"Malformed ABC score: unmatched closing brace '}}' on line {line_num}"
                    )
                bracket_stack.pop()
            elif ch == "(":
                # In ABC, (2, (3, (4 etc are tuplets and do not need closing parens.
                # Slurs like (c d e) do need closing parens.
                if i + 1 < len(line_code) and line_code[i + 1].isdigit():
                    # tuplet, skip
                    pass
                else:
                    bracket_stack.append(("(", line_num))
            elif ch == ")":
                if not bracket_stack or bracket_stack[-1][0] != "(":
                    raise InvalidRequestError(
                        f"Malformed ABC score: unmatched closing paren ')' on line {line_num}"
                    )
                bracket_stack.pop()

            i += 1

    if bracket_stack:
        delimiter, line_no = bracket_stack[-1]
        raise InvalidRequestError(
            f"Malformed ABC score: unclosed delimiter '{delimiter}' opened near line {line_no}"
        )


def validate_abc_score(abc_text: str) -> Dict[str, Any]:
    """
    Validates ABC notation score structure and headers.
    Enforces required headers, valid meter (M:), key (K:), tempo (Q:), note length (L:),
    and balanced delimiters.
    Returns parsed score structure or raises InvalidRequestError.
    """
    if not isinstance(abc_text, str) or not abc_text.strip():
        raise InvalidRequestError("ABC score cannot be empty or non-string")

    clean_text = abc_text.strip()
    headers = extract_headers(clean_text)

    # K: (Key) is mandatory for any valid ABC tune
    if "K" not in headers or not headers["K"]:
        raise InvalidRequestError("ABC score must include a valid Key header (K:)")

    # Validate Key
    key_val = headers["K"]
    if not KEY_REGEX.match(key_val):
        raise InvalidRequestError(f"Invalid key header format: K:{key_val}")

    # Validate Meter if present
    if "M" in headers and headers["M"]:
        meter_val = headers["M"]
        if not METER_REGEX.match(meter_val):
            raise InvalidRequestError(f"Invalid meter header format: M:{meter_val}")

    # Validate Tempo if present
    if "Q" in headers and headers["Q"]:
        tempo_val = headers["Q"]
        if not TEMPO_REGEX.match(tempo_val):
            raise InvalidRequestError(f"Invalid tempo header format: Q:{tempo_val}")
        tempo_part = tempo_val.split("=")[-1] if "=" in tempo_val else tempo_val
        num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", tempo_part)
        if num_match:
            try:
                bpm = float(num_match.group(1))
                if bpm <= 0.0:
                    raise InvalidRequestError("Tempo must be strictly greater than 0 BPM")
            except ValueError:
                pass

    # Validate Note Length if present
    if "L" in headers and headers["L"]:
        len_val = headers["L"]
        if not NOTE_LENGTH_REGEX.match(len_val):
            raise InvalidRequestError(f"Invalid unit note length header format: L:{len_val}")

    # Check balanced delimiters
    _check_balanced_delimiters(clean_text)

    # Check that music body contains musical notes or rests
    body_lines: List[str] = []
    voices: Set[str] = set()
    measure_count = 0

    for line in clean_text.splitlines():
        line_code = line.split("%")[0].strip()
        if not line_code:
            continue
        if HEADER_LINE_REGEX.match(line_code):
            match = HEADER_LINE_REGEX.match(line_code)
            if match and match.group(1).upper() == "V":
                voices.add(match.group(2).strip())
            continue
        body_lines.append(line_code)
        measure_count += line_code.count("|")

    body_content = "\n".join(body_lines)
    if not NOTE_OR_REST_REGEX.search(body_content):
        raise InvalidRequestError("ABC score must contain musical notes or rests in tune body")

    chords = extract_chords(clean_text)

    return {
        "valid": True,
        "headers": headers,
        "chords": chords,
        "voices": sorted(list(voices)),
        "measures_count": measure_count,
        "body_lines": len(body_lines),
    }


def diff_scores(score_a: str, score_b: str) -> Dict[str, Any]:
    """
    Diffs two ABC scores and returns header changes, chord changes, and unified line diff.
    Both scores are validated prior to diffing.
    """
    val_a = validate_abc_score(score_a)
    val_b = validate_abc_score(score_b)

    headers_a = val_a["headers"]
    headers_b = val_b["headers"]

    header_changes: Dict[str, Dict[str, Optional[str]]] = {}
    all_keys = set(headers_a.keys()) | set(headers_b.keys())
    for k in sorted(all_keys):
        va = headers_a.get(k)
        vb = headers_b.get(k)
        if va != vb:
            header_changes[k] = {"from": va, "to": vb}

    chords_a = set(val_a["chords"])
    chords_b = set(val_b["chords"])

    added_chords = sorted(list(chords_b - chords_a))
    removed_chords = sorted(list(chords_a - chords_b))
    preserved_chords = sorted(list(chords_a & chords_b))

    lines_a = [line.strip() for line in score_a.splitlines() if line.strip()]
    lines_b = [line.strip() for line in score_b.splitlines() if line.strip()]
    line_diff = list(difflib.unified_diff(lines_a, lines_b, fromfile="base.abc", tofile="edited.abc", lineterm=""))

    return {
        "header_changes": header_changes,
        "chord_diff": {
            "added": added_chords,
            "removed": removed_chords,
            "preserved": preserved_chords,
        },
        "measures_delta": val_b["measures_count"] - val_a["measures_count"],
        "line_diff": line_diff,
    }


def edit_score(base_score: str, modifications: Dict[str, Any]) -> str:
    """
    Applies modifications (tempo, key, meter, title, chords, body) to an ABC score.
    Returns the modified ABC score and verifies validity.
    """
    validate_abc_score(base_score)

    lines = base_score.splitlines()
    new_lines: List[str] = []
    headers_updated: Set[str] = set()

    # Map modifications to header updates
    header_updates: Dict[str, str] = {}
    if "headers" in modifications and isinstance(modifications["headers"], dict):
        for k, v in modifications["headers"].items():
            header_updates[k.upper()] = str(v).strip()

    if "tempo" in modifications:
        header_updates["Q"] = str(modifications["tempo"]).strip()
    if "key" in modifications:
        header_updates["K"] = str(modifications["key"]).strip()
    if "meter" in modifications:
        header_updates["M"] = str(modifications["meter"]).strip()
    if "title" in modifications:
        header_updates["T"] = str(modifications["title"]).strip()
    if "unit_length" in modifications:
        header_updates["L"] = str(modifications["unit_length"]).strip()

    replace_chords: Dict[str, str] = modifications.get("replace_chords", {})
    body_replacement: Optional[str] = modifications.get("body")
    append_body: Optional[str] = modifications.get("append_body")

    k_line_index = -1

    for idx, line in enumerate(lines):
        match = HEADER_LINE_REGEX.match(line.strip())
        if match:
            h_name = match.group(1).upper()
            if h_name == "K":
                k_line_index = len(new_lines)

            if h_name in header_updates:
                new_val = header_updates[h_name]
                new_lines.append(f"{h_name}:{new_val}")
                headers_updated.add(h_name)
            else:
                new_lines.append(line)
        else:
            # Body line
            if body_replacement is not None:
                continue  # Skip existing body lines if replacing body
            modified_line = line
            for old_chord, new_chord in replace_chords.items():
                pattern = rf'"{re.escape(old_chord)}"'
                replacement = f'"{new_chord}"'
                modified_line = re.sub(pattern, replacement, modified_line)
            new_lines.append(modified_line)

    # Insert any new headers not yet added before K: or at the end
    remaining_headers = set(header_updates.keys()) - headers_updated
    if remaining_headers:
        insert_idx = k_line_index if k_line_index != -1 else 0
        new_header_lines = [f"{k}:{header_updates[k]}" for k in sorted(remaining_headers) if k != "K"]
        new_lines[insert_idx:insert_idx] = new_header_lines

    if body_replacement is not None:
        new_lines.append(body_replacement.strip())

    if append_body:
        new_lines.append(append_body.strip())

    result_score = "\n".join(new_lines)
    # Ensure edited score is valid
    validate_abc_score(result_score)
    return result_score


def generate_sample_abc(
    title: str = "YuE2 Sample Melody",
    key: str = "C",
    meter: str = "4/4",
    tempo: int = 120,
    unit_length: str = "1/8",
) -> str:
    """Generates a valid standard ABC melody score for testing and defaults."""
    return f"""X:1
T:{title}
M:{meter}
L:{unit_length}
Q:{tempo}
K:{key}
"C" C2 E2 G2 c2 | "G" B2 G2 D2 B,2 | "Am" A2 c2 e2 a2 | "F" F2 A2 c2 f2 |]"""
