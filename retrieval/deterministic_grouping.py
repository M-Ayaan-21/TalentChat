"""
Deterministic grouping & advanced candidate name extraction (v2.2)
Includes:
- Line segmentation around email/phone anchors
- Anchor pass extracting inline names (after/before anchors)
- Multi-pass fallback (pure line / role-stripping / ALL CAPS / email-combo)
- File name hint override for generic picks
- Optional spaCy fallback (if installed) when low confidence

Public API:
    group_chunks(chunks, references)
    extract_candidate_name(chunks) -> (name, confidence, source, debug_dict)
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple, Any, Optional, Set

# ---------------- Configuration ----------------
TOP_SCORE_CHUNK_COUNT = 10
MAX_CHAR_SCAN = 18000
MAX_LINES_INSPECTED = 140   # increased
MIN_NAME_TOKENS = 2
MAX_NAME_TOKENS = 5

ROLE_TRAILING_CUTOFF_WORDS = {
    "developer","engineer","architect","specialist","lead","manager","consultant",
    "designer","programmer","intern","student","trainee","analyst","administrator","admin"
}

TECH_OR_HEADING = {
    "objective","summary","professional","experience","experiences","skills","skill","technology",
    "technologies","tech","stack","education","profile","career","certification","certifications",
    "contact","contacts","phone","email","e-mail","github","linkedin","project","projects",
    "achievements","strengths","core","highlights","software","engineer","engineering","developer",
    "development","programmer","architect","android","kotlin","java","python","javascript","react",
    "node","flutter","spring","django","sql","mysql","postgres","nosql","mongodb","aws","gcp","azure",
    "cloud","microservices","kafka","system","systems","ide","jira","git","gitlab","bitbucket",
    "docker","kubernetes","linux","windows","macos","ci","cd","cicd","testing","qa","automation",
    "resume","curriculum","vitae","cv","mobile","application","applications","app","apps","fullstack",
    "full-stack","devops","security","ai","ml","data","science","student","analyst"
}

FORBIDDEN_SUBSTRINGS = {"http","www.","@", "/", "\\", "|", "api"}

NAME_TOKEN_REGEX = re.compile(r"^[A-Z][a-z][a-zA-Z'\-]{0,30}$")
ALL_CAPS_LINE = re.compile(r"^[A-Z][A-Z \-']{2,}$")
ALL_CAPS_TOKEN = re.compile(r"^[A-Z]{2,}$")
INITIAL_TOKEN = re.compile(r"^[A-Z]\.?$")

EMAIL_REGEX = re.compile(r"[A-Za-z0-9.\-_+]+@[A-Za-z0-9\.\-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"\+?\d[\d \-\(\)]{6,}\d")

EMAIL_TOKEN_SPLIT = r"[._\-+]"
COMMON_PREFIX_DISCARD = {"mr","mrs","ms","miss","dr","prof","sir","madam"}

# Optional spaCy
try:
    import spacy
    _SPACY_NLP = None
    def get_spacy():
        global _SPACY_NLP
        if _SPACY_NLP is None:
            try:
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except Exception:
                _SPACY_NLP = False
        return _SPACY_NLP
except ImportError:
    def get_spacy():
        return False

# ---------------- Grouping ----------------
def group_chunks(chunks: List[Dict], references: List[str]) -> Dict[str, Dict]:
    groups: Dict[str, Dict] = {}
    for i, ch in enumerate(chunks):
        gid = str(ch.get("group_id") or f"chunk::{i}")
        g = groups.setdefault(gid, {"chunks": [], "reference": None, "file_stems": set(), "ref_stem": None})
        g["chunks"].append(ch)
        fn = ch.get("file_name") or ch.get("source")
        if isinstance(fn, str):
            stem = _stem(fn)
            if stem:
                g["file_stems"].add(stem)
    stem_to_ref = {}
    for ref in references or []:
        st = _stem(ref)
        if st:
            stem_to_ref[st] = ref
    for g in groups.values():
        for st in g["file_stems"]:
            if st in stem_to_ref:
                g["reference"] = stem_to_ref[st]
                g["ref_stem"] = st
                break
    return groups

def _stem(path: str) -> str:
    if not path:
        return ""
    name = path.rsplit("/", 1)[-1]
    name = re.sub(r"\.[a-zA-Z0-9]+$", "", name)
    return re.sub(r"[^a-zA-Z0-9]+", "", name).lower()

# ---------------- Line Segmentation ----------------
def _segment_lines(raw_text: str) -> List[str]:
    """
    Produce finer-grained 'lines' by:
      - Splitting original lines
      - Further splitting around email + phone anchors into left/anchor/right subsegments.
    Ensures names appended to emails / phones become isolated candidates.
    """
    segments: List[str] = []
    for raw_line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        # Hard split on large separators to reduce noise
        base_parts = re.split(r"\s{2,}| \| ", line)
        for part in base_parts:
            p = part.strip()
            if not p:
                continue
            # Find anchors inside part
            anchors = []
            for m in EMAIL_REGEX.finditer(p):
                anchors.append(("email", m.start(), m.end()))
            for m in PHONE_REGEX.finditer(p):
                anchors.append(("phone", m.start(), m.end()))
            if not anchors:
                segments.append(p)
                continue

            pos = 0
            for typ, a_s, a_e in sorted(anchors, key=lambda x: x[1]):
                before = p[pos:a_s].strip()
                anchor_txt = p[a_s:a_e].strip()
                pos = a_e
                if before:
                    segments.append(before)
                segments.append(anchor_txt)
            tail = p[pos:].strip()
            if tail:
                segments.append(tail)
    # De-duplicate consecutive identical lines to shrink noise
    collapsed: List[str] = []
    prev = None
    for s in segments:
        if s != prev:
            collapsed.append(s)
        prev = s
        if len(collapsed) >= MAX_LINES_INSPECTED:
            break
    return collapsed

# ---------------- Anchor Pass ----------------
ANCHOR_ROLE_STOP = ROLE_TRAILING_CUTOFF_WORDS
ANCHOR_NAME_TOKEN = re.compile(r"[A-Z][A-Za-z'\-]{1,30}|[A-Z]{2,}")

def _anchor_pass(lines: List[str]) -> Dict[str, Any]:
    hits = []
    best = None
    best_score = 0.0
    for idx, line in enumerate(lines):
        # If line is purely anchor (email or phone) skip; but we still look at neighbor lines later
        has_email = bool(EMAIL_REGEX.search(line))
        has_phone = bool(PHONE_REGEX.search(line))
        if not (has_email or has_phone):
            # Look for composite: tokens chain with typical pattern (NAME tokens ended by role)
            continue

        # After anchor case handled by segmentation; treat next line also
        # Try to build a candidate from the next immediate line if current is anchor-only.
        # (Handled outside: we only parse inline after anchor tokens here.)
        # Extract potential name tokens following the anchor tokens within SAME line if any.
        tokens = line.split()
        # If the line itself mixes anchor + name tokens
        # Strategy: drop early tokens that are email/phone until we reach first plausible name token
        filtered = []
        skipping = True
        for t in tokens:
            if EMAIL_REGEX.match(t) or PHONE_REGEX.match(t):
                continue
            # first non-anchor token -> treat as start of name region
            if skipping:
                if _name_like_token(t):
                    skipping = False
                    filtered.append(t)
            else:
                filtered.append(t)
        if len(filtered) >= 2:
            candidate_tokens = []
            for t in filtered:
                clean = t.strip(",;|/")
                if clean.lower() in ANCHOR_ROLE_STOP and len(candidate_tokens) >= 2:
                    break
                if _name_like_token(clean):
                    candidate_tokens.append(clean)
                else:
                    # break on first obviously non-name after name sequence started
                    break
            if len(candidate_tokens) >= 2:
                recased = [_recase_name_token(x) for x in candidate_tokens[:MAX_NAME_TOKENS]]
                low_toks = [x.lower() for x in recased]
                if not any(t in TECH_OR_HEADING for t in low_toks):
                    pattern_ok = sum(1 for x in recased if NAME_TOKEN_REGEX.match(x) or INITIAL_TOKEN.match(x)) / len(recased)
                    pos_bonus = 1.0 - (idx / max(1,len(lines))) * 0.5
                    score = 0.55 * pattern_ok + 0.25 * pos_bonus + 0.2
                    hits.append({"candidate":" ".join(recased),"score":round(score,4),"line_index":idx,"raw":line[:160]})
                    if score > best_score:
                        best_score = score
                        best = hits[-1]

    if best:
        confidence = 0.60 + min(0.30, best_score * 0.20)
        return {
            "candidate": best["candidate"],
            "confidence": round(confidence,3),
            "origin": "anchor_pass",
            "debug": {"hits": hits[:25], "selected": best}
        }
    return {"candidate": None, "confidence": 0.0, "origin": "anchor_pass", "debug": {"hits": []}}

def _name_like_token(tok: str) -> bool:
    if not tok:
        return False
    if EMAIL_REGEX.match(tok) or PHONE_REGEX.match(tok):
        return False
    base = tok.strip(",.;:()")
    if not base:
        return False
    if ALL_CAPS_TOKEN.match(base) and len(base) > 1:
        return True
    if NAME_TOKEN_REGEX.match(base):
        return True
    if INITIAL_TOKEN.match(base):
        return True
    return False

def _recase_name_token(tok: str) -> str:
    base = tok.strip(",.;:()")
    if ALL_CAPS_TOKEN.match(base) and len(base) > 1:
        return base.capitalize()
    if base.islower():
        return base.capitalize()
    return base

# ---------------- Helpers for later passes ----------------
def _looks_like_name_token(token: str) -> bool:
    return bool(NAME_TOKEN_REGEX.match(token) or INITIAL_TOKEN.match(token))

def _is_generic(name: str) -> bool:
    low = name.lower()
    if low in {"candidate","profile","resume"}:
        return True
    if any(low.startswith(pref) for pref in ("android","aws","kafka","system","email","skills")):
        return True
    return False

def _file_name_hint(file_name: str) -> Optional[str]:
    base = re.sub(r"\.[A-Za-z0-9]+$", "", file_name)
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\d{2,}", " ", base)
    tokens = [t for t in base.split() if 2 <= len(t) <= 25]
    tokens = [t for t in tokens if t.lower() not in {"resume","final","updated","profile","cv","new","latest"}]
    cand = [t for t in tokens if re.match(r"^[A-Z][a-z]+$", t)]
    if 1 < len(cand) <= 4:
        return " ".join(cand[:3])
    return None

def _clean_role_tail(parts: List[str]) -> List[str]:
    while parts:
        if parts[-1].lower() in ROLE_TRAILING_CUTOFF_WORDS and len(parts) > 2:
            parts = parts[:-1]
        else:
            break
    return parts

def _analyze_line(line: str, allow_role_strip: bool):
    raw = line.strip()
    low = raw.lower()
    if any(fs in low for fs in FORBIDDEN_SUBSTRINGS):
        return False, "", "forbidden_substring"
    if len(raw) > 80:
        return False, "", "too_long"
    if raw.count("|") >= 2:
        return False, "", "multi_pipe"
    if raw.count(",") >= 1:
        if not re.match(r"^[A-Z][a-z]+,\s+[A-Z][a-z]+$", raw):
            return False, "", "comma_list"
    tokens = [t.strip(",;") for t in raw.split()]
    tokens = [t for t in tokens if t.lower().rstrip(".") not in COMMON_PREFIX_DISCARD]
    if not (MIN_NAME_TOKENS <= len(tokens) <= MAX_NAME_TOKENS + 2):
        return False, "", f"token_len_{len(tokens)}"

    if allow_role_strip:
        for delim in [" - ", " – ", " — ", " | ", " : "]:
            if delim in raw:
                left = raw.split(delim, 1)[0].strip()
                ltoks = left.split()
                if 1 < len(ltoks) <= MAX_NAME_TOKENS + 1:
                    tokens = ltoks
                    break

    eval_tokens = [t.lower().strip(".") for t in tokens]
    if sum(1 for t in eval_tokens if t in TECH_OR_HEADING) >= 1 and not allow_role_strip:
        return False, "", "tech_hit"

    if allow_role_strip:
        trimmed = _clean_role_tail(tokens)
        if len(trimmed) >= MIN_NAME_TOKENS:
            tokens = trimmed

    caps_tokens = sum(1 for t in tokens if t.isupper() and len(t) > 2)
    if caps_tokens == len(tokens) and len(tokens) > 1:
        return False, "", "all_caps_line"

    invalid = 0
    clean_parts = []
    for t in tokens:
        canon = t.strip(",;")
        if _looks_like_name_token(canon.capitalize()):
            clean_parts.append(canon.capitalize().rstrip("."))
        else:
            invalid += 1
    if invalid > 1:
        return False, "", "invalid_tokens"
    if not (MIN_NAME_TOKENS <= len(clean_parts) <= MAX_NAME_TOKENS):
        return False, "", "final_token_len"

    return True, " ".join(clean_parts), ""

# ---------------- Main Extraction ----------------
def extract_candidate_name(chunks: List[Dict]) -> Tuple[str, float, str, Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "passes": {},
        "selected": None,
        "file_name_hints": [],
        "spacy_used": False
    }

    if not chunks:
        debug["reason"] = "no_chunks"
        return "Candidate", 0.30, "fallback_no_chunks", debug

    # Collect text
    ordered = sorted(chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True)
    collected, total_chars = [], 0
    for ch in ordered[:TOP_SCORE_CHUNK_COUNT]:
        txt = (ch.get("text") or ch.get("content") or "")
        if not txt:
            continue
        remain = MAX_CHAR_SCAN - total_chars
        if remain <= 0:
            break
        piece = txt[:remain]
        collected.append(piece)
        total_chars += len(piece)
    combined = "\n".join(collected)
    if not combined.strip():
        debug["reason"] = "empty_combined"
        return "Candidate", 0.30, "fallback_empty", debug

    # Segment lines (improved)
    lines = _segment_lines(combined)
    debug["passes"]["segmented_line_count"] = len(lines)

    # File name hints
    hints = []
    for ch in chunks:
        fn = ch.get("file_name")
        if fn:
            hn = _file_name_hint(fn)
            if hn:
                hints.append(hn)
    hints = list(dict.fromkeys(hints))
    debug["file_name_hints"] = hints

    # Anchor Pass
    anchor_result = _anchor_pass(lines)
    debug["passes"]["anchor_pass"] = anchor_result["debug"]

    # If high-confidence anchor hit
    if anchor_result["candidate"] and anchor_result["confidence"] >= 0.62:
        cand = anchor_result["candidate"]
        if _is_generic(cand) and hints:
            cand = hints[0]
        debug["selected"] = {"name": cand, "origin": "anchor_pass"}
        return cand, anchor_result["confidence"], "anchor_pass", debug

    # Prepare for other passes
    # Reconstruct a condensed lines list for traditional passes (avoid pure anchor lines)
    condensed = [l for l in lines if not EMAIL_REGEX.fullmatch(l) and not PHONE_REGEX.fullmatch(l)]
    condensed = condensed[:MAX_LINES_INSPECTED]

    # Pass A
    pass_a, dropped_a = [], []
    for idx, line in enumerate(condensed):
        verdict, clean, reason = _analyze_line(line, allow_role_strip=False)
        if verdict:
            pass_a.append({"clean": clean, "line_index": idx, "origin": "pass_a", "raw": line})
        else:
            dropped_a.append({"line": line, "reason": reason})
    debug["passes"]["pass_a_candidates"] = pass_a
    debug["passes"]["pass_a_dropped"] = dropped_a[:40]

    # Pass B
    pass_b = []
    if not pass_a:
        for idx, line in enumerate(condensed):
            verdict, clean, reason = _analyze_line(line, allow_role_strip=True)
            if verdict:
                pass_b.append({"clean": clean, "line_index": idx, "origin": "pass_b", "raw": line})
        debug["passes"]["pass_b_candidates"] = pass_b

    # Pass C (ALL CAPS salvage)
    pass_c = []
    if not pass_a and not pass_b:
        for idx, line in enumerate(condensed[:40]):
            if ALL_CAPS_LINE.match(line) and 4 <= len(line) <= 60:
                toks = [t for t in line.split() if t.isalpha()]
                if 1 < len(toks) <= MAX_NAME_TOKENS:
                    rec = " ".join(t.capitalize() for t in toks)
                    if all(_looks_like_name_token(x) for x in rec.split()):
                        pass_c.append({"clean": rec, "line_index": idx, "origin": "pass_c", "raw": line})
        debug["passes"]["pass_c_candidates"] = pass_c

    # Pass D (email token combos)
    pass_d = []
    if not (pass_a or pass_b or pass_c):
        email_tokens = set()
        for e in EMAIL_REGEX.findall(combined):
            local = e.split("@",1)[0]
            for part in re.split(EMAIL_TOKEN_SPLIT, local):
                p = part.strip().lower()
                if 2 <= len(p) <= 25 and p not in {"gmail","email","mail","yahoo","outlook","hotmail"}:
                    email_tokens.add(p)
        etoks = [t for t in email_tokens if t not in TECH_OR_HEADING]
        if 2 <= len(etoks) <= 6:
            etoks_sorted = sorted(etoks, key=len)
            first, last = etoks_sorted[0], etoks_sorted[-1]
            combo1 = f"{first} {last}"
            pass_d.append({"clean": combo1.title(), "line_index": -1, "origin":"email_combo"})
            if len(etoks_sorted) >= 3:
                mid = etoks_sorted[1]
                combo2 = f"{first} {mid} {last}"
                pass_d.append({"clean": combo2.title(), "line_index": -1, "origin":"email_combo"})
        debug["passes"]["pass_d_email"] = pass_d

    candidates = pass_a or pass_b or pass_c or pass_d

    if not candidates and anchor_result["candidate"]:
        # anchor candidate with lower confidence ( < 0.62 ) fallback
        cand = anchor_result["candidate"]
        if _is_generic(cand) and hints:
            cand = hints[0]
        conf = max(0.55, anchor_result["confidence"])
        debug["selected"] = {"name": cand, "origin": "anchor_pass_low"}
        return cand, conf, "anchor_pass_low", debug

    if not candidates:
        if hints:
            return hints[0], 0.55, "file_name_hint_only", debug
        return "Candidate", 0.30, "fallback_none", debug

    # Scoring
    scored = []
    for cand in candidates:
        name = cand["clean"]
        parts = name.split()
        pattern_score = sum(1 for p in parts if _looks_like_name_token(p)) / len(parts)
        idx = cand["line_index"]
        if idx >= 0:
            position_score = 1.0 - (idx / max(1,len(condensed))) * 0.65
        else:
            position_score = 0.55

        suspicious = 0.0
        for p in parts:
            pl = p.lower()
            if pl in TECH_OR_HEADING:
                suspicious += 0.6
            if ALL_CAPS_TOKEN.match(p) and len(p) > 2:
                suspicious += 0.4
            if INITIAL_TOKEN.match(p):
                suspicious += 0.1
        suspicious_penalty = min(1.0, suspicious)

        base = 0.55 * pattern_score + 0.25 * position_score - 0.25 * suspicious_penalty
        final_score = max(0.0, min(1.0, base))
        scored.append({
            "name": name,
            "origin": cand["origin"],
            "final_score": round(final_score,4),
            "pattern_score": round(pattern_score,4),
            "position_score": round(position_score,4),
            "suspicious_penalty": round(suspicious_penalty,4),
            "line_index": idx,
            "raw_line": cand.get("raw","")
        })

    debug["passes"]["scored"] = scored
    best = max(scored, key=lambda x: x["final_score"])

    if _is_generic(best["name"]) and hints:
        best["name"] = hints[0]
        best["origin"] = best["origin"] + "+file_hint"
        best["final_score"] = max(best["final_score"], 0.55)

    deterministic_conf = 0.40 + best["final_score"] * 0.53
    deterministic_conf = round(min(0.93, deterministic_conf),3)

    # spaCy fallback if very low
    if deterministic_conf < 0.55:
        nlp = get_spacy()
        if nlp:
            doc = nlp(combined[:6000])
            person_ents = [ent.text.strip() for ent in doc.ents if ent.label_=="PERSON"]
            cleaned_ents = []
            for e in person_ents:
                toks = [t for t in e.split() if t]
                if 1 < len(toks) <= 4 and all(_looks_like_name_token(t.capitalize()) for t in toks):
                    cleaned_ents.append(" ".join(t.capitalize() for t in toks))
            if cleaned_ents:
                spacy_name = cleaned_ents[0]
                if not _is_generic(spacy_name):
                    best["name"] = spacy_name
                    best["origin"] = best["origin"] + "+spacy"
                    deterministic_conf = max(deterministic_conf, 0.62)
                    debug["spacy_used"] = True

    debug["selected"] = best
    return best["name"], deterministic_conf, best["origin"], debug
