"""
Text normalization for business names and addresses.

The raw data is messy - abbreviations, legal suffixes, punctuation differences,
transliteration variants, missing address bits, landmark references. We don't
try to solve entity resolution with rules here. Normalization just strips out
the easy systematic noise so the similarity features downstream (Jaccard,
Levenshtein, TF-IDF) compare apples to apples. The actual matching decisions
are left to the model.

Pure Python + regex only. No network calls, no external gazetteers or geocoding
(the challenge rules disallow that).
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Legal-suffix / common business-word normalization
# ---------------------------------------------------------------------------
# Map lots of spellings -> one canonical token. Order doesn't matter; applied
# as whole-word replacements after basic cleanup.
_NAME_WORD_MAP = {
    "corporation": "",
    "incorporated": "",
    "limited": "",
    "company": "",
    "companies": "",
    "private": "",
    "enterprises": "",
    "enterprise": "",
    "industries": "",
    "industry": "",
    "international": "",
    "associates": "",
    "brothers": "",
    "and": "&",  # collapse "and" <-> "&" onto one symbol, stripped later
    "the": "",   # leading articles carry no matching signal
    # Map common legal suffixes to empty (remove them entirely)
    "ltd": "",
    "pvt": "",
    "llc": "",
    "inc": "",
    "corp": "",
    "co": "",
    "gmbh": "",
    "sa": "",
    "sas": "",
    "srl": "",
    "bv": "",
    "plc": "",
    "group": "",
    "holdings": "",
    "services": "",
    "solutions": "",
    "technologies": "",
    "tech": "",
    "intl": "",
    "global": "",
    "national": "",
    "assoc": "",
    "bros": "",
    "ent": "",
    "ind": "",
}

_ADDR_WORD_MAP = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "lane": "ln",
    "drive": "dr",
    "court": "ct",
    "circle": "cir",
    "highway": "hwy",
    "apartment": "apt",
    "apartments": "apt",
    "building": "bldg",
    "floor": "fl",
    "suite": "ste",
    "sector": "sec",
    "block": "blk",
    "colony": "col",
    "nagar": "ngr",
    "cross": "x",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "post": "po",
    "office": "off",
    "near": "near",  # keep as landmark marker, handled separately
    "opposite": "near",
    "opp": "near",
    "behind": "near",
}

_STOPWORDS_NAME = {"", "&", "co", "the", "of", "a", "an",
    # Common legal suffixes / generic business words that carry little matching signal
    "ltd", "pvt", "llc", "inc", "corp", "gmbh", "sa", "sas", "srl", "bv", "plc",
    "group", "holdings", "services", "solutions", "technologies", "tech",
    "intl", "global", "national", "assoc", "bros", "ent", "ind", "corporation",
    "incorporated", "limited", "company", "companies", "private", "enterprises",
    "enterprise", "industries", "industry", "international", "associates", "brothers",
    # Country names / generic location words
    "india", "indian", "usa", "us", "uk", "uk", "canada", "australia"}

_STOPWORDS_ADDRESS = {"", "no", "rd", "st", "dr", "ave", "new", "fl", "city",
    "a", "n", "c", "s", "w", "e", "h", "p", "l", "b", "m", "x", "y", "z",
    "ngr", "col", "blk", "sec", "blvd", "ln", "ct", "cir", "hwy", "apt",
    "bldg", "ste", "po", "off", "rd", "st", "dr", "ave", "ln", "ct", "cir",
    # Single digits (often noise)
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Common noise tokens
    "near", "nan", "null", "unit", "plot", "o", "d"}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s&]", re.UNICODE)


def _strip_accents(text):
    """Fold transliteration/diacritic variants (e.g. cafe vs café) together."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _base_clean(text):
    if text is None:
        return ""
    text = str(text)
    text = _strip_accents(text)
    text = text.lower()
    text = text.replace("-", " ").replace("/", " ").replace(".", " ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(raw):
    """Return a cleaned, suffix-normalized business name string."""
    text = _base_clean(raw)
    tokens = text.split(" ")
    out = []
    for tok in tokens:
        tok = _NAME_WORD_MAP.get(tok, tok)
        if tok:
            out.append(tok)
    # drop trailing "&"/"co" noise tokens with no signal, keep order otherwise
    out = [t for t in out if t not in ("&",)]
    return " ".join(out).strip()


def name_tokens(raw):
    """Token set used for Jaccard / blocking, with stopwords removed."""
    norm = normalize_name(raw)
    return [t for t in norm.split(" ") if t and t not in _STOPWORDS_NAME]


def normalize_address(raw):
    """Return a cleaned, abbreviation-normalized address string."""
    text = _base_clean(raw)
    tokens = text.split(" ")
    out = []
    for tok in tokens:
        tok = _ADDR_WORD_MAP.get(tok, tok)
        if tok:
            out.append(tok)
    return " ".join(out).strip()


def address_tokens(raw):
    norm = normalize_address(raw)
    return [t for t in norm.split(" ") if t and t not in _STOPWORDS_ADDRESS]


_PIN_RE = re.compile(r"\b\d{5,6}\b")
_LANDMARK_RE = re.compile(r"\bnear\b.*$")


def extract_postal_code(raw):
    """Pull a 5-6 digit ZIP / PIN code out of a raw address, if present."""
    if not raw:
        return None
    m = _PIN_RE.search(str(raw))
    return m.group(0) if m else None


def strip_landmark(raw_norm_address):
    """Remove a trailing 'near <landmark>' clause used only as a hint."""
    return _LANDMARK_RE.sub("", raw_norm_address).strip()


def char_ngrams(text, n=3):
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}
