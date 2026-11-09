"""Slot parser: regular expressions + a small gazetteer (word list).

Fixes from the starter slots.py:
  * the starter's time regex matched ANY 1-2 digits (it grabbed "33" out of a ZIP code); time now needs am/pm
  * "3pm" and "3am" were treated the same; times are converted to a 24-hour value
  * building names were case sensitive; every match here is case-insensitive and normalized
Slots: specialty, zip, max_miles, language, provider_name, time_hour
"""
import re

# keyword -> canonical specialty. Longer phrases are checked first so "primary care" wins over "care".
SPECIALTY_LEXICON = [
    (r"primary care|family (?:doctor|medicine)|general doctor|\bpcps?\b|main doctor", "primary care"),
    (r"cardiolog\w*|heart (?:doctor|specialist)", "cardiology"),
    (r"dermatolog\w*|skin doctor", "dermatology"),
    (r"endocrinolog\w*|diabetes (?:doctor|care|specialist)", "endocrinology"),
    (r"ob-?gyn\w*|gyn(?:a|e)colog\w*|women'?s health", "ob-gyn"),
    (r"orthop(?:a)?edi\w*|bone doctor|knee surgeon", "orthopedic surgery"),
    (r"psychiatr\w*|mental health doctor", "psychiatry"),
]
LANGUAGE_LEXICON = [(r"haitian creole|creole", "Haitian Creole"), (r"spanish", "Spanish"), (r"portuguese", "Portuguese")]

ZIP_RE = re.compile(r"\b(3\d{4})\b")
MILES_RE = re.compile(r"(?:within|under|less than|no more than|max(?:imum)?(?: of)?)\s+(\d{1,3})\s*(?:miles?|mi)\b", re.I)
TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)
NAME_RE = re.compile(r"\bdr\.?\s+([a-z][a-z'\-]+)", re.I)

# which slots each intent is allowed to carry (intent-conditioned slot schema)
SCHEMA = {
    "find_provider": ["specialty", "zip", "max_miles", "language", "time_hour"],
    "check_network": ["provider_name"],
    "change_pcp": ["provider_name", "time_hour"],
    "report_access_issue": ["specialty", "zip"],
}
ALL_SLOTS = ["specialty", "zip", "max_miles", "language", "provider_name", "time_hour"]


def _name(token):
    token = re.sub(r"'s$", "", token)                    # Dr. Rios's office -> Rios
    return "Dr. " + "-".join(p.capitalize() for p in token.split("-"))


def parse_slots(text):
    """Return every slot found in the text, before the intent schema is applied."""
    t = text.lower()
    slots = {}
    for pat, canon in SPECIALTY_LEXICON:
        if re.search(pat, t):
            slots["specialty"] = canon
            break
    m = ZIP_RE.search(text)
    if m:
        slots["zip"] = m.group(1)
    m = MILES_RE.search(text)
    if m:
        slots["max_miles"] = int(m.group(1))
    for pat, canon in LANGUAGE_LEXICON:
        if re.search(pat, t):
            slots["language"] = canon
            break
    m = NAME_RE.search(text)
    if m:
        slots["provider_name"] = _name(m.group(1))
    m = TIME_RE.search(text)
    if m:
        h, ampm = int(m.group(1)), m.group(3).lower()
        if 1 <= h <= 12:
            slots["time_hour"] = h % 12 + (12 if ampm == "pm" else 0)
    return slots


def apply_schema(intent, slots):
    """Keep only the slots that belong to the predicted intent."""
    return {k: v for k, v in slots.items() if k in SCHEMA.get(intent, [])}
