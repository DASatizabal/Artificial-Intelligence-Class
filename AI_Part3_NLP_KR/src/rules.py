"""Logical rule base for request validation (knowledge representation).

How it works
  1. ground_facts() turns the intent, slots and a knowledge-base lookup into true atoms
     (for example "has_zip", "provider_in_network").
  2. forward_chain() fires Horn rules until nothing new can be derived. A premise written "not X"
     means X is not in the fact set (negation as failure, a closed-world assumption). Negation is only
     used on ground facts, never on derived ones, so the result does not depend on rule order.
  3. decide() picks the strongest action: block > ask_clarify > reject > escalate > inform > accept.
Every fired rule is kept as a proof trace, so each decision can be explained.

Fixes from the starter rules.py: it crashed on any building missing from OPEN_HOURS (KeyError),
ignored am/pm, and only coded rule R1; R2 (missing slot) and R3 (conflict) from rule_catalog.csv were never checked.
"""
import re

INJECTION_PATTERNS = [
    r"ignore (?:all |the )?(?:previous|prior|above)?\s*(?:rules|instructions|policy)", r"ignore the \w+ rules",
    r"disregard", r"\[?system\]?\s*:", r"\[system\]", r"you are now", r"override", r"bypass", r"new instructions",
    r"reveal", r"delete the", r"print the .*database", r"approve (?:all|this|every)", r"skip the checks",
    r"without checking", r"every (?:doctor|provider) is in network", r"mark every provider",
]
LOW_CONFIDENCE = 0.50

# (rule id, premises, conclusion, message). Conclusions starting with "action:" are decisions.
RULES = [
    ("R8", ["injection_detected"], "action:block", "Request contains instructions aimed at the system; blocked and logged for review."),
    ("R0", ["low_confidence"], "action:ask_clarify", "I'm not sure what you need. Are you looking for a doctor, checking coverage, changing your PCP, or reporting an access problem?"),
    ("R2a", ["intent_find_provider", "not has_specialty"], "action:ask_clarify", "What type of doctor are you looking for?"),
    ("R2b", ["intent_find_provider", "not has_zip"], "action:ask_clarify", "What ZIP code should I search near?"),
    ("R2c", ["intent_report_access_issue", "not has_zip"], "action:ask_clarify", "What ZIP code do you live in?"),
    ("R2d", ["intent_report_access_issue", "not has_specialty"], "action:ask_clarify", "Which type of doctor were you trying to reach?"),
    ("R2e", ["needs_provider", "not has_provider"], "action:ask_clarify", "Which provider do you mean?"),
    ("Rn1", ["intent_check_network"], "needs_provider", ""),
    ("Rn2", ["intent_change_pcp"], "needs_provider", ""),
    ("R9", ["has_provider", "not provider_known"], "action:reject", "That provider is not in our directory."),
    ("R3", ["intent_change_pcp", "provider_known", "not provider_is_pcp"], "action:reject", "That provider is a specialist and cannot be assigned as a PCP."),
    ("R4", ["intent_change_pcp", "provider_known", "not provider_in_network"], "action:reject", "That provider is out of network, so they cannot be your PCP."),
    ("R5", ["intent_change_pcp", "provider_in_network", "not provider_accepting"], "action:reject", "That provider is not accepting new patients."),
    ("R10", ["intent_change_pcp", "provider_accepting", "not provider_has_capacity"], "action:reject", "That provider's panel is full."),
    ("Rt1", ["not has_time"], "time_ok", ""),
    ("Rt2", ["time_in_hours"], "time_ok", ""),
    ("R1", ["intent_change_pcp", "provider_known", "has_time", "not time_in_hours"], "action:reject", "The office is closed at the time you asked for."),
    ("R16", ["intent_change_pcp", "provider_is_pcp", "provider_in_network", "provider_accepting", "provider_has_capacity", "time_ok"], "action:accept", "PCP change can be submitted."),
    ("R11", ["intent_check_network", "provider_known", "provider_in_network"], "action:accept", "Yes, that provider is in network."),
    ("R12", ["intent_check_network", "provider_known", "not provider_in_network"], "action:inform", "That provider is out of network. Ask me for an in-network provider by specialty and ZIP."),
    ("Rs1", ["intent_find_provider", "has_specialty", "has_zip"], "can_search", ""),
    ("Rs2", ["intent_report_access_issue", "has_specialty", "has_zip"], "can_search", ""),
    ("R6", ["can_search", "not within_standard"], "action:escalate", "Access gap: no in-network provider accepting patients within the CMS distance standard. Logged for the network team."),
    ("R7", ["can_search", "has_language", "not language_match"], "action:inform", "No provider speaking that language was found nearby; closest options are shown."),
    ("R1b", ["intent_find_provider", "can_search", "has_time", "not open_at_time"], "action:reject", "No matching provider is open at that hour."),
    ("R14", ["intent_find_provider", "can_search", "not candidates_found"], "action:reject", "No in-network provider matches within the distance you asked for."),
    ("R13", ["intent_find_provider", "candidates_found", "time_ok_find"], "action:accept", "Here are in-network providers that match."),
    ("Rt3", ["not has_time"], "time_ok_find", ""),
    ("Rt4", ["open_at_time"], "time_ok_find", ""),
    ("R15", ["intent_report_access_issue", "within_standard"], "action:inform", "Complaint logged. In-network options inside the distance standard are listed."),
]
PRIORITY = ["block", "ask_clarify", "reject", "escalate", "inform", "accept"]


def detect_injection(text):
    t = text.lower()
    return [p for p in INJECTION_PATTERNS if re.search(p, t)]


def ground_facts(intent, slots, kb, text, confidence=1.0):
    """Build the fact set and the knowledge-base context (provider record, search results)."""
    f, ctx = {f"intent_{intent}"}, {}
    if confidence < LOW_CONFIDENCE:
        f.add("low_confidence")
    hits = detect_injection(text)
    if hits:
        f.add("injection_detected"); ctx["injection_patterns"] = hits
    for s in ("specialty", "zip", "language"):
        if slots.get(s):
            f.add(f"has_{s}")
    if slots.get("time_hour") is not None:
        f.add("has_time")
    if slots.get("provider_name"):
        f.add("has_provider")
        p = kb.provider(slots["provider_name"])
        if p:
            ctx["provider"] = {k: p[k] for k in ("provider_id", "name", "specialty", "in_network", "accepting_new", "open_hour", "close_hour")}
            f.add("provider_known")
            if p["specialty"] == "primary care": f.add("provider_is_pcp")
            if p["in_network"]: f.add("provider_in_network")
            if p["accepting_new"]: f.add("provider_accepting")
            if p["panel_size"] < p["panel_capacity"]: f.add("provider_has_capacity")
            h = slots.get("time_hour")
            if h is not None and p["open_hour"] <= h < p["close_hour"]:
                f.add("time_in_hours")
    if slots.get("specialty") and slots.get("zip") and kb.zip_cell(slots["zip"]):
        spec, z = slots["specialty"], slots["zip"]
        std = kb.standards[spec]
        nearest = kb.nearest_providers(z, spec, k=1)
        ctx["standard_miles"] = std
        ctx["nearest_in_network_miles"] = nearest[0]["miles"] if nearest else None
        if nearest and nearest[0]["miles"] <= std:
            f.add("within_standard")
        limit = slots.get("max_miles") or std
        lang = slots.get("language")
        res = kb.nearest_providers(z, spec, language=lang, k=3)
        if lang and res and res[0]["miles"] <= limit:
            f.add("language_match")
        if lang and not (res and res[0]["miles"] <= limit):
            res = kb.nearest_providers(z, spec, k=3)          # fall back to any language
        res = [r for r in res if r["miles"] <= limit]
        if res:
            f.add("candidates_found")
        if slots.get("time_hour") is not None:
            timed = [r for r in kb.nearest_providers(z, spec, language=lang if "language_match" in f else None, k=3,
                                                     open_hour=slots["time_hour"]) if r["miles"] <= limit]
            if timed:
                f.add("open_at_time"); res = timed
        ctx["providers"] = res
    return f, ctx


def forward_chain(facts):
    facts, trace, changed = set(facts), [], True
    while changed:
        changed = False
        for rid, prem, concl, msg in RULES:
            if concl in facts:
                continue
            ok = all((p[4:] not in facts) if p.startswith("not ") else (p in facts) for p in prem)
            if ok:
                facts.add(concl); changed = True
                trace.append({"rule": rid, "if": prem, "then": concl, **({"message": msg} if msg else {})})
    return facts, trace


def decide(facts, trace):
    actions = [c.split(":")[1] for c in facts if c.startswith("action:")]
    if not actions:
        return "ask_clarify", "I could not complete that request. Can you say it another way?"
    best = min(actions, key=PRIORITY.index)
    msg = next(t["message"] for t in trace if t["then"] == f"action:{best}")
    return best, msg


def check_rules(intent, slots, kb, text, confidence=1.0):
    """Starter-compatible entry point: returns (valid, message) plus the full structured result."""
    facts, ctx = ground_facts(intent, slots, kb, text, confidence)
    facts, trace = forward_chain(facts)
    action, msg = decide(facts, trace)
    return action == "accept", msg, {"action": action, "facts": sorted(f for f in facts if not f.startswith("action:")),
                                      "proof_trace": trace, **ctx}
