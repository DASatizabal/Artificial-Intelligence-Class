"""Stage 3: the front door. A request (typed by a member, or created from a scanned referral) goes
through AI Part 3 and is then matched to a doctor who has a seat left after the AI Part 1 plan.

AI Part 3 alone answers "who is the nearest in-network doctor that is accepting patients?".
The planner already filled many of those seats, so this stage adds integration rules (I1 to I7).
They are written in the same if/then form as AI Part 3's rules and added to the same proof trace:

  I1  member already seated with a doctor of this specialty        -> give that doctor back
  I2  AI Part 3's pick has a seat left (and a high-cost seat if needed) -> keep the pick, take the seat
  I3  AI Part 3's pick is full, a farther doctor inside the limit has a seat -> switch, take the seat
  I4  no doctor inside the limit has a seat                         -> escalate: capacity gap
  I5  PCP change to a doctor whose seats are all used by the plan   -> reject
  I6  MI Part 3 MC Dropout is unsure about this member's risk        -> add care-team review note
  I7  MI Part 3 HMM says a high-cost spell is likely, not flagged    -> add watch-list note
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parts import load, AI3, SEED

SEARCH_DEPTH = 60      # how many nearest doctors to look through for an open seat


def ai3_model():
    """AI Part 3 intent model, trained on the same 70% split, and checked against its reported accuracy."""
    ib = load(AI3, "intent_baseline")
    req = pd.read_csv(AI3 / "data/nlp/requests.csv", dtype={"gold_zip": str})
    chal = pd.read_csv(AI3 / "data/nlp/challenge_requests.csv", dtype={"gold_zip": str})
    train, test = train_test_split(req, test_size=0.30, stratify=req["intent"], random_state=SEED)
    m3 = json.load(open(AI3 / "src/part3_nlp_interface/metrics.json"))
    champ = m3["champion"]
    model = ib.IntentClassifier(champ).fit(train["utterance"], train["intent"])
    acc = accuracy_score(test["intent"], model.predict(test["utterance"]))
    assert abs(acc - m3["intent_eval"][f"{champ}_test"]["accuracy"]) < 1e-4, "AI Part 3 intent model did not reproduce"
    return model, test.assign(split="test"), chal.assign(split="challenge"), round(acc, 4)


class FrontDoor:
    def __init__(self, plan, scores):
        self.slots_mod = load(AI3, "slots"); self.rules = load(AI3, "rules")
        self.kb = plan["kb"]                                   # same directory and A* as every Part
        self.assign = plan["assign"]
        self.members = plan["members"].set_index("member_id")
        self.scores = scores.set_index("member_id")
        self.seats = plan["seats"].set_index("provider_id")["seats_left"].to_dict()
        self.hc_seats = plan["seats"].set_index("provider_id")["high_cost_seats_left"].to_dict()
        self.spec = self.kb.providers.set_index("provider_id")["specialty"].to_dict()

    def member_card(self, mid):
        s = self.scores.loc[mid]
        return {"member_id": mid, "high_cost_flag": int(s.high_cost_flag), "p_high_cost": float(s.p_high_cost),
                "pred_cost_next_3m": float(s.pred_cost_next_3m), "needs_review": int(s.needs_review),
                "watch_list": int(s.watch_list), "p_spell_3m_hmm": float(s.p_spell_3m_hmm)}

    def _fits(self, pid, high_cost):
        return self.seats.get(pid, 0) > 0 and (not high_cost or self.hc_seats.get(pid, 0) > 0)

    def _take(self, pid, high_cost):
        self.seats[pid] -= 1
        if high_cost:
            self.hc_seats[pid] -= 1

    def handle(self, text, intent, conf, slots, mid, request_id, source):
        valid, msg, res = self.rules.check_rules(intent, slots, self.kb, text, conf)
        action, trace = res["action"], list(res["proof_trace"])
        card = self.member_card(mid)
        hc = card["high_cost_flag"] == 1
        top = (res.get("providers") or [None])[0]
        matched, top_had_seat = None, None

        def fire(rid, prem, concl, message):
            trace.append({"rule": rid, "if": prem, "then": concl, "message": message})

        if action in ("accept", "inform") and intent in ("find_provider", "report_access_issue") and res.get("providers"):
            spec, z = slots["specialty"], slots["zip"]
            limit = slots.get("max_miles") or self.kb.standards[spec]
            cur = self.assign.get(mid)
            top_had_seat = self._fits(top["provider_id"], hc)
            if cur and self.spec[cur] == spec:
                p = self.kb.providers.set_index("provider_id").loc[cur]
                miles = self.kb.astar(self.kb.zip_cell(z), (int(p.row), int(p.col)))
                matched = {"provider_id": cur, "name": p["name"], "miles": int(miles) if miles is not None else None, "seat": "already seated by the plan"}
                fire("I1", ["member_already_seated_same_specialty"], "action:accept", f"You are already matched with {p['name']}.")
                action, msg = "accept", f"You are already matched with {p['name']}."
            else:
                lang = slots.get("language") if "language_match" in res["facts"] else None
                hour = slots.get("time_hour") if "open_at_time" in res["facts"] else None
                pool = [r for r in self.kb.nearest_providers(z, spec, language=lang, k=SEARCH_DEPTH, open_hour=hour) if r["miles"] <= limit]
                pick = next((r for r in pool if self._fits(r["provider_id"], hc)), None)
                if pick and pick["provider_id"] == top["provider_id"]:
                    fire("I2", ["ai3_pick_has_seat"] + (["member_high_cost", "high_cost_seat_left"] if hc else []), "seat_taken",
                         f"{pick['name']} has a seat left after the plan.")
                elif pick:
                    fire("I3", ["not ai3_pick_has_seat", "farther_doctor_inside_limit_has_seat"], "seat_taken",
                         f"{top['name']} is full after the plan; {pick['name']} ({pick['miles']} mi) has a seat.")
                    msg = f"{top['name']} is full. {pick['name']} is {pick['miles']} miles away and can take you."
                else:
                    fire("I4", ["not any_doctor_inside_limit_has_seat"], "action:escalate",
                         "Capacity gap: every matching doctor inside the limit is full after the plan. Logged for the network team.")
                    action, msg = "escalate", "Every matching doctor inside the distance limit is full. Logged for the network team."
                if pick:
                    self._take(pick["provider_id"], hc)
                    matched = {**{k: pick[k] for k in ("provider_id", "name", "miles")}, "seat": "new seat taken",
                               "seats_left_after": self.seats[pick["provider_id"]]}
        elif action == "accept" and intent == "change_pcp" and res.get("provider"):
            pid = res["provider"]["provider_id"]
            if not self._fits(pid, hc):
                fire("I5", ["intent_change_pcp", "not provider_has_seat_after_plan"], "action:reject",
                     "That doctor's open seats were all used by this month's plan.")
                action, msg = "reject", "That doctor's open seats were all used by this month's plan."
            else:
                self._take(pid, hc)
                matched = {"provider_id": pid, "name": res["provider"]["name"], "miles": None, "seat": "new seat taken",
                           "seats_left_after": self.seats[pid]}
        if intent == "report_access_issue" and action != "block" and not msg.startswith("Complaint logged"):
            msg = "Complaint logged. " + msg          # a complaint is always logged, whatever the match
            if action == "accept":
                action = "inform"
        notes = []
        if card["needs_review"] and action != "block":
            fire("I6", ["mc_dropout_unsure"], "note:care_team_review", "Risk model is unsure about this member; a care manager reviews the match.")
            notes.append("care team review: the risk model is unsure about this member")
        if card["watch_list"] and action != "block":
            fire("I7", ["hmm_spell_likely", "not high_cost_flag"], "note:watch_list", "Rising-risk watch list (HMM).")
            notes.append("watch list: HMM sees a likely high-cost spell")
        return {"request_id": request_id, "source": source, "text": text, "member": card, "intent": intent,
                "confidence": round(float(conf), 4), "slots": slots, "action": action, "message": msg,
                "matched_provider": matched, "ai3_top_pick": ({k: top[k] for k in ("provider_id", "name", "miles")} if top else None),
                "ai3_top_pick_had_seat": top_had_seat, "notes": notes, "proof_trace": trace}


def pair_members(reqs, members, rng):
    """The NLP requests have no member attached, so each is paired with a planner member who lives in the
    same ZIP (chosen at random with seed 42). This pairing is synthetic and only lets the risk scores ride along."""
    by_zip = members.groupby("zip")["member_id"].apply(list).to_dict()
    out = []
    for z in reqs["gold_zip"].fillna(""):
        pool = by_zip.get(z) or list(members["member_id"])
        out.append(pool[int(rng.integers(len(pool)))])
    return out


def run_requests(plan, scores):
    model, test, chal = ai3_model()[:3]
    reqs = pd.concat([test, chal]).reset_index(drop=True)
    reqs["member_id"] = pair_members(reqs, plan["members"], np.random.default_rng(SEED))
    door = FrontDoor(plan, scores)
    labels = list(model.classes_); probs = model.predict_proba(reqs["utterance"])
    results = []
    for (_, r), pr in zip(reqs.iterrows(), probs):
        intent, conf = labels[int(pr.argmax())], float(pr.max())
        slots = door.slots_mod.apply_schema(intent, door.slots_mod.parse_slots(r["utterance"]))
        out = door.handle(r["utterance"], intent, conf, slots, r["member_id"], r["request_id"], f"chat ({r['split']})")
        results.append(out)
    return door, results
