"""Integrated pipeline: Parts 1 to 3 of AI and MI as one system.

  0  sync check         every Part's copy of the shared data and code is identical
  1  member scoring     MI1 cost dollars, MI2 high-cost flag, MI3 uncertainty and HMM watch list
  2  planner + cost     AI1 seats members (using the MI2 flag) and puts MI1 dollars on each panel
  3  front door         AI3 reads chat requests; integration rules match each one to a seat left after the plan
  4  documents          AI2 softmax routes scans (MI3 CNN second opinion); referrals go through the front door
Every answer is one JSON object, checked against a schema. Run:  python src/run_pipeline.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parts import OUT, SEED
import sync_check
from member_scores import score_members
from panel_cost import run as plan_run
from front_door import ai3_model, run_requests
from doc_router import train_readers, route, referrals_to_requests

PROVIDER = {"type": ["object", "null"], "required": ["provider_id", "name"],
            "properties": {"provider_id": {"type": "string", "pattern": "^P\\d{4}$"}, "name": {"type": "string"},
                           "miles": {"type": ["integer", "null"], "minimum": 0}}}
ANSWER_SCHEMA = {
    "type": "object",
    "required": ["request_id", "source", "text", "member", "intent", "confidence", "slots", "action", "message",
                 "matched_provider", "ai3_top_pick", "notes", "proof_trace"],
    "properties": {
        "source": {"type": "string"},
        "member": {"type": "object", "required": ["member_id", "high_cost_flag", "p_high_cost", "pred_cost_next_3m", "needs_review", "watch_list"],
                   "properties": {"member_id": {"type": "string", "pattern": "^M\\d{4}$"}, "high_cost_flag": {"enum": [0, 1]},
                                  "p_high_cost": {"type": "number", "minimum": 0, "maximum": 1},
                                  "pred_cost_next_3m": {"type": "number", "minimum": 0}}},
        "intent": {"enum": ["find_provider", "check_network", "change_pcp", "report_access_issue"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "action": {"enum": ["accept", "inform", "reject", "escalate", "ask_clarify", "block"]},
        "matched_provider": PROVIDER, "ai3_top_pick": PROVIDER,
        "notes": {"type": "array", "items": {"type": "string"}},
        "proof_trace": {"type": "array", "minItems": 1,
                        "items": {"type": "object", "required": ["rule", "if", "then"]}},
    },
}


def diagram(path):
    boxes = [("Chat request", 0.03, 0.78, "#c8d3e6"), ("Scanned document", 0.03, 0.22, "#c8d3e6"),
             ("AI 3\nintent + slots\n+ rules", 0.24, 0.78, "#d7ece9"), ("AI 2 softmax\n(MI 3 CNN\nsecond opinion)", 0.24, 0.22, "#d7ece9"),
             ("Front door\nintegration rules\nI1 to I7", 0.47, 0.5, "#f3e3c8"),
             ("MI 1 cost $\nMI 2 high-cost flag\nMI 3 review + watch", 0.47, 0.06, "#f2d4d0"),
             ("AI 1 planner\nA* + CSP seats\n+ panel cost", 0.47, 0.88, "#f2d4d0"),
             ("JSON answer\n+ proof trace", 0.76, 0.5, "#e6e0f0")]
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off")
    for t, x, y, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y - 0.09), 0.17, 0.18, boxstyle="round,pad=0.01", fc=c, ec="#333"))
        ax.text(x + 0.085, y, t, ha="center", va="center", fontsize=8.5)
    arrows = [((0.20, 0.78), (0.24, 0.78)), ((0.20, 0.22), (0.24, 0.22)), ((0.41, 0.74), (0.47, 0.56)),
              ((0.41, 0.26), (0.47, 0.44), "referral"), ((0.555, 0.15), (0.555, 0.41), "risk"), ((0.555, 0.79), (0.555, 0.59), "seats left"),
              ((0.64, 0.5), (0.76, 0.5)), ((0.64, 0.06), (0.64, 0.80), "flag + $")]
    for a in arrows:
        ax.annotate("", xy=a[1], xytext=a[0], arrowprops=dict(arrowstyle="->", color="#333"))
        if len(a) == 3:
            ax.text((a[0][0] + a[1][0]) / 2 + 0.01, (a[0][1] + a[1][1]) / 2, a[2], fontsize=7.5, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0)
    ax.set_title("Integrated pipeline: how AI Parts 1 to 3 and MI Parts 1 to 3 pass work to each other", fontsize=11)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def cap_plot(sens, path):
    d = pd.DataFrame(sens)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = (d.flagged_share * 100).astype(int).astype(str) + "%"
    ax[0].bar(np.arange(len(d)) - 0.2, d.panels_over_real_cap_no_cap, 0.4, label="cap switched off", color="#b85c00")
    ax[0].bar(np.arange(len(d)) + 0.2, d.panels_over_real_cap_cap, 0.4, label="cap on", color="#0e7c86")
    ax[0].set_xticks(range(len(d)), x); ax[0].set_xlabel("share of members flagged high-cost"); ax[0].set_ylabel("panels over their high-cost cap")
    ax[0].legend(fontsize=8); ax[0].set_title("What the cap prevents", fontsize=10)
    ax[1].plot(x, d.placed_no_cap, "o-", color="#b85c00", label="cap switched off"); ax[1].plot(x, d.placed_cap, "o-", color="#0e7c86", label="cap on")
    ax[1].set_xlabel("share of members flagged high-cost"); ax[1].set_ylabel("members placed (of 600)"); ax[1].legend(fontsize=8)
    ax[1].set_title("What the cap costs", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def main():
    t0 = time.perf_counter()
    print("Stage 0: sync check"); sync = sync_check.check(verbose=False)
    print("Stage 1: member scoring"); scores, score_info = score_members()
    print("Stage 2: planner + cost"); plan = plan_run(scores)
    print("Stage 3: front door"); door, chat = run_requests(plan, scores)
    intent_acc = ai3_model()[3]
    print("Stage 4: documents"); ds, p_soft, p_cnn, reader_check = train_readers()
    doc_rows, doc_summary = [], {}
    for s in ("test", "messy"):
        rows, doc_summary[s] = route(ds, p_soft, p_cnn, s); doc_rows += rows
    members = plan["members"].sort_values("member_id").reset_index(drop=True)
    referrals = referrals_to_requests(doc_rows, door, members, np.random.default_rng(SEED))

    answers = chat + referrals
    # open seats by specialty: after the plan, and after the chat requests and referrals took theirs
    seats = plan["seats"].assign(specialty=plan["seats"].provider_id.map(door.spec),
                                 after_requests=plan["seats"].provider_id.map(door.seats))
    seats_by_spec = seats.groupby("specialty")[["seats_left", "after_requests"]].sum().rename(
        columns={"seats_left": "after_plan"}).astype(int).to_dict("index")
    ref_by_spec = pd.DataFrame([{"specialty": r["slots"]["specialty"], "action": r["action"]} for r in referrals])
    ref_by_spec = pd.crosstab(ref_by_spec.specialty, ref_by_spec.action).to_dict("index")
    for a in answers:
        jsonschema.validate(a, ANSWER_SCHEMA)          # every answer must fit the contract

    def tally(rows):
        rules = Counter(t["rule"] for r in rows for t in r["proof_trace"] if t["rule"].startswith("I"))
        seat_q = [r for r in rows if r["ai3_top_pick_had_seat"] is not None]
        return {"answers": len(rows), "actions": dict(Counter(r["action"] for r in rows)),
                "integration_rules_fired": dict(sorted(rules.items())),
                "searches_matched_against_seats": len(seat_q),
                "ai3_top_pick_was_full": sum(1 for r in seat_q if not r["ai3_top_pick_had_seat"]),
                "new_seats_taken": sum(1 for r in rows if (r["matched_provider"] or {}).get("seat") == "new seat taken"),
                "high_cost_members": sum(r["member"]["high_cost_flag"] for r in rows),
                "with_review_or_watch_note": sum(1 for r in rows if r["notes"])}

    metrics = {"seed": SEED, "sync_check": {"files_checked": len(sync), "all_identical": all(r["identical"] for r in sync)},
               "reproduced_from_parts": {"mi1_linear_test_mae": score_info["mi1_linear_test_mae_check"],
                                         "mi3_mlp_test_roc_auc": score_info["mlp_test_roc_auc"],
                                         "ai1_placed": plan["info"]["placed"], "ai1_avg_miles": plan["info"]["avg_miles"],
                                         "ai3_intent_test_accuracy": intent_acc, **reader_check},
               "member_scoring": score_info, "planner_cost": plan["info"],
               "front_door_chat": tally(chat), "front_door_referrals": tally(referrals),
               "open_seats_by_specialty": seats_by_spec, "referral_actions_by_specialty": ref_by_spec,
               "documents": doc_summary, "all_answers_pass_schema": True,
               "runtime_seconds": round(time.perf_counter() - t0, 1)}

    scores.to_csv(OUT / "member_scores.csv", index=False)
    plan["panels"].to_csv(OUT / "panel_costs.csv", index=False)
    pd.DataFrame(doc_rows).to_csv(OUT / "document_routing.csv", index=False)
    with open(OUT / "answers.json", "w") as fh:
        json.dump(answers, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(OUT / "answer_schema.json", "w") as fh:
        json.dump(ANSWER_SCHEMA, fh, indent=1)
    with open(OUT / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    diagram(OUT / "pipeline_diagram.png")
    cap_plot(plan["info"]["cap_sensitivity"], OUT / "cap_sensitivity.png")
    print(json.dumps({k: v for k, v in metrics.items() if k != "planner_cost"}, indent=1))
    print(json.dumps({k: v for k, v in plan["info"].items() if k != "cap_sensitivity"}, indent=1))
    print(pd.DataFrame(plan["info"]["cap_sensitivity"]).to_string(index=False))


if __name__ == "__main__":
    main()
