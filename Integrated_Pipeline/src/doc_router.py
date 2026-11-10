"""Stage 4: scanned documents. Two readers look at every scan:
  AI Part 2  softmax regression trained with augmentation (best on messy scans: 0.997)
  MI Part 3  CNN trained with augmentation (0.970 on messy scans)
The softmax is the main reader because it scored higher. A scan is auto-routed when the softmax is
at least 70% sure (AI Part 2's routing line); anything else goes to a person. The CNN is a second
opinion: when it disagrees, the scan keeps its route but gets a "second reader disagrees" note.
A stricter rule (auto-route only when both readers agree and both are 70% sure) was also measured.
It is reported, not used: on the messy scans it sent 10 more scans to a person and caught no extra
mistakes, because the softmax's only mistake was already under the 70% line.

Where an auto-routed scan goes:
  claim_form       -> claims queue (the claims feed the MI cost data)
  enrollment_form  -> directory update queue
  referral         -> becomes a find-provider request in the front door (Stage 3), so the
                      member gets a doctor with a seat left after the plan.
The images carry no readable text (they are synthetic layouts), so each referral is paired with a
planner member (seed 42) and asks for a specialist that member does not already need.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parts import load, AI2, MI3, SEED

ROUTE_LINE = 0.70
NEXT_STEP = {"claim_form": "claims queue", "enrollment_form": "directory update queue", "referral": "front door: find a provider"}


def train_readers():
    ds = load(AI2, "dataset"); sl = load(AI2, "simple_linear"); r2 = load(AI2, "run_part2")
    Xtr, ytr, _ = ds.load("train", AI2); Xva, yva, _ = ds.load("val", AI2)
    # AI Part 2 softmax + augmentation, same steps and seed as run_part2.py
    Xaug = r2.augment(Xtr, np.random.default_rng(SEED))
    Xa, ya = np.concatenate([Xtr, Xaug]), np.concatenate([ytr, ytr, ytr])
    sc = ds.Scaler().fit(ds.flatten(Xa))
    soft = sl.SoftmaxClassifier(seed=SEED).fit(sc.transform(ds.flatten(Xa)), ya, sc.transform(ds.flatten(Xva)), yva)
    # MI Part 3 CNN + augmentation, same steps and seed as run_part3.py
    common = load(MI3, "common"); cnnm = load(MI3, "cnn_vision"); r3 = load(MI3, "run_part3")
    Xa3 = np.concatenate([Xtr, r3.augment(Xtr, np.random.default_rng(SEED))])
    common.set_seed()
    cnn, _, _ = cnnm.train(cnnm.TinyCNN(), Xa3, np.concatenate([ytr] * 3), Xva, yva)
    p_soft = lambda X: soft.predict_proba(sc.transform(ds.flatten(X)))
    p_cnn = lambda X: cnnm.predict_proba(cnn, X)
    # check both readers reproduce their Parts' messy-scan accuracy
    Xms, yms, _ = ds.load("messy", AI2)
    a_soft = float((p_soft(Xms).argmax(1) == yms).mean()); a_cnn = float((p_cnn(Xms).argmax(1) == yms).mean())
    rep2 = json.load(open(AI2 / "outputs/metrics.json"))["models"]["Softmax + augmentation"]["messy"]["accuracy"]
    rep3 = next(r["messy_accuracy"] for r in json.load(open(MI3 / "outputs/metrics.json"))["vision"] if r["model"] == "CNN + augmentation")
    assert abs(a_soft - rep2) < 1e-3 and abs(a_cnn - rep3) < 1e-3, f"readers did not reproduce: {a_soft} {a_cnn}"
    return ds, p_soft, p_cnn, {"softmax_messy_accuracy": round(a_soft, 4), "cnn_messy_accuracy": round(a_cnn, 4)}


def route(ds, p_soft, p_cnn, set_name):
    X, y, files = ds.load(set_name, AI2)
    ps, pc = p_soft(X), p_cnn(X)
    s_pred, c_pred = ps.argmax(1), pc.argmax(1)
    auto = ps.max(1) >= ROUTE_LINE                                             # the policy in use
    strict = (s_pred == c_pred) & auto & (pc.max(1) >= ROUTE_LINE)              # measured, not used
    rows = []
    for i, f in enumerate(files):
        lab = ds.CLASSES[s_pred[i]]
        rows.append({"file": f, "set": set_name, "true": ds.CLASSES[y[i]], "softmax": lab, "softmax_conf": round(float(ps[i].max()), 4),
                     "cnn": ds.CLASSES[c_pred[i]], "cnn_conf": round(float(pc[i].max()), 4), "readers_agree": bool(s_pred[i] == c_pred[i]),
                     "auto_routed": bool(auto[i]), "next_step": NEXT_STEP[lab] if auto[i] else "human review",
                     "note": "" if s_pred[i] == c_pred[i] else "second reader (CNN) disagrees"})
    summary = {"documents": len(files), "auto_routed": int(auto.sum()), "auto_share": round(float(auto.mean()), 4),
               "wrong_auto_routes": int((auto & (s_pred != y)).sum()), "sent_to_human": int((~auto).sum()),
               "human_queue_that_softmax_had_wrong": int((~auto & (s_pred != y)).sum()),
               "readers_disagree": int((s_pred != c_pred).sum()),
               "strict_both_agree_rule": {"auto_routed": int(strict.sum()), "wrong_auto_routes": int((strict & (s_pred != y)).sum())},
               "cnn_alone": {"auto_routed": int((pc.max(1) >= ROUTE_LINE).sum()),
                             "wrong_auto_routes": int(((pc.max(1) >= ROUTE_LINE) & (c_pred != y)).sum())}}
    return rows, summary


def referrals_to_requests(rows, door, members, rng):
    """Turn every auto-routed referral into a structured find-provider request."""
    specs = sorted(s for s in door.kb.standards if s != "primary care")
    out = []
    for r in rows:
        if not (r["auto_routed"] and r["softmax"] == "referral"):
            continue
        m = members.iloc[int(rng.integers(len(members)))]
        spec = [s for s in specs if s != m["need_specialty"]][int(rng.integers(len(specs) - 1))]
        slots = {"specialty": spec, "zip": m["zip"]}
        text = f"Referral scan {r['file']}: {spec} for member {m['member_id']} near {m['zip']}"
        res = door.handle(text, "find_provider", min(r["softmax_conf"], r["cnn_conf"]), slots, m["member_id"],
                          r["file"].replace(".png", ""), "document (referral)")
        out.append(res)
    return out
