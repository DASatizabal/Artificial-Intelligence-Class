# Integrated Pipeline: Parts 1 to 3 as one system
Provider Network Access and Cost Planner · David Satizabal · CAI4505C and CAI4510C

Every Part folder still runs and can be graded on its own. This folder imports their code in place (no copies) and passes work between them. Each stage retrains its model with the Part's own code and seed, and checks that the result matches the number that Part reported before the model gets used. All data is synthetic and seeded (seed 42).

## Run it
```
pip install -r ../MI_Part3_Neural/requirements.txt networkx jsonschema
python src/sync_check.py      # stage 0 on its own: are all shared copies identical?
python src/run_pipeline.py    # all stages, about 1 minute on a CPU -> outputs/
```
Or open `Integrated_Pipeline.ipynb` in Colab and run all cells. It clones the repo (finalProject branch) if the files are missing. This folder needs every Part folder next to it.

## Stages
| Stage | File | Parts used | What it hands on |
|---|---|---|---|
| 0 Sync check | `sync_check.py` | all | Fails if any of the 15 shared files (data, code, images, hand-offs) differs between Parts |
| 1 Member scoring | `member_scores.py` | MI 1, MI 2, MI 3 | Predicted dollars, high-cost flag, MC Dropout "unsure" flag, HMM watch list for the 600 planner members |
| 2 Planner + cost | `panel_cost.py` | AI 1, MI 1, MI 2 | Seats members (MI 2 flag), puts MI 1 dollars on every panel, tests the high-cost cap, reports the seats left |
| 3 Front door | `front_door.py` | AI 3, AI 1, MI 3 | AI 3 reads the request; integration rules I1 to I7 match it to a doctor who still has a seat |
| 4 Documents | `doc_router.py` | AI 2, MI 3, AI 3 | Softmax routes scans, CNN gives a second opinion, referrals become front-door requests |
| Runner | `run_pipeline.py` | | Runs everything, checks every answer against the JSON schema, writes outputs |

`parts.py` loads each Part's files under unique names (AI 3 and MI 3 both have a `run_part3.py`).

## Integration rules (added to AI 3's proof trace)
| Rule | If | Then |
|---|---|---|
| I1 | member is already seated with a doctor of this specialty | give that doctor back |
| I2 | AI 3's pick has a seat left (and a high-cost seat if the member is flagged) | keep it, take the seat |
| I3 | AI 3's pick is full, a farther doctor inside the limit has a seat | switch, take the seat |
| I4 | no doctor inside the limit has a seat | escalate: capacity gap |
| I5 | PCP change to a doctor whose seats the plan used up | reject |
| I6 | MI 3 MC Dropout is unsure about the member | add a care-team review note |
| I7 | MI 3 HMM says a spell is likely, MI 2 did not flag | add a watch-list note |

## Results (seed 42)
**Reproduced from the Parts:** MI 1 linear MAE $2,167.62 · MI 3 MLP ROC-AUC 0.7553 · AI 1 465 placed at 4.85 mi · AI 3 intent accuracy 0.9945 · AI 2 softmax 0.9967 and MI 3 CNN 0.970 on messy scans.

**Stage 1, member scoring (600 members, Feb 2026):**
- MI 2 flags 60 members, and they carry 26.3% of the $1.12M predicted next-quarter cost.
- MI 3 marks 60 members as "unsure"; only 7 of them are also flagged.
- The HMM puts 22 members, none of them flagged by MI 2, on a watch list.
- 25 negative cost predictions were set to $0.

**Stage 2, planner + cost:**
- Placement matches AI 1: 465 placed, 4.85 mi.
- The 135 unplaced members carry $306,859 of predicted cost.
- At 10% flagged, the high-cost cap never binds (max 2 flagged per panel).

| Flagged | Panels over cap, cap off | Panels over cap, cap on | Placed, cap off | Placed, cap on |
|---|---|---|---|---|
| 10% | 0 | 0 | 465 | 465 |
| 20% | 1 | 0 | 465 | 465 |
| 30% | 9 | 0 | 465 | 463 |
| 40% | 13 | 0 | 465 | 452 |

**Stage 3, front door (218 AI 3 requests):**
- In 66 of the 92 searches, AI 3's top pick had no seat left after the plan.
- I3 rerouted 22 of those, and I4 escalated 36 as capacity gaps.
- I5 caught 10 PCP changes to doctors with no seats left.
- I6 and I7 added review or watch-list notes to 25 answers.

**Stage 4, documents:**

| Policy (messy scans, n=300) | Auto-routed | Wrong auto-routes |
|---|---|---|
| **Softmax at 70% (used)** | **298** | **0** |
| Both readers agree, both at 70% | 288 | 0 |
| CNN alone at 70% | 297 | 7 |

The stricter rule cost 10 extra human reviews and caught nothing, so the CNN only adds a note when it disagrees.

**Stage 4, referrals:**
- 160 referrals became requests; 21 got a seat and 120 escalated.
- After the plan, only 87 specialist seats were left (endocrinology 7, OB-GYN 13), and both specialties ran out completely.

**Contract:** all 378 answers pass `outputs/answer_schema.json`.

## Honest limits
- The NLP requests have no member ID, so each one is paired at random (seed 42) with a member from the same ZIP. The referral images have no readable text, so each is paired with a member and a specialist. The pairings are synthetic; they let the risk scores and seat counts ride along.
- Seats are taken in request order (chat first, then referrals), so a different order would seat different people.
- Everything runs on one month's snapshot; there is no time dimension yet.

## Outputs
`answers.json` (378 answers), `answer_schema.json`, `member_scores.csv`, `panel_costs.csv`, `document_routing.csv`, `metrics.json`, `pipeline_diagram.png`, `cap_sensitivity.png`
