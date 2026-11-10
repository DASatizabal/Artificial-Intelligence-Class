# AI Part 1 (healthcare version): Search and Constraint Satisfaction
Provider Network Access and Cost Planner · David Satizabal · CAI4505C Artificial Intelligence

A rational agent that assigns health plan members to in-network doctors. A* measures travel miles on a county grid, and a constraint solver assigns members without going over any doctor's open slots or high-cost cap. All data is synthetic and seeded (seed 42).

## Run it
```
pip install pandas numpy matplotlib networkx
python src/gen_members.py     # rebuild members and open slots from seed 42
python src/run_part1.py       # search comparison, final assignment, stress test, map
```
Or open `AI_Part1_Search_CSP.ipynb` in Colab and run all cells. It clones this repo (finalProject branch) if the files are missing.

## Folder
```
data/members.csv                600 members (home cell, ZIP, needed specialty, age, starter high-cost flag)
data/predicted_high_cost_flags.csv  MI Part 2's predicted high-cost flag (riskiest 10%); used instead of the starter flag
data/kb/                        county grid, ZIP cells, providers, CMS standards, provider_slots.csv
src/kb.py                       knowledge base + A* (Manhattan heuristic)
src/planner.py                  CSP: domains, backtracking, MRV, forward checking, LCV, max-flow look-ahead, min-conflicts
src/run_part1.py                runs everything and writes outputs/
outputs/                        assignments.csv, search_comparison.csv, metrics.json, assignment_map.png
```

## CSP formulation
| Piece | In this project |
|---|---|
| Variables | Members who need a doctor |
| Domains | In-network doctors of the needed specialty, accepting patients, with open slots, within the CMS distance standard (A* miles), closest first |
| Constraints | Open slots per doctor; high-cost cap per doctor |
| Decomposition | Each specialty is solved on its own (no constraint links them) |

## Results (seed 42)
| Method | Steps (all 7 specialties) | Backtracks | Specialties solved |
|---|---|---|---|
| Plain backtracking | 552,382 | 552,029 | 5 of 7 |
| Backtracking + forward checking | 481,019 | 341,552 | 5 of 7 |
| MRV + forward checking | 200,306 | 160,764 | 6 of 7 |
| MRV + FC + LCV | 200,354 | 142,385 | 6 of 7 |
| **MRV + FC + look-ahead** | **479** | **0** | **7 of 7** |

Step limit: 200,000 per specialty. Backtrack counts come from `outputs/search_comparison.csv`.

- High-cost flag: MI Part 2's prediction (60 members, 10%), not the starter flag (64 members); the two lists share only 10 members.
- Placed 465 of 600 members (77.5%) inside the CMS standard, 4.85 miles on average, zero capacity or high-cost cap violations.
- 77 distance gaps (no doctor inside the standard) and 58 capacity gaps (nearby doctors full).
- Stress test (Pinecrest Medical Group, 15 doctors, leaves; 73 members displaced): min-conflicts repair ends with 401 placed, and 12 unaffected members lose their doctor and 4 switch; solving from scratch ends with 402 placed, and 12 unaffected members lose their doctor and 8 switch. Min-conflicts used its full 3,000-step budget without settling.
