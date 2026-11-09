# Provider Network Access and Cost Planner
David Satizabal · Miami Dade College · CAI4505C Artificial Intelligence and CAI4510C Machine Intelligence · Fall 2026

One agent that matches health plan members to doctors they can reach, who have room for them, and who offer the care they need, while predicting which members will cost the most. Each course Part builds one piece of it. All data is synthetic and seeded (seed 42). No real members, providers, or claims.

**Why AI and MI need each other**
```
AI (A*) measures distance  ->  MI uses distance to predict cost
MI predicts cost           ->  AI (CSP) uses cost to assign doctors fairly
```

## Status
| Part | AI | Status | MI | Status |
|---|---|---|---|---|
| 1 | Search + CSP: A* travel miles, assign members to doctors | Done (healthcare version) | EDA + regression: predict member cost | Not started |
| 2 | Perception: sort paperwork images | Not started | Classification: high-cost flag | Not started |
| 3 | NLP + rules: understand requests, check policy, block injections | Done | Neural models | Not started |
| 4 | Probabilistic reasoning, RL, adversarial search | Waiting on instructions | Sequences + transformers | Waiting on instructions |
| 5 | Multi-agent defense swarm | Opens Nov 16 | Final integrated system (capstone) | Opens Nov 23 |

## Folders
| Folder | What's inside |
|---|---|
| `AI_Part1_Search_CSP/` | A* + constraint satisfaction planner, search-method comparison, stress test, map, notebook |
| `AI_Part3_NLP_KR/` | Intent model, slot parser, rule base with proof traces, JSON outputs, notebook, APA report |
| `docs/` | Revised term project proposal (Word) and a plain-language explanation of every Part |

## How to run
Each Part folder runs on its own.

In Colab: open the Part's notebook and run all cells. If the files aren't there, the first cell clones this repo (finalProject branch) for you.

Locally:
```
git clone -b finalProject https://github.com/DASatizabal/Artificial-Intelligence-Class
cd Artificial-Intelligence-Class/AI_Part1_Search_CSP
pip install pandas numpy scikit-learn matplotlib networkx
python src/gen_members.py
python src/run_part1.py
```
For Part 3, use `AI_Part3_NLP_KR`, `python src/gen_data.py`, then `python src/run_part3.py`.

## Headline results
- **AI Part 1:** MRV + forward checking + a max-flow look-ahead solved all 7 specialty puzzles in 480 steps with zero backtracks. Plain backtracking solved 5 of 7 in 416,759 steps. 465 of 600 members (77.5%) placed inside the CMS distance standard, 4.88 miles on average, no rule broken.
- **AI Part 3:** intent accuracy .9945 on held-out requests and .8333 on new wording; slots exact when the intent is right; all 10 injection attempts detected with no false alarms.
