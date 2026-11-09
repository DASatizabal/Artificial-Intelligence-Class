# AI Part 3: NLP and Knowledge Representation
Provider Network Access and Cost Planner · David Satizabal · CAI4505C Artificial Intelligence

A natural-language front door for a health plan's provider network. A member types a request, the system finds the intent and the details (slots), checks them against a rule base and the provider directory, and returns structured JSON with a proof trace.

All data is synthetic and seeded (seed 42). No real patients, providers or claims.

## Run it
```
pip install pandas numpy scikit-learn matplotlib
python src/gen_data.py      # rebuild data/ from seed 42
python src/run_part3.py     # train, evaluate, write src/part3_nlp_interface/*.json + confusion_matrix.png
```
Or open `AI_Part3_NLP_KR.ipynb` in Colab and run all cells. It clones this repo (finalProject branch) if the files are missing.

## Folder
```
data/kb/      county_grid.csv, zip_cells.csv, providers.csv, access_standards.csv (CMS 42 CFR 422.116 large metro)
data/nlp/     requests.csv (606 labeled), challenge_requests.csv (36 hand-written), nlp_label_map.csv, rule_catalog.csv
src/          gen_data.py, kb.py (A*), intent_baseline.py, slots.py, rules.py, run_part3.py
src/part3_nlp_interface/   nlp_outputs.json, metrics.json, confusion_matrix.png
```

## Starter code fixes
| File | Problem | Fix |
|---|---|---|
| run_part3.py | trained and predicted on the same sentences | 70/30 stratified split, 5-fold CV on train for model choice, separate challenge set |
| slots.py | time regex matched any 1 to 2 digits (grabbed digits from ZIPs), no am/pm | time requires am/pm, converted to 24-hour |
| rules.py | KeyError on any building not in OPEN_HOURS | every lookup checks the directory first (rule R9) |
| rules.py | only R1 was coded; R2 (missing slot) and R3 (conflict) were not | full Horn rule base with forward chaining and proof traces |

## Results (seed 42)
| Measure | Held-out test (n=182) | Challenge (n=36) |
|---|---|---|
| Intent accuracy (BOW + LR) | 0.9945 | 0.8333 |
| Intent macro F1 | 0.9941 | 0.8336 |
| Slot frame match, true intent | 1.0000 | 1.0000 |
| Slot frame match, predicted intent | 1.0000 | 0.9722 |
| Injection detection (all 642 requests) | precision 1.0, recall 1.0 (10 of 10) | |
