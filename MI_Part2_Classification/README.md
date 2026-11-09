# MI Part 2: Classification Pipeline (the high-cost flag)
Provider Network Access and Cost Planner · David Satizabal · CAI4510C Machine Intelligence

Predict whether a member will be high-cost next quarter (top 10% of next-quarter cost; line set from training months = $3,904). The flag feeds the AI Part 1 planner. Same synthetic, seeded data as MI Part 1.

## Run it
```
pip install pandas numpy scikit-learn matplotlib
python src/gen_claims.py      # same data as MI Part 1 (seed 42)
python src/run_mi_part2.py    # CV, test metrics, plots, flags -> outputs/
```
Or open `MI_Part2_Classification.ipynb` in Colab and run all cells. It clones this repo (finalProject branch) if the files are missing.

## Evaluation plan
- Split: train Jan to Sep 2025, 3-month gap, test Jan to Feb 2026.
- Cross-validation: StratifiedGroupKFold, 5 folds, same high-cost share per fold, each member's months kept in one fold.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC. Recall is the priority.
- Pipeline: ColumnTransformer (StandardScaler for numbers; fill 'Unknown' + OneHotEncoder for categories), fit on training rows only. See `outputs/pipeline_diagram.png`.

## Results (test months: 3,000 rows, 305 high-cost)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Majority baseline (always "no") | 0.898 | 0.000 | 0.000 | 0.000 | 0.500 | 0.102 |
| Logistic regression | 0.903 | 0.724 | 0.069 | 0.126 | 0.753 | 0.337 |
| **Logistic regression, balanced** | **0.705** | **0.206** | **0.669** | **0.315** | **0.752** | **0.337** |
| Decision tree, balanced | 0.621 | 0.173 | 0.721 | 0.279 | 0.712 | 0.256 |

Champion: balanced logistic regression (best F1 and PR-AUC; the tree catches 16 more members but raises 268 more false alarms).

## Hand-off
`outputs/predicted_high_cost_flags.csv`: predicted probability and flag for the 600 AI Part 1 members (latest test month). The riskiest 60 (10%) are flagged; 195 would be flagged at a 0.5 cut-off, too many for the planner's per-doctor cap. The AI Part 1 planner does not read this file yet (next integration step).
