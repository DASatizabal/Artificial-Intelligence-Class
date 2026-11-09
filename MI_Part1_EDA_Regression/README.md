# MI Part 1: Data Audit, EDA and Baseline Regression
Provider Network Access and Cost Planner · David Satizabal · CAI4510C Machine Intelligence

Predict each member's health plan cost over the next three months (supervised regression). All data is synthetic and seeded (seed 42); no real people.

## Run it
```
pip install pandas numpy scikit-learn matplotlib
python src/gen_claims.py      # build data/member_months.csv from seed 42
python src/run_mi_part1.py    # audit, EDA charts, baseline models -> outputs/
```
Or open `MI_Part1_EDA_Regression.ipynb` in Colab and run all cells. It clones this repo (finalProject branch) if the files are missing.

## Data: `data/member_months.csv`
1,500 members x 21 months (Jan 2025 to Sep 2026) = 31,563 rows including 63 planted duplicates. Members M0001 to M0600 are the AI Part 1 members. `miles_to_pcp` is the A* distance to the nearest in-network PCP. On 2026-06-01 Pinecrest Medical Group leaves the network (drift event for the capstone).

| Column | Meaning |
|---|---|
| member_id, month | who and when |
| age, sex, zip, county, plan_type | member facts (plan_type missing in 1.5% of rows on purpose) |
| chronic_conditions | count of ongoing conditions |
| miles_to_pcp | A* miles to the nearest in-network primary care doctor |
| saw_pcp_last_3m | 1 if they saw a PCP in the last 3 months |
| er_visits_6m, admits_12m | recent ER visits and hospital admissions |
| cost_month, cost_prior_3m, cost_prior_12m | dollars paid this month, last 3 months, last 12 months (40 negative reversals planted) |
| **cost_next_3m** | **target: dollars paid in the next 3 months** |

## Split
Train Jan to Sep 2025 · gap Oct to Dec 2025 · test Jan to Feb 2026. The gap keeps training answers (which cover the next 3 months) from overlapping the test months.

## Results (test months)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Training average | $2,446 | $7,758 | 0.000 |
| Last quarter repeats | $2,610 | $8,837 | -0.298 |
| **Linear regression** | **$2,168** | **$7,426** | **0.084** |
| Log model (typical member) | $1,834 | $8,099 | -0.090 |

- Top 10% of rows = 64.8% of all cost (median $472, mean $1,967). EDA charts and these descriptive numbers use all months; models only learn from training months.
- Members 9+ miles from a PCP: 13.8% saw a PCP recently (vs 69.3% within 2 miles), 1.16 ER visits in 6 months (vs 0.54), $2,704 next-quarter cost (vs $1,839).
