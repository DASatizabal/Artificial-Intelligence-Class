# MI Part 3: Neural Models Integration
Provider Network Access and Cost Planner · David Satizabal · CAI4510C Machine Intelligence

Extends the MI Part 2 pipeline with neural and probabilistic models, uncertainty, calibration, ablations and error analysis. All data is synthetic and seeded.

## Reproduce (exact commands, seed 42)
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/gen_claims.py      # member-month data (same as MI Parts 1-2)
python src/gen_docs.py        # document images (same as AI Part 2)
python src/run_part3.py       # every model, plots, tables -> outputs/   (about 1 minute on a laptop CPU)
```
Seeds: `torch.manual_seed(42)`, NumPy seed 42, `torch.use_deterministic_algorithms(True)`. Re-running gives identical tables. Or open `MI_Part3_Neural.ipynb` in Colab and run all cells (it clones this repo if needed; Colab already has PyTorch). Hardware notes: CPU only, no GPU needed. Checked on a 2-core Intel Xeon (2.10 GHz) with Python 3 and the pinned versions in `requirements.txt`: the three commands together take about 40 seconds, and a fresh run regenerates the data and reproduces every table in `outputs/` exactly (only the train-time columns change).

## Code (src/)
| File | What it does |
|---|---|
| `common.py` | Seeds, tabular data and split (same as MI Part 2 + member-level validation set), metrics, ECE |
| `mlp_baseline.py` | MLP: Dense(128) > ReLU > Dropout(0.2) > Dense(64) > ReLU > Dense(2); Adam 1e-3, batch 64, up to 50 epochs, early stopping on validation ROC-AUC; MC Dropout |
| `cnn_vision.py` | CNN: Conv(8,3x3) > ReLU > MaxPool > Conv(16,3x3) > ReLU > MaxPool > Dense(32) > ReLU > Dense(3); saliency |
| `bayes_hmm.py` | 2-state HMM with +1 pseudo-counts, forward filter, Viterbi, 3-month-ahead risk |
| `run_part3.py` | Trains everything, Gaussian NB, ablations, error analysis, subgroup check, plots |

## Report
`docs/MI_Part3_Neural_Report.pdf` (and the editable `.docx`): APA, 10 body pages, covering every section in the brief plus the four reflection prompts (architecture rationale, probabilistic justification, uncertainty discussion, ethics and impact).

## Outputs (outputs/)
`comparison_tabular.csv`, `comparison_vision.csv`, `training_logs.csv`, `metrics.json`, `training_curves.png`, `reliability_diagram.png`, `mlp_confusion_test.png`, `cnn_confusion_messy.png`, `cnn_saliency.png`, `report_fig_calibration_saliency.png` (Figure 2 in the report)

## Comparison vs MI Part 2 (high-cost flag, test months)
| Model | Features | Val Acc | F1 (macro) | ROC-AUC | ECE | Params | Train time | Notes |
|---|---|---|---|---|---|---|---|---|
| Project 2 best (logistic regression, balanced) | Tabular (17 features) | 0.713 | 0.565 | 0.755 | 0.019 | 18 | < 0.1 s | Traditional baseline |
| MLP (class weights) | Tabular | 0.732 | 0.565 | 0.755 | 0.033 | 10,690 | about 1 s | Kept epoch 1 |
| MLP + MC Dropout (20 passes) | Tabular | 0.733 | 0.564 | 0.755 | 0.036 | 10,690 | about 1 s | Variance = uncertainty |
| Gaussian Naive Bayes | Tabular | 0.844 | 0.616 | 0.736 | 0.108 | 68 | fast | Interpretable |
| HMM (2 states) | Monthly cost sequence | 0.829 | 0.590 | 0.673 | 0.010* | 14 | about 2 s | Interpretable tables |

ECE follows the assignment's definition (confidence of the chosen answer). *For the HMM that answer is almost always "no", so its low ECE says little; `reliability_diagram.png` checks P(high-cost) directly and shows the class-weighted models overstate risk about 2 to 3 times. "Project 2 best" is refit on the same 80% of training members as the MLP (test ROC-AUC 0.755 here vs 0.752 in MI Part 2). The MLP keeps its epoch-1 weights: validation ROC-AUC peaks after one pass and training loss keeps falling while validation loss rises (see `training_curves.png`). MLP test confusion matrix: `mlp_confusion_test.png` (199 of 305 high-cost caught, 763 false alarms).

**Win / loss vs Project 2:**
- **Tie:** the MLP matches the Project 2 champion on ranking (ROC-AUC 0.755 vs 0.755) with about 600 times the parameters.
- **Small loss:** the MLP's recall is slightly lower (0.653 vs 0.672).
- **Insight:** the signal in this data is close to additive (age, conditions, prior cost), so the simpler logistic regression stays champion.
- **Other models:** Naive Bayes and the HMM rank worse, but they are fast and readable. MC Dropout adds an uncertainty score without changing the ranking.

## Deep track: CNN on document images
| Model | Clean test | Messy scans |
|---|---|---|
| AI Part 2 softmax, clean training | 1.000 | 0.780 |
| CNN, clean training | 1.000 | 0.903 |
| AI Part 2 softmax + augmentation | 1.000 | 0.997 |
| CNN + augmentation | 1.000 | 0.970 |

## Ablations (MLP, change vs full model)
| Change | Recall | ROC-AUC |
|---|---|---|
| No dropout | -0.020 | -0.001 |
| Half hidden units | -0.013 | -0.000 |
| No class weights | -0.613 | -0.009 |
