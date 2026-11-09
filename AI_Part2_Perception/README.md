# AI Part 2: Perception (reading the paperwork)
Provider Network Access and Cost Planner · David Satizabal · CAI4505C Artificial Intelligence

Classify scanned documents so each one goes to the right next step: claim form (to claims and MI cost data), referral (start a provider search with the AI Part 1 planner), or provider enrollment form (update the directory). Images are synthetic and seeded (seed 42); text is drawn as gray bars, so no real form or person appears.

## Run it
```
pip install pandas numpy scikit-learn matplotlib pillow
python src/gen_docs.py     # build data/images/*.png and data/labels.csv
python src/run_part2.py    # train, evaluate, augmentation, routing -> outputs/
```
Or open `AI_Part2_Perception.ipynb` in Colab and run all cells. It clones this repo (finalProject branch) if the files are missing.

## Files
| File | What it does |
|---|---|
| `src/gen_docs.py` | Draws the three document types and simulates scanning (tilt, blur, noise, shadows) |
| `src/dataset.py` | Loads the PNGs, flattens 64 x 64 into 4,096 numbers, standardizes with training statistics only |
| `src/simple_linear.py` | Softmax regression from scratch in NumPy (cross-entropy, L2, mini-batch gradient descent, early stopping) |
| `src/run_part2.py` | Augmentation, training, confusion matrices, weight images, confidence routing, metrics JSON |
| `outputs/metrics.json` | All scores, routing table and example routes |
| `outputs/augmented_samples.png` | Original vs augmented training images vs messy scans |

## Data
1,500 images (64 x 64 grayscale), 500 per class: 280 train, 60 validation, 60 test, 100 messy scans (heavier distortion, never trained on).

## Results
| Model | Training images | Clean test accuracy | Messy scan accuracy |
|---|---|---|---|
| Softmax, clean training | 840 | 1.000 | 0.780 |
| **Softmax + augmentation** | **2,520** | **1.000** | **0.997** |

Routing at a 70% confidence threshold on the messy scans: 99.3% routed automatically with zero wrong routes, 2 sent to a person.

Caveats: the augmentation uses the same kinds of distortion as the messy-scan generator, so 0.997 is optimistic for real scans; the 70% threshold was chosen on the same messy set it is reported on. Per-class F1 and confusion counts are in `outputs/metrics.json`.
