# Experiment results

These files record the final validation-selected CNN and ResNet-18 runs described in
the repository README. Model selection used only the official PathMNIST validation
split. The selected checkpoint for each model was evaluated once on the official
test split.

## Headline results

| Model | Selected epoch | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Custom CNN | 20 | 0.9935 | 0.9148 | 0.8934 |
| ResNet-18 | 8 | 0.9904 | 0.9467 | 0.9300 |

The configured early-stopping monitor was validation macro-F1 with
`min_delta=0.001`. Therefore, “selected epoch” means the last checkpoint whose
improvement exceeded that threshold, not necessarily the numerically highest later
value in `history.csv`.

## Contents

Each model directory contains:

- `history.csv`: epoch-level training and validation metrics;
- `training_summary.json`: selection split, monitor, and selected value;
- `metrics.json`: selected-checkpoint validation metadata, aggregate test metrics,
  and the full per-class classification report;
- `per_class_metrics.csv`: tabular per-class precision, recall, F1, and support;
- `confusion_matrix.png`: count and row-normalized test confusion matrices;
- `representative_errors.png`: highest-confidence test-set errors.

The ResNet-18 directory additionally contains `gradcam_test_42.png`, a qualitative
Grad-CAM example generated from the selected checkpoint. It uses a 4×4 feature map
from `layer3`; interpretation is necessarily coarse for 64×64 inputs.

Raw per-image prediction CSV files are omitted to keep the repository compact.
Model checkpoints and PathMNIST data are also omitted. All of them can be recreated
with the version-controlled commands and configurations in this repository.
