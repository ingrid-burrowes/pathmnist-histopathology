# PathMNIST histopathology classification

A reproducible PyTorch project for nine-class colorectal tissue classification on
[PathMNIST](https://medmnist.com/). It compares a custom convolutional neural
network with an ImageNet-pretrained ResNet-18, selects models using validation data
only, and produces clinically cautious performance and interpretability artifacts.

> **Research/education only.** This project is not a medical device and must not be
> used for diagnosis, treatment, or patient-level decisions.

## Why this project

Histopathology models can appear strong when the train, validation, and test roles
are blurred or when only overall accuracy is reported. This repository makes those
choices explicit:

- the official PathMNIST train/validation/test splits are preserved;
- checkpoints are selected by validation macro-F1 with early stopping;
- the test set is accessed only by the separate evaluation command;
- accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices are saved;
- high-confidence errors are shown rather than hiding failure cases;
- Grad-CAM is provided as a qualitative inspection tool, not an explanation of
  clinical reasoning;
- experiment settings are stored in version-controlled YAML files.

## Dataset

PathMNIST contains **107,180 H&E-stained colorectal histology patches** from nine
tissue classes. The official split contains 89,996 training, 10,004 validation, and
7,180 test images. The test set comes from a different clinical centre, making it a
more meaningful distribution-shift check than a random held-out sample.

This project defaults to the 64×64 MedMNIST+ release. The dataset is downloaded by
the official `medmnist` package and is excluded from Git. The label order is:

1. adipose
2. background
3. debris
4. lymphocytes
5. mucus
6. smooth muscle
7. normal colon mucosa
8. cancer-associated stroma
9. colorectal adenocarcinoma epithelium

## Methods

### Models

**Custom CNN.** Three convolutional blocks, each containing two 3×3 convolutions,
batch normalization, ReLU activations, and 2×2 max pooling, followed by adaptive
average pooling, dropout, and a nine-class linear head.

**ResNet-18.** ImageNet-pretrained ResNet-18 with a new dropout and nine-class
classification head. All layers are fine-tuned. Inputs retain the official 64×64
resolution; no artificial upscaling to 224×224 is performed.

### Training

Training uses AdamW, cross-entropy with light label smoothing, ImageNet
normalization, and conservative augmentations: horizontal/vertical flips, small
rotations, and mild colour jitter. The orientation of these tissue patches is not
intrinsically meaningful, while restrained colour changes simulate limited staining
variation without claiming full stain normalization.

Early stopping monitors validation macro-F1. `ReduceLROnPlateau` lowers the learning
rate when the same validation measure stops improving. Randomness is seeded across
Python, NumPy, PyTorch, and data-loader workers.

## Quick start

Python 3.10 or newer is required. A CUDA-enabled environment such as Google Colab is
recommended for full training.

For a guided GPU run, open `notebooks/run_in_colab.ipynb` and run its cells in order
after creating the GitHub repository. The CNN and ResNet-18 experiments are
deliberately separate so their outputs can be checked before continuing.

```bash
git clone https://github.com/YOUR_USERNAME/pathmnist-histopathology.git
cd pathmnist-histopathology
python -m pip install -e .
```

Train the custom CNN:

```bash
pathmnist-train --config configs/cnn.yaml
```

Train ResNet-18:

```bash
pathmnist-train --config configs/resnet18.yaml
```

The training command downloads PathMNIST automatically, creates `history.csv`, and
saves only the checkpoint with the best validation macro-F1. It does **not** load the
test split.

Evaluate a selected checkpoint once:

```bash
pathmnist-evaluate --checkpoint outputs/resnet18/best_checkpoint.pt
```

This creates:

```text
outputs/resnet18/test/
├── confusion_matrix.png
├── metrics.json
├── per_class_metrics.csv
├── predictions.csv
└── representative_errors.png
```

Create Grad-CAM for a test image:

```bash
pathmnist-gradcam \
  --checkpoint outputs/resnet18/best_checkpoint.pt \
  --split test \
  --index 42
```

Every important training choice can be changed in `configs/cnn.yaml` or
`configs/resnet18.yaml`. Use a new `output_dir` for every experiment so outputs do not
overwrite one another.

## Results

Results will be added here only after the final configurations have been chosen on
the validation split and each selected checkpoint has been evaluated once on the
test split. No performance values are fabricated or copied from the MedMNIST
benchmark.

| Model | Image size | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Custom CNN | 64×64 | Pending | Pending | Pending |
| ResNet-18 | 64×64 | Pending | Pending | Pending |

## Tests

The tests use synthetic tensors and do not download PathMNIST.

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

GitHub Actions runs the same lint and test checks on every push and pull request.

## Limitations and responsible interpretation

- PathMNIST contains small pre-extracted patches, not whole-slide images. This
  project does not address tissue detection, slide tiling, aggregation, or
  patient-level prediction.
- Patch accuracy cannot establish clinical safety, clinical utility, or
  generalization to other scanners, laboratories, staining protocols, populations,
  specimen types, or disease settings.
- Class labels are inherited from the source dataset and are not independently
  reviewed in this project.
- The official test set is from a different centre, but a single external centre is
  insufficient for broad external validation.
- Augmentation is not a substitute for explicit stain normalization or multi-centre
  training.
- Grad-CAM is a coarse localization method. A plausible heatmap does not prove that
  a model used biologically valid features or provide a causal explanation.
- This is a patch-classification study for technical demonstration and education,
  not a clinical claim.

## Reproducibility notes

Exact reproducibility can still vary across hardware and PyTorch/CUDA versions even
with deterministic settings. Each checkpoint records its complete resolved config,
class order, selected epoch, and validation metrics. Dataset files, model weights,
and generated outputs are intentionally excluded from version control.

## Attribution

PathMNIST is distributed by MedMNIST under **CC BY 4.0**. Please cite both MedMNIST
and the original colorectal histology dataset when using it:

- Yang J, Shi R, Wei D, et al. [MedMNIST v2: A large-scale lightweight benchmark for
  2D and 3D biomedical image classification](https://doi.org/10.1038/s41597-022-01721-8).
  *Scientific Data*. 2023;10:41.
- Kather JN, Halama N, Marx A. [100,000 histological images of human colorectal
  cancer and healthy tissue](https://doi.org/10.5281/zenodo.1214456). Zenodo. 2018.

The source code in this repository is released under the MIT License. The dataset
retains its own license and is not redistributed here.
