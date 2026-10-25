# Imperceptible Backdoor Attacks on Self-Supervised Learning

A final-year research project studying whether learned, visually subtle triggers can redirect self-supervised image representations toward a target class while preserving useful features and clean-image classification accuracy.

The experiments cover **SimCLR** and **MoCo v1**, with CIFAR-10 and STL-10 used for both upstream encoder training and downstream classification. The implementation includes a StegaStamp-inspired trigger, alternating encoder/hider optimization, and the TRAD target-relative disentanglement objective.

## Method

Both frameworks use a ResNet-18 adapted to 32 × 32 images, with a 3 × 3 first convolution, stride 1, and no max-pooling. Downstream classifiers use L2-normalized, 512-dimensional backbone features before projection.

- **SimCLR:** contrastive training on paired augmented views with an MLP projection head.
- **MoCo v1:** query and momentum-updated key encoders, a FIFO negative queue, and a **single linear projection head**. The retained implementation is a single-process, small-image adaptation of MoCo v1.

The **StegaStamp-inspired hider** produces an image-dependent residual carrying a fixed alternating **16-bit message**. The main pipeline bounds this residual by **8/255** per channel and clips encoded images to [0, 1], before normalization. Message warm-up, weak-stealth, and mild-corruption training establish the initialization. Message recovery and image-quality losses support visual subtlety; a residual bound alone does not establish perceptual invisibility.

**TRAD** compares clean and triggered images under the same sampled SSL augmentation. Let `s_a` and `s_ta` be their feature cosine similarities to a frozen clean-encoder target-class centroid:

```text
L_TRAD = 0.1 × mean(max(0, 0.1 - |s_ta - s_a|))
       + 0.2 × mean(1 - s_ta)
```

The gap term separates the two branches relative to the target; the attraction term continuously pulls triggered features toward it. The augmented clean branch and centroid are detached anchors. TRAD operates in both alternating phases: gradients update the hider with decoder/encoder parameters frozen, then update the compromised backbone with the hider, projection head, and BatchNorm state frozen. Alignment, feature-preservation, message, and image-quality terms complement this objective.

The final TRAD runs disable the older HSV/HSL color-disentanglement loss. The **no-TRAD ablation** uses the alternating SimCLR trainer with both its color- and feature-disentanglement weights set to zero. See the [TRAD loss](src/main/stegastamp_target_relative_disentangle_dualphase.py) and [alternating trainer](src/main/train_inactive_alternating_c2_target_relative_dualphase.py).

## Experimental results

The eight main settings evaluate each framework/upstream pair on both downstream datasets; the ninth group is the no-TRAD ablation. STL-10 is resized to 32 × 32, and its clean SSL pretraining uses `train+unlabeled`. The target is **label 0 (airplane)** in both datasets.

Each row links to its original aggregate JSON. Values are percentages: CA and BA are means; ASR statistics are **mean ± sample standard deviation** across downstream MLP seeds 42–46.

- **CA:** clean-encoder classifier accuracy on clean test images.
- **BA:** backdoored-encoder classifier accuracy on clean test images.
- **ASR:** target prediction rate on triggered non-target test images.
- **Adjusted ASR:** `max(0, ASR - baseline target rate)`, calculated per repetition before aggregation. The baseline is the compromised classifier's target rate on clean non-target images.

| Framework / method | Upstream → downstream / saved record | CA | BA | ASR | Adjusted ASR |
|---|---|---:|---:|---:|---:|
| SimCLR / TRAD | [CIFAR-10 → CIFAR-10](results/original/b4/results_C2_dualphase_strict/downstream_cifar10/inactive_downstream_aggregate.json) | 90.764 | 90.546 | 94.16 ± 0.24 | 93.14 ± 0.20 |
| SimCLR / TRAD | [CIFAR-10 → STL-10](results/original/b4/results_C2_dualphase_strict/downstream_stl10/inactive_downstream_aggregate.json) | 77.200 | 77.418 | 97.55 ± 0.35 | 96.05 ± 0.39 |
| SimCLR / TRAD | [STL-10 → CIFAR-10](results/original/b4/results_C2_dualphase_strict_stl10_upstream/downstream_cifar10/inactive_downstream_aggregate.json) | 86.956 | 86.628 | 94.45 ± 1.40 | 93.07 ± 1.32 |
| SimCLR / TRAD | [STL-10 → STL-10](results/original/b4/results_C2_dualphase_strict_stl10_upstream/downstream_stl10/inactive_downstream_aggregate.json) | 80.588 | 80.590 | 99.33 ± 0.11 | 98.36 ± 0.10 |
| MoCo v1 / TRAD | [CIFAR-10 → CIFAR-10](results/original/b4/results_eval_moco_c2_cifar10_to_cifar10/inactive_downstream_aggregate.json) | 87.822 | 87.868 | 60.76 ± 35.41 | 59.42 ± 35.23 |
| MoCo v1 / TRAD | [CIFAR-10 → STL-10](results/original/b4/results_eval_moco_c2_cifar10_to_stl10/inactive_downstream_aggregate.json) | 74.165 | 74.638 | 99.21 ± 0.09 | 97.41 ± 0.15 |
| MoCo v1 / TRAD | [STL-10 → CIFAR-10](results/original/b4/results_eval_moco_c2_stl10_to_cifar10/inactive_downstream_aggregate.json) | 86.278 | 86.250 | 2.06 ± 0.92 | 0.72 ± 0.63 |
| MoCo v1 / TRAD | [STL-10 → STL-10](results/original/b4/results_eval_moco_c2_stl10_to_stl10/inactive_downstream_aggregate.json) | 78.035 | 77.740 | 99.60 ± 0.08 | 98.35 ± 0.08 |
| SimCLR / no TRAD | [CIFAR-10 → CIFAR-10](results/original/b4/results_B_strict_cifar10_upstream/downstream_cifar10/inactive_downstream_aggregate.json) | 90.764 | 90.528 | 92.23 ± 0.47 | 91.19 ± 0.46 |

The low and variable MoCo results on CIFAR-10 downstream are part of the findings. Repetitions vary downstream MLP training, rather than independently retraining the upstream attack. The saved MoCo/CIFAR-10 attack pair is from **epoch 16** for both downstream tasks; the other attack pairs are from epoch 20.

The MLP architecture is `512 → 512 → 256 → 10`, with dropout 0.5 and selection by best validation accuracy. The global experiment seed is 42 and the downstream split seed is 53.

| Downstream dataset | Classifier fit | Classifier validation | Test | Non-target test |
|---|---:|---:|---:|---:|
| CIFAR-10 | 45,000 | 5,000 | 10,000 | 9,000 |
| STL-10 | 4,500 | 500 | 8,000 | 7,200 |

## Repository structure

```text
src/
├── main/                  # Trainers, evaluators, models, losses, dataset loaders
├── plots/                 # Charts using embedded saved per-seed data
└── historical/ca2/        # Isolated older t-SNE experiment and matching helpers
configs/original/b4/       # Original training configurations
results/original/          # Aggregate, per-seed, validation, and historical metrics
logs/original/b4/          # Machine-generated training/validation CSV logs
figures/original/          # Saved figure assets
environment/              # Main, historical, and legacy requirements
docs/                     # Reproduction and supporting documentation
artifacts/                # Local checkpoints (ignored)
data/                     # External datasets (ignored)
outputs/                  # New training/replay outputs (ignored)
```

**Model checkpoints and datasets are intentionally excluded from GitHub.** Private thesis/paper files and Word experiment logs are also excluded by [the ignore rules](.gitignore). Numerical results, configurations, and machine-generated CSV logs remain available.

## Setup

The [main requirements](environment/main/requirements.txt) declare `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`, `pillow`, `tqdm`, and `lpips`. Python 3.11.4 was observed in the project environments; the requirements are unpinned. LPIPS computation is optional and disabled in the selected attack configurations.

From the repository root, using **PowerShell**:

```powershell
python -m venv .venv
. ./.venv/Scripts/Activate.ps1
```

Install a compatible PyTorch/torchvision build using the [official installation instructions](https://pytorch.org/get-started/locally/), then install the declared dependencies:

```powershell
python -m pip install -r environment/main/requirements.txt
```

Obtain the [CIFAR-10 Python dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and [STL-10 binary dataset](https://cs.stanford.edu/~acoates/stl10/), extracting them under `data/cifar-10-batches-py/` and `data/stl10_binary/`. Supply separately obtained local checkpoints under `artifacts/` with the expected run-directory and filename suffixes.

See the **[reproduction guide](docs/REPRODUCTION.md)** for checkpoint paths, dataset requirements, and verified CLI examples covering clean pretraining, trigger/attack training, downstream fitting, saved validation/test inference, and figure generation.

## Reproducibility limitations

- Exact numerical reproduction is unverified. Dependencies are unpinned, original runtime/hardware metadata is incomplete, and the bare clean CIFAR SimCLR checkpoint lacks full historical provenance.
- **Downstream fitting trains new MLPs.** Saved validation replay supports both frameworks/datasets but evaluates the classifier-selection partition. Existing saved-MLP test inference is limited to the legacy SimCLR/CIFAR path; there is **no general saved-MLP test replay CLI for all nine groups**.
- Historical phase options, normalization differences, derived seed streams, and checkpoint-selection rules matter. New runs can select different epochs; evaluation exports do not establish complete training-resume state.
- Historical CA2 t-SNE uses a separate **30-bit / 0.125 residual-bound** hider. It is isolated from the main pipeline, and exact stochastic regeneration is not established.

The reproduction guide details these limits and keeps new outputs separate from the saved experimental records.
