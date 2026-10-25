# Reproduction guide

This guide provides training, saved-model evaluation, and figure-generation examples for **Imperceptible Backdoor Attacks on Self-Supervised Learning**. See the [project README](../README.md) for the method and [all nine experimental results](../README.md#experimental-results).

Commands use the retained CLI interfaces and must be run from the **repository root** in PowerShell. They have been checked statically against the code; they have not been executed to establish numerical reproduction.

Choose the workflow that matches your objective:

- **New training:** clean SSL, StegaStamp phases, attack training, and downstream MLP fitting create new models.
- **Saved-model inference:** validation replay loads saved pair MLPs; the legacy SimCLR/CIFAR test runner loads explicitly supplied clean/pair MLPs.
- **Figures:** embedded-data charts need no model inference; trigger-image and historical t-SNE regeneration do.

**There is no general saved-MLP test replay CLI for all nine groups.** Keep saved inputs and original results unchanged, and use fresh output directories.

## Setup

The main dependencies are declared in [environment/main/requirements.txt](../environment/main/requirements.txt):

```text
torch
torchvision
numpy
matplotlib
scikit-learn
pillow
tqdm
lpips
```

Python **3.11.4** was observed in the recorded project environments. The main environment observation records PyTorch **2.7.1+cu118** and torchvision **0.22.1+cu118**; the separate legacy clean environment records a different pair. These observations are not a complete experiment-time lockfile or a guarantee of compatibility.

The following examples use **PowerShell**, starting in the repository root. Create an environment and activate it:

```powershell
python -m venv .venv
. ./.venv/Scripts/Activate.ps1
```

Choose a compatible PyTorch/torchvision build for your platform using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the declared dependencies:

```powershell
python -m pip install -r environment/main/requirements.txt
```

The requirements are unpinned; this command does not reconstruct the historical environment. LPIPS computation is optional in the code and disabled in the selected attack configurations; enabling it may require auxiliary network weights. The [historical CA2 requirements](../environment/historical_ca2/requirements.txt) omit LPIPS. The UTF-16 legacy freeze in `environment/legacy_clean/requirements.txt` includes additional packages and is preserved separately, rather than serving as the main requirements file.

Define the root-relative paths used below:

```powershell
$Project = (Get-Location).Path
$Data = Join-Path $Project "data"
$Artifacts = Join-Path $Project "artifacts"
$SimCLRCifar = "$Artifacts/code/results/simclr_cifar10_ep1000.pth"
$SimCLRStl = "$Artifacts/b4/results_simclr_stl10/simclr_stl10_ep1000.pth"
$MoCoCifar = "$Artifacts/b4/results_moco_cifar10/moco_cifar10_ep1000.pth"
$MoCoStl = "$Artifacts/b4/results_moco_stl10/moco_stl10_ep1000.pth"
$TriggerStart = "$Artifacts/b4/results_stegastamp_corruption_mild/checkpoints/advanced_stegastamp_best.pth"
```

Run entry points by their file paths, such as `python src/main/train_simclr.py`. Their sibling imports depend on this layout. Keep the main and historical CA2 helper directories separate; do not combine them on `PYTHONPATH`.

## External datasets and local checkpoints

Obtain the **CIFAR-10 Python version** from the [official CIFAR dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) and the **STL-10 binary version** from the [official STL-10 dataset page](https://cs.stanford.edu/~acoates/stl10/). Extract them beneath the dataset root:

```text
data/
├── cifar-10-batches-py/
│   ├── data_batch_1 ... data_batch_5
│   ├── test_batch
│   └── batches.meta
└── stl10_binary/
    └── original extracted binary dataset files
```

CIFAR-10 provides 50,000 training and 10,000 test images. STL-10 provides 5,000 labeled training, 8,000 test, and 100,000 unlabeled images. The project resizes STL-10 to 32 × 32; its downstream protocol uses the split above, rather than the dataset's standard predefined-fold benchmark. Clean STL pretraining uses `train+unlabeled`; STL attack training also needs the unlabeled data. See [torchvision's STL-10 interface](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html) for split support. Some training loaders can download missing datasets; the replay examples below disable downloads.

To use saved models, obtain the original checkpoints separately and supply the clean encoder, matched attack encoder/trigger exports, and downstream MLPs locally. No hosted checkpoint-download URL is provided here. Preserve the original run-directory and filename suffixes beneath `artifacts/b4`, with the clean CIFAR SimCLR file under `artifacts/code/results`. This retains the original `../code/results` relationship.

| Local checkpoint group beneath `artifacts/b4/` | Purpose |
|---|---|
| `results_simclr_stl10/`, `results_moco_cifar10/`, `results_moco_stl10/` | Other clean encoders |
| `results_stegastamp_message_warmup_50epoch/checkpoints/` | Warm-up best checkpoint, selected epoch 48 |
| `results_stegastamp_weak_stealth/checkpoints/` | Weak-stealth final checkpoint, epoch 25 |
| `results_stegastamp_corruption_mild/checkpoints/` | Mild-corruption best initialization, selected epoch 6 |
| Original attack-run `checkpoints/` directories listed below | Matched encoder and trigger exports |
| Original evaluation paths followed by `classifiers/repeat_XX_seed_YY/` | Saved clean and pair MLPs |

Original JSON paths are retained as provenance. They are not generally relative to the JSON's new physical directory. Validation replay resolves them with `--project-root artifacts/b4`; other workflows need explicit checkpoint arguments. The downstream fitting evaluators require an explicit clean-checkpoint argument; no clean checkpoint is selected implicitly.

## Training workflows

These commands create **new trained models**. They describe actual CLI interfaces and relocated paths; they do not assert exact numerical reproduction. Use fresh output directories and preserve `configs/original`, `results/original`, and `logs/original`.

There is no general `--config` JSON-loading option. The saved configurations document the settings; arguments must be passed through the corresponding script's CLI.

### Clean SSL pretraining

```powershell
python src/main/train_simclr.py --dataset cifar10 --epochs 1000 --seed 100 --data_dir "$Data" --save_dir "$Project/outputs/clean/simclr_cifar10"
python src/main/train_simclr.py --dataset stl10 --epochs 1000 --seed 100 --data_dir "$Data" --save_dir "$Project/outputs/clean/simclr_stl10"
python src/main/train_moco.py --dataset cifar10 --epochs 1000 --seed 100 --data_dir "$Data" --save_dir "$Project/outputs/clean/moco_cifar10"
python src/main/train_moco.py --dataset stl10 --epochs 1000 --seed 100 --data_dir "$Data" --save_dir "$Project/outputs/clean/moco_stl10"
```

These trainers use underscore options `--data_dir` and `--save_dir`. The exported names are `simclr_<dataset>_ep1000.pth` and `moco_<dataset>_ep1000.pth`. To use newly trained clean models below, update the clean-checkpoint variables to those new exports. The original bare CIFAR SimCLR checkpoint does not establish its historical seed or full pretraining provenance.

### StegaStamp-inspired initialization

| Phase | Saved configuration | Training epochs | Original selected checkpoint |
|---|---|---:|---|
| Message warm-up | [Warm-up config](../configs/original/b4/results_stegastamp_message_warmup_50epoch/config.json) | 50 | Best export at epoch 48 |
| Weak stealth | [Weak-stealth config](../configs/original/b4/results_stegastamp_weak_stealth/config.json) | 25 | Final export at epoch 25 |
| Mild corruption | [Mild-corruption config](../configs/original/b4/results_stegastamp_corruption_mild/config.json) | 20 | Best export at epoch 6 |

The following are new-run examples transcribed from the recorded phase settings. The weak-stealth and mild-corruption examples use the original selected boundary checkpoints as inputs. For a new chain from scratch, replace these inputs with appropriately selected exports from the preceding new run; the original selected epochs do not guarantee the same selection in a new run.

```powershell
python src/main/train_stegastamp_like.py `
  --data-dir "$Data" --output-dir "$Project/outputs/stegastamp/warmup" `
  --epochs 50 --batch-size 64 --train-subset 5000 --val-subset 1000 --num-workers 0 --seed 42 `
  --lr 0.0001 --disable-corruption --message-weight 10 `
  --image-l1-weight 0 --image-mse-weight 0 --ssim-weight 0 --psnr-weight 0 `
  --residual-l2-weight 0 --residual-linf-weight 0 --residual-tv-weight 0 `
  --frequency-weight 0 --texture-weight 0 --residual-mask-l1-weight 0 `
  --corruption-warmup-epochs 10 --corruption-ramp-epochs 40 --max-corruption-strength 0.75 `
  --print-every 20 --max-eval-batches 5

python src/main/train_stegastamp_like.py `
  --data-dir "$Data" --output-dir "$Project/outputs/stegastamp/weak_stealth" `
  --initialize-from "$Artifacts/b4/results_stegastamp_message_warmup_50epoch/checkpoints/advanced_stegastamp_best.pth" `
  --epochs 25 --batch-size 64 --train-subset 5000 --val-subset 1000 --num-workers 0 --seed 42 `
  --lr 0.00005 --disable-corruption --message-weight 10 `
  --image-l1-weight 0.2 --image-mse-weight 0.1 --ssim-weight 0.1 --psnr-weight 0.05 --target-psnr-db 38 `
  --residual-l2-weight 0.05 --residual-linf-weight 0.1 --residual-tv-weight 0.01 `
  --frequency-weight 0.01 --texture-weight 0.01 --residual-mask-l1-weight 0.001 `
  --corruption-warmup-epochs 10 --corruption-ramp-epochs 40 --max-corruption-strength 0.75 `
  --print-every 20 --save-every 5 --max-eval-batches 0

python src/main/train_stegastamp_like.py `
  --data-dir "$Data" --output-dir "$Project/outputs/stegastamp/mild_corruption" `
  --initialize-from "$Artifacts/b4/results_stegastamp_weak_stealth/checkpoints/advanced_stegastamp_final.pth" `
  --epochs 20 --batch-size 64 --train-subset 5000 --val-subset 1000 --num-workers 2 --seed 42 `
  --lr 0.00002 --message-weight 10 --clean-message-weight 2 `
  --corruption-profile combined --initial-corruption-strength 0.05 `
  --corruption-warmup-epochs 2 --corruption-ramp-epochs 13 `
  --max-corruption-strength 0.25 --eval-corruption-strength 0.25 --freeze-encoder-epochs 5 `
  --print-every 20
```

Unspecified options use the retained trainer's defaults. The older warm-up and weak-stealth configs omit newer options, including clean-message loss and stricter checkpoint-selection thresholds. Their historical values cannot be recovered from those configs alone, so these examples are not verified exact reconstructions of the original phases. `--initialize-from` starts a new phase from model weights and resets training state; `--resume` continues a compatible interrupted phase.

### TRAD attack training

Each command uses a clean encoder and the selected mild-corruption hider/decoder initialization. The following run directories identify the original configuration families:

| Framework / upstream | Saved configuration / original run |
|---|---|
| SimCLR / CIFAR-10 | [results_C2_dualphase_strict](../configs/original/b4/results_C2_dualphase_strict/config.json) |
| SimCLR / STL-10 | [results_C2_dualphase_strict_stl10_upstream](../configs/original/b4/results_C2_dualphase_strict_stl10_upstream/config.json) |
| MoCo v1 / CIFAR-10 | [results_moco_inactive_alternating_c2_dualphase_cifar10](../configs/original/b4/results_moco_inactive_alternating_c2_dualphase_cifar10/config.json) |
| MoCo v1 / STL-10 | [results_moco_inactive_alternating_c2_dualphase_stl10](../configs/original/b4/results_moco_inactive_alternating_c2_dualphase_stl10/config.json) |

```powershell
python src/main/train_inactive_alternating_c2_target_relative_dualphase.py `
  --upstream-dataset cifar10 --clean-simclr-checkpoint "$SimCLRCifar" `
  --stegastamp-checkpoint "$TriggerStart" --data-dir "$Data" `
  --epochs 20 --batch-size 64 --train-subset 5000 --val-subset 1000 --seed 42 `
  --c2-separation-margin 0.1 --c2-separation-weight 0.1 --c2-target-weight 0.2 `
  --output-dir "$Project/outputs/attacks/simclr_cifar10"

python src/main/train_inactive_alternating_c2_target_relative_dualphase.py `
  --upstream-dataset stl10 --clean-simclr-checkpoint "$SimCLRStl" `
  --stegastamp-checkpoint "$TriggerStart" --data-dir "$Data" `
  --epochs 20 --batch-size 64 --train-subset 5000 --val-subset 1000 --seed 42 `
  --c2-separation-margin 0.1 --c2-separation-weight 0.1 --c2-target-weight 0.2 `
  --output-dir "$Project/outputs/attacks/simclr_stl10"

python src/main/train_inactive_alternating_c2_target_relative_dualphase_moco.py `
  --upstream-dataset cifar10 --clean-moco-checkpoint "$MoCoCifar" `
  --stegastamp-checkpoint "$TriggerStart" --data-dir "$Data" `
  --epochs 20 --batch-size 64 --train-subset 5000 --val-subset 1000 --seed 42 `
  --c2-separation-margin 0.1 --c2-separation-weight 0.1 --c2-target-weight 0.2 `
  --output-dir "$Project/outputs/attacks/moco_cifar10"

python src/main/train_inactive_alternating_c2_target_relative_dualphase_moco.py `
  --upstream-dataset stl10 --clean-moco-checkpoint "$MoCoStl" `
  --stegastamp-checkpoint "$TriggerStart" --data-dir "$Data" `
  --epochs 20 --batch-size 64 --train-subset 5000 --val-subset 1000 --seed 42 `
  --c2-separation-margin 0.1 --c2-separation-weight 0.1 --c2-target-weight 0.2 `
  --output-dir "$Project/outputs/attacks/moco_stl10"
```

The remaining saved attack settings match the retained defaults: hider/encoder learning rates 1e-5/2e-5, alignment weight 1, and clean/reference preservation weights 5. Best exports are selected subject to validation and quality constraints; a new run may select a different best epoch than the one recorded in the original checkpoint.

The SimCLR encoder export is `backdoored_simclr_encoder_alternating_best.pth`; the MoCo export is `backdoored_moco_encoder_alternating_best.pth`. The companion trigger is `stegastamp_inactive_alternating_best.pth`. Keep each encoder and trigger from the same run and epoch.

### No-TRAD ablation

This command follows the [saved no-TRAD configuration](../configs/original/b4/results_B_strict_cifar10_upstream/config.json), including 20 epochs instead of the trainer's default 10:

```powershell
python src/main/train_inactive_alternating.py `
  --upstream-dataset cifar10 --clean-simclr-checkpoint "$SimCLRCifar" `
  --stegastamp-checkpoint "$TriggerStart" --data-dir "$Data" `
  --epochs 20 --injector-color-disentangle-weight 0 --encoder-feature-disentangle-weight 0 `
  --output-dir "$Project/outputs/attacks/no_trad_simclr_cifar10"
```

## Downstream fitting and saved-model evaluation

The filenames beginning with `evaluate_` do not all mean inference-only operation:

| Entry point | What it does |
|---|---|
| `evaluate_ssl_transfer.py` | Fits new downstream MLPs on frozen features and evaluates them; supports SimCLR/MoCo and both downstream datasets |
| `evaluate_inactive_downstream.py` | Fits new MLPs for the legacy SimCLR/CIFAR downstream protocol |
| `evaluate_validation_asr.py` | Loads saved pair MLPs and evaluates the original downstream classifier-validation partition; supports both frameworks/datasets |
| `evaluate_inactive_robustness.py` | Loads explicitly supplied saved clean/pair MLPs for the legacy SimCLR/CIFAR test path |

**There is no existing general saved-MLP test-set replay CLI for all nine evaluation groups.** Validation replay is not test-set replay. In particular, `evaluate_ssl_transfer.py --clean-only` still fits a new clean classifier.

### Fit new downstream MLPs

For example, fit new classifiers for the saved MoCo/CIFAR-10 encoder–trigger pair on STL-10:

```powershell
$MoCoPair = "$Artifacts/b4/results_moco_inactive_alternating_c2_dualphase_cifar10/checkpoints"

python src/main/evaluate_ssl_transfer.py `
  --ssl-framework moco --upstream-dataset cifar10 --downstream-dataset stl10 `
  --clean-ssl-checkpoint "$MoCoCifar" `
  --pair moco_c2_cifar10_best "$MoCoPair/backdoored_moco_encoder_alternating_best.pth" "$MoCoPair/stegastamp_inactive_alternating_best.pth" `
  --data-dir "$Data" --no-download --validation-size 500 `
  --mlp-epochs 500 --mlp-hidden-1 512 --mlp-hidden-2 256 --mlp-dropout 0.5 `
  --mlp-lr 0.0001 --mlp-weight-decay 0.0001 `
  --seed 42 --num-repeats 5 --repeat-seeds 42 43 44 45 46 `
  --output-dir "$Project/outputs/downstream/moco_cifar10_to_stl10"
```

Choose `--ssl-framework simclr` or `moco`, the appropriate upstream/downstream selectors, and the matching clean and pair checkpoint paths for other settings. CIFAR-10 downstream uses validation size 5,000; STL-10 uses 500. The original SimCLR/CIFAR downstream groups were recorded by the legacy `evaluate_inactive_downstream.py` protocol; use that retained runner when reconstructing those groups.

A corresponding legacy fitting example is:

```powershell
$SimCLRPair = "$Artifacts/b4/results_C2_dualphase_strict/checkpoints"

python src/main/evaluate_inactive_downstream.py `
  --clean-simclr-checkpoint "$SimCLRCifar" `
  --pair c2_dualphase_strict_best "$SimCLRPair/backdoored_simclr_encoder_alternating_best.pth" "$SimCLRPair/stegastamp_inactive_alternating_best.pth" `
  --data-dir "$Data" --no-download --validation-size 5000 --mlp-epochs 500 `
  --seed 42 --num-repeats 5 --repeat-seeds 42 43 44 45 46 `
  --output-dir "$Project/outputs/downstream/simclr_cifar10_to_cifar10"
```

### Evaluate saved MLPs on the downstream validation partition

The validation runner writes `validation_asr/` beneath `--results-dir`. First prepare a **fresh** replay directory containing unchanged copies of the five per-seed JSONs. This keeps the original saved records intact:

```powershell
$SourceResults = "$Project/results/original/b4/results_eval_moco_c2_cifar10_to_cifar10"
$ReplayResults = "$Project/outputs/replays/moco_cifar10_to_cifar10"
New-Item -ItemType Directory -Path "$ReplayResults/per_seed" -ErrorAction Stop | Out-Null
Copy-Item -Path "$SourceResults/per_seed/repeat_*.json" -Destination "$ReplayResults/per_seed"

python src/main/evaluate_validation_asr.py `
  --results-dir "$ReplayResults" --project-root "$Artifacts/b4" `
  --data-dir "$Data" --pair-name moco_c2_cifar10_best `
  --device cpu --num-workers 0 --check-only
```

`--check-only` checks metadata and checkpoint paths without loading models or datasets. Once the required checkpoints and data are available, run inference by omitting that flag:

```powershell
python src/main/evaluate_validation_asr.py `
  --results-dir "$ReplayResults" --project-root "$Artifacts/b4" `
  --data-dir "$Data" --pair-name moco_c2_cifar10_best `
  --device cpu --num-workers 0
```

For another group, use its own per-seed JSONs and recorded pair name. The artifact-root fallback resolves the separated classifier files. The evaluated validation partition was used to select the original MLP; it is not an independent test set or the attack-training validation split.

The original trigger-comparison metadata.json and the two original validation_asr_results.json files contain personal paths and remain local-only. Both validation_asr_summary.csv files, all aggregate/per-seed results, comparison PNGs, and preview.html remain eligible for publication. Regeneration does not read the original comparison metadata; validation replay reads per-seed records and checkpoints rather than the excluded validation-result JSONs. The preserved HTML footer still names metadata.json as a source record; that record is available only in the local copy.

### Supported saved-MLP test inference: SimCLR/CIFAR-10

This example uses the saved TRAD pair and both matching MLPs from repetition 1, seed 42. The `none` corruption condition evaluates the untransformed case:

```powershell
$SimCLRPair = "$Artifacts/b4/results_C2_dualphase_strict/checkpoints"
$SavedMLPs = "$Artifacts/b4/results_C2_dualphase_strict/downstream_cifar10/classifiers/repeat_01_seed_42"

python src/main/evaluate_inactive_robustness.py `
  --clean-simclr-checkpoint "$SimCLRCifar" `
  --clean-classifier-checkpoint "$SavedMLPs/downstream_mlp_clean_baseline.pth" `
  --pair c2_dualphase_strict_best "$SimCLRPair/backdoored_simclr_encoder_alternating_best.pth" "$SimCLRPair/stegastamp_inactive_alternating_best.pth" "$SavedMLPs/downstream_mlp_c2_dualphase_strict_best.pth" `
  --data-dir "$Data" --no-download --suites robustness --corruptions none --skip-plots `
  --output-dir "$Project/outputs/test_replay/simclr_cifar10_seed42"
```

This loads saved classifiers and does not fit them. Select the corresponding clean/pair MLPs for each additional seed. It does not provide the missing general MoCo/STL test replay functionality.

## Figures

The three chart scripts use embedded saved per-seed arrays and perform no model inference:

```powershell
python src/plots/plot_simclr_section4_4_updated.py
python src/plots/plot_moco_section4_5.py
python src/plots/plot_framework_comparison_section4_6.py
```

They write PNG/SVG files **beside the scripts under `src/plots`**, regardless of the working directory. Those exact output names are ignored. NumPy and Matplotlib suffice for these charts. MoCo chart arrays are rounded transcriptions; the original result JSONs retain higher precision.

The saved trigger comparison can be regenerated with a local dataset and checkpoint. This performs a hider forward pass and requires a new output directory:

```powershell
python src/main/compare_final_trigger_image.py `
  --root "$Artifacts/b4" `
  --results-dir "$Project/results/original/b4/results_C2_dualphase_strict/downstream_cifar10" `
  --pair-name c2_dualphase_strict_best --dataset cifar10 --index 67 `
  --data-dir "$Data" --output-dir "$Project/outputs/figures/trigger_index67" `
  --device cpu --no-open
```

The preserved example is test image 67; the script default is image 0.

Historical CA2 t-SNE is a separate experiment using a **30-bit / 0.125 residual-bound** hider, not the advanced main checkpoint. Its six matching modules remain isolated:

```powershell
python src/historical/ca2/tsne_stegastamp_learned_vs_ssl_aug.py `
  --data_dir "$Data" --dataset cifar10 --num_samples 1000 `
  --clean_encoder "$SimCLRCifar" `
  --stegastamp_checkpoint "$Artifacts/ca2/results_stegastamp_learned/stegastamp_like_plain_issba.pth" `
  --figure_path "$Project/outputs/figures/historical_ca2_tsne.png" `
  --metrics_path "$Project/outputs/figures/historical_ca2_tsne_metrics.json"
```

This performs model inference and a new t-SNE computation. The saved metrics specify 1,000 points per group, while the script defaults to 3,000. Exact stochastic reconstruction of the preserved figure is not established; the original image remains the historical evidence.

## Reproducibility limits

- **Verification scope:** the command interfaces have been checked against the retained scripts. Training and inference have not been rerun to establish numerical reproduction.
- **Environment and clean-model provenance:** requirements are unpinned; original hardware, driver/kernel details, and every experiment-time package version are unavailable. The bare clean CIFAR SimCLR checkpoint lacks complete historical metadata.
- **Training versus replay:** upstream/attack training and downstream MLP fitting create new models. Saved validation inference and the limited legacy test runner load existing MLPs. None establishes a general saved-MLP test replay for all nine groups.
- **Phase history and checkpoint selection:** older warm-up/weak-stealth configs omit newer trainer options. The warm-up CSV contains appended training blocks whose first two identities are uncertain. New runs may select different epochs or fail the best-export constraints.
- **Normalization and protocol differences:** clean CIFAR pretraining and advanced attack/evaluation helpers retain different historical CIFAR standard deviations. Preserve the implemented transforms and split routines rather than harmonizing them. Derived initialization, loader, and dropout seed streams are not interchangeable with the repeat seed.
- **Resume support:** selected attack exports are encoder/trigger evaluation weights, not joint optimizer resume states. Clean trainers expose no resume CLI. Advanced StegaStamp has phase-specific resume support, but complete RNG-stream restoration and identical uninterrupted trajectories are not established.
- **Record portability:** original configurations/results retain historical path strings. Use explicit relocated paths and the documented artifact-root relationship. Some optional confusion PNG/CSV exports were intentionally omitted; numerical confusion matrices remain in per-seed JSON.
- **Historical scope:** CA2 t-SNE uses an older architecture and settings. Its trainer/loss are outside this retained subset, and the figure's exact stochastic projection settings are not fully established.

The original JSON/CSV records are the reference for the reported results; new runs should write to separate output directories.


