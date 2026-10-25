# Replay, path relocation and environment requirements

Run examples from the repository root using the appropriate experiment environment. Paths below are repository-relative unless an explicit input path is supplied. Numerical reproduction has not been verified.

## One source tree and explicit inputs

Use src/main for the 26-file B4 pipeline, src/plots for three self-contained recovered charts, and src/historical/ca2 for the six old Figure 3 files. Execute a script by its canonical repository-relative path from the repository root; do not copy source into runtime directories or put both main and CA2 helper directories together on PYTHONPATH. Local artifacts and any future Git checkout use those same files.

Original records are immutable provenance. Keep all 8 config.json files, 13 training/validation CSV logs, saved aggregate/per-seed/validation files, checkpoint names, pair names and seed plans unchanged. Use the copy manifest and PATH_RELOCATION_AUDIT.csv as external mappings. The public/local split changes storage paths only, not model settings or numerical values.

## A. Evaluation replay with saved checkpoints

There are two different goals: reading already-saved results requires no model execution; numerical replay performs fresh inference with the saved encoder, trigger and MLP. Neither means training another classifier.

| Entry point | Actual behavior | Proposed use / limit |
|---|---|---|
| src/main/evaluate_validation_asr.py | loads original per-seed records and saved pair MLPs; reconstructs the original downstream validation partition | supports both frameworks and both downstream datasets; retained Table 8 results use MoCo/CIFAR-downstream groups |
| src/main/evaluate_inactive_robustness.py | loads explicitly supplied saved clean/pair classifiers and runs inference | legacy SimCLR/CIFAR test path; choose robustness suite with corruption none for the untransformed condition, and use a fresh output directory |
| src/main/evaluate_ssl_transfer.py | fits new clean and pair MLPs, then evaluates | training/evaluation reproduction, NOT saved-MLP replay, including --clean-only |
| src/main/evaluate_inactive_downstream.py | legacy CIFAR downstream classifier fitting plus shared helper APIs | source dependency / original protocol; NOT a general saved-MLP test runner |

No supplied CLI is a general saved-MLP test-set replay for every MoCo/STL setting. The recovered validation script closes the validation replay gap only. Do not claim the selected files make all nine test rows directly rerunnable through one existing command. Full test replay would need a separately reviewed new inference entry point that uses the unchanged saved models, classifier states, transforms and split protocol; that implementation is outside this planning stage. No new Python is authored here.

### Validation path resolution without record edits

All 225 live model/classifier references in the 45 selected per-seed JSONs are relative and map to selected manifest destinations:

| Original recorded form | New storage relationship |
|---|---|
| ./results_<run>/checkpoints/<original>.pth | artifacts/b4/results_<run>/checkpoints/<original>.pth |
| results_<evaluation>/classifiers/repeat_XX_seed_YY/<original>.pth | artifacts/b4/results_<evaluation>/classifiers/repeat_XX_seed_YY/<original>.pth |
| ../code/results/simclr_cifar10_ep1000.pth | artifacts/code/results/simclr_cifar10_ep1000.pth, relative to artifacts/b4 |
| ./results_moco_cifar10 or ./results_moco_stl10 clean files | same original folder suffix below artifacts/b4 |
| ./results_simclr_stl10 clean file | same original folder suffix below artifacts/b4 |

Set the existing --project-root argument to artifacts/b4. The validation script tries --results-dir/classifiers first, then the original recorded classifier path under that project root. The first location can be absent because the selected MLPs are in the parallel artifact tree. Do not rename any pair or classifier file. The shared evaluator checks preserved checkpoint basenames; the legacy SimCLR checker verifies the final three path components, which this layout also preserves. No verification bypass flags are proposed.

The script writes validation_asr under --results-dir and has no separate output-root argument. Therefore a future approved replay should:

1. Create a NEW outputs/replays/<run>/<original-evaluation-name>/per_seed directory.
2. Populate it with unchanged copies of only the five relevant original per-seed JSONs. This is disposable metadata, not a second Python tree and not a rewrite of records.
3. Invoke the canonical src/main/evaluate_validation_asr.py with this new results directory, --project-root pointing to artifacts/b4, and explicit --data-dir.
4. Compare newly generated results with the preserved original validation records and test ASRs. Never point an inference run at results/original when it would overwrite its original validation_asr files.

Nothing in these steps has been performed now. The initial copy manifest excludes disposable replay metadata. A future documented command form is:

```powershell
python src/main/evaluate_validation_asr.py `
  --results-dir outputs/replays/new-run/results_eval_moco_c2_cifar10_to_cifar10 `
  --project-root artifacts/b4 `
  --data-dir data `
  --pair-name moco_c2_cifar10_best `
  --device cpu --num-workers 0
```

This is an inference command for a later separately authorized stage, not a command run here. --check-only is metadata-only but was also not executed. CPU/CUDA execution can differ numerically; the script checks clean validation accuracy against the saved classifier-selection value with a one-decision tolerance. The original reported numbers remain authoritative records.

### Saved downstream protocol

| Downstream dataset | Fit | Validation | Test | Non-target test |
|---|---:|---:|---:|---:|
| CIFAR-10 | 45,000 | 5,000 | 10,000 | 9,000 |
| STL-10 | 4,500 | 500 | 8,000 | 7,200 |

Preserve global seed 42, split seed 53, repeats 42-46, their derived initialization/DataLoader/dropout seed plans and the actual selected MLP epoch for each file. Do not substitute repeat seed for every derived RNG seed. The saved validation records contain 5,000 actual CIFAR indices, SHA256 4029a55f7bf2c011cf587215ab31b0b4e4512f08642a7cb0fea32ef14fe15998, and 4,513 non-target validation examples. Validation ASR uses the original classifier-selection validation partition; it is not a new independent test set or the joint attack-training validation set.

The MoCo/CIFAR attack export selected by both downstream tasks remains epoch **16**. Other attack pairs are epoch 20. Retained MLPs use the backbone before projection and L2-normalized 512-dimensional features; target label 0 and original target-class exclusions stay unchanged. Saved standard deviations use sample SD.

## B. Reproducing figures using saved results

| Figure/source | Inputs | Executes model inference? | Output/path treatment |
|---|---|---|---|
| Three scripts under src/plots | embedded saved per-seed ASR arrays; no external result JSON input required | No | writes adjacent PNG/SVG via __file__, regardless of cwd; exact generated names are ignored by proposed .gitignore |
| Preserved bird comparison PNGs and preview.html | saved comparison images and HTML preview | No, when simply viewing originals | keep saved PNGs and the HTML preview together in figures/original/b4/final_trigger_comparison_20261024_162348 |
| src/main/compare_final_trigger_image.py | selected aggregate, trigger checkpoint and CIFAR test index 67 | Yes, a new hider forward pass | --root artifacts/b4; absolute --results-dir, --data-dir and NEW --output-dir; --index 67 and --no-open as appropriate |
| Preserved Figure 3 PNG | existing original matching t-SNE image | No | keep saved plot as original evidence |
| src/historical/ca2/tsne_stegastamp_learned_vs_ssl_aug.py | old CA2 hider, clean CIFAR SimCLR weights and CIFAR images | Yes, plus new t-SNE computation | explicit --clean_encoder, --stegastamp_checkpoint, --figure_path and --metrics_path; output only to a new location |

The three plot scripts use sample SD (ddof=1). SimCLR arrays match the selected full-precision records; MoCo arrays were transcribed at two decimal places. Plot regeneration therefore preserves the supplied script behavior but does not replace the higher-precision original aggregate JSON. The final PNG/SVG exports for these chart scripts were not located; do not pretend they are included in the manifest.

The chart output filenames beside src/plots are: simclr_section4_4_asr_updated.png/.svg, moco_section4_5_asr.png/.svg and framework_comparison_section4_6.png/.svg. They are not another source-code version. A future decision can select generated figures into the tracked figures tree without changing the canonical scripts.

For the bird script, default --index 0 produces a different image. The selected original is CIFAR test index 67, bird/class 2, using the selected SimCLR/CIFAR epoch-20 trigger. Its saved trigger hash is b9189db2eee8b292fe012f8d8ad4299145b303945070b1eaae1e20e50f8a8443. Original metadata contains personal historical paths and remains unchanged locally; the exact metadata.json file is excluded from GitHub. --root resolves the live relative trigger and companion-encoder paths, while --results-dir may be an absolute path to the separated saved aggregate.

Figure 3 keeps its historical checkpoint at artifacts/ca2/results_stegastamp_learned/stegastamp_like_plain_issba.pth, epoch 59, **30 bits / max_residual 0.125**. Do not load it with advanced main helpers or apply the final 16-bit/8/255 settings. Saved metrics describe 1,000 points per group, 2,000 total, CIFAR-10, SSL augmentation simclr, target label 0 and non_target_only=false. The script default is 3,000 points per group; it must not silently replace the saved figure settings. Seed 123/perplexity 30 are script defaults, not proof of the exact historical projection settings. The saved matching PNG is preserved because full stochastic regeneration is not established. The old CA2 trainer/loss are not selected; historical Figure 3 training from scratch is outside this source subset.

The original Figure 3 metrics refer to a CA2 clean checkpoint copy. It is full-SHA256-identical to the retained CA3 clean file; pass the retained artifacts/code/results path explicitly without editing the original metrics. No extra checkpoint copy is added.

## C. Reproducing main training from scratch

The retained B4 trainers support clean SimCLR/MoCo v1, progressive advanced StegaStamp phases, final dual-phase SimCLR/MoCo attacks and the no-TRAD SimCLR ablation. The historical experiment remains MoCo v1; do not migrate the projection head or convert checkpoint state.

| Stage | Canonical source | Original configuration evidence |
|---|---|---|
| Clean SSL | src/main/train_simclr.py and train_moco.py | selected epoch-1000 checkpoint metadata; legacy clean requirements; bare CIFAR SimCLR provenance is incomplete |
| Warm-up -> weak stealth -> mild corruption | src/main/train_stegastamp_like.py | three original configs/logs, selected boundary epochs 48 -> 25 -> 6 |
| No-TRAD SimCLR/CIFAR | src/main/train_inactive_alternating.py | original results_B_strict_cifar10_upstream config/logs; both disentanglement terms zero |
| SimCLR TRAD, CIFAR/STL upstream | src/main/train_inactive_alternating_c2_target_relative_dualphase.py | the two strict final configs/logs |
| MoCo v1 TRAD, CIFAR/STL upstream | src/main/train_inactive_alternating_c2_target_relative_dualphase_moco.py | the two original MoCo attack configs/logs |
| Downstream MLP fitting | src/main/evaluate_inactive_downstream.py for original legacy CIFAR records; evaluate_ssl_transfer.py for shared transfer cases | original per-seed protocol, architecture, seed plans, counts and 500-epoch histories |

Do not run these stages for saved-MLP replay. They create new trained models even with frozen encoders during downstream fitting. configs/original is evidence, not a directory to overwrite. The runners do not provide a general --config JSON interface; future launch arguments must be transcribed from each exact saved config with explicit path overrides and checked against the original CLI. Clean trainer options use --data_dir/--save_dir, while main attack/replay options commonly use hyphens; Figure 3 uses underscores. Do not assume one spelling convention.

The no-TRAD trainer default is 10 epochs, while its selected original run uses 20. A future reconstruction must explicitly use the saved config rather than changing defaults. Main attack settings retain batch 64, subsets 5,000/1,000, seed 42, hider/encoder LR 1e-5/2e-5, preservation 5 and alignment 1. TRAD uses margin 0.1, separation 0.1 and target 0.2. Phase schedules and all other coefficients remain as originally saved. New run output paths may change; hyperparameters and datasets/splits must not.

Exact from-scratch numerical reproduction is not established: original hardware/drivers and every historical library version are not recorded, the bare clean CIFAR SimCLR checkpoint lacks embedded epoch/seed provenance, and stochastic operation order matters. The supplied code/configs describe the experiment without guaranteeing bitwise reproduction. The earlier B3 strict-protocol questions are resolved for the selected final numeric records; do not substitute B3 test-validation attack code for those B4 strict runs.

### Preserve existing normalization differences

| Stage/helper | Mean | Standard deviation |
|---|---|---|
| Main clean CIFAR dataset transforms | (0.4914, 0.4822, 0.4465) | (0.2023, 0.1994, 0.2010) |
| Advanced main CIFAR helpers used by attack/shared evaluation | (0.4914, 0.4822, 0.4465) | (0.2470, 0.2435, 0.2616) |
| Main STL transforms/helpers | (0.4467, 0.4398, 0.4066) | (0.2603, 0.2566, 0.2713) |
| Historical CA2 CIFAR helper | (0.4914, 0.4822, 0.4465) | (0.2023, 0.1994, 0.2010) |

These are observations of the retained source, not a harmonization proposal. Preserve each original stage. Trigger insertion remains on raw [0,1] images before the corresponding normalization. Preserve original STL resizing/augmentation implementation and all classifier/attack dataset selection routines.

## D. Resuming training from checkpoint state

| Stage | What the selection contains | Resume support and limit |
|---|---|---|
| Three advanced StegaStamp phases | selected best/final boundary checkpoints; small data.pkl metadata contains optimizer_state_dict and scheduler_state_dict | trainer --resume restores epoch+1, optimizer, scheduler and best metrics; only same-phase continuation with original settings. --initialize-from instead resets optimizer/scheduler/epoch for the next phase |
| Five final attack pairs | evaluation encoder/trigger exports | exports are not joint optimizer states. The five optional inactive_alternating_best.pth joint files are not in the approved selection |
| Clean MoCo | epoch-1000 payload contains optimizer and scheduler keys | unchanged train_moco.py exposes no --resume CLI; full training continuation is not provided merely by retaining these keys |
| Clean SimCLR | selected final clean file; CIFAR file is bare state dict | unchanged trainer has no --resume CLI; full optimizer/RNG continuation is not established |
| Downstream MLPs | 90 selected best classifier exports | inference weights and selection metadata, not an approved generic MLP-resume workflow |
| Historical Figure 3 | old hider checkpoint with optimizer metadata | old trainer/loss excluded; no historical training-resume workflow is promised |

The advanced warm-up epoch-48 and mild epoch-6 states can describe continuation within their original 50/20-epoch phases. Weak-stealth epoch 25 is the completed boundary of its 25-epoch phase; starting mild is initialize-from, not resuming the old optimizer into a new phase. Changing total epochs to continue past a completed run is a new experiment, not replay of the original training. No Python/NumPy/torch/CUDA RNG-state keys were found in the inspected checkpoint metadata, and the selected save/load code does not restore full RNG streams; optimizer resume is not a guarantee of an identical uninterrupted trajectory.

The five original joint attack-resume checkpoints are not included. If supplied separately, use these repository-relative local destinations:
- `artifacts/b4/results_B_strict_cifar10_upstream/checkpoints/inactive_alternating_best.pth` — 203,686,475 bytes.
- `artifacts/b4/results_C2_dualphase_strict/checkpoints/inactive_alternating_best.pth` — 203,688,139 bytes.
- `artifacts/b4/results_C2_dualphase_strict_stl10_upstream/checkpoints/inactive_alternating_best.pth` — 203,688,139 bytes.
- `artifacts/b4/results_moco_inactive_alternating_c2_dualphase_cifar10/checkpoints/inactive_alternating_best.pth` — 203,704,907 bytes.
- `artifacts/b4/results_moco_inactive_alternating_c2_dualphase_stl10/checkpoints/inactive_alternating_best.pth` — 203,704,907 bytes.

They total 1,018,472,567 bytes and contain separate hider/encoder optimizer keys. Keep their original run/checkpoints/filename suffixes if later added.

## External datasets and auxiliary assets

| Dataset/assets | Needed for | Location/selection rule |
|---|---|---|
| CIFAR-10 Python batches: official 50,000 training and 10,000 test examples | CIFAR downstream validation/test replay, clean CIFAR pretraining, main CIFAR/trigger training and Figure 3 | external data/cifar-10-batches-py; preserve torchvision data ordering and original split/seed routines |
| STL-10 binary labeled train (5,000) and test (8,000) | STL downstream validation/test replay and labeled references for the STL attack | external data/stl10_binary; preserve original 32x32 resizing path |
| STL-10 unlabeled (100,000) | clean STL train+unlabeled SSL and original STL attack shadow selection | same external dataset tree; not needed merely to load saved downstream validation images |
| Optional LPIPS network weights | only explicitly enabled LPIPS calculations/training | not required for the selected basic replay/plots; do not silently enable LPIPS or download auxiliary weights |

One complete external dataset tree is sufficient; do not copy every duplicated original data directory. Existing inventory lists about 6,121,394,153 bytes in the B4 data folder, including its stored dataset files/containers; exact minimum dataset storage is not asserted. No raw data is selected in the copy manifest. Existing CIFAR/STL files are expected to be supplied later, with downloads disabled for replay. The CPU/GPU used for inference and library kernels must be verified in a future numerical validation stage.

## Dependencies and observed environments

Original B4 requirements list torch, torchvision, numpy, matplotlib, scikit-learn, pillow, tqdm and lpips without pins. The original CA2 list contains the same base packages without lpips. NumPy/Matplotlib suffice for the three embedded-data charts. PyTorch/torchvision and image/array support are necessary for checkpoint inference; scikit-learn is needed for t-SNE. Keep the legacy clean requirements freeze separately; it is UTF-16LE and contains unrelated packages, not a replacement main lockfile.

Read-only metadata observed on 25 October 2026:

| Package | B4 main / historical CA2 observed | Legacy CA3 clean observed |
|---|---|---|
| torch | 2.7.1+cu118 | 2.5.1+cu121 |
| torchvision | 0.22.1+cu118 | 0.20.1+cu121 |
| numpy | 2.4.4 | 2.4.3 |
| matplotlib | 3.10.9 | 3.10.8 |
| scikit-learn | 1.8.0 | 1.8.0 |
| pillow | 12.2.0 | 12.1.1 |
| tqdm | 4.67.3 | 4.67.3 |
| lpips | 0.1.4 (main only; absent in CA2) | 0.1.4 |
| scipy | 1.17.1 | 1.17.1 |

The B4 and CA2 pyvenv.cfg files record Python **3.11.4**; the legacy clean file also records 3.11.4. The B4/CA2 package metadata reports torch 2.7.1+cu118 / torchvision 0.22.1+cu118; main torch/version.py records CUDA 11.8. Legacy clean metadata reports torch 2.5.1+cu121 / torchvision 0.20.1+cu121. These are currently installed observations, not proof of the environment that produced every checkpoint. The creation commands inside copied pyvenv.cfg files still refer to older CA2 locations; do not treat those paths as portable executables or activate a copied environment.

The manifest proposes preserving only original requirements and three small pyvenv.cfg observations, never .venv binaries. Do not merge the two CUDA/PyTorch versions into one invented tested lockfile. No NVIDIA driver, GPU model, original cuDNN build or complete experiment-time freeze was recovered. A future approved replay should verify an environment against the preserved records before claiming numerical reproduction; no environment creation, pip invocation or import of experiment libraries occurred here.

## Path audit and immutable originals

PATH_RELOCATION_AUDIT.csv enumerates **487 JSON/CSV path fields and 8 script assumptions**. Its categories are 289 selected-target mappings, 1 exact-duplicate clean-checkpoint redirect, 180 omitted optional confusion exports, 9 external dataset references, 8 historical output paths and 8 script assumptions. The 225 per-seed model/classifier paths all map to selected artifacts. An old absolute provenance path may no longer be clickable after relocation; its text remains preserved and the audit provides the new target.

Keep numerical confusion matrices from original per-seed JSONs; missing optional confusion PNG/CSV paths are not reasons to edit those JSONs. Keep the saved comparison PNGs and preview.html together; original metadata.json remains local-only. The earlier approved results audit and source audit are retained as dated local provenance; this guide supersedes their now-resolved missing-artifact/replay statements without altering those files.

These path mappings are static documentation and do not establish runtime compatibility or numerical reproduction. Preserve original experimental records and write new run outputs separately.
