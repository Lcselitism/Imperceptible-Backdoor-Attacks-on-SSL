"""Evaluate downstream validation ASR with saved SimCLR/MoCo classifiers.

Place this file beside evaluate_ssl_transfer.py and run from your project root:
    python evaluate_validation_asr.py --results-dir RESULTS_FOLDER --data-dir ./data

RESULTS_FOLDER must contain the original per_seed JSONs and classifiers/ weights.
Several results folders can follow --results-dir. Framework, upstream/downstream
datasets, split sizes, seeds and model paths are read from those JSONs.
--check-only checks the recorded metadata and checkpoint paths without inference.

No encoder, trigger or MLP is trained. Images come only from the downstream
training split's existing validation partition, and original target-class images
are excluded from ASR. This is post-hoc downstream validation ASR: it is neither
the 1,000-image joint attack-validation diagnostic nor a new independent test.
Classifiers were already selected using clean accuracy on this validation set.
Results are saved in RESULTS_FOLDER/validation_asr/; source results stay intact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def portable_parts(value):
    return str(value).replace("\\", "/").split("/")


def resolve_checkpoint(value, project_root, preferred=None):
    """Resolve recorded Windows/POSIX paths, including a moved project folder."""
    if not value:
        raise ValueError("A required checkpoint path is absent from the result JSON.")
    parts = portable_parts(value)
    normalized = Path(str(value).replace("\\", "/"))
    candidates = [preferred] if preferred is not None else []
    candidates += [project_root / normalized, normalized]
    # An absolute path may refer to the previous location of this same project.
    for index, part in enumerate(parts):
        if part.lower().startswith("results"):
            candidates.append(project_root.joinpath(*parts[index:]))
            break
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    tried = "\n  ".join(str(p) for p in candidates if p is not None)
    raise FileNotFoundError(f"Checkpoint not found: {value}\nTried:\n  {tried}")


def canonical_dataset(value):
    name = re.sub(r"[\s_-]", "", str(value)).lower()
    if name not in ("cifar10", "stl10"):
        raise ValueError(f"Unsupported or absent recorded dataset: {value!r}")
    return name.upper()


def resolve_dataset_protocol(recorded):
    """Recover legacy SimCLR datasets from saved metadata, never folder names."""
    protocol = dict(recorded)
    sources = {}
    downstream = recorded.get("downstream_dataset")
    if downstream is None:
        if recorded.get("feature_source") != "model.f":
            raise ValueError("Missing downstream_dataset in non-SimCLR results.")
        downstream = recorded.get("dataset")
        sources["downstream_dataset"] = "protocol.dataset"
    protocol["downstream_dataset"] = canonical_dataset(downstream)

    upstream = recorded.get("upstream_dataset")
    if upstream is None:
        if recorded.get("feature_source") != "model.f":
            raise ValueError("Missing upstream_dataset in non-SimCLR results.")
        matches = re.findall(
            r"\b(cifar[\s_-]*10|stl[\s_-]*10)(?:\s+encoder)?\s+normalization\b",
            str(recorded.get("normalization_order", "")), flags=re.IGNORECASE)
        datasets = {canonical_dataset(value) for value in matches}
        if len(datasets) != 1:
            raise ValueError("Cannot determine legacy upstream dataset from the recorded "
                             "normalization_order; provide the original per-seed JSON for review.")
        upstream = datasets.pop()
        sources["upstream_dataset"] = "protocol.normalization_order"
    protocol["upstream_dataset"] = canonical_dataset(upstream)
    return protocol, sources


def make_job(results_dir, project_root, pair_name=None):
    """Read original per-seed records; preflight every checkpoint before work."""
    results_dir = Path(results_dir).resolve()
    files = sorted((results_dir / "per_seed").glob("repeat_*.json"))
    if not files:
        raise FileNotFoundError(f"No per_seed/repeat_*.json files in {results_dir}")
    records = [(path, read_json(path)) for path in files]
    first = records[0][1]
    if pair_name is None:
        names = list(first.get("pairs", {}))
        if len(names) != 1:
            raise ValueError("This results folder contains multiple pairs; specify --pair-name.")
        pair_name = names[0]
    recorded_protocol = first["protocol"]
    protocol, dataset_sources = resolve_dataset_protocol(recorded_protocol)
    counts = first["data_counts"]
    source = protocol["feature_source"]
    framework = {"encoder_q_backbone_pre_projection": "moco", "model.f": "simclr"}.get(source)
    if framework is None:
        raise ValueError(f"Unsupported feature source: {source!r}")
    if protocol.get("projection_head_used") or not protocol.get("feature_l2_normalized"):
        raise ValueError("Expected normalized backbone features before projection.")
    if protocol.get("message_pattern") != "alternating" or protocol.get("message_length") != 16:
        raise ValueError("Expected the project's fixed alternating 16-bit trigger.")
    if not math.isclose(float(protocol["maximum_residual"]), 8 / 255, abs_tol=1e-8):
        raise ValueError("Expected the 8/255 trigger residual bound.")
    if int(protocol.get("image_size", -1)) != 32 or int(protocol["target_label"]) != 0:
        raise ValueError("Expected 32x32 inputs and airplane target label 0.")
    seed = int(protocol["global_experiment_seed"])
    if int(protocol["data_split_seed"]) != seed + 11:
        raise ValueError("Unexpected downstream split seed; cannot reproduce this run.")
    if int(counts["classifier_fit"]) < 1 or int(counts["classifier_validation"]) < 1:
        raise ValueError("Invalid classifier fit/validation counts.")
    runs = []
    model_paths = None
    for path, record in records:
        if record["protocol"] != recorded_protocol or record["data_counts"] != counts:
            raise ValueError(f"Mixed protocols or dataset splits in {path}")
        repetition = record["repetition"]
        index, repeat_seed = int(repetition["index"]), int(repetition["repeat_seed"])
        if int(repetition["global_experiment_seed"]) != seed:
            raise ValueError(f"Repetition/global split seed disagreement in {path}")
        pair = record["pairs"][pair_name]
        if pair["pair_name"] != pair_name:
            raise ValueError(f"Pair identity disagreement in {path}")
        baseline = record["clean_baseline"]
        clean_path = baseline.get("clean_ssl_checkpoint", baseline.get("clean_simclr_checkpoint"))
        paths = (
            resolve_checkpoint(pair["encoder_checkpoint"], project_root),
            resolve_checkpoint(pair["trigger_checkpoint"], project_root),
            resolve_checkpoint(clean_path, project_root),
        )
        if model_paths is not None and paths != model_paths:
            raise ValueError("The per-seed files do not reference the same attack pair.")
        model_paths = paths
        training = pair["downstream_classifier_training"]
        filename = portable_parts(training["checkpoint"])[-1]
        preferred = results_dir / "classifiers" / f"repeat_{index:02d}_seed_{repeat_seed}" / filename
        classifier_path = resolve_checkpoint(training["checkpoint"], project_root, preferred)
        runs.append(dict(source_json=str(path), record=record, pair=pair,
                         seed=repeat_seed, repetition=index, classifier_path=classifier_path))
    runs.sort(key=lambda run: run["repetition"])
    expected = [int(s) for s in protocol["repeat_seeds"]]
    if ([run["seed"] for run in runs] != expected or
            [run["repetition"] for run in runs] != list(range(1, len(expected) + 1))):
        raise ValueError("Missing, duplicated or unexpected per-seed result files.")
    if int(protocol["num_repeats"]) != len(runs):
        raise ValueError("num_repeats disagrees with the per-seed files.")
    return dict(results_dir=results_dir, pair_name=pair_name, protocol=protocol,
                recorded_protocol=recorded_protocol, dataset_metadata_sources=dataset_sources,
                counts=counts, framework=framework, runs=runs, model_paths=model_paths)


def build_validation_data(job, cli, ev):
    """Use the original evaluator's split helpers, without opening a test split."""
    downstream = job["protocol"]["downstream_dataset"].lower()
    if downstream == "cifar10":
        base = ev.datasets.CIFAR10(root=cli.data_dir, train=True,
                                  download=cli.download, transform=ev.transforms.ToTensor())
    elif downstream == "stl10":
        transform = ev.transforms.Compose([
            ev.transforms.Resize((32, 32)), ev.transforms.ToTensor()])
        base = ev.datasets.STL10(root=cli.data_dir, split="train",
                                download=cli.download, transform=transform)
    else:
        raise ValueError(f"Unsupported downstream dataset: {downstream}")
    counts = job["counts"]
    selected_size = int(counts["classifier_fit"]) + int(counts["classifier_validation"])
    if selected_size > len(base):
        raise ValueError("Recorded classifier data count exceeds this dataset.")
    protocol = job["protocol"]
    selected = ev.deterministic_indices(len(base), selected_size,
                                        int(protocol["global_experiment_seed"]))
    fit, validation = ev.split_train_validation_indices(
        selected, int(counts["classifier_validation"]), int(protocol["data_split_seed"]))
    if len(fit) != int(counts["classifier_fit"]) or set(fit).intersection(validation):
        raise RuntimeError("Could not reconstruct the disjoint fit/validation split.")
    loader = ev.make_loader(base, validation, cli.feature_batch_size, cli.num_workers,
                            int(protocol["global_experiment_seed"]) + 102)
    return loader, validation


def extract_validation_features(model, trigger, config, loader, args, device, ev, torch):
    """Extract clean features and triggered non-target features once per pair."""
    clean_features, labels_all, triggered_features = [], [], []
    model.eval()
    trigger.eval()
    with torch.no_grad():
        for batch_index, (raw_images, labels) in enumerate(loader):
            raw_images = raw_images.to(device, non_blocking=True)
            if raw_images.min().item() < -1e-6 or raw_images.max().item() > 1 + 1e-6:
                raise ValueError("Validation images must be raw tensors in [0,1].")
            clean = ev.encode_features(model, ev.normalize_for_upstream(
                raw_images, args.upstream_dataset))
            clean_features.append(clean.cpu())
            labels_all.append(labels.long().cpu())
            mask = labels.to(device).ne(args.target_label)
            if bool(mask.any()):
                outputs = ev.generate_trigger_batch(
                    clean_image=raw_images[mask], model=trigger, config=config,
                    message_pattern="alternating", decode_message=False,
                    return_cpu=False, check_range=True)
                if outputs["residual"].abs().max().item() > 8 / 255 + 1e-6:
                    raise RuntimeError("Generated validation trigger exceeded 8/255.")
                encoded = ev.normalize_for_upstream(outputs["encoded_image"], args.upstream_dataset)
                triggered_features.append(ev.encode_features(model, encoded).cpu())
            print(f"  Features: batch {batch_index + 1}/{len(loader)}", flush=True)
    if not triggered_features:
        raise ValueError("No non-target validation images remain.")
    return (torch.cat(clean_features), torch.cat(labels_all), torch.cat(triggered_features))


def load_classifier(run, feature_dim, device, ev):
    payload = ev.safe_torch_load(str(run["classifier_path"]), "cpu")
    training = run["pair"]["downstream_classifier_training"]
    for key in ("best_epoch", "feature_dim", "seed_plan"):
        if payload.get(key) != training.get(key):
            raise ValueError(f"Classifier {key} does not match its JSON: {run['classifier_path']}")
    if int(payload["feature_dim"]) != feature_dim:
        raise ValueError("Saved MLP feature dimension does not match the encoder.")
    if not math.isclose(float(payload["best_validation_accuracy"]),
                        float(training["best_validation_accuracy"]), abs_tol=1e-8):
        raise ValueError("Saved classifier validation accuracy does not match its JSON.")
    arch = payload["architecture"]
    if int(arch["num_classes"]) != 10:
        raise ValueError("Expected a 10-class downstream classifier.")
    args = SimpleNamespace(mlp_hidden_1=int(arch["hidden_1"]),
                           mlp_hidden_2=int(arch["hidden_2"]), mlp_dropout=float(arch["dropout"]))
    return ev.load_saved_classifier(payload["classifier_state_dict"], feature_dim, args, device)


def run_job(job, cli, ev, torch):
    started = time.perf_counter()
    protocol = job["protocol"]
    seed = int(protocol["global_experiment_seed"])
    ev.set_random_seed(seed)
    ev.CLASS_NAMES = (ev.CIFAR10_CLASS_NAMES if protocol["downstream_dataset"].lower() == "cifar10"
                      else ev.STL10_CLASS_NAMES)
    device = ev.resolve_device(cli.device)
    encoder_path, trigger_path, clean_path = job["model_paths"]
    args = SimpleNamespace(ssl_framework=job["framework"], arch="resnet18", projection_dim=128,
                           upstream_dataset=protocol["upstream_dataset"].lower(),
                           clean_ssl_checkpoint=str(clean_path), clean_simclr_checkpoint=str(clean_path),
                           target_label=int(protocol["target_label"]),
                           allow_unverified_pair=False, allow_unsafe_finish=False)
    print(f"\n{job['framework'].upper()} {protocol['upstream_dataset']} -> "
          f"{protocol['downstream_dataset']} | pair={job['pair_name']} | device={device}", flush=True)
    model, encoder_checkpoint = ev.load_backdoored_ssl_model(
        str(encoder_path), args, device, role="validation ASR encoder")
    trigger, config, trigger_checkpoint = ev.load_stegastamp_model(
        checkpoint_path=str(trigger_path), device=device, freeze=True, strict=True)
    pair_spec = ev.PairSpec(job["pair_name"], str(encoder_path), str(trigger_path))
    metadata = ev.validate_matched_pair_metadata(pair_spec, encoder_checkpoint, trigger_checkpoint, args)
    ev.validate_trigger_config(job["pair_name"], config, trigger_checkpoint)
    saved_metadata = job["runs"][0]["pair"]["pair_metadata"]
    for key in ("encoder_epoch", "trigger_epoch", "encoder_stage", "trigger_stage"):
        if metadata.get(key) != saved_metadata.get(key):
            raise ValueError(f"Loaded attack {key} differs from the evaluated pair.")
    loader, indices = build_validation_data(job, cli, ev)
    feature_started = time.perf_counter()
    clean_features, labels, triggered_features = extract_validation_features(
        model, trigger, config, loader, args, device, ev, torch)
    feature_seconds = time.perf_counter() - feature_started
    mask = labels.ne(args.target_label)
    if triggered_features.size(0) != int(mask.sum()):
        raise RuntimeError("Triggered validation features and non-target labels disagree.")
    rows, details = [], []
    print("  Seed  Clean val  Val ASR  Baseline  Adjusted  Test ASR  Val-test (pp)")
    for run in job["runs"]:
        classifier = load_classifier(run, int(clean_features.size(1)), device, ev)
        clean = ev.evaluate_classifier(classifier, clean_features, labels,
                                       cli.mlp_eval_batch_size, device)
        training = run["pair"]["downstream_classifier_training"]
        expected = float(training["best_validation_accuracy"])
        # Allow one changed decision for device-dependent floating-point rounding.
        if abs(clean.accuracy - expected) > 1 / len(indices) + 1e-8:
            raise ValueError(f"Seed {run['seed']}: clean validation accuracy is "
                             f"{100 * clean.accuracy:.4f}%, expected {100 * expected:.4f}%. "
                             "Check the saved classifier, encoder and original split before reporting ASR.")
        baseline = ev.evaluate_target_rate(classifier, clean_features[mask], args.target_label,
                                            cli.mlp_eval_batch_size, device)
        asr = ev.evaluate_target_rate(classifier, triggered_features, args.target_label,
                                       cli.mlp_eval_batch_size, device)
        dominant = max(range(10), key=lambda index: asr.prediction_distribution[index])
        row = dict(seed=run["seed"], selected_mlp_epoch=int(training["best_epoch"]),
                   clean_validation_accuracy_percent=100 * clean.accuracy,
                   saved_clean_validation_accuracy_percent=100 * expected,
                   validation_baseline_target_rate_percent=100 * baseline.target_rate,
                   validation_asr_percent=100 * asr.target_rate,
                   validation_adjusted_asr_percent=100 * ev.compute_adjusted_asr(
                       asr.target_rate, baseline.target_rate),
                   original_test_asr_percent=float(run["pair"]["asr"]["target_rate_percent"]),
                   dominant_triggered_prediction=ev.CLASS_NAMES[dominant],
                   dominant_triggered_prediction_percent=100 * asr.prediction_distribution[dominant] / asr.total_samples)
        row["validation_minus_test_asr_pp"] = row["validation_asr_percent"] - row["original_test_asr_percent"]
        print(f"  {run['seed']:4d}  {row['clean_validation_accuracy_percent']:9.2f} "
              f"{row['validation_asr_percent']:8.2f} "
              f"{row['validation_baseline_target_rate_percent']:9.2f} "
              f"{row['validation_adjusted_asr_percent']:9.2f} "
              f"{row['original_test_asr_percent']:9.2f} "
              f"{row['validation_minus_test_asr_pp']:13.2f}", flush=True)
        rows.append(row)
        details.append(dict(metrics=row, source_json=run["source_json"],
                            classifier_checkpoint=str(run["classifier_path"]),
                            validation_asr=ev.target_rate_result_to_dict(asr),
                            validation_baseline_target_rate=ev.target_rate_result_to_dict(baseline),
                            clean_validation=ev.classification_result_to_dict(clean)))
        del classifier
    aggregate = {}
    for key in ("clean_validation_accuracy_percent", "validation_asr_percent",
                "validation_baseline_target_rate_percent", "validation_adjusted_asr_percent",
                "original_test_asr_percent", "validation_minus_test_asr_pp"):
        values = [row[key] for row in rows]
        aggregate[key] = dict(mean=statistics.mean(values),
                              std_sample=statistics.stdev(values) if len(values) > 1 else 0.0,
                              minimum=min(values), maximum=max(values))
    result = dict(experiment="posthoc_downstream_validation_asr", protocol=protocol,
                  recorded_protocol=job["recorded_protocol"],
                  dataset_metadata_sources=job["dataset_metadata_sources"],
                  pair_name=job["pair_name"], pair_metadata=metadata,
                  interpretation="Final saved classifiers; existing clean-validation selection set; "
                  "no retraining; not the joint attack-validation set or an independent test.",
                  validation_samples=len(indices), non_target_validation_samples=int(mask.sum()),
                  validation_indices=indices,
                  validation_indices_sha256=hashlib.sha256(
                      json.dumps(indices, separators=(",", ":")).encode()).hexdigest(),
                  feature_extraction_seconds=feature_seconds,
                  elapsed_seconds=time.perf_counter() - started, per_seed=details, aggregate=aggregate)
    output_dir = job["results_dir"] / "validation_asr"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_asr_results.json"
    csv_path = output_dir / "validation_asr_summary.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    score = aggregate["validation_asr_percent"]
    print(f"  Validation ASR: {score['mean']:.2f} +/- {score['std_sample']:.2f}% | "
          f"{result['elapsed_seconds']:.1f} seconds\n  Saved: {json_path}\n  Saved: {csv_path}", flush=True)
    del model, trigger, clean_features, triggered_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", nargs="+", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--pair-name", default=None)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--mlp-eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--download", action="store_true", help="Download missing training data.")
    parser.add_argument("--check-only", action="store_true", help="Check metadata/paths without inference.")
    cli = parser.parse_args()
    if min(cli.feature_batch_size, cli.mlp_eval_batch_size) < 1 or cli.num_workers < 0:
        parser.error("Batch sizes must be positive; num-workers cannot be negative.")
    jobs = [make_job(folder, cli.project_root.resolve(), cli.pair_name) for folder in cli.results_dir]
    if len({job["results_dir"] for job in jobs}) != len(jobs):
        parser.error("A results folder was supplied more than once.")
    for job in jobs:
        print(f"Ready: {job['framework']} {job['protocol']['upstream_dataset']} -> "
              f"{job['protocol']['downstream_dataset']} | pair={job['pair_name']} | "
              f"{len(job['runs'])} saved classifiers | "
              f"{job['counts']['classifier_validation']} validation images", flush=True)
        test_asr = [float(run["pair"]["asr"]["target_rate_percent"]) for run in job["runs"]]
        print(f"  Encoder: {job['model_paths'][0]}\n"
              f"  Trigger: {job['model_paths'][1]}\n"
              f"  Recorded test ASR mean: {statistics.mean(test_asr):.2f}%", flush=True)
        if job["dataset_metadata_sources"]:
            print("  Legacy SimCLR dataset fields recovered from the saved dataset "
                  "and normalization metadata.", flush=True)
    if cli.check_only:
        print("Metadata and checkpoint paths checked. No datasets or models were loaded.")
        return
    try:
        import torch
        import evaluate_ssl_transfer as ev
    except ImportError as error:
        raise SystemExit("Place this script beside evaluate_ssl_transfer.py and use your "
                         f"existing project Python environment. Import failed: {error}") from error
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    for job in jobs:
        run_job(job, cli, ev, torch)


if __name__ == "__main__":
    main()
