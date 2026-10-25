#!/usr/bin/env python3
r"""Show ONE local test image before and after its selected final TRAD trigger.

PowerShell, from your project folder (default: SimCLR/CIFAR-10 -> CIFAR-10):
    python .\compare_final_trigger_image.py --root .

Different test sample:
    python .\compare_final_trigger_image.py --root . --index 12

MoCo, CIFAR-10 upstream, CIFAR-10 test image:
    python .\compare_final_trigger_image.py --root . --results-dir results_eval_moco_c2_cifar10_to_cifar10 --pair-name moco_c2_cifar10_best

STL-10 test image with the SimCLR/CIFAR-10-upstream trigger:
    python .\compare_final_trigger_image.py --root . --results-dir results_C2_dualphase_strict/downstream_stl10 --pair-name c2_dualphase_strict_best --dataset stl10

This is a NEW visualization task, separate from verify_reported_results.py.
It reads a saved downstream result JSON, follows its recorded trigger checkpoint,
checks TRAD export metadata and epoch, and runs ONE frozen hider forward pass.
The trigger export contains wrapper weights; only its hider is instantiated/run.
No training, optimizer updates, decoder execution, SSL encoder or classifier model
loads, ASR evaluation, dataset downloads, or project imports. Local datasets only.
It creates a NEW output directory; existing results/checkpoints are never changed.

Dependencies: torch, torchvision, Pillow (already used by the supplied project).
Standard-library-only --inspect-only checks JSON selection without importing them.
The hider definitions below are copied from the supplied stegastamp_like_model.py;
no training scripts are imported. Strict weight loading rejects incompatible models.

Default source mapping was established during the preceding results verification:
results_C2_dualphase_strict/downstream_cifar10/inactive_downstream_aggregate.json
  $["pairs"]["c2_dualphase_strict_best"]["trigger_checkpoint"]
Folder names do not prove experiment identity: saved metadata is checked as well.
The script refuses an initial hider/decoder checkpoint or other training stage.
It does not recompute checkpoint selection or prove the original experiment correct.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import html
import io
import json
import math
from pathlib import Path, PureWindowsPath
import sys
import webbrowser

STAGE = "inactive_alternating_c2_dualphase_trigger_export"
ENCODER_STAGE = "inactive_alternating_c2_dualphase_encoder_export"
DEFAULT_RESULTS = "results_C2_dualphase_strict/downstream_cifar10"
DEFAULT_PAIR = "c2_dualphase_strict_best"
CONFIG_KEYS = ("message_length", "image_size", "message_map_size", "message_channels", "base_channels", "max_residual")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + str(key))
        result[key] = value
    return result


def read_json(path):
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Result JSON exceeds 32 MiB; select the supported aggregate file.")
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("Result JSON must be an object.")
    return value


def dataset_name(value):
    name = str(value).lower().replace("-", "").replace("_", "")
    if name not in ("cifar10", "stl10"):
        raise ValueError("Unrecognized dataset metadata: " + repr(value))
    return name


def local_path(root, value):
    value = str(value).replace("\\", "/")
    path = Path(value).expanduser()
    # Windows absolute paths cannot be resolved on a non-Windows host by guessing.
    if PureWindowsPath(value).is_absolute() and not path.is_absolute():
        raise ValueError("Recorded Windows path cannot be resolved on this OS: " + value)
    return (path if path.is_absolute() else root / path).resolve()


def path_identity(root, value):
    return str(local_path(root, value)).replace("\\", "/").casefold()


def framework_name(protocol):
    feature = protocol.get("feature_source")
    if feature == "model.f":
        return "simclr"
    if feature == "encoder_q_backbone_pre_projection":
        return "moco"
    raise ValueError("Cannot establish framework from saved protocol.feature_source: " + repr(feature))


def select_result(root, results_dir, pair_name, input_dataset):
    directory = local_path(root, results_dir)
    if directory.is_file():
        path = directory
    else:
        path = directory / "inactive_downstream_aggregate.json"
    if not path.is_file():
        raise ValueError("Saved result JSON not found: {}. Use --results-dir to select the actual final downstream result directory or JSON file.".format(path))
    data = read_json(path)
    pairs = data.get("pairs")
    if not isinstance(pairs, dict) or pair_name not in pairs:
        available = ", ".join(str(k) for k in pairs) if isinstance(pairs, dict) else "none"
        raise ValueError("Pair {!r} is unavailable. Saved pair names: {}. Set --pair-name explicitly.".format(pair_name, available))
    pair = pairs[pair_name]
    if not isinstance(pair, dict):
        raise ValueError("Selected pair record is not an object.")
    protocol = data.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Saved result has no protocol object.")
    framework = framework_name(protocol)
    upstream_values = [protocol.get("upstream_dataset")]
    downstream = dataset_name(protocol.get("downstream_dataset", protocol.get("dataset")))
    if downstream != input_dataset:
        raise ValueError("Selected results are for {}, but --dataset is {}. Select the matching downstream result directory.".format(downstream, input_dataset))
    metadata = pair.get("pair_metadata")
    if not isinstance(metadata, dict) or metadata.get("verified") is not True:
        raise ValueError("Saved pair_metadata does not identify a verified encoder-trigger pair.")
    if metadata.get("trigger_stage") != STAGE or metadata.get("encoder_stage") != ENCODER_STAGE:
        raise ValueError("Saved pair is not a final dual-phase TRAD export: {} / {}".format(metadata.get("encoder_stage"), metadata.get("trigger_stage")))
    epoch = metadata.get("trigger_epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1 or epoch != metadata.get("encoder_epoch"):
        raise ValueError("Saved encoder and trigger epochs must be positive, equal integers.")
    upstream_values.append(metadata.get("upstream_dataset"))
    upstreams = {dataset_name(v) for v in upstream_values if v is not None}
    if len(upstreams) > 1:
        raise ValueError("Pair metadata and protocol upstream datasets disagree.")
    # The legacy CIFAR-10 writer omitted this field. The selected final trigger
    # export will establish it from saved metadata; never infer it from a folder.
    upstream = next(iter(upstreams)) if upstreams else None
    for key in ("trigger_checkpoint", "encoder_checkpoint"):
        if not isinstance(pair.get(key), str) or not pair[key].strip():
            raise ValueError("Selected result has no usable " + key)
    trigger_path = local_path(root, pair["trigger_checkpoint"])
    return {"result_path": path, "pair_name": pair_name, "pair": pair,
            "framework": framework, "upstream": upstream, "dataset": downstream,
            "epoch": epoch, "trigger_path": trigger_path}


def validate_checkpoint(root, checkpoint, selection):
    if not isinstance(checkpoint, dict):
        raise ValueError("Trigger checkpoint is not a dictionary.")
    extra = checkpoint.get("extra")
    if not isinstance(extra, dict) or extra.get("stage") != STAGE:
        raise ValueError("This is not the selected final TRAD trigger export. Initial hider-decoder and other stage checkpoints are refused.")
    if checkpoint.get("epoch") != selection["epoch"]:
        raise ValueError("Trigger checkpoint epoch differs from the saved result pair.")
    args = extra.get("args")
    if not isinstance(args, dict):
        raise ValueError("Trigger export has no architecture arguments.")
    upstream = dataset_name(extra.get("upstream_dataset", args.get("upstream_dataset")))
    if selection["upstream"] is not None and upstream != selection["upstream"]:
        raise ValueError("Trigger checkpoint upstream dataset differs from saved results.")
    frameworks = set()
    if extra.get("ssl_framework") in ("simclr", "moco"):
        frameworks.add(extra["ssl_framework"])
    if args.get("clean_simclr_checkpoint"):
        frameworks.add("simclr")
    if args.get("clean_moco_checkpoint"):
        frameworks.add("moco")
    if frameworks != {selection["framework"]}:
        raise ValueError("Trigger framework metadata is missing or disagrees with saved results: " + repr(frameworks))
    if extra.get("fixed_message_pattern") != "alternating":
        raise ValueError("Expected recorded fixed_message_pattern='alternating'.")
    recorded_encoder = extra.get("paired_encoder_checkpoint")
    if not isinstance(recorded_encoder, str) or path_identity(root, recorded_encoder) != path_identity(root, selection["pair"]["encoder_checkpoint"]):
        raise ValueError("Trigger export points to a different paired encoder than the selected result record.")
    missing = [key for key in CONFIG_KEYS if key not in args]
    if missing:
        raise ValueError("Missing trigger architecture metadata: " + ", ".join(missing))
    config = {}
    for key in CONFIG_KEYS:
        value = args[key]
        if key == "max_residual":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid max_residual metadata.")
            config[key] = float(value)
        else:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Invalid architecture metadata: " + key)
            config[key] = value
    if config["message_length"] != 16 or config["image_size"] != 32 or not math.isclose(config["max_residual"], 8 / 255, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Checkpoint does not match the inspected 16-bit, 32px, 8/255 protocol.")
    saved_config = selection["pair"].get("trigger_config")
    if not isinstance(saved_config, dict):
        raise ValueError("Saved result has no trigger_config to cross-check.")
    for key, value in config.items():
        observed = saved_config.get(key)
        if not isinstance(observed, (int, float)) or isinstance(observed, bool) or not math.isclose(value, observed, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Result and checkpoint trigger_config disagree at " + key)
    weights = checkpoint.get("model_state_dict")
    if not isinstance(weights, dict):
        raise ValueError("Trigger export has no model_state_dict.")
    hider_weights = {key[len("encoder."):]: value for key, value in weights.items() if isinstance(key, str) and key.startswith("encoder.")}
    if not hider_weights:
        raise ValueError("No encoder.* hider weights found in trigger export.")
    selection["upstream"] = upstream
    return config, hider_weights


# Exact hider operations from the supplied project. Definitions are local to this
# factory, so importing this script does not import torch or any project module.
def build_hider(config, torch):
    nn = torch.nn
    F = torch.nn.functional
    def check_image_tensor(image):
        if not torch.is_tensor(image):
            raise TypeError('image must be a torch.Tensor')
        if image.dim() != 4:
            raise ValueError(f'Expected image shape (B, 3, H, W), but got {tuple(image.shape)}')
        if image.size(1) != 3:
            raise ValueError(f'Expected image to have 3 channels, but got {image.size(1)}')
        if not image.is_floating_point():
            raise TypeError('image must be a floating point tensor')

    def check_message_tensor(message, message_length):
        if message.dim() != 2:
            raise ValueError(f'Expected message shape (B, {message_length}), but got {tuple(message.shape)}')
        if message.size(1) != message_length:
            raise ValueError(f'Expected message length {message_length}, but got {message.size(1)}')

    def get_group_count(channels, preferred_groups=8):
        if channels <= 0:
            raise ValueError('channels must be greater than 0')
        for group_count in range(min(preferred_groups, channels), 0, -1):
            if channels % group_count == 0:
                return group_count
        return 1

    class ConvGNAct(nn.Module):

        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=True):
            super().__init__()
            if in_channels <= 0:
                raise ValueError('in_channels must be greater than 0')
            if out_channels <= 0:
                raise ValueError('out_channels must be greater than 0')
            padding = kernel_size // 2
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), nn.GroupNorm(num_groups=get_group_count(out_channels), num_channels=out_channels)]
            if activation:
                layers.append(nn.SiLU(inplace=True))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class ResidualConvBlock(nn.Module):

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = ConvGNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, activation=True)
            self.conv2 = ConvGNAct(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, activation=False)
            if in_channels == out_channels:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.activation = nn.SiLU(inplace=True)

        def forward(self, x):
            identity = self.skip(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = x + identity
            x = self.activation(x)
            return x

    class SpatialMessageMap(nn.Module):

        def __init__(self, message_length, message_channels, message_map_size):
            super().__init__()
            self.message_length = message_length
            self.message_channels = message_channels
            self.message_map_size = message_map_size
            flattened_size = message_channels * message_map_size * message_map_size
            self.projection = nn.Sequential(nn.Linear(message_length, flattened_size), nn.SiLU(inplace=True))

        def forward(self, message, output_size):
            batch_size = message.size(0)
            message_map = self.projection(message)
            message_map = message_map.view(batch_size, self.message_channels, self.message_map_size, self.message_map_size)
            message_map = F.interpolate(message_map, size=output_size, mode='bilinear', align_corners=False)
            return message_map

    class MessageFiLM(nn.Module):

        def __init__(self, message_length, feature_channels, hidden_channels=128, modulation_strength=0.1):
            super().__init__()
            self.feature_channels = feature_channels
            self.modulation_strength = float(modulation_strength)
            self.conditioner = nn.Sequential(nn.Linear(message_length, hidden_channels), nn.SiLU(inplace=True), nn.Linear(hidden_channels, feature_channels * 2))
            final_layer = self.conditioner[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

        def forward(self, features, message):
            parameters = self.conditioner(message)
            gamma, beta = parameters.chunk(2, dim=1)
            gamma = gamma.view(-1, self.feature_channels, 1, 1)
            beta = beta.view(-1, self.feature_channels, 1, 1)
            gamma = self.modulation_strength * torch.tanh(gamma)
            beta = self.modulation_strength * torch.tanh(beta)
            return features * (1.0 + gamma) + beta

    class AdvancedDownBlock(nn.Module):

        def __init__(self, in_channels, out_channels, message_length):
            super().__init__()
            self.downsample = ConvGNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, activation=True)
            self.residual = ResidualConvBlock(in_channels=out_channels, out_channels=out_channels)
            self.message_film = MessageFiLM(message_length=message_length, feature_channels=out_channels)

        def forward(self, x, message):
            x = self.downsample(x)
            x = self.residual(x)
            x = self.message_film(x, message)
            return x

    class AdvancedUpBlock(nn.Module):

        def __init__(self, in_channels, skip_channels, out_channels, message_length):
            super().__init__()
            self.fusion = ResidualConvBlock(in_channels=in_channels + skip_channels, out_channels=out_channels)
            self.message_film = MessageFiLM(message_length=message_length, feature_channels=out_channels)

        def forward(self, x, skip, message):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.fusion(x)
            x = self.message_film(x, message)
            return x

    class StegaStampEncoder(nn.Module):

        def __init__(self, message_length=16, image_size=32, message_map_size=8, message_channels=8, base_channels=32, max_residual=8 / 255):
            super().__init__()
            if message_length <= 0:
                raise ValueError('message_length must be greater than 0')
            if image_size <= 0:
                raise ValueError('image_size must be greater than 0')
            if message_map_size <= 0:
                raise ValueError('message_map_size must be greater than 0')
            if message_channels <= 0:
                raise ValueError('message_channels must be greater than 0')
            if base_channels <= 0:
                raise ValueError('base_channels must be greater than 0')
            if max_residual <= 0.0:
                raise ValueError('max_residual must be greater than 0')
            self.message_length = message_length
            self.image_size = image_size
            self.message_map_size = message_map_size
            self.message_channels = message_channels
            self.base_channels = base_channels
            self.max_residual = float(max_residual)
            self.message_map = SpatialMessageMap(message_length=message_length, message_channels=message_channels, message_map_size=message_map_size)
            input_channels = 3 + message_channels
            self.input_block = ResidualConvBlock(in_channels=input_channels, out_channels=base_channels)
            self.input_message_film = MessageFiLM(message_length=message_length, feature_channels=base_channels)
            self.down1 = AdvancedDownBlock(in_channels=base_channels, out_channels=base_channels * 2, message_length=message_length)
            self.down2 = AdvancedDownBlock(in_channels=base_channels * 2, out_channels=base_channels * 4, message_length=message_length)
            self.down3 = AdvancedDownBlock(in_channels=base_channels * 4, out_channels=base_channels * 8, message_length=message_length)
            self.bottleneck = nn.Sequential(ResidualConvBlock(in_channels=base_channels * 8, out_channels=base_channels * 8), ResidualConvBlock(in_channels=base_channels * 8, out_channels=base_channels * 8))
            self.bottleneck_message_film = MessageFiLM(message_length=message_length, feature_channels=base_channels * 8)
            self.up2 = AdvancedUpBlock(in_channels=base_channels * 8, skip_channels=base_channels * 4, out_channels=base_channels * 4, message_length=message_length)
            self.up1 = AdvancedUpBlock(in_channels=base_channels * 4, skip_channels=base_channels * 2, out_channels=base_channels * 2, message_length=message_length)
            self.up0 = AdvancedUpBlock(in_channels=base_channels * 2, skip_channels=base_channels, out_channels=base_channels, message_length=message_length)
            self.residual_content_head = nn.Sequential(ResidualConvBlock(in_channels=base_channels, out_channels=base_channels), nn.Conv2d(base_channels, 3, kernel_size=1, stride=1, padding=0))
            mask_hidden_channels = max(base_channels // 2, 8)
            self.residual_mask_head = nn.Sequential(ConvGNAct(in_channels=base_channels, out_channels=mask_hidden_channels, kernel_size=3, stride=1, activation=True), nn.Conv2d(mask_hidden_channels, 1, kernel_size=1, stride=1, padding=0))
            self.initialize_output_heads()

        def initialize_output_heads(self):
            residual_layer = self.residual_content_head[-1]
            nn.init.normal_(residual_layer.weight, mean=0.0, std=0.001)
            nn.init.zeros_(residual_layer.bias)
            mask_layer = self.residual_mask_head[-1]
            nn.init.zeros_(mask_layer.weight)
            nn.init.constant_(mask_layer.bias, -1.5)

        def generate_trigger_with_mask(self, image, message):
            check_image_tensor(image)
            check_message_tensor(message, self.message_length)
            if image.size(0) != message.size(0):
                raise ValueError(f'Image batch size and message batch size must match. Got image batch {image.size(0)} and message batch {message.size(0)}.')
            _, _, height, width = image.shape
            message_map = self.message_map(message=message, output_size=(height, width))
            x = torch.cat([image, message_map], dim=1)
            skip0 = self.input_block(x)
            skip0 = self.input_message_film(skip0, message)
            skip1 = self.down1(skip0, message)
            skip2 = self.down2(skip1, message)
            x = self.down3(skip2, message)
            x = self.bottleneck(x)
            x = self.bottleneck_message_film(x, message)
            x = self.up2(x, skip2, message)
            x = self.up1(x, skip1, message)
            x = self.up0(x, skip0, message)
            residual_content = self.residual_content_head(x)
            residual_content = torch.tanh(residual_content)
            residual_mask = self.residual_mask_head(x)
            residual_mask = torch.sigmoid(residual_mask)
            bounded_residual = residual_content * residual_mask * self.max_residual
            encoded_image = torch.clamp(image + bounded_residual, min=0.0, max=1.0)
            effective_residual = encoded_image - image
            return (encoded_image, effective_residual, residual_mask)

        def forward(self, image, message):
            encoded_image, effective_residual, _ = self.generate_trigger_with_mask(image, message)
            return (encoded_image, effective_residual)

        def forward_with_mask(self, image, message):
            return self.generate_trigger_with_mask(image, message)
    return StegaStampEncoder(**config)


def new_output_directory(root, requested=None):
    if requested:
        path = local_path(root, requested)
        path.mkdir(exist_ok=False)
        return path
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        tail = "" if index == 0 else "_" + str(index)
        path = root / ("final_trigger_comparison_" + stamp + tail)
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError("Could not choose a new output directory.")


def build_preview(clean_png, triggered_png, caption):
    uris = ["data:image/png;base64," + base64.b64encode(value).decode("ascii") for value in (clean_png, triggered_png)]
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'">
<title>Final trigger: one image comparison</title><style>
body{font:17px system-ui,Segoe UI,sans-serif;color:#172833;background:white;margin:35px auto;padding:20px;max-width:850px}
h1{text-align:center;font-size:25px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:30px;text-align:center}
img{width:100%;max-width:320px;image-rendering:pixelated}p{text-align:center;line-height:1.5}.note{font-size:14px;color:#52636f}
</style><h1>Same image, without and with final TRAD trigger</h1><div class="pair">
<div><h2>Without trigger</h2><img alt="Without trigger" src="''' + uris[0] + '''"></div>
<div><h2>With trigger</h2><img alt="With final TRAD trigger at its normal strength" src="''' + uris[1] + '''"></div>
</div><p>''' + html.escape(caption) + '''</p><p class="note">Trigger at normal strength. Both images are enlarged from 32 x 32 pixels.<br>
This visual comparison does not measure attack success. Exact source metadata are in metadata.json.</p></html>'''


def tensor_to_image(tensor, Image):
    pixels = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).mul(255).add(0.5).clamp(0, 255).byte().contiguous()
    return Image.frombytes("RGB", (32, 32), bytes(pixels.reshape(-1).tolist()))


def save_images(output, clean_image, triggered_image, metadata, Image, ImageDraw, ImageFont):
    images = [clean_image, triggered_image]
    pngs = []
    for image, name in zip(images, ("without_trigger.png", "with_trigger.png")):
        image.save(output / name, format="PNG")
        buffer = io.BytesIO(); image.save(buffer, format="PNG"); pngs.append(buffer.getvalue())
    caption = "{} | upstream {} | {} test image {} | selected epoch {}".format(
        "SimCLR" if metadata["framework"] == "simclr" else "MoCo",
        metadata["upstream_dataset"].upper(), metadata["input_dataset"].upper(), metadata["test_index"], metadata["trigger_epoch"])
    canvas = Image.new("RGB", (800, 460), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default(); small = font
    draw.text((200, 35), "Without trigger", fill="#172833", font=font, anchor="mm")
    draw.text((600, 35), "With trigger", fill="#172833", font=font, anchor="mm")
    nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    canvas.paste(images[0].resize((320, 320), nearest), (40, 75))
    canvas.paste(images[1].resize((320, 320), nearest), (440, 75))
    draw.text((400, 420), caption, fill="#172833", font=small, anchor="mm")
    canvas.save(output / "comparison.png", format="PNG")
    (output / "preview.html").write_text(build_preview(*pngs, caption), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="Project folder.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS, help="Exact final downstream result directory or aggregate JSON (relative to root).")
    parser.add_argument("--pair-name", default=DEFAULT_PAIR, help="Exact saved pair name.")
    parser.add_argument("--dataset", choices=("cifar10", "stl10"), default="cifar10", help="Local test dataset; must match selected result protocol.")
    parser.add_argument("--index", type=int, default=0, help="One test-set index, starting at 0 (default 0).")
    parser.add_argument("--data-dir", default="data", help="Existing torchvision dataset directory, relative to root.")
    parser.add_argument("--output-dir", help="NEW output folder; must not already exist.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="One hider pass; default CPU.")
    parser.add_argument("--inspect-only", action="store_true", help="Read result JSON selection only; no torch, dataset, checkpoint or forward pass.")
    parser.add_argument("--no-open", action="store_true", help="Save pictures without opening the browser.")
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir() or args.index < 0:
        parser.error("--root must exist and --index must be non-negative")
    try:
        selected = select_result(root, args.results_dir, args.pair_name, args.dataset)
        print("ONE IMAGE WITH THE SELECTED FINAL TRAD TRIGGER")
        print("Result JSON:", selected["result_path"])
        print('Source key: $["pairs"][{}]["trigger_checkpoint"]'.format(json.dumps(args.pair_name)))
        print("Pair:", selected["pair_name"])
        print("Framework / upstream:", selected["framework"], "/", selected["upstream"] or "not recorded in this legacy JSON; will check trigger export")
        print("Recorded selected epoch:", selected["epoch"])
        print("Recorded trigger:", selected["trigger_path"])
        print("Input:", args.dataset, "test set, index", args.index)
        if args.inspect_only:
            print("Inspection only: checkpoint contents and local images have not been loaded or verified.")
            return 0
        if not selected["trigger_path"].is_file():
            raise ValueError("The trigger path recorded in your result JSON does not exist locally. Restore that exact checkpoint or correct the project location; no other checkpoint will be chosen.")
        try:
            import torch
            from torchvision import datasets, transforms
            from PIL import Image, ImageDraw, ImageFont
        except (ImportError, RuntimeError) as error:
            raise ValueError("Use the same Python environment as your project (torch, torchvision and Pillow required). Dependency error: " + str(error)) from error
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu.")
        # Safe tensor/primitive metadata loader, with no unsafe pickle fallback.
        try:
            checkpoint = torch.load(selected["trigger_path"], map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError("Could not read the recorded trigger export with weights_only=True. No unsafe loader fallback was attempted. " + str(error)) from error
        config, weights = validate_checkpoint(root, checkpoint, selected)
        print("TRAD export metadata checked:", selected["framework"], "/", selected["upstream"], "/ epoch", selected["epoch"])
        model = build_hider(config, torch)
        model.load_state_dict(weights, strict=True)
        model.eval().requires_grad_(False).to(args.device)
        data_dir = local_path(root, args.data_dir)
        try:
            if args.dataset == "cifar10":
                dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=transforms.ToTensor())
            else:
                transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
                dataset = datasets.STL10(root=str(data_dir), split="test", download=False, transform=transform)
        except Exception as error:
            raise ValueError("Local test dataset could not be opened at {}. No download was attempted. {}".format(data_dir, error)) from error
        if args.index >= len(dataset):
            raise ValueError("--index must be between 0 and {} for this test set.".format(len(dataset) - 1))
        image, label = dataset[args.index]
        clean = image.unsqueeze(0).to(args.device)
        message = torch.tensor([i % 2 for i in range(16)], dtype=clean.dtype, device=args.device).view(1, 16)
        # The only forward pass. No corruption, normalization or residual scaling.
        with torch.inference_mode():
            triggered, residual = model(clean, message)
        if clean.shape != (1, 3, 32, 32) or triggered.shape != clean.shape or not torch.isfinite(triggered).all():
            raise ValueError("Unexpected or non-finite triggered image; no images saved.")
        if float(triggered.min()) < 0 or float(triggered.max()) > 1 or float(residual.abs().max()) > config["max_residual"] + 1e-6:
            raise ValueError("Triggered image violates the inspected pixel-range/residual constraints; no images saved.")
        with selected["trigger_path"].open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata = {"purpose": "single-image visualization, not an ASR evaluation", "result_json": str(selected["result_path"]),
                    "result_json_key": '$["pairs"][{}]["trigger_checkpoint"]'.format(json.dumps(args.pair_name)),
                    "pair_name": args.pair_name, "framework": selected["framework"], "upstream_dataset": selected["upstream"],
                    "trigger_checkpoint": str(selected["trigger_path"]), "trigger_checkpoint_sha256": digest.hexdigest(),
                    "trigger_stage": STAGE, "trigger_epoch": selected["epoch"], "paired_encoder_path_from_metadata": checkpoint["extra"]["paired_encoder_checkpoint"],
                    "encoder_note": "Recorded path only; SSL encoder checkpoint was not loaded or executed.",
                    "input_dataset": args.dataset, "split": "test", "test_index": args.index, "class_label": int(label), "data_dir": str(data_dir),
                    "input_preprocessing": "CIFAR10: ToTensor; STL10: PIL Resize((32,32)) with torchvision defaults then ToTensor; no normalization or random augmentation",
                    "message": [i % 2 for i in range(16)], "trigger_config": config,
                    "png_encoding": "Clamp to [0,255] after x*255+0.5, convert to uint8; display is rounded 8-bit RGB, not original float tensors.",
                    "comparison_png": "Nearest-neighbor enlargement of saved-strength images; no residual amplification."}
        clean_image = tensor_to_image(clean, Image)
        triggered_image = tensor_to_image(triggered, Image)
        output = new_output_directory(root, args.output_dir)
        save_images(output, clean_image, triggered_image, metadata, Image, ImageDraw, ImageFont)
        print("Saved one comparison:", output / "comparison.png")
        print("Original-sized pictures:", output / "without_trigger.png", "and", output / "with_trigger.png")
        print("Full trigger applied at 8/255 bound, with the recorded final weights. No training or ASR evaluation.")
        if not args.no_open:
            try:
                if not webbrowser.open((output / "preview.html").as_uri()):
                    print("Open comparison.png or preview.html manually.")
            except (OSError, webbrowser.Error):
                print("Open comparison.png or preview.html manually.")
        return 0
    except Exception as error:
        print("CANNOT CREATE COMPARISON:", error, file=sys.stderr)
        print("No other experiment or checkpoint was substituted.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
