from __future__ import annotations
import argparse
import glob
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SERVER_GIGATIME_SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
LOCAL_GIGATIME_SCRIPT_DIR = ROOT / "scripts"
DEFAULT_CALIBRATION = ROOT / "channel_calibration_trend.json"


def default_gigatime_script_dir() -> Path:
    if SERVER_GIGATIME_SCRIPT_DIR.exists():
        return SERVER_GIGATIME_SCRIPT_DIR
    return LOCAL_GIGATIME_SCRIPT_DIR


def load_gigatime_module(script_dir: Path):
    script_dir = script_dir.resolve()
    script = script_dir / "custom_inference.py"
    if not script.exists():
        raise FileNotFoundError(f"Cannot find original GigaTIME script: {script}")
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("gigatime_custom_inference", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gi = None


def get_gigatime(script_dir: Path):
    global gi
    if gi is None:
        gi = load_gigatime_module(script_dir)
    return gi


def load_calibration(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for marker, cfg in data.items():
        out[marker] = {
            "background": float(cfg.get("background", 0.0)),
            "threshold": float(cfg.get("threshold", 0.0)),
            "gamma": float(cfg.get("gamma", 1.0)),
            "gain": float(cfg.get("gain", 1.0)),
        }
    return out


def apply_channel_calibration(
    pred: np.ndarray,
    channel_to_index: dict[str, int],
    calibration: dict[str, dict[str, float]],
) -> np.ndarray:
    calibrated = np.clip(pred.astype(np.float32, copy=True), 0.0, 1.0)
    for marker, cfg in calibration.items():
        if marker not in channel_to_index:
            continue
        ch = channel_to_index[marker]
        x = calibrated[ch]
        background = cfg.get("background", 0.0)
        threshold = cfg.get("threshold", 0.0)
        gamma = cfg.get("gamma", 1.0)
        gain = cfg.get("gain", 1.0)
        if background > 0:
            x = np.clip(x - background, 0.0, 1.0)
        if threshold > 0:
            denom = max(1.0 - threshold, 1e-6)
            x = np.where(x >= threshold, (x - threshold) / denom, 0.0)
        if gamma != 1.0:
            x = np.power(np.clip(x, 0.0, 1.0), gamma)
        if gain != 1.0:
            x = x * gain
        calibrated[ch] = np.clip(x, 0.0, 1.0)
    return calibrated


def iter_prediction_coords(
    npy_files: list[str], tile_size: int
) -> tuple[list[tuple[int, int, str]], int, int]:
    coords = []
    max_x, max_y = 0, 0
    for f in npy_files:
        parts = Path(f).stem.split("_")
        if len(parts) < 3:
            continue
        x, y = int(parts[1]), int(parts[2])
        coords.append((x, y, f))
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    if not coords:
        raise RuntimeError("No valid prediction files named like tile_x_y.npy were found.")
    return sorted(coords, key=lambda item: (item[1], item[0])), max_x + tile_size, max_y + tile_size


def generate_calibrated_heatmaps(
    npy_dir: Path,
    output_viz_dir: Path,
    calibration: dict[str, dict[str, float]],
    script_dir: Path,
    tile_size: int = 512,
    max_raw_mb: int = 220,
    max_side: int = 14000,
    save_calibrated_npy: bool = False,
) -> None:
    gi = get_gigatime(script_dir)
    print(f"\n>>> Generating calibrated virtual images -> {output_viz_dir}")
    npy_files = glob.glob(str(npy_dir / "*.npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy files found in {npy_dir}")
    png_dir = output_viz_dir / "high_quality_png"
    tiles_dir = output_viz_dir / "merge_tiles"
    channel_tiles_dir = output_viz_dir / "channel_tiles"
    calibrated_npy_dir = output_viz_dir / "calibrated_predictions"
    png_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    channel_tiles_dir.mkdir(parents=True, exist_ok=True)
    if save_calibrated_npy:
        calibrated_npy_dir.mkdir(parents=True, exist_ok=True)
    coords, canvas_w_full, canvas_h_full = iter_prediction_coords(npy_files, tile_size)
    output_scale = gi.estimate_output_scale(
        canvas_w_full, canvas_h_full, max_raw_mb=max_raw_mb, max_side=max_side
    )
    canvas_w = int(np.ceil(canvas_w_full / output_scale))
    canvas_h = int(np.ceil(canvas_h_full / output_scale))
    print(f"Original canvas: {canvas_w_full} x {canvas_h_full}")
    print(f"Output scale: 1/{output_scale}")
    print(f"PNG canvas: {canvas_w} x {canvas_h}")
    channel_to_index = {name: idx for idx, name in enumerate(gi.CHANNEL_NAMES)}
    target_indices = {
        name: channel_to_index[name] for name in gi.TARGET_CHANNELS if name in channel_to_index
    }
    dapi_ch = target_indices["DAPI"]
    dapi_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    print("\n>>> Building calibrated DAPI base ...")
    for x, y, f in tqdm(coords, unit="tile"):
        pred = np.load(f)
        pred = apply_channel_calibration(pred, channel_to_index, calibration)
        if save_calibrated_npy:
            np.save(calibrated_npy_dir / Path(f).name, pred)
        dapi_rgb = gi.colorize_patch_uint8(pred[dapi_ch], gi.TARGET_CHANNELS["DAPI"])
        gi.paste_scaled_patch(dapi_canvas, dapi_rgb, x, y, output_scale)
    gi.save_png(dapi_canvas, str(png_dir / "Map_00_DAPI_calibrated.png"))
    overlay_canvas = dapi_canvas.copy()
    print("\n>>> Building calibrated marker + DAPI PNGs ...")
    for marker_name, ch in target_indices.items():
        if marker_name == "DAPI":
            continue
        print(f"\nProcessing marker: {marker_name}")
        marker_canvas = dapi_canvas.copy()
        for x, y, f in tqdm(coords, unit="tile"):
            pred = np.load(f)
            pred = apply_channel_calibration(pred, channel_to_index, calibration)
            marker_rgb = gi.colorize_patch_uint8(pred[ch], gi.TARGET_CHANNELS[marker_name])
            gi.paste_scaled_patch(marker_canvas, marker_rgb, x, y, output_scale)
            gi.paste_scaled_patch(overlay_canvas, marker_rgb, x, y, output_scale)
        ch_idx = target_indices[marker_name]
        gi.save_png(
            marker_canvas,
            str(
                png_dir / f"Map_{ch_idx:02d}_{gi.safe_name(marker_name)}_DAPI_merge_calibrated.png"
            ),
        )
    print("\n>>> Saving calibrated total overlay PNG ...")
    gi.save_png(overlay_canvas, str(png_dir / "Overlay_Merge_DAPI_all_markers_calibrated.png"))
    print("\n>>> Saving calibrated selectable tiles ...")
    manifest_tiles = []
    for x, y, f in tqdm(coords, unit="tile"):
        pred = np.load(f)
        pred = apply_channel_calibration(pred, channel_to_index, calibration)
        tile_canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        channel_files = {}
        dapi_rgb = gi.colorize_patch_uint8(pred[target_indices["DAPI"]], gi.TARGET_CHANNELS["DAPI"])
        for marker_name, ch in target_indices.items():
            rgb = gi.colorize_patch_uint8(pred[ch], gi.TARGET_CHANNELS[marker_name])
            np.maximum(tile_canvas, rgb, out=tile_canvas)
            channel_dir_name = gi.safe_name(marker_name)
            channel_dir = channel_tiles_dir / channel_dir_name
            channel_dir.mkdir(parents=True, exist_ok=True)
            if marker_name == "DAPI":
                channel_canvas = rgb
            else:
                channel_canvas = dapi_rgb.copy()
                np.maximum(channel_canvas, rgb, out=channel_canvas)
            channel_filename = f"{channel_dir_name}_tile_{x}_{y}.png"
            gi.save_png(
                channel_canvas, str(channel_dir / channel_filename), compress_level=1, verbose=False
            )
            channel_files[marker_name] = f"channel_tiles/{channel_dir_name}/{channel_filename}"
        filename = f"merge_tile_{x}_{y}.png"
        gi.save_png(tile_canvas, str(tiles_dir / filename), compress_level=1, verbose=False)
        manifest_tiles.append(
            {
                "x": x,
                "y": y,
                "w": tile_size,
                "h": tile_size,
                "file": f"merge_tiles/{filename}",
                "channels": channel_files,
            }
        )
    manifest = {
        "tileSize": tile_size,
        "canvasWidth": canvas_w_full,
        "canvasHeight": canvas_h_full,
        "channels": gi.TARGET_CHANNELS_HEX,
        "calibration": calibration,
        "tiles": sorted(manifest_tiles, key=lambda t: (t["y"], t["x"])),
    }
    with (output_viz_dir / "patch_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with (output_viz_dir / "applied_channel_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {output_viz_dir / 'patch_manifest.json'}")


def run_inference(
    input_tiff: Path, output_dir: Path, gpu_id: int, batch_size: int, script_dir: Path
) -> Path:
    import albumentations as A
    import torch
    from torch.utils.data import DataLoader

    gi = get_gigatime(script_dir)
    temp_tiles_dir = output_dir / "temp_tiles"
    predictions_dir = output_dir / "predictions_raw"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if not glob.glob(str(temp_tiles_dir / "*.png")):
        gi.tile_image(str(input_tiff), str(temp_tiles_dir))
    img_paths = glob.glob(str(temp_tiles_dir / "*.png"))
    if not img_paths:
        raise RuntimeError("No valid tiles found after tiling.")
    if gi.archs is None:
        raise RuntimeError("GigaTIME archs module is not available.")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    local_dir = gi.snapshot_download(repo_id="prov-gigatime/GigaTIME")
    weights_path = os.path.join(local_dir, "model.pth")
    model = gi.archs.__dict__["gigatime"](num_classes=23, input_channels=3)
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint.items()})
    model = model.to(device).eval()
    dataset = gi.InferenceDataset(
        img_paths, transform=A.Compose([A.Resize(512, 512), A.Normalize()])
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=gi.collate_skip_none,
    )
    print("Starting raw GigaTIME inference ...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if not batch:
                continue
            imgs, names = batch
            preds = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            for i, name in enumerate(names):
                np.save(predictions_dir / name.replace(".png", ".npy"), preds[i])
    gi.generate_he_overview(str(input_tiff), str(temp_tiles_dir))
    return predictions_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GigaTIME and generate trend-calibrated virtual IF tiles without modifying raw predictions."
    )
    parser.add_argument(
        "--input_tiff",
        type=Path,
        default=None,
        help="Input HE/IF TIFF. Required unless --predictions_dir is provided.",
    )
    parser.add_argument(
        "--predictions_dir", type=Path, default=None, help="Existing raw prediction .npy directory."
    )
    parser.add_argument("--output_dir", type=Path, default=Path("calibrated_gigatime_results"))
    parser.add_argument("--calibration_json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--gigatime_script_dir",
        type=Path,
        default=default_gigatime_script_dir(),
        help="Folder containing custom_inference.py and archs.py. Defaults to tools/GigaTIME/scripts relative to this project.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--max_png_raw_mb", type=int, default=220)
    parser.add_argument("--max_png_side", type=int, default=14000)
    parser.add_argument("--save_calibrated_npy", action="store_true")
    args = parser.parse_args()
    calibration = load_calibration(args.calibration_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.predictions_dir is not None:
        predictions_dir = args.predictions_dir
    else:
        if args.input_tiff is None:
            raise SystemExit("Please provide either --input_tiff or --predictions_dir.")
        predictions_dir = run_inference(
            args.input_tiff, args.output_dir, args.gpu_id, args.batch_size, args.gigatime_script_dir
        )
    visualizations_dir = args.output_dir / "visualizations_calibrated"
    if visualizations_dir.exists():
        shutil.rmtree(visualizations_dir)
    generate_calibrated_heatmaps(
        predictions_dir,
        visualizations_dir,
        calibration,
        args.gigatime_script_dir,
        tile_size=args.tile_size,
        max_raw_mb=args.max_png_raw_mb,
        max_side=args.max_png_side,
        save_calibrated_npy=args.save_calibrated_npy,
    )


if __name__ == "__main__":
    main()
