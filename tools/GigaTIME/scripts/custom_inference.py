import matplotlib

matplotlib.use("Agg")
import argparse
import glob
import json
import os
import shutil
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import archs
except ImportError:
    archs = None
    print("Warning: archs module not found. Please ensure archs.py is present.")
CHANNEL_NAMES = [
    "DAPI",
    "TRITC",
    "Cy5",
    "PD-1",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3",
    "CD8",
    "PD-L1",
    "CK",
    "Ki67",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
]
TARGET_CHANNELS_HEX = {
    "DAPI": "#0800EC",
    "CD68": "#00E6FF",
    "CD8": "#C200BD",
    "PD-L1": "#FDCD0F",
    "CK": "#00F77B",
    "Ki67": "#FFFFFF",
    "CD4": "#B30017",
}


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


TARGET_CHANNELS = {k: hex_to_rgb(v) for k, v in TARGET_CHANNELS_HEX.items()}


def to_uint8_rgb(arr):
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[0] < a.shape[-1]:
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 2:
        a = a[:, :, None]
    elif a.ndim != 3:
        raise ValueError(f"Unsupported array shape: {a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] >= 3:
        a = a[:, :, :3]
    else:
        a = np.concatenate([a, a[:, :, 0:1]], axis=2)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    amax = float(np.nanmax(a)) if a.size else 0.0
    amin = float(np.nanmin(a)) if a.size else 0.0
    if amax <= 1.5:
        a = np.clip(a, 0.0, 1.0) * 255.0
    elif amax == amin:
        a = np.zeros_like(a)
    else:
        a = (a - amin) / (amax - amin) * 255.0
    return np.clip(a, 0.0, 255.0).astype(np.uint8)


def safe_name(s):
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def is_background(patch, threshold=220, background_ratio=0.85):
    rgb = to_uint8_rgb(patch)
    gray = rgb.mean(axis=2)
    return (np.sum(gray > threshold) / max(gray.size, 1)) > background_ratio


def colorize_patch_uint8(mask, rgb_tuple):
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    m16 = mask_u8.astype(np.uint16)
    r, g, b = rgb_tuple
    out = np.empty((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = (m16 * r // 255).astype(np.uint8)
    out[:, :, 1] = (m16 * g // 255).astype(np.uint8)
    out[:, :, 2] = (m16 * b // 255).astype(np.uint8)
    return out


def estimate_output_scale(canvas_w, canvas_h, max_raw_mb=220, max_side=14000):
    raw_bytes = canvas_w * canvas_h * 3
    max_raw_bytes = max_raw_mb * 1024 * 1024
    scale_by_size = (
        int(np.ceil(np.sqrt(raw_bytes / max_raw_bytes))) if raw_bytes > max_raw_bytes else 1
    )
    scale_by_side = (
        int(np.ceil(max(canvas_w, canvas_h) / max_side))
        if max(canvas_w, canvas_h) > max_side
        else 1
    )
    return max(scale_by_size, scale_by_side, 1)


def paste_scaled_patch(canvas, patch_rgb, x, y, output_scale):
    xs = x // output_scale
    ys = y // output_scale
    xe = int(np.ceil((x + patch_rgb.shape[1]) / output_scale))
    ye = int(np.ceil((y + patch_rgb.shape[0]) / output_scale))
    xe = min(xe, canvas.shape[1])
    ye = min(ye, canvas.shape[0])
    out_w = max(xe - xs, 1)
    out_h = max(ye - ys, 1)
    patch_small = np.array(Image.fromarray(patch_rgb).resize((out_w, out_h), Image.BILINEAR))
    region = canvas[ys:ye, xs:xe, :]
    np.maximum(region, patch_small[: region.shape[0], : region.shape[1], :], out=region)


def save_png(arr, path, compress_level=1, verbose=True):
    Image.fromarray(arr).save(path, compress_level=compress_level)
    if verbose:
        print(f"Saved PNG: {path}")


class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Read error {img_path}: {e}")
            return None
        if self.transform:
            img = self.transform(image=img)["image"]
        img = img.astype("float32").transpose(2, 0, 1)
        return img, os.path.basename(img_path)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, names = zip(*batch)
    return torch.from_numpy(np.stack(imgs, axis=0)), list(names)


def tile_image(tiff_path, output_dir, tile_size=512):
    print(f"Reading TIFF: {tiff_path}")
    try:
        tif = np.asarray(tifffile.memmap(tiff_path))
    except Exception:
        with tifffile.TiffFile(tiff_path) as f:
            tif = np.asarray(f.asarray())
    if tif.ndim == 3 and tif.shape[0] < 10 and tif.shape[1] > 100:
        print(f"Multi-page TIFF detected {tif.shape}, using page 0.")
        tif = tif[0]
    h, w = tif.shape[0], tif.shape[1]
    print(f"Dims: {w}x{h}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    y_steps = range(0, h, tile_size)
    x_steps = range(0, w, tile_size)
    with tqdm(total=len(y_steps) * len(x_steps), unit="tile") as pbar:
        for y in y_steps:
            for x in x_steps:
                patch = tif[y : min(y + tile_size, h), x : min(x + tile_size, w)]
                if patch.shape[0] != tile_size or patch.shape[1] != tile_size:
                    pbar.update(1)
                    continue
                if is_background(patch):
                    pbar.update(1)
                    continue
                Image.fromarray(to_uint8_rgb(patch)).save(
                    os.path.join(output_dir, f"tile_{x}_{y}.png")
                )
                saved_count += 1
                pbar.update(1)
    print(f"Tiling done. Saved {saved_count} tiles.")


def generate_he_overview(tiff_path, temp_tiles_dir, tile_size=512, scale_factor=4):
    print(f"\n>>> Generating HE overview images -> {temp_tiles_dir}")
    try:
        tif = np.asarray(tifffile.memmap(tiff_path))
    except Exception:
        with tifffile.TiffFile(tiff_path) as f:
            tif = np.asarray(f.asarray())
    if tif.ndim == 3 and tif.shape[0] < 10 and tif.shape[1] > 100:
        tif = tif[0]
    h_full, w_full = tif.shape[0], tif.shape[1]
    w_small = max(w_full // scale_factor, 1)
    h_small = max(h_full // scale_factor, 1)
    he_small = np.array(
        Image.fromarray(to_uint8_rgb(tif)).resize((w_small, h_small), Image.BILINEAR)
    )
    roi_coords = []
    for f in glob.glob(os.path.join(temp_tiles_dir, "tile_*.png")):
        parts = os.path.basename(f).replace(".png", "").split("_")
        roi_coords.append((int(parts[1]), int(parts[2])))
    fig, ax = plt.subplots(figsize=(max(w_small / 100, 6), max(h_small / 100, 6)), dpi=150)
    ax.imshow(he_small)
    ax.set_title("HE - Before ROI Selection", fontsize=14, fontweight="bold", pad=8)
    ax.axis("off")
    plt.savefig(
        os.path.join(temp_tiles_dir, "HE_before.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    fig, ax = plt.subplots(figsize=(max(w_small / 100, 6), max(h_small / 100, 6)), dpi=150)
    ax.imshow(he_small)
    scaled_tile = max(tile_size // scale_factor, 1)
    for x, y in roi_coords:
        ax.add_patch(
            plt.Rectangle(
                (x // scale_factor, y // scale_factor),
                scaled_tile,
                scaled_tile,
                linewidth=1.5,
                edgecolor="#00FF88",
                facecolor="none",
                alpha=0.85,
            )
        )
    ax.set_title(
        f"HE - After ROI Selection ({len(roi_coords)} tiles)", fontsize=14, fontweight="bold", pad=8
    )
    ax.axis("off")
    plt.savefig(
        os.path.join(temp_tiles_dir, "HE_after.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def generate_heatmaps(npy_dir, output_viz_dir, tile_size=512, max_raw_mb=220, max_side=14000):
    print(f"\n>>> Generating high-quality PNG heatmaps -> {output_viz_dir}")
    npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))
    if not npy_files:
        print("No .npy files found.")
        return
    png_dir = os.path.join(output_viz_dir, "high_quality_png")
    tiles_dir = os.path.join(output_viz_dir, "merge_tiles")
    channel_tiles_dir = os.path.join(output_viz_dir, "channel_tiles")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(channel_tiles_dir, exist_ok=True)
    coords = []
    max_x, max_y = 0, 0
    for f in npy_files:
        parts = os.path.basename(f).replace(".npy", "").split("_")
        x, y = int(parts[1]), int(parts[2])
        coords.append((x, y, f))
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    canvas_w_full = max_x + tile_size
    canvas_h_full = max_y + tile_size
    output_scale = estimate_output_scale(
        canvas_w_full, canvas_h_full, max_raw_mb=max_raw_mb, max_side=max_side
    )
    canvas_w = int(np.ceil(canvas_w_full / output_scale))
    canvas_h = int(np.ceil(canvas_h_full / output_scale))
    approx_raw_mb = canvas_w * canvas_h * 3 / 1024 / 1024
    print(f"Original canvas: {canvas_w_full} x {canvas_h_full}")
    print(f"Output scale: 1/{output_scale}")
    print(f"PNG canvas: {canvas_w} x {canvas_h}, approx raw RGB {approx_raw_mb:.1f} MB")
    channel_to_index = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}
    target_indices = {
        name: channel_to_index[name] for name in TARGET_CHANNELS if name in channel_to_index
    }
    if "DAPI" not in target_indices:
        print("DAPI channel not found.")
        return
    dapi_ch = target_indices["DAPI"]
    dapi_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    print("\n>>> Building DAPI base PNG ...")
    for x, y, f in tqdm(coords, unit="tile"):
        pred = np.load(f)
        dapi_rgb = colorize_patch_uint8(
            np.clip(pred[dapi_ch].astype(np.float32), 0.0, 1.0), TARGET_CHANNELS["DAPI"]
        )
        paste_scaled_patch(dapi_canvas, dapi_rgb, x, y, output_scale)
    save_png(dapi_canvas, os.path.join(png_dir, "Map_00_DAPI.png"))
    overlay_canvas = dapi_canvas.copy()
    print("\n>>> Building each marker + DAPI PNG ...")
    for marker_name, ch in target_indices.items():
        if marker_name == "DAPI":
            continue
        print(f"\nProcessing marker: {marker_name}")
        marker_canvas = dapi_canvas.copy()
        color_rgb = TARGET_CHANNELS[marker_name]
        for x, y, f in tqdm(coords, unit="tile"):
            pred = np.load(f)
            marker_rgb = colorize_patch_uint8(
                np.clip(pred[ch].astype(np.float32), 0.0, 1.0), color_rgb
            )
            paste_scaled_patch(marker_canvas, marker_rgb, x, y, output_scale)
            paste_scaled_patch(overlay_canvas, marker_rgb, x, y, output_scale)
        ch_idx = target_indices[marker_name]
        save_png(
            marker_canvas,
            os.path.join(png_dir, f"Map_{ch_idx:02d}_{safe_name(marker_name)}_DAPI_merge.png"),
        )
        del marker_canvas
    print("\n>>> Saving total overlay PNG ...")
    save_png(overlay_canvas, os.path.join(png_dir, "Overlay_Merge_DAPI_all_markers.png"))
    print("\n>>> Saving selectable merge tiles for frontend ...")
    manifest_tiles = []
    for x, y, f in tqdm(coords, unit="tile"):
        pred = np.load(f)
        tile_canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        channel_files = {}
        dapi_rgb = colorize_patch_uint8(
            np.clip(pred[target_indices["DAPI"]].astype(np.float32), 0.0, 1.0),
            TARGET_CHANNELS["DAPI"],
        )
        for marker_name, ch in target_indices.items():
            rgb = colorize_patch_uint8(
                np.clip(pred[ch].astype(np.float32), 0.0, 1.0), TARGET_CHANNELS[marker_name]
            )
            np.maximum(tile_canvas, rgb, out=tile_canvas)
            channel_dir_name = safe_name(marker_name)
            channel_dir = os.path.join(channel_tiles_dir, channel_dir_name)
            os.makedirs(channel_dir, exist_ok=True)
            if marker_name == "DAPI":
                channel_canvas = rgb
            else:
                channel_canvas = dapi_rgb.copy()
                np.maximum(channel_canvas, rgb, out=channel_canvas)
            channel_filename = f"{channel_dir_name}_tile_{x}_{y}.png"
            save_png(
                channel_canvas,
                os.path.join(channel_dir, channel_filename),
                compress_level=1,
                verbose=False,
            )
            channel_files[marker_name] = f"channel_tiles/{channel_dir_name}/{channel_filename}"
        filename = f"merge_tile_{x}_{y}.png"
        save_png(tile_canvas, os.path.join(tiles_dir, filename), compress_level=1, verbose=False)
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
        "channels": TARGET_CHANNELS_HEX,
        "tiles": sorted(manifest_tiles, key=lambda t: (t["y"], t["x"])),
    }
    manifest_path = os.path.join(output_viz_dir, "patch_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {manifest_path}")
    print(f"Open patch_stitcher.html and choose this manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tiff", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./my_results")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_png_raw_mb", type=int, default=220)
    parser.add_argument("--max_png_side", type=int, default=14000)
    args = parser.parse_args()
    temp_tiles_dir = os.path.join(args.output_dir, "temp_tiles")
    predictions_dir = os.path.join(args.output_dir, "predictions")
    visualizations_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(predictions_dir, exist_ok=True)
    if not glob.glob(os.path.join(temp_tiles_dir, "*.png")):
        tile_image(args.input_tiff, temp_tiles_dir)
    img_paths = glob.glob(os.path.join(temp_tiles_dir, "*.png"))
    if not img_paths:
        print("No valid tiles found.")
        return
    if archs is None:
        print("Model load failed: archs module not found.")
        return
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        local_dir = snapshot_download(repo_id="prov-gigatime/GigaTIME")
        weights_path = os.path.join(local_dir, "model.pth")
        model = archs.__dict__["gigatime"](num_classes=23, input_channels=3)
        checkpoint = torch.load(weights_path, map_location="cpu")
        model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint.items()})
        model = model.to(device).eval()
    except Exception as e:
        print(f"Model load failed: {e}")
        return
    dataset = InferenceDataset(img_paths, transform=A.Compose([A.Resize(512, 512), A.Normalize()]))
    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_skip_none
    )
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if not batch:
                continue
            imgs, names = batch
            preds = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            for i, name in enumerate(names):
                np.save(os.path.join(predictions_dir, name.replace(".png", ".npy")), preds[i])
    generate_he_overview(args.input_tiff, temp_tiles_dir)
    generate_heatmaps(
        predictions_dir,
        visualizations_dir,
        max_raw_mb=args.max_png_raw_mb,
        max_side=args.max_png_side,
    )


if __name__ == "__main__":
    main()
