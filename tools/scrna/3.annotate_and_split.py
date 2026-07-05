import re
import time
import argparse
import shutil
import os
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["svg.fonttype"] = "none"
plt.switch_backend("agg")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(m):
    print(f"[{_now()}] {m}", flush=True)


def smart_find_cluster_key(adata: ad.AnnData, csv_clusters: set, default_key=None) -> str:
    if default_key and default_key in adata.obs.columns:
        obs_vals = set(adata.obs[default_key].astype(str))
        missing = csv_clusters - obs_vals
        if not missing:
            log(f"message '{default_key}' message CSV message ")
            return default_key
        else:
            ratio = len(csv_clusters - missing) / len(csv_clusters) if len(csv_clusters) > 0 else 0
            log(f"message message '{default_key}' message {ratio:.1%} message CSV message ID ")
    log("Status update.")
    candidates = []
    potential_cols = [c for c in adata.obs.columns if len(adata.obs) > 0]
    for col in potential_cols:
        try:
            col_vals = set(adata.obs[col].dropna().astype(str))
            if len(col_vals) < 2:
                continue
            common = csv_clusters & col_vals
            recall = len(common) / len(csv_clusters) if len(csv_clusters) > 0 else 0
            if recall > 0.5:
                candidates.append((col, recall, len(col_vals)))
        except Exception:
            continue
    target_count = len(csv_clusters)

    def score_candidate(item):
        col, recall, n_cats = item
        diff = abs(n_cats - target_count)
        return (recall, -diff)

    candidates.sort(key=score_candidate, reverse=True)
    if candidates:
        best_col, best_recall, best_n = candidates[0]
        log(f"message '{best_col}' (message {best_recall:.1%}, message {best_n})")
        return best_col
    log("Status update.")
    return _fallback_autodetect(adata)


def _fallback_autodetect(adata: ad.AnnData) -> str:
    best = adata.uns.get("best_cluster_key", None)
    if isinstance(best, str) and best in adata.obs.columns:
        return best
    best2 = adata.uns.get("best_leiden_key", None)
    if isinstance(best2, str) and best2 in adata.obs.columns:
        return best2
    if "leiden" in adata.obs.columns:
        return "leiden"
    cand = [c for c in adata.obs.columns if isinstance(c, str) and c.startswith("leiden_")]
    if cand:

        def _safe_float(x: str) -> float:
            try:
                parts = x.split("_")
                if len(parts) >= 3:
                    return float(parts[1] + "." + parts[2])
                return 0.0
            except Exception:
                return 0.0

        cand_sorted = sorted(cand, key=_safe_float, reverse=True)
        return cand_sorted[0]
    return "leiden"


def autodetect_sample_key(adata: ad.AnnData) -> str:
    for k in ["project", "sampleid", "sample_id", "Sample", "sample", "orig.ident"]:
        if k in adata.obs.columns:
            log(f"message {k}")
            return k
    for k in adata.obs.columns:
        if pd.api.types.is_object_dtype(adata.obs[k]) or pd.api.types.is_categorical_dtype(
            adata.obs[k]
        ):
            return k
    raise ValueError("Status update.")


def load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    cls_col = None
    for c in df.columns:
        if c.lower() == "cluster":
            cls_col = c
            break
    if not cls_col:
        raise ValueError(f"CSVmessage 'cluster' message message {df.columns}")
    type_col = None
    for c in df.columns:
        if c.lower() in ["predicted_cell_type", "cell_type", "annotation"]:
            type_col = c
            break
    if not type_col:
        raise ValueError(f"CSVmessage 'predicted_cell_type' message ")
    df = df.rename(columns={cls_col: "cluster", type_col: "predicted_cell_type"})
    df["cluster"] = df["cluster"].astype(str).str.strip()
    df["predicted_cell_type"] = df["predicted_cell_type"].astype(str).str.strip()
    return df


def sanitize_obs_strings(adata: ad.AnnData):
    for c in adata.obs.columns:
        if adata.obs[c].dtype == "object":
            adata.obs[c] = adata.obs[c].astype(str)


def save_slices_by_key(adata: ad.AnnData, outdir: Path, key: str, prefix: str):
    subdir = outdir / f"annotated_by_{prefix}"
    subdir.mkdir(parents=True, exist_ok=True)
    if key not in adata.obs:
        return
    groups = adata.obs.groupby(key)
    for val, idx in groups.groups.items():
        if pd.isna(val) or str(val) == "nan":
            continue
        safe_val = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(val))
        sub = adata[idx].copy()
        sanitize_obs_strings(sub)
        try:
            sub.write(subdir / f"{prefix}_{safe_val}.h5ad")
        except Exception as e:
            log(f"Warning: message {safe_val} message: {e}")


def ensure_umap(adata: ad.AnnData, neighbors_n=15):
    has_umap = "X_umap" in adata.obsm_keys()
    has_neighbors = "neighbors" in adata.uns
    if has_umap:
        log("Status update.")
        return
    log("Status update.")
    if not has_neighbors:
        rep = "X_pca"
        if "X_pca_harmony" in adata.obsm_keys():
            rep = "X_pca_harmony"
        elif "X_pca" not in adata.obsm_keys():
            log("Status update.")
            sc.tl.pca(adata, n_comps=min(30, adata.n_vars), svd_solver="arpack")
        log(f"  > message {rep} message Neighbors...")
        sc.pp.neighbors(adata, use_rep=rep, n_neighbors=neighbors_n)
    log("Status update.")
    sc.tl.umap(adata)


def plot_contour_umap(adata: ad.AnnData, outdir: Path, key: str, filename: str):
    if key not in adata.obs.columns:
        return
    try:
        if f"{key}_colors" not in adata.uns:
            palette = sc.pl.palettes.godsnot_102
            sc.pl.umap(adata, color=key, palette=palette, show=False)
            plt.close()
        colors = adata.uns[f"{key}_colors"]
        if not pd.api.types.is_categorical_dtype(adata.obs[key]):
            adata.obs[key] = adata.obs[key].astype("category")
        categories = adata.obs[key].cat.categories
        color_map = dict(zip(categories, colors))
        counts = adata.obs[key].value_counts()
        umap_coords = adata.obsm["X_umap"]
        data_df = pd.DataFrame(umap_coords, columns=["umap_1", "umap_2"])
        data_df["label"] = adata.obs[key].values
        fig, ax = plt.subplots(figsize=(8, 8))
        sorted_cats = counts.sort_values(ascending=False).index
        for cat in sorted_cats:
            if cat not in categories:
                continue
            sub_df = data_df[data_df["label"] == cat]
            color = color_map.get(cat, "#888888")
            n_points = len(sub_df)
            if n_points == 0:
                continue
            if n_points >= 5:
                try:
                    sns.kdeplot(
                        data=sub_df,
                        x="umap_1",
                        y="umap_2",
                        fill=True,
                        color=color,
                        alpha=0.15,
                        thresh=0.05,
                        levels=1,
                        ax=ax,
                        legend=False,
                        zorder=1,
                        warn_singular=False,
                    )
                    sns.kdeplot(
                        data=sub_df,
                        x="umap_1",
                        y="umap_2",
                        fill=False,
                        color=color,
                        linewidth=1.5,
                        alpha=0.9,
                        thresh=0.05,
                        levels=1,
                        ax=ax,
                        legend=False,
                        zorder=1,
                        warn_singular=False,
                    )
                except Exception:
                    pass
            point_size = np.clip(8000 / max(len(data_df), 1), 3, 25)
            ax.scatter(
                sub_df["umap_1"],
                sub_df["umap_2"],
                c=[color],
                s=point_size,
                label=None,
                alpha=0.8,
                edgecolors="none",
                zorder=2,
            )
        ax.set_xlabel("umap_1", fontsize=12)
        ax.set_ylabel("umap_2", fontsize=12)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.text(
            0.02,
            1.02,
            f"N = {adata.n_obs}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="normal",
            ha="left",
        )
        legend_handles = []
        for cat in categories:
            count = counts.get(cat, 0)
            if count == 0:
                continue
            label_str = f"{cat}({count})"
            handle = mpatches.Patch(color=color_map.get(cat, "#888888"), label=label_str)
            legend_handles.append(handle)
        legend = ax.legend(
            handles=legend_handles,
            title=key,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            fontsize=11,
            title_fontsize=13,
            markerscale=1.2,
        )
        try:
            legend.get_title().set_ha("left")
        except Exception:
            pass
        plt.savefig(outdir / filename, dpi=300, bbox_inches="tight")
        plt.close()
        log(f"message: {filename}")
    except Exception as e:
        log(f"message ({filename}): {e}")
        import traceback

        traceback.print_exc()


def _try_make_step4_link_or_copy(src: Path, dst: Path):
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except Exception:
        pass
    try:
        os.symlink(str(src), str(dst))
        return "symlink"
    except Exception:
        pass
    try:
        shutil.copyfile(str(src), str(dst))
        return "copy"
    except Exception:
        return "skip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_data", required=False)
    ap.add_argument("--annotations_csv", required=False)
    ap.add_argument("--adata_h5ad", required=False, help="Alias of --ann_data")
    ap.add_argument("--csv", required=False, help="Alias of --annotations_csv")
    ap.add_argument("--cluster_key", default=None, help="Status update.")
    ap.add_argument("--sample_key", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample_only", action="store_true", default=False, help="Status update.")
    ap.add_argument("--do_split", action="store_true", default=False, help="Status update.")
    ap.add_argument("--cell_type_key", default="cell_type")
    args = ap.parse_args()
    ann_data = args.ann_data or args.adata_h5ad
    annotations_csv = args.annotations_csv or args.csv
    if not ann_data or not annotations_csv:
        ap.error(
            "the following arguments are required: --ann_data/--adata_h5ad and --annotations_csv/--csv"
        )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"message AnnData: {ann_data}")
    adata = sc.read_h5ad(ann_data)
    log(f"message CSV: {annotations_csv}")
    ann_df = load_annotations(Path(annotations_csv))
    csv_clusters = set(ann_df["cluster"].unique())
    log(f"CSV message {len(csv_clusters)} message ID")
    cluster_key = smart_find_cluster_key(adata, csv_clusters, default_key=args.cluster_key)
    log(f"message: ")
    sample_key = None
    if args.sample_key and args.sample_key in adata.obs.columns:
        sample_key = args.sample_key
    else:
        try:
            sample_key = autodetect_sample_key(adata)
        except Exception as e:
            log(f"[WARN] message sample_key message sample message/message  {e}")
            sample_key = None
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
    clust2type = dict(zip(ann_df["cluster"], ann_df["predicted_cell_type"]))
    new_col = adata.obs[cluster_key].map(clust2type)
    n_unknown = new_col.isna().sum()
    if n_unknown > 0:
        log(f"message: message {n_unknown} message CSV message ID message Unknown ")
        new_col = new_col.fillna("Unknown")
    adata.obs[args.cell_type_key] = new_col.astype("category")
    if (adata.obs[args.cell_type_key] == "Unknown").any():
        log("Status update.")
        ensure_umap(adata)
        if "connectivities" in adata.obsp:
            conn = adata.obsp["connectivities"]
        else:
            sc.pp.neighbors(adata, n_neighbors=15)
            conn = adata.obsp["connectivities"]
        labels = adata.obs[args.cell_type_key].values.copy()
        is_unknown = labels == "Unknown"
        unknown_indices = np.where(is_unknown)[0]
        known_indices = np.where(~is_unknown)[0]
        if len(known_indices) > 0:
            log(f"  > message: {len(unknown_indices)}")
            fixed_count = 0
            conn_csr = conn.tocsr()
            for idx in unknown_indices:
                row = conn_csr[idx]
                if row.nnz == 0:
                    continue
                indices = row.indices
                weights = row.data
                neigh_labels = labels[indices]
                valid_mask = neigh_labels != "Unknown"
                if not np.any(valid_mask):
                    continue
                valid_labels = neigh_labels[valid_mask]
                valid_weights = weights[valid_mask]
                unique_l = np.unique(valid_labels)
                best_l = None
                best_score = -1
                for l in unique_l:
                    score = np.sum(valid_weights[valid_labels == l])
                    if score > best_score:
                        best_score = score
                        best_l = l
                if best_l:
                    labels[idx] = best_l
                    fixed_count += 1
            adata.obs[args.cell_type_key] = labels
            adata.obs[args.cell_type_key] = adata.obs[args.cell_type_key].astype("category")
            log(f"  > message message {fixed_count} message ")
    ensure_umap(adata)
    log("Status update.")
    sc.pl.umap(adata, color=[args.cell_type_key], show=False, frameon=False, legend_loc="on data")
    plt.savefig(outdir / "umap_simple.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("Status update.")
    plot_contour_umap(adata, outdir, args.cell_type_key, "umap_celltype_contour.png")
    log("Status update.")
    for k in [cluster_key, args.cell_type_key, sample_key]:
        if k:
            counts = adata.obs[k].value_counts().reset_index()
            counts.columns = [k, "count"]
            safe_k = str(k).replace("/", "_")
            counts.to_csv(outdir / f"stats_{safe_k}.csv", index=False)
    try:
        adata.uns["step3_info"] = {
            "timestamp": _now(),
            "cluster_key_used": str(cluster_key),
            "cell_type_key": str(args.cell_type_key),
            "annotations_csv": str(Path(annotations_csv).resolve()),
            "ann_data": str(Path(ann_data).resolve()),
        }
    except Exception:
        pass
    final_h5ad = outdir / "annotated_complete.h5ad"
    sanitize_obs_strings(adata)
    adata.write(final_h5ad)
    log(f"message: {final_h5ad}")
    step4_h5ad = outdir / "annotated_for_step4.h5ad"
    mode = _try_make_step4_link_or_copy(final_h5ad, step4_h5ad)
    if mode != "skip":
        log(f"[Step4] message {step4_h5ad}  message {mode} ")
    else:
        log(
            f"[Step4] message annotated_for_step4.h5ad message Step4 message annotated_complete.h5ad "
        )
    do_split = bool(args.do_split or args.sample_only)
    if not do_split:
        log("Status update.")
    else:
        if args.sample_only:
            log("Status update.")
            if sample_key:
                save_slices_by_key(adata, outdir, sample_key, "sample")
            else:
                log("Status update.")
        else:
            log("Status update.")
            if sample_key:
                save_slices_by_key(adata, outdir, sample_key, "sample")
            else:
                log("Status update.")
            save_slices_by_key(adata, outdir, cluster_key, "cluster")
            save_slices_by_key(adata, outdir, args.cell_type_key, "celltype")
    log("Status update.")


if __name__ == "__main__":
    main()
