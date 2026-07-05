import os


def _early_configure_threads():
    n_cpu = os.cpu_count() or 8
    try:
        max_threads = int(os.environ.get("SCRNA_MAX_THREADS", str(n_cpu)))
    except Exception:
        max_threads = n_cpu
    n_use = int(min(n_cpu, max_threads))
    numba_n = int(max(1, min(8, n_use)))

    def _maybe_set(k, v):
        if os.environ.get(k) is None:
            os.environ[k] = str(v)

    for k in [
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        _maybe_set(k, n_use)
    os.environ["NUMBA_NUM_THREADS"] = str(numba_n)
    return n_use, numba_n


_EARLY_N_USE, _EARLY_NUMBA_N = _early_configure_threads()
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

sc.settings.verbosity = 2


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{_now()}] {msg}", flush=True)


def _set_fast_threads():
    n_use = int(_EARLY_N_USE)
    numba_env = int(os.environ.get("NUMBA_NUM_THREADS", str(_EARLY_NUMBA_N)))
    safe_jobs = int(max(1, min(n_use, numba_env)))
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(n_use)
        tp_msg = f"BLAS/OMP={n_use}"
    except Exception:
        tp_msg = f"message BLAS/OMP={n_use}"
    try:
        sc.settings.n_jobs = safe_jobs
    except Exception:
        pass
    numba_msg = f"NUMBA={numba_env}"
    try:
        from numba import config as numba_config, set_num_threads as numba_set_num_threads

        numba_use = int(
            max(1, min(safe_jobs, int(getattr(numba_config, "NUMBA_NUM_THREADS", numba_env))))
        )
        numba_set_num_threads(numba_use)
        numba_msg = f"NUMBA={numba_use}"
    except Exception as e:
        numba_msg = f"NUMBA message {e} "
    log(f"[THREADS] {tp_msg} message scanpy_n_jobs={safe_jobs} {numba_msg} message ")


_set_fast_threads()


def list_10x_children(parent: Path) -> List[Path]:
    subs = []
    for p in sorted(parent.iterdir()):
        if not p.is_dir():
            continue
        has_mtx = (p / "matrix.mtx").exists() or (p / "matrix.mtx.gz").exists()
        has_filtered = (p / "filtered_feature_bc_matrix").exists()
        has_raw = (p / "raw_feature_bc_matrix").exists()
        if has_mtx or has_filtered or has_raw:
            subs.append(p)
    return subs


def read_one_10x(subdir: Path) -> ad.AnnData:
    adata = sc.read_10x_mtx(str(subdir), var_names="gene_symbols", cache=False)
    adata.obs["project"] = str(subdir.name)
    return adata


def load_from_10x_parent(parent: Path, sample_key: str = "project") -> ad.AnnData:
    log(f"message 10x message   {parent}")
    subs = list_10x_children(parent)
    if len(subs) == 0:
        raise FileNotFoundError(f"message {parent} message 10x message ")
    log(f"message {len(subs)} message 10x message ")
    for i, p in enumerate(subs, 1):
        log(f"  [{i}/{len(subs)}] {p}")
    adatas = []
    for p in subs:
        t0 = time.time()
        a = read_one_10x(p)
        log(f"  message {p.name} message {time.time() - t0:.2f}s")
        adatas.append(a)
    log("Status update.")
    t0 = time.time()
    adata = ad.concat(
        adatas, axis=0, join="outer", label="batch", keys=[x.obs["project"][0] for x in adatas]
    )
    adata.obs[sample_key] = adata.obs["batch"].astype(str).values
    log(f"  message message {time.time() - t0:.2f}s")
    return adata


def _normalize_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def resolve_batch_key(adata: ad.AnnData, preferred: str) -> str:
    obs_cols = list(adata.obs.columns)
    if preferred in obs_cols:
        return preferred
    norm_map = {c: _normalize_colname(c) for c in obs_cols}
    want = _normalize_colname(preferred)
    for c, n in norm_map.items():
        if n == want:
            return c
    aliases = [
        "orig.dent",
        "orig.ident",
        "origident",
        "origdent",
        "project",
        "sample",
        "batch",
        "donor",
        "patient",
        "library",
    ]
    viable = []
    for alias in aliases:
        n_alias = _normalize_colname(alias)
        for c, n in norm_map.items():
            if n == n_alias:
                try:
                    nunq = int(pd.Series(adata.obs[c].astype(str)).nunique())
                except Exception:
                    nunq = 0
                viable.append((c, nunq))
    if len(viable) > 0:
        viable.sort(key=lambda x: x[1], reverse=True)
        if viable[0][1] > 1:
            return viable[0][0]
    adata.obs["sample"] = "sample"
    return "sample"


def sanitize_X_inplace(adata: ad.AnnData):
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr(copy=False)
        data = X.data
        if data is None:
            return
        if np.isnan(data).any():
            log("Status update.")
            np.nan_to_num(data, copy=False, nan=0.0)
        if (data < 0).any():
            log("Status update.")
            data[data < 0] = 0
        adata.X = X
    else:
        X = np.asarray(X, dtype=np.float32)
        if np.isnan(X).any():
            log("Status update.")
            np.nan_to_num(X, copy=False, nan=0.0)
        if (X < 0).any():
            log("Status update.")
            X[X < 0] = 0
        adata.X = X


def load_input(
    input_10x_parent: Optional[str], input_h5ad: Optional[str], batch_key: str
) -> ad.AnnData:
    if input_10x_parent:
        adata = load_from_10x_parent(Path(input_10x_parent), sample_key=batch_key)
        real_key = batch_key
    else:
        log(f"message message h5ad   {input_h5ad}")
        adata = sc.read_h5ad(input_h5ad)
        log("Status update.")
        try:
            sanitize_X_inplace(adata)
        except Exception as e:
            log(f"[WARN] sanitize_X_inplace message {e} message ")
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.filter_genes(adata, min_cells=1)
        real_key = resolve_batch_key(adata, batch_key)
        if real_key != batch_key:
            log(f"[INFO] batch_key='{batch_key}' message obs['{real_key}'] ")
        adata.obs[real_key] = adata.obs[real_key].astype(str).values
    log(f"  message (Cells={adata.n_obs}, Genes={adata.n_vars})")
    adata.uns["_resolved_batch_key"] = real_key
    return adata


def ensure_basic_qc_fields(adata: ad.AnnData):
    if "mt" not in adata.var.columns:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    if "ribo" not in adata.var.columns:
        vup = adata.var_names.str.upper()
        adata.var["ribo"] = vup.str.startswith("RPL") | vup.str.startswith("RPS")
    if "hb" not in adata.var.columns:
        vup = adata.var_names.str.upper()
        adata.var["hb"] = (
            vup.str.startswith("HB") | vup.str.startswith("HBA") | vup.str.startswith("HBB")
        )
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=False, inplace=True
    )


def _soft_qc_thresholds(sub: pd.DataFrame) -> Dict[str, float]:
    ng = sub["n_genes_by_counts"].astype(float).values
    tc = sub["total_counts"].astype(float).values
    mt = sub["pct_counts_mt"].astype(float).values if "pct_counts_mt" in sub else np.zeros(len(sub))
    rb = (
        sub["pct_counts_ribo"].astype(float).values
        if "pct_counts_ribo" in sub
        else np.zeros(len(sub))
    )
    hb = sub["pct_counts_hb"].astype(float).values if "pct_counts_hb" in sub else np.zeros(len(sub))

    def low_iqr(v):
        q1, q3 = np.quantile(v, [0.25, 0.75])
        iqr = max(q3 - q1, 1e-9)
        return q1 - 1.5 * iqr

    ng_low = max(100.0, min(np.quantile(ng, 0.01), low_iqr(ng)))
    tc_low = max(300.0, min(np.quantile(tc, 0.01), low_iqr(tc)))
    mt_high = max(10.0, np.quantile(mt, 0.98)) if mt.size else 100.0
    rb_high = min(60.0, np.quantile(rb, 0.99)) if rb.size else 100.0
    hb_high = min(15.0, np.quantile(hb, 0.99)) if hb.size else 100.0
    return dict(
        ng_low=float(ng_low),
        tc_low=float(tc_low),
        mt_high=float(mt_high),
        rb_high=float(rb_high),
        hb_high=float(hb_high),
    )


def _apply_thresholds(sub: pd.DataFrame, th: Dict[str, float]) -> np.ndarray:
    conds = []
    conds.append(sub["n_genes_by_counts"].values > th["ng_low"])
    conds.append(sub["total_counts"].values > th["tc_low"])
    if "pct_counts_mt" in sub:
        conds.append(sub["pct_counts_mt"].values < th["mt_high"])
    if "pct_counts_ribo" in sub:
        conds.append(sub["pct_counts_ribo"].values < th["rb_high"])
    if "pct_counts_hb" in sub:
        conds.append(sub["pct_counts_hb"].values < th["hb_high"])
    m = np.ones(len(sub), dtype=bool)
    for c in conds:
        m &= c
    return m


def _adaptive_keep_mask_per_batch(
    sub: pd.DataFrame, target_min_keep: float = 0.60, max_relax_rounds: int = 3
) -> Tuple[np.ndarray, Dict[str, float], float]:
    th = _soft_qc_thresholds(sub)
    mask = _apply_thresholds(sub, th)
    keep_rate = mask.mean() if len(mask) else 1.0
    round_id = 0
    while keep_rate < target_min_keep and round_id < max_relax_rounds:
        round_id += 1
        ng = sub["n_genes_by_counts"].values
        tc = sub["total_counts"].values
        if round_id == 1:
            th["ng_low"] = max(80.0, min(th["ng_low"], np.quantile(ng, 0.005)))
            th["tc_low"] = max(200.0, min(th["tc_low"], np.quantile(tc, 0.005)))
            th["mt_high"] = max(15.0, th["mt_high"])
        elif round_id == 2 and "pct_counts_mt" in sub:
            mt = sub["pct_counts_mt"].values
            th["mt_high"] = max(th["mt_high"], np.quantile(mt, 0.995))
        else:
            if "pct_counts_ribo" in sub:
                th["rb_high"] = max(th["rb_high"], 70.0)
            if "pct_counts_hb" in sub:
                th["hb_high"] = max(th["hb_high"], 20.0)
        mask = _apply_thresholds(sub, th)
        keep_rate = mask.mean() if len(mask) else 1.0
    return mask.astype(bool), th, float(keep_rate)


def maybe_qc(adata: ad.AnnData, species: str, batch_key: str, auto_qc: bool = False):
    ensure_basic_qc_fields(adata)
    if not auto_qc:
        log("Status update.")
        return
    if batch_key not in adata.obs.columns:
        adata.obs[batch_key] = "sample"
    log("Status update.")
    before = adata.n_obs
    keep_mask = np.zeros(adata.n_obs, dtype=bool)
    obs_index = adata.obs.index
    for b, sub in adata.obs.groupby(batch_key, sort=False):
        m, th, rate = _adaptive_keep_mask_per_batch(sub)
        pos = obs_index.get_indexer(sub.index[m])
        pos = pos[pos >= 0]
        if len(pos) > 0:
            keep_mask[pos] = True
        kept = int(m.sum())
        total = int(len(sub))
        rate_pct = f"{rate * 100:.1f}%"
        log(
            f"  - message {b}: message>ng>{th['ng_low']:.0f}, >tc>{th['tc_low']:.0f}, mt<{th['mt_high']:.1f}%, "
            f"ribo<{th['rb_high']:.1f}%, hb<{th['hb_high']:.1f}%   message {kept:,}/{total:,} ({rate_pct})"
        )
    if keep_mask.sum() == 0:
        log("Status update.")
    else:
        adata._inplace_subset_obs(keep_mask)
    log(f"QC message {before:,}   {adata.n_obs:,} (-{before - adata.n_obs:,})")


def stash_counts_and_normlayer(adata: ad.AnnData):
    if hasattr(adata.X, "tocsr"):
        adata.layers["counts"] = adata.X.tocsr()
    else:
        adata.layers["counts"] = sparse.csr_matrix(adata.X)
    log("Normalize_total(target_sum=1e4) + log1p   layers['log1p_cp10k'] ...")
    tmp = ad.AnnData(adata.layers["counts"].copy(), obs=adata.obs.copy(), var=adata.var.copy())
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    tmp.X = (
        tmp.X.tocsr().astype(np.float32)
        if sparse.issparse(tmp.X)
        else sparse.csr_matrix(tmp.X.astype(np.float32))
    )
    adata.layers["log1p_cp10k"] = tmp.X
    del tmp


def _starts_any_upper(index_like, prefixes) -> np.ndarray:
    up = pd.Index([str(s).upper() for s in index_like])
    mask = np.zeros(len(up), dtype=bool)
    for pre in prefixes:
        res = up.str.startswith(pre)
        res = res.to_numpy() if hasattr(res, "to_numpy") else np.asarray(res)
        mask |= res.astype(bool)
    return mask


def _is_sparse_int_counts(X) -> bool:
    try:
        return (sparse.issparse(X) and np.issubdtype(X.data.dtype, np.integer)) or (
            not sparse.issparse(X) and np.issubdtype(np.asarray(X).dtype, np.integer)
        )
    except Exception:
        return False


def normalize_hvg_scale(
    adata: ad.AnnData, batch_key: str, flavor_try=("seurat", "seurat_v3"), n_top_genes=3000
):
    stash_counts_and_normlayer(adata)
    hv = None
    tried = []
    for flv in flavor_try:
        try:
            if flv == "seurat_v3" and not _is_sparse_int_counts(adata.layers["counts"]):
                tried.append(f"{flv}(message counts message)")
                continue
            X_for_hvg = (
                adata.layers["counts"] if flv == "seurat_v3" else adata.layers["log1p_cp10k"]
            )
            tmp = ad.AnnData(X_for_hvg, obs=adata.obs.copy(), var=adata.var.copy())
            log(f"HVG flavor={flv} n_top_genes={n_top_genes} batch_key={batch_key}  ...")
            sc.pp.highly_variable_genes(
                tmp, flavor=flv, n_top_genes=n_top_genes, batch_key=batch_key
            )
            hv_series = tmp.var["highly_variable"]
            hv = hv_series.to_numpy() if hasattr(hv_series, "to_numpy") else np.asarray(hv_series)
            tried.append(flv)
            break
        except Exception as e:
            tried.append(f"{flv}(message:{e})")
            log(f"[WARN] HVG flavor={flv} message {e}")
    if hv is None:
        log(f"[WARN] HVG message message {'; '.join(map(str, tried))}  message ")
        X = adata.layers["log1p_cp10k"]
        X = X.A if hasattr(X, "A") else (X.toarray() if sparse.issparse(X) else np.asarray(X))
        var = np.asarray(np.var(X, axis=0)).ravel()
        top_idx = np.argsort(-var)[: min(n_top_genes, adata.n_vars)]
        hv = np.zeros(adata.n_vars, dtype=bool)
        hv[top_idx] = True
    bad = _starts_any_upper(adata.var_names, ["MT-", "RPL", "RPS", "HB"])
    hv = hv & (~bad)
    if hv.sum() < 200:
        log(f"[WARN] HVG message {hv.sum()}  message 3000 message ")
        X = adata.layers["log1p_cp10k"]
        X = X.A if hasattr(X, "A") else (X.toarray() if sparse.issparse(X) else np.asarray(X))
        var = np.asarray(np.var(X, axis=0)).ravel()
        top_idx = np.argsort(-var)[: min(3000, adata.n_vars)]
        hv = np.zeros(adata.n_vars, dtype=bool)
        hv[top_idx] = True
        hv = hv & (~bad)
    adata.var["highly_variable"] = hv
    log(f"HVG message {int(hv.sum()):,}/{adata.n_vars:,} message MT/RPL/RPS/HB ")
    adata.uns["_do_regress_out"] = True


S_GENES = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "POLA1",
    "CHAF1B",
    "BRIP1",
    "E2F8",
]
G2M_GENES = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
    "TMPO",
    "CENPF",
    "TACC3",
    "FAM64A",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPA",
]


def _match_genes_present(var_names: pd.Index, genes: List[str]) -> List[str]:
    up2orig: Dict[str, str] = {}
    for g in var_names:
        up = str(g).upper()
        if up not in up2orig:
            up2orig[up] = g
    found = []
    for g in genes:
        up = g.upper()
        if up in up2orig:
            found.append(up2orig[up])
    seen = set()
    uniq = []
    for g in found:
        if g not in seen:
            uniq.append(g)
            seen.add(g)
    return uniq


_DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024


def _build_covariate_Z(work: ad.AnnData, keys_reg: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    cols = [np.ones((work.n_obs, 1), dtype=np.float32)]
    for k in dict.fromkeys(keys_reg):
        v = work.obs[k].values.astype(np.float32)
        v = np.nan_to_num(v, copy=False)
        mu = float(v.mean())
        sd = float(v.std() + 1e-8)
        v = (v - mu) / sd
        cols.append(v.reshape(-1, 1))
    Z = np.concatenate(cols, axis=1)
    ZtZ = Z.T @ Z
    A = np.linalg.pinv(ZtZ) @ Z.T
    return Z, A


def _auto_chunk_cols(n_cells: int, target_bytes: int, safety_factor: float = 3.5) -> int:
    bytes_per_col = n_cells * 4 * safety_factor
    cols = int(target_bytes // max(1.0, bytes_per_col))
    return int(max(32, min(2048, cols)))


def regress_out_chunked(
    work: ad.AnnData, keys_reg: List[str], max_bytes_per_chunk: int = _DEFAULT_CHUNK_BYTES
):
    if work.n_vars == 0 or work.n_obs == 0:
        return
    Z, A = _build_covariate_Z(work, keys_reg)
    n, g = int(work.n_obs), int(work.n_vars)
    cols = _auto_chunk_cols(n, target_bytes=max_bytes_per_chunk)
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".regress_residuals.f32")
    os.close(fd)
    R = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, g))
    log(f"Regress_out message n_cells={n}, n_genes={g}, chunk={cols} memmap='{path}' ")
    X = work.X
    for c0 in range(0, g, cols):
        c1 = min(g, c0 + cols)
        if sparse.issparse(X):
            Y = X[:, c0:c1].toarray().astype(np.float32, copy=False)
        else:
            Y = np.asarray(X[:, c0:c1], dtype=np.float32, order="C")
        coef = A @ Y
        Y_hat = Z @ coef
        R[:, c0:c1] = Y - Y_hat
    work.X = R
    work.uns["_memmap_X_path"] = path


def scale_chunked(
    work: ad.AnnData, max_value: float = 10.0, max_bytes_per_chunk: int = _DEFAULT_CHUNK_BYTES
):
    if work.n_vars == 0 or work.n_obs == 0:
        return
    n, g = int(work.n_obs), int(work.n_vars)
    cols = _auto_chunk_cols(n, target_bytes=max_bytes_per_chunk)
    X = work.X
    means = np.zeros((g,), dtype=np.float32)
    for c0 in range(0, g, cols):
        c1 = min(g, c0 + cols)
        data = np.asarray(X[:, c0:c1], dtype=np.float32)
        means[c0:c1] = data.mean(axis=0, dtype=np.float64).astype(np.float32)
    eps = 1e-8
    for c0 in range(0, g, cols):
        c1 = min(g, c0 + cols)
        data = np.asarray(X[:, c0:c1], dtype=np.float32)
        diff = data - means[c0:c1]
        var = (diff.astype(np.float64) ** 2).mean(axis=0)
        std = np.sqrt(var).astype(np.float32) + eps
        data = diff / std
        np.clip(data, -max_value, max_value, out=data)
        X[:, c0:c1] = data


def pca_neighbors_umap(
    adata: ad.AnnData,
    use_harmony: bool,
    batch_key: str,
    auto_dim: bool,
    random_state: int,
    species: str,
):
    hv = adata.var.get(
        "highly_variable", pd.Series([True] * adata.n_vars, index=adata.var_names)
    ).values
    if hv.sum() == 0:
        log("Status update.")
        hv = np.ones(adata.n_vars, dtype=bool)
    Xh = adata.layers["log1p_cp10k"][:, hv]
    work = ad.AnnData(Xh.copy(), obs=adata.obs.copy(), var=adata.var.loc[hv].copy())
    keys_reg = []
    if {"total_counts", "pct_counts_mt"}.issubset(work.obs.columns):
        keys_reg += ["total_counts", "pct_counts_mt"]
    for k in ["pct_counts_ribo", "pct_counts_hb"]:
        if k in work.obs.columns:
            keys_reg.append(k)
    s_present = _match_genes_present(work.var_names, S_GENES)
    g_present = _match_genes_present(work.var_names, G2M_GENES)
    if len(s_present) > 0 and len(g_present) > 0:
        log(f"message S={len(s_present)}, G2M={len(g_present)} ...")
        try:
            sc.tl.score_genes_cell_cycle(work, s_genes=s_present, g2m_genes=g_present)
            keys_reg += ["S_score", "G2M_score"]
        except Exception as e:
            log(f"[WARN] message {e}")
    log(f"Regress_out keys={keys_reg} ...")
    regress_out_chunked(work, keys_reg=keys_reg)
    log("Scale (clip=10) ...")
    scale_chunked(work)
    log("PCA (auto_dim) ...")
    sc.tl.pca(work, svd_solver="arpack", random_state=random_state)
    work.obsm["X_pca"] = np.asarray(work.obsm["X_pca"], dtype=np.float32)
    try:
        path = work.uns.get("_memmap_X_path", None)
        if path and isinstance(work.X, np.memmap):
            try:
                work.X._mmap.close()
            except Exception:
                pass
            if path and os.path.exists(path):
                os.remove(path)
            log(f"[INFO] message memmap {path}")
    except Exception:
        pass
    if "pca" in work.uns and "variance_ratio" in work.uns["pca"]:
        var_ratio = work.uns["pca"]["variance_ratio"]
        csum = np.cumsum(var_ratio)
        n90 = int(np.searchsorted(csum, 0.90) + 1)
        elbow = (
            int(np.argmax(var_ratio[1:] < (var_ratio[0] * 0.02)) + 2) if len(var_ratio) > 2 else n90
        )
        n_use = (
            int(max(20, min(35, n90, elbow))) if auto_dim else min(35, work.obsm["X_pca"].shape[1])
        )
    else:
        n_use = min(35, work.obsm["X_pca"].shape[1])
    if use_harmony:
        if batch_key in work.obs.columns and pd.Series(work.obs[batch_key]).nunique() <= 1:
            log(f"[INFO] batch_key='{batch_key}' message 1 message message Harmony ")
        else:
            log(f"Harmony message (batch_key={batch_key}) ...")
            try:
                import harmonypy as hm

                ho = hm.run_harmony(
                    work.obsm["X_pca"][:, :n_use],
                    work.obs,
                    [batch_key],
                    max_iter_harmony=20,
                    random_state=random_state,
                )
                work.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr.T, dtype=np.float32)
            except Exception as e:
                log(f"[WARN] Harmony message {e} message PCA ")
    if "X_pca_harmony" in work.obsm:
        adata.obsm["X_pca_harmony"] = work.obsm["X_pca_harmony"]
    adata.obsm["X_pca"] = work.obsm["X_pca"]
    adata.uns["pca"] = work.uns.get("pca", {})
    if "neighbors" not in adata.uns or "connectivities" not in adata.obsp:
        log("Status update.")
        rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm_keys() else "X_pca"
        n = adata.n_obs
        n_neighbors = 12 if n < 20000 else (15 if n < 80000 else 15)
        n_pcs = adata.obsm[rep].shape[1] if rep in adata.obsm else 35
        sc.pp.neighbors(
            adata, use_rep=rep, n_neighbors=n_neighbors, n_pcs=n_pcs, metric="cosine", method="umap"
        )
    if "X_umap" not in adata.obsm:
        log("UMAP ...")
        sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=random_state)
        adata.obsm["X_umap"] = np.asarray(adata.obsm["X_umap"], dtype=np.float32)


def _run_leiden_compat(
    adata,
    resolution: float,
    random_state: int,
    key_added: str,
    n_iterations: int = 2,
    directed: bool = False,
):
    try:
        sc.tl.leiden(
            adata,
            resolution=float(resolution),
            random_state=random_state,
            key_added=key_added,
            flavor="igraph",
            n_iterations=n_iterations,
            directed=directed,
        )
        return
    except TypeError as e:
        msg = str(e)
        if (
            "unexpected keyword argument 'flavor'" in msg
            or "got an unexpected keyword argument 'flavor'" in msg
        ):
            log("Status update.")
    try:
        sc.tl.leiden(
            adata,
            resolution=float(resolution),
            random_state=random_state,
            key_added=key_added,
            n_iterations=n_iterations,
            directed=directed,
        )
        return
    except TypeError as e:
        msg = str(e)
        if (
            "unexpected keyword argument 'directed'" in msg
            or "got an unexpected keyword argument 'directed'" in msg
        ):
            log("Status update.")
        elif (
            "unexpected keyword argument 'n_iterations'" in msg
            or "got an unexpected keyword argument 'n_iterations'" in msg
        ):
            log("Status update.")
    try:
        sc.tl.leiden(
            adata,
            resolution=float(resolution),
            random_state=random_state,
            key_added=key_added,
            n_iterations=n_iterations,
        )
        return
    except TypeError:
        pass
    sc.tl.leiden(
        adata, resolution=float(resolution), random_state=random_state, key_added=key_added
    )


def _intra_cluster_edge_ratio(adata, key):
    try:
        A = adata.obsp["connectivities"]
        labels = adata.obs[key].values
        return 0.5
    except Exception:
        return 0.0


def leiden_grid(
    adata,
    res_min=0.4,
    res_max=1.4,
    res_step=0.2,
    random_state=42,
    min_cells_per_cluster=10,
    min_n_clusters=5,
):
    log(f"Leiden message: res=[{res_min}, {res_max}], step={res_step} ...")
    candidate_keys = []
    rs = np.arange(res_min, res_max + 1e-8, res_step)
    for r in rs:
        key = f"leiden_{str(r).replace('.', '_')}"
        _run_leiden_compat(
            adata,
            resolution=float(r),
            random_state=random_state,
            key_added=key,
            n_iterations=2,
            directed=False,
        )
        vc = adata.obs[key].value_counts()
        n_clusters = int(vc.shape[0])
        min_cells = int(vc.min())
        if n_clusters >= min_n_clusters and min_cells >= min_cells_per_cluster:
            freqs = (vc.values / vc.sum()).astype(float)
            gini = 1 - np.sum(freqs**2)
            same = _intra_cluster_edge_ratio(adata, key)
            candidate_keys.append((key, n_clusters, min_cells, gini, same))
    if not candidate_keys:
        all_keys = []
        for r in rs:
            key = f"leiden_{str(r).replace('.', '_')}"
            vc = adata.obs[key].value_counts()
            all_keys.append((key, int(vc.shape[0]), int(vc.min())))
        key = sorted(all_keys, key=lambda x: (x[1], x[2]))[-1][0]
        log(
            f"[WARN] message(min_n={min_n_clusters}, min_cells={min_cells_per_cluster})message message: {key}"
        )
        return key
    candidate_keys.sort(key=lambda x: (x[1] >= min_n_clusters, x[3]), reverse=True)
    best = candidate_keys[0][0]
    log(
        f"message: {best} (n={candidate_keys[0][1]}, min={candidate_keys[0][2]}, gini={candidate_keys[0][3]:.2f})"
    )
    return best


def _filter_groups_by_size(labels, min_cells=3):
    vc = pd.Series(labels).value_counts()
    valid = vc[vc >= min_cells].index.tolist()
    return valid


def stratified_subsample_idx(labels, cap_total=50000, seed=42):
    n = len(labels)
    if n <= cap_total:
        return np.arange(n)
    np.random.seed(seed)
    vc = pd.Series(labels).value_counts()
    idx_keep = []
    import math

    ratio = cap_total / n
    for grp in vc.index:
        idx_grp = np.where(labels == grp)[0]
        n_grp = len(idx_grp)
        n_sel = math.ceil(n_grp * ratio)
        sel = np.random.choice(idx_grp, min(n_grp, n_sel), replace=False)
        idx_keep.append(sel)
    return np.concatenate(idx_keep)


def safe_find_markers(adata, cluster_key, outdir, min_cells_per_group=3, cap_total=50000, seed=42):
    log(
        f"message Markers (key={cluster_key}, log1p_cp10k message wilcoxon+tie_correct message<{min_cells_per_group} message "
    )
    hv = adata.var.get(
        "highly_variable", pd.Series([True] * adata.n_vars, index=adata.var_names)
    ).values
    if hv.sum() == 0:
        hv = np.ones(adata.n_vars, dtype=bool)
    Xh = adata.layers["log1p_cp10k"][:, hv]
    work = ad.AnnData(Xh.copy(), obs=adata.obs[[cluster_key]].copy(), var=adata.var.loc[hv].copy())
    labels_all = work.obs[cluster_key].astype(str).values
    keep_groups = _filter_groups_by_size(labels_all, min_cells=min_cells_per_group)
    if len(keep_groups) < 2:
        log("Status update.")
        return pd.DataFrame(columns=["cluster", "gene", "score", "logFC", "pval_adj", "rank"])
    mask_groups = np.isin(labels_all, keep_groups)
    work = work[mask_groups, :].copy()
    labels = work.obs[cluster_key].astype(str).values
    idx_keep = stratified_subsample_idx(labels, cap_total=cap_total, seed=seed)
    work = work[idx_keep, :].copy()
    labels2 = work.obs[cluster_key].astype(str).values
    keep_groups2 = _filter_groups_by_size(labels2, min_cells=2)
    if len(keep_groups2) < 2:
        log("Status update.")
        return pd.DataFrame(columns=["cluster", "gene", "score", "logFC", "pval_adj", "rank"])
    mask2 = np.isin(labels2, keep_groups2)
    work = work[mask2, :].copy()
    try:
        sc.tl.rank_genes_groups(
            work,
            groupby=cluster_key,
            groups=keep_groups2,
            method="wilcoxon",
            n_genes=min(5000, work.n_vars),
            use_raw=False,
            tie_correct=True,
        )
    except TypeError:
        sc.tl.rank_genes_groups(
            work,
            groupby=cluster_key,
            groups=keep_groups2,
            method="wilcoxon",
            n_genes=min(5000, work.n_vars),
            use_raw=False,
        )
    res = sc.get.rank_genes_groups_df(work, group=None)
    if res is None:
        return pd.DataFrame()
    res = res.rename(
        columns={
            "names": "gene",
            "scores": "score",
            "logfoldchanges": "logFC",
            "pvals_adj": "pval_adj",
            "group": "cluster",
        }
    )
    res["rank"] = res.groupby("cluster").cumcount() + 1
    res = res[(res["logFC"] > 0.25) & (res["pval_adj"] < 0.05)]
    return res


def _centroids_and_spread(X, labels):
    labels = np.asarray(labels).astype(str)
    ulabels = np.unique(labels)
    dims = int(X.shape[1])
    centroids = np.zeros((len(ulabels), dims), dtype=np.float64)
    r90 = np.zeros((len(ulabels),), dtype=np.float64)
    r50 = np.zeros((len(ulabels),), dtype=np.float64)
    for i, lb in enumerate(ulabels):
        mask = labels == lb
        sub = np.asarray(X[mask], dtype=np.float64)
        if sub.shape[0] == 0:
            cent = np.zeros((dims,), dtype=np.float64)
            centroids[i] = cent
            r90[i] = np.nan
            r50[i] = np.nan
            continue
        cent = np.median(sub, axis=0)
        centroids[i] = cent
        dists = np.sqrt(np.sum((sub - cent) ** 2, axis=1))
        r90[i] = np.quantile(dists, 0.9)
        r50[i] = np.quantile(dists, 0.5)
    df_cent = pd.DataFrame(
        centroids, index=ulabels, columns=[f"dim_{i}" for i in range(dims)], dtype=float
    )
    df_spread = pd.DataFrame({"radius_90": r90, "radius_50": r50}, index=ulabels).astype(float)
    return df_cent, df_spread


def _pairwise_dist(centroids):
    from scipy.spatial.distance import pdist, squareform

    if isinstance(centroids, pd.DataFrame):
        X = np.asarray(centroids.values, dtype=np.float64)
        idx = centroids.index
        cols = centroids.index
    else:
        X = np.asarray(centroids, dtype=np.float64)
        idx = None
        cols = None
    D = squareform(pdist(X, metric="euclidean"))
    if idx is not None:
        return pd.DataFrame(D, index=idx, columns=cols)
    return pd.DataFrame(D)


def export_cluster_geometry(adata, cluster_key, outdir: Path):
    geom_dir = outdir
    geom_dir.mkdir(parents=True, exist_ok=True)
    labels = adata.obs[cluster_key].astype(str).values
    uns_payload = {"spaces": {}}
    if "X_umap" in adata.obsm:
        Xu = np.asarray(adata.obsm["X_umap"], dtype=float)
        dfC_u, dfS_u = _centroids_and_spread(Xu, labels)
        Du = _pairwise_dist(dfC_u)
        if len(dfC_u) > 1:
            nn_u = Du.replace(0, np.nan).idxmin(axis=1).to_frame("nearest_cluster")
            nn_u["nearest_dist"] = [Du.loc[i, nn_u.loc[i, "nearest_cluster"]] for i in nn_u.index]
        else:
            nn_u = pd.DataFrame(
                {"nearest_cluster": [None], "nearest_dist": [np.nan]}, index=dfC_u.index
            )
        p_cent = geom_dir / "umap_cluster_centroids.csv"
        p_sp = geom_dir / "umap_cluster_spread.csv"
        p_dist = geom_dir / "umap_cluster_dists.csv"
        p_nn = geom_dir / "umap_cluster_nearest.csv"
        dfC_u.to_csv(p_cent)
        dfS_u.to_csv(p_sp)
        Du.to_csv(p_dist)
        nn_u.to_csv(p_nn)
        uns_payload["spaces"]["umap"] = {
            "centroids_csv": str(p_cent),
            "spread_csv": str(p_sp),
            "dists_csv": str(p_dist),
            "nearest_csv": str(p_nn),
            "n_clusters": int(len(dfC_u)),
            "dims": 2,
        }
        log(f"message UMAP  {p_cent.name}, {p_sp.name}, {p_dist.name}, {p_nn.name}")
    else:
        log("Status update.")
    rep = (
        "X_pca_harmony"
        if "X_pca_harmony" in adata.obsm_keys()
        else ("X_pca" if "X_pca" in adata.obsm_keys() else None)
    )
    if rep is not None:
        Xp = np.asarray(adata.obsm[rep], dtype=float)
        dim_use = max(1, min(30, Xp.shape[1]))
        Xp = Xp[:, :dim_use]
        dfC_p, dfS_p = _centroids_and_spread(Xp, labels)
        Dp = _pairwise_dist(dfC_p)
        if len(dfC_p) > 1:
            nn_p = Dp.replace(0, np.nan).idxmin(axis=1).to_frame("nearest_cluster")
            nn_p["nearest_dist"] = [Dp.loc[i, nn_p.loc[i, "nearest_cluster"]] for i in nn_p.index]
        else:
            nn_p = pd.DataFrame(
                {"nearest_cluster": [None], "nearest_dist": [np.nan]}, index=dfC_p.index
            )
        p_cent = geom_dir / "pca_cluster_centroids.csv"
        p_sp = geom_dir / "pca_cluster_spread.csv"
        p_nn = geom_dir / "pca_cluster_nearest.csv"
        dfC_p.to_csv(p_cent)
        dfS_p.to_csv(p_sp)
        nn_p.to_csv(p_nn)
        uns_payload["spaces"]["pca"] = {
            "centroids_csv": str(p_cent),
            "spread_csv": str(p_sp),
            "nearest_csv": str(p_nn),
            "n_clusters": int(len(dfC_p)),
            "dims": dim_use,
        }
    adata.uns["best_cluster_geometry"] = uns_payload
    return uns_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--input_10x_parent", default=None)
    parser.add_argument("--input_h5ad", default=None)
    parser.add_argument("--batch_key", default="sample")
    parser.add_argument("--species", default="human")
    parser.add_argument("--auto_qc", action="store_true", default=True)
    parser.add_argument("--run_harmony", action="store_true", default=True)
    parser.add_argument("--auto_dim", action="store_true", default=True)
    parser.add_argument("--auto_cluster", action="store_true", default=True)
    parser.add_argument("--auto_cluster_res_min", type=float, default=0.4)
    parser.add_argument("--auto_cluster_res_max", type=float, default=1.4)
    parser.add_argument("--auto_cluster_res_step", type=float, default=0.2)
    parser.add_argument("--min_per_cluster_after", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    adata = load_input(args.input_10x_parent, args.input_h5ad, args.batch_key)
    bk = str(adata.uns.get("_resolved_batch_key", args.batch_key))
    if not adata.is_view and not adata.isbacked:
        adata.var_names_make_unique()
    maybe_qc(adata, species=args.species, batch_key=bk, auto_qc=args.auto_qc)
    normalize_hvg_scale(adata, batch_key=bk)
    pca_neighbors_umap(
        adata,
        use_harmony=args.run_harmony,
        batch_key=bk,
        auto_dim=args.auto_dim,
        random_state=args.random_state,
        species=args.species,
    )
    if args.auto_cluster:
        res_min = max(0.4, float(args.auto_cluster_res_min))
        res_max = max(res_min, float(args.auto_cluster_res_max))
        best_key = leiden_grid(
            adata,
            res_min=res_min,
            res_max=res_max,
            res_step=args.auto_cluster_res_step,
            random_state=args.random_state,
            min_cells_per_cluster=max(20, args.min_per_cluster_after),
            min_n_clusters=5,
        )
    else:
        _run_leiden_compat(
            adata,
            resolution=1.0,
            random_state=args.random_state,
            key_added="leiden",
            n_iterations=2,
            directed=False,
        )
        best_key = "leiden"
    adata.uns["best_cluster_key"] = best_key
    df_mk = safe_find_markers(adata, best_key, outdir, min_cells_per_group=3)
    p_mk = outdir / "markers_per_cluster_capped.csv"
    df_mk.to_csv(p_mk, index=False)
    log(f"message Markers: {p_mk}")
    log("Status update.")
    import matplotlib.pyplot as plt

    sc.pl.umap(adata, color=[best_key, bk], show=False, legend_loc="on data", frameon=False)
    plt.savefig(outdir / "umap_best_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    export_cluster_geometry(adata, best_key, outdir)
    final_h5ad = outdir / "final_preannotation.h5ad"
    log(f"message AnnData message {final_h5ad}")
    for c in adata.obs.columns:
        if pd.api.types.is_string_dtype(
            adata.obs[c].dtype
        ) and not pd.api.types.is_categorical_dtype(adata.obs[c].dtype):
            adata.obs[c] = adata.obs[c].astype(str)
    if "counts" in adata.layers:
        if not sparse.issparse(adata.layers["counts"]):
            adata.layers["counts"] = sparse.csr_matrix(adata.layers["counts"])
        adata.X = adata.layers["counts"]
    else:
        log("Status update.")
    if "log1p_cp10k" in adata.layers:
        try:
            del adata.layers["log1p_cp10k"]
        except Exception:
            pass
    try:
        adata.write_h5ad(final_h5ad, compression="gzip")
    except Exception as e:
        log(f"[WARN] message h5ad message message  {e}")
        adata.write_h5ad(final_h5ad)
    geom_json = outdir / "cluster_geometry_index.json"
    with open(geom_json, "w", encoding="utf-8") as f:
        json.dump(adata.uns.get("best_cluster_geometry", {}), f, indent=2, ensure_ascii=False)
    log(f"Step1 message Best Key: {best_key}")


if __name__ == "__main__":
    main()
