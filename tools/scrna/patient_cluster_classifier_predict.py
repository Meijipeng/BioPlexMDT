from __future__ import annotations
import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

warnings.filterwarnings("ignore")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str):
    print(f"[{now_ts()}] {msg}", flush=True)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_csv_safe(df: pd.DataFrame, out_path: str):
    ensure_dir(Path(out_path).parent)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def is_sparse(x) -> bool:
    return sparse.issparse(x)


def _csr(x):
    if is_sparse(x):
        return x.tocsr()
    return sparse.csr_matrix(np.asarray(x))


def _row_sum(X):
    if is_sparse(X):
        return np.asarray(X.sum(axis=1)).reshape(-1)
    return np.asarray(X).sum(axis=1)


def _get_X(adata):
    X = adata.X
    if is_sparse(X):
        return X
    return np.asarray(X)


def cp10k_log1p_sparse(
    mat_csr: sparse.csr_matrix, libsize: Optional[np.ndarray] = None
) -> sparse.csr_matrix:
    X = mat_csr.tocsr()
    if libsize is None:
        libsize = _row_sum(X) + 1e-8
    inv = sparse.diags(1.0 / libsize)
    Xn = inv.dot(X) * 1e4
    Xn = Xn.tocoo()
    Xn.data = np.log1p(Xn.data)
    return Xn.tocsr()


AGENT_JSON_PREFIX = "[AGENT_OUTPUT] "


def cluster_id_to_name(cid: int) -> str:
    try:
        return f"cluster{int(cid)}"
    except Exception:
        return "clusterNA"


def print_agent_json(payload: Dict[str, Any]):
    try:
        s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = json.dumps(
            {"error": "failed_to_dump_agent_json"}, ensure_ascii=False, separators=(",", ":")
        )
    print(AGENT_JSON_PREFIX + s, flush=True)


try:
    import scanpy as sc
except Exception:
    sc = None
import anndata as ad


def read_h5ad_compat(path: str):
    if sc is not None:
        return sc.read_h5ad(path)
    return ad.read_h5ad(path)


def _cuda_sanity_check() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.tensor([1.0], device="cuda")
        y = x * 2.0
        _ = y.item()
        return True
    except Exception as e:
        log(f"[CUDA] [WARN] sanity check failed -> fallback CPU. err={e}")
        return False


def choose_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available() and _cuda_sanity_check():
        name = torch.cuda.get_device_name(0)
        log(f"[Device] using GPU: {name}")
        return torch.device("cuda")
    log("[Device] using CPU")
    return torch.device("cpu")


def _patch_numpy_pickle_paths():
    import numpy as _np

    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _np.core
    try:
        sys.modules.setdefault("numpy._core.multiarray", _np.core.multiarray)
    except Exception:
        pass
    try:
        sys.modules.setdefault("numpy._core._multiarray_umath", _np.core._multiarray_umath)
    except Exception:
        pass
    try:
        sys.modules.setdefault("numpy._core.numerictypes", _np.core.numerictypes)
    except Exception:
        pass
    try:
        sys.modules.setdefault("numpy._core.fromnumeric", _np.core.fromnumeric)
    except Exception:
        pass
    try:
        sys.modules.setdefault("numpy._core._dtype", getattr(_np.core, "_dtype", _np.dtype))
    except Exception:
        pass


def _torch_load_safely(pt_path: str, map_location="cpu") -> Any:
    _patch_numpy_pickle_paths()
    try:
        return torch.load(pt_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(pt_path, map_location=map_location)
    except Exception as e:
        log(f"[CKPT] weights_only=True load failed: {e}")
    try:
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "safe_globals"):
            import numpy as _np

            allowed = []
            try:
                allowed.append(_np.core.multiarray._reconstruct)
            except Exception:
                pass
            try:
                if (
                    hasattr(_np, "_core")
                    and hasattr(_np._core, "multiarray")
                    and hasattr(_np._core.multiarray, "_reconstruct")
                ):
                    allowed.append(_np._core.multiarray._reconstruct)
            except Exception:
                pass
            try:
                allowed.append(_np.dtype)
            except Exception:
                pass
            try:
                allowed.append(_np.ndarray)
            except Exception:
                pass
            uniq = []
            seen = set()
            for a in allowed:
                if a is None:
                    continue
                if id(a) in seen:
                    continue
                seen.add(id(a))
                uniq.append(a)
            if uniq:
                log("[CKPT] retrying checkpoint load with safe globals")
                with torch.serialization.safe_globals(uniq):
                    return torch.load(pt_path, map_location=map_location, weights_only=True)
    except Exception as e2:
        log(f"[CKPT] safe globals load failed: {e2}")
    log("[CKPT] retrying checkpoint load with weights_only=False")
    _patch_numpy_pickle_paths()
    try:
        return torch.load(pt_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(pt_path, map_location=map_location)


def _load_checkpoint(pt_path: str, map_location="cpu") -> Dict[str, Any]:
    obj = _torch_load_safely(pt_path, map_location=map_location)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "state_dict"):
        return {"model": obj, "model_state_dict": obj.state_dict()}
    raise ValueError(f"Unsupported checkpoint format: {pt_path}")


class BigMLPClassifier(nn.Module):
    def __init__(
        self, input_dim: int, width: int, depth: int, num_classes: int, dropout: float = 0.1
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d_in = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(d_in, width))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(width))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = width
        layers.append(nn.Linear(d_in, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _infer_input_dim_from_state_dict(state: Dict[str, Any]) -> Optional[int]:
    if not isinstance(state, dict):
        return None
    for k, v in state.items():
        if k.endswith("net.0.weight") and hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[1])
    best = None
    for k, v in state.items():
        if hasattr(v, "shape") and len(v.shape) == 2 and str(k).endswith(".weight"):
            inp = int(v.shape[1])
            if best is None or inp > best:
                best = inp
    return best


def _infer_num_classes_from_state_dict(state: Dict[str, Any]) -> Optional[int]:
    if not isinstance(state, dict):
        return None
    candidates = []
    for k, v in state.items():
        if hasattr(v, "shape") and len(v.shape) == 2 and str(k).endswith(".weight"):
            out_dim = int(v.shape[0])
            in_dim = int(v.shape[1])
            candidates.append((out_dim, in_dim, k))
    if not candidates:
        return None
    candidates = [c for c in candidates if c[0] > 1 and c[0] <= 200]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return int(candidates[0][0])


def _build_model_from_info(
    sc_info: Dict[str, Any], ckpt: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, Any]]:
    cfg: Dict[str, Any] = {}
    if isinstance(sc_info.get("model_config"), dict):
        cfg.update(sc_info["model_config"])
    if isinstance(ckpt.get("model_config"), dict):
        cfg.update(ckpt["model_config"])
    if isinstance(ckpt.get("config"), dict):
        cfg.update(ckpt["config"])
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    input_dim = int(cfg.get("input_dim", sc_info.get("feature_dim", 0) or 0))
    if input_dim <= 0 and state is not None:
        inferred = _infer_input_dim_from_state_dict(state)
        if inferred:
            input_dim = int(inferred)
    width = int(cfg.get("width", cfg.get("hidden_dim", 4096)))
    depth = int(cfg.get("depth", 4))
    num_classes = int(cfg.get("num_classes", cfg.get("n_classes", 0) or 0))
    if num_classes <= 0 and state is not None:
        inf_nc = _infer_num_classes_from_state_dict(state)
        if inf_nc:
            num_classes = int(inf_nc)
    if num_classes <= 0:
        num_classes = 4
    dropout = float(cfg.get("dropout", 0.1))
    model_obj = ckpt.get("model", None)
    if model_obj is not None and hasattr(model_obj, "forward"):
        return model_obj, {
            "input_dim": input_dim,
            "width": width,
            "depth": depth,
            "num_classes": num_classes,
            "dropout": dropout,
            "from": "ckpt.model",
        }
    if input_dim <= 0:
        raise ValueError("Cannot infer model input dimension from checkpoint or model info")
    model = BigMLPClassifier(
        input_dim=input_dim, width=width, depth=depth, num_classes=num_classes, dropout=dropout
    )
    return model, {
        "input_dim": input_dim,
        "width": width,
        "depth": depth,
        "num_classes": num_classes,
        "dropout": dropout,
        "from": "rebuild_BigMLPClassifier",
    }


FIXED_CELL_TYPES_DEFAULT = [
    "AC-like",
    "Astrocyte",
    "MDMs",
    "MES-like",
    "Microglia",
    "NPC-like",
    "Neural progenitor cell",
    "Neuron",
    "OPC-like",
    "Oligodendrocyte",
    "Oligodendrocyte precursor cell (OPC)",
    "Stromal cell",
]


def _as_list(x) -> Optional[List[Any]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return None


def _try_get_from_specdict(
    spec: Dict[str, Any],
) -> Tuple[
    Optional[List[str]],
    Optional[List[str]],
    Optional[List[Tuple[str, str]]],
    Optional[Dict[str, List[str]]],
]:
    fixed = (
        _as_list(spec.get("fixed_cell_types"))
        or _as_list(spec.get("fixed_celltypes"))
        or _as_list(spec.get("cell_types"))
        or _as_list(spec.get("celltype_list"))
        or _as_list(spec.get("celltype"))
    )
    union = (
        _as_list(spec.get("union_genes"))
        or _as_list(spec.get("genes_union"))
        or _as_list(spec.get("gene_union"))
        or _as_list(spec.get("gene_list"))
        or _as_list(spec.get("genes"))
    )
    pairs = (
        _as_list(spec.get("pairs_order"))
        or _as_list(spec.get("ct_gene_pairs_order"))
        or _as_list(spec.get("pairs"))
        or _as_list(spec.get("ct_gene_pairs"))
    )
    genes_by_ct = (
        spec.get("genes_by_celltype") or spec.get("genes_by_ct") or spec.get("genes_by_cell_type")
    )
    if genes_by_ct is not None and isinstance(genes_by_ct, dict):
        genes_by_ct = {str(k): [str(g) for g in _as_list(v) or []] for k, v in genes_by_ct.items()}
    else:
        genes_by_ct = None
    fixed_s = [str(x) for x in fixed] if fixed else None
    union_s = [str(x) for x in union] if union else None
    pairs_t: Optional[List[Tuple[str, str]]] = None
    if pairs:
        pairs_t = []
        for it in pairs:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                pairs_t.append((str(it[0]), str(it[1])))
    return fixed_s, union_s, pairs_t, genes_by_ct


def _parse_feature_spec(
    sc_info: Dict[str, Any], ckpt: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    spec_dicts: List[Dict[str, Any]] = []
    if isinstance(sc_info.get("feature_spec"), dict):
        spec_dicts.append(sc_info["feature_spec"])
    spec_dicts.append(sc_info)
    if ckpt is not None:
        for k in ["feature_spec", "spec", "features", "feature", "feature_info"]:
            if isinstance(ckpt.get(k), dict):
                spec_dicts.append(ckpt[k])
        spec_dicts.append(ckpt)
    fixed_cell_types = None
    union_genes = None
    pairs_order = None
    genes_by_ct = None
    for sd in spec_dicts:
        f, u, p, gbc = _try_get_from_specdict(sd)
        if fixed_cell_types is None and f:
            fixed_cell_types = f
        if union_genes is None and u:
            union_genes = u
        if pairs_order is None and p and len(p) > 0:
            pairs_order = p
        if genes_by_ct is None and gbc:
            genes_by_ct = gbc
    if pairs_order is None and genes_by_ct is not None:
        if fixed_cell_types is None:
            fixed_cell_types = FIXED_CELL_TYPES_DEFAULT
        pairs_order = []
        for ct in fixed_cell_types:
            for g in genes_by_ct.get(ct, []):
                pairs_order.append((ct, g))
    if union_genes is None and pairs_order is not None:
        seen = set()
        union_genes = []
        for _, g in pairs_order:
            if g not in seen:
                seen.add(g)
                union_genes.append(g)
    if fixed_cell_types is None:
        fixed_cell_types = FIXED_CELL_TYPES_DEFAULT
    if union_genes is None:
        union_genes = []
    if pairs_order is None:
        pairs_order = []
    return {
        "fixed_cell_types": [str(x) for x in fixed_cell_types],
        "union_genes": [str(x) for x in union_genes],
        "pairs_order": [(str(a), str(b)) for (a, b) in pairs_order],
        "has_real_spec": (len(pairs_order) > 0 and len(union_genes) > 0),
    }


def _maybe_apply_feature_scaler(feat: np.ndarray, sc_info: Dict[str, Any]) -> np.ndarray:
    mean = sc_info.get("scaler_mean", None) or sc_info.get("feature_mean", None)
    scale = sc_info.get("scaler_scale", None) or sc_info.get("feature_std", None)
    if mean is None or scale is None:
        return feat
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    if mean.shape[0] != feat.shape[0] or scale.shape[0] != feat.shape[0]:
        log("[Features] scaler shape mismatch; skipping feature scaling")
        return feat
    scale = np.where(scale == 0, 1.0, scale)
    return (feat - mean) / scale


def _fix_celltypes_no_unk(
    adata,
    cell_type_key: str,
    fixed_cell_types: List[str],
    seed: int = 1234,
) -> np.ndarray:
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"h5ad.obs does not contain column '{cell_type_key}'")
    raw = adata.obs[cell_type_key].astype(str).values
    fixed_set = set(fixed_cell_types)
    bad = np.array([x not in fixed_set for x in raw], dtype=bool)
    if not bad.any():
        return raw
    counts = pd.Series(raw[~bad]).value_counts()
    if len(counts) > 0:
        fallback = str(counts.index[0])
    else:
        rng = np.random.RandomState(seed)
        fallback = str(rng.choice(fixed_cell_types, size=1)[0])
    mapped = raw.copy()
    mapped[bad] = fallback
    log(f"[CellType] [WARN] remapped {bad.sum()} unsupported labels to '{fallback}'")
    return mapped


def _extract_expr_for_genes(
    adata, genes: List[str]
) -> Tuple[sparse.csr_matrix, List[str], np.ndarray]:
    X = _csr(_get_X(adata))
    lib = _row_sum(X) + 1e-8
    var = adata.var_names.astype(str).tolist()
    g2i = {g: i for i, g in enumerate(var)}
    cols = [g2i[g] for g in genes if g in g2i]
    genes_found = [g for g in genes if g in g2i]
    if len(cols) == 0:
        return sparse.csr_matrix((adata.n_obs, 0), dtype=np.float32), [], lib
    sub = X[:, cols].astype(np.float32)
    return sub, genes_found, lib


def featurize_one_h5ad(
    h5ad_path: str,
    cell_type_key: str,
    fixed_cell_types: List[str],
    pairs_order: List[Tuple[str, str]],
    expected_input_dim: int,
    max_cells: int = 0,
    seed: int = 1234,
) -> np.ndarray:
    adata = read_h5ad_compat(h5ad_path)
    if max_cells and max_cells > 0 and adata.n_obs > max_cells:
        rng = np.random.RandomState(seed)
        idx = rng.choice(np.arange(adata.n_obs), size=max_cells, replace=False)
        adata = adata[idx].copy()
    ct = _fix_celltypes_no_unk(
        adata, cell_type_key=cell_type_key, fixed_cell_types=fixed_cell_types, seed=seed
    )
    n_total = len(ct)
    feats: List[float] = []
    for c in fixed_cell_types:
        feats.append(float((ct == c).sum()) / max(1, n_total))
    if pairs_order and len(pairs_order) > 0:
        genes_all = []
        seen = set()
        for _, g in pairs_order:
            if g not in seen:
                genes_all.append(g)
                seen.add(g)
        sub_raw, genes_found, lib = _extract_expr_for_genes(adata, genes_all)
        sub_norm = cp10k_log1p_sparse(sub_raw, libsize=lib)
        g2col = {g: i for i, g in enumerate(genes_found)}
        ct2rows: Dict[str, np.ndarray] = {c: np.where(ct == c)[0] for c in fixed_cell_types}
        for c, g in pairs_order:
            rows = ct2rows.get(c, None)
            if rows is None or rows.size == 0:
                feats.append(0.0)
                continue
            j = g2col.get(g, None)
            if j is None:
                feats.append(0.0)
                continue
            val = sub_norm[rows, j].mean()
            if hasattr(val, "A"):
                val = float(np.asarray(val).reshape(-1)[0])
            else:
                val = float(val)
            feats.append(val)
    feat = np.asarray(feats, dtype=np.float32)
    if expected_input_dim > 0:
        if feat.shape[0] < expected_input_dim:
            pad = np.zeros((expected_input_dim - feat.shape[0],), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        elif feat.shape[0] > expected_input_dim:
            feat = feat[:expected_input_dim].copy()
    return feat


def _infer_outdir_for_predict(h5ad_path: str, outdir_arg: Optional[str]) -> Path:
    if outdir_arg:
        return Path(outdir_arg)
    p = Path(h5ad_path).resolve()
    parent = p.parent
    if parent.name.startswith("03_") and parent.parent.exists():
        return parent.parent / "04_PatientClassification"
    return parent / "04_PatientClassification"


def predict_one(
    model_pt: str,
    h5ad_path: str,
    sc_model_info_json: str,
    sample_id: str,
    cell_type_key: str,
    outdir: Optional[str] = None,
    max_cells: int = 0,
    prefer_cuda: bool = True,
) -> Dict[str, Any]:
    sc_info = read_json(sc_model_info_json)
    ckpt = _load_checkpoint(model_pt, map_location="cpu")
    model, used_cfg = _build_model_from_info(sc_info, ckpt)
    expected_input_dim = int(used_cfg.get("input_dim", 0))
    feat_spec = _parse_feature_spec(sc_info, ckpt=ckpt)
    if not feat_spec["has_real_spec"]:
        log("[Features] feature specification not found; using defaults")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is not None:
        model.load_state_dict(state, strict=False)
    device = choose_device(prefer_cuda=prefer_cuda)
    model.to(device)
    model.eval()
    outdir_path = _infer_outdir_for_predict(h5ad_path, outdir)
    ensure_dir(outdir_path)
    feat = featurize_one_h5ad(
        h5ad_path=h5ad_path,
        cell_type_key=cell_type_key,
        fixed_cell_types=feat_spec["fixed_cell_types"],
        pairs_order=feat_spec["pairs_order"],
        expected_input_dim=expected_input_dim,
        max_cells=max_cells,
    )
    feat = _maybe_apply_feature_scaler(feat, sc_info)
    x = torch.from_numpy(feat[None, :]).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
        pred = int(prob.argmax())
    df_label = pd.DataFrame([{"SampleID": sample_id, "PatientCluster": pred}])
    df_prob = pd.DataFrame(
        [{"SampleID": sample_id, **{f"prob_{i}": float(prob[i]) for i in range(len(prob))}}]
    )
    out_labels = outdir_path / "patient_cluster_labels.csv"
    out_probs = outdir_path / "patient_cluster_probs.csv"
    to_csv_safe(df_label, str(out_labels))
    to_csv_safe(df_prob, str(out_probs))
    summary = {
        "mode": "predict_one",
        "sample_id": sample_id,
        "h5ad": str(Path(h5ad_path).resolve()),
        "model_pt": str(Path(model_pt).resolve()),
        "sc_model_info": str(Path(sc_model_info_json).resolve()),
        "outdir": str(outdir_path.resolve()),
        "cell_type_key": cell_type_key,
        "predicted_cluster": pred,
        "prob": [float(x) for x in prob.tolist()],
        "feature_dim_used": int(feat.shape[0]),
        "feature_spec_found": bool(feat_spec["has_real_spec"]),
        "feature_spec": {
            "n_fixed_cell_types": len(feat_spec["fixed_cell_types"]),
            "n_pairs_order": len(feat_spec["pairs_order"]),
            "n_union_genes": len(feat_spec["union_genes"]),
        },
        "model_config_used": used_cfg,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": str(device),
        "outputs": {
            "labels_csv": str(out_labels.resolve()),
            "probs_csv": str(out_probs.resolve()),
        },
    }
    write_json(summary, str(outdir_path / "predict_one_summary.json"))
    log(f"[Predict] wrote: {out_labels}")
    log(f"[Predict] wrote: {out_probs}")
    log(f"[Predict] wrote: {outdir_path / 'predict_one_summary.json'}")
    agent_payload = {
        "sample_id": sample_id,
        "patient_cluster_id": pred,
        "patient_cluster_name": cluster_id_to_name(pred),
        "probs": {cluster_id_to_name(i): float(prob[i]) for i in range(len(prob))},
        "outputs": {
            "labels_csv": str(out_labels.resolve()),
            "probs_csv": str(out_probs.resolve()),
            "summary_json": str((outdir_path / "predict_one_summary.json").resolve()),
        },
    }
    print_agent_json(agent_payload)
    return summary


def _discover_h5ad_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".h5ad":
        return [str(p)]
    out = []
    for root, _, files in os.walk(str(p)):
        for fn in files:
            if fn.lower().endswith(".h5ad"):
                out.append(str(Path(root) / fn))
    return sorted(out)


def predict_dir(
    model_pt: str,
    sc_data_dir: str,
    sc_model_info_json: str,
    cell_type_key: str,
    outdir: str,
    max_cells: int = 0,
    prefer_cuda: bool = True,
) -> Dict[str, Any]:
    sc_info = read_json(sc_model_info_json)
    ckpt = _load_checkpoint(model_pt, map_location="cpu")
    model, used_cfg = _build_model_from_info(sc_info, ckpt)
    expected_input_dim = int(used_cfg.get("input_dim", 0))
    feat_spec = _parse_feature_spec(sc_info, ckpt=ckpt)
    if not feat_spec["has_real_spec"]:
        log("[Features] feature specification not found; using defaults")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is not None:
        model.load_state_dict(state, strict=False)
    device = choose_device(prefer_cuda=prefer_cuda)
    model.to(device)
    model.eval()
    outdir_path = Path(outdir)
    ensure_dir(outdir_path)
    files = _discover_h5ad_files(sc_data_dir)
    log(f"[PredictDir] found h5ad files: {len(files)}")
    rows = []
    prob_rows = []
    agent_rows = []
    for hp in files:
        sid = Path(hp).stem
        feat = featurize_one_h5ad(
            h5ad_path=hp,
            cell_type_key=cell_type_key,
            fixed_cell_types=feat_spec["fixed_cell_types"],
            pairs_order=feat_spec["pairs_order"],
            expected_input_dim=expected_input_dim,
            max_cells=max_cells,
        )
        feat = _maybe_apply_feature_scaler(feat, sc_info)
        x = torch.from_numpy(feat[None, :]).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = F.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
            pred = int(prob.argmax())
        rows.append({"SampleID": sid, "PatientCluster": pred})
        prob_rows.append(
            {"SampleID": sid, **{f"prob_{i}": float(prob[i]) for i in range(len(prob))}}
        )
        agent_rows.append(
            {
                "sample_id": sid,
                "patient_cluster_id": pred,
                "patient_cluster_name": cluster_id_to_name(pred),
                "probs": {cluster_id_to_name(i): float(prob[i]) for i in range(len(prob))},
            }
        )
    df_label = pd.DataFrame(rows)
    df_prob = pd.DataFrame(prob_rows)
    out_labels = outdir_path / "patient_cluster_labels.csv"
    out_probs = outdir_path / "patient_cluster_probs.csv"
    to_csv_safe(df_label, str(out_labels))
    to_csv_safe(df_prob, str(out_probs))
    summary = {
        "mode": "predict_dir",
        "sc_data_dir": str(Path(sc_data_dir).resolve()),
        "model_pt": str(Path(model_pt).resolve()),
        "sc_model_info": str(Path(sc_model_info_json).resolve()),
        "outdir": str(outdir_path.resolve()),
        "cell_type_key": cell_type_key,
        "n_files": len(files),
        "feature_dim_used": expected_input_dim,
        "feature_spec_found": bool(feat_spec["has_real_spec"]),
        "feature_spec": {
            "n_fixed_cell_types": len(feat_spec["fixed_cell_types"]),
            "n_pairs_order": len(feat_spec["pairs_order"]),
            "n_union_genes": len(feat_spec["union_genes"]),
        },
        "model_config_used": used_cfg,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": str(device),
        "outputs": {
            "labels_csv": str(out_labels.resolve()),
            "probs_csv": str(out_probs.resolve()),
        },
    }
    write_json(summary, str(outdir_path / "predict_dir_summary.json"))
    log(f"[PredictDir] wrote: {out_labels}")
    log(f"[PredictDir] wrote: {out_probs}")
    log(f"[PredictDir] wrote: {outdir_path / 'predict_dir_summary.json'}")
    agent_payload = {
        "mode": "predict_dir",
        "n_files": len(files),
        "results": agent_rows,
        "outputs": {
            "labels_csv": str(out_labels.resolve()),
            "probs_csv": str(out_probs.resolve()),
            "summary_json": str((outdir_path / "predict_dir_summary.json").resolve()),
        },
    }
    print_agent_json(agent_payload)
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp_one = sub.add_parser("predict_one", help="predict one h5ad")
    sp_one.add_argument("--model", required=True, help="model .pt")
    sp_one.add_argument("--h5ad", required=True, help="single .h5ad file")
    sp_one.add_argument("--sc_model_info", required=True, help="summary json")
    sp_one.add_argument("--sample_id", required=True, help="sample id")
    sp_one.add_argument("--cell_type_key", default="cell_type", help="obs column name")
    sp_one.add_argument("--outdir", default=None, help="optional; inferred if not set")
    sp_one.add_argument(
        "--max_cells", type=int, default=0, help="optional subsample cells for speed"
    )
    sp_one.add_argument("--cpu", action="store_true", help="force cpu")
    sp_dir = sub.add_parser("predict_dir", help="batch predict h5ad under a directory")
    sp_dir.add_argument("--model", required=True)
    sp_dir.add_argument("--sc_data_dir", required=True)
    sp_dir.add_argument("--sc_model_info", required=True)
    sp_dir.add_argument("--cell_type_key", default="cell_type")
    sp_dir.add_argument("--outdir", required=True)
    sp_dir.add_argument("--max_cells", type=int, default=0)
    sp_dir.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cmd == "predict_one":
        predict_one(
            model_pt=args.model,
            h5ad_path=args.h5ad,
            sc_model_info_json=args.sc_model_info,
            sample_id=args.sample_id,
            cell_type_key=args.cell_type_key,
            outdir=args.outdir,
            max_cells=args.max_cells,
            prefer_cuda=(not args.cpu),
        )
        return
    if args.cmd == "predict_dir":
        predict_dir(
            model_pt=args.model,
            sc_data_dir=args.sc_data_dir,
            sc_model_info_json=args.sc_model_info,
            cell_type_key=args.cell_type_key,
            outdir=args.outdir,
            max_cells=args.max_cells,
            prefer_cuda=(not args.cpu),
        )
        return


if __name__ == "__main__":
    main()
