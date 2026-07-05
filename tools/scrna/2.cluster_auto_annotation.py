import os
import re
import json
import argparse
import sys
import warnings
import urllib.request
import ssl
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import scanpy as sc

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import infercnvpy as cnv
except ImportError:
    cnv = None


def log(msg: str):
    print(msg, flush=True)


FIXED_CELL_TYPES = [
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
_KEYWORD_MAPPING_RULES: List[tuple] = [
    ("ac-like", "AC-like"),
    ("mes-like", "MES-like"),
    ("npc-like", "NPC-like"),
    ("opc-like", "OPC-like"),
    ("malignant (unspecified)", "AC-like"),
    ("malignant", "AC-like"),
    ("monocyte-derived macrophage", "MDMs"),
    ("mdm", "MDMs"),
    ("macrophage", "MDMs"),
    ("monocyte", "MDMs"),
    ("dendritic", "MDMs"),
    ("mast cell", "MDMs"),
    ("neutrophil", "MDMs"),
    ("natural killer", "MDMs"),
    ("nk cell", "MDMs"),
    ("t cell", "MDMs"),
    ("b cell", "MDMs"),
    ("lymphocyte", "MDMs"),
    ("plasma cell", "MDMs"),
    ("microglia", "Microglia"),
    ("microglial", "Microglia"),
    ("neural progenitor", "Neural progenitor cell"),
    ("neuronal progenitor", "Neural progenitor cell"),
    ("radial glia", "Neural progenitor cell"),
    ("neuroblast", "NPC-like"),
    ("neuron", "Neuron"),
    ("neuronal", "Neuron"),
    ("oligodendrocyte precursor", "Oligodendrocyte precursor cell (OPC)"),
    ("opc", "Oligodendrocyte precursor cell (OPC)"),
    ("oligodendrocyte", "Oligodendrocyte"),
    ("oligodendroglia", "Oligodendrocyte"),
    ("astrocyte", "Astrocyte"),
    ("astroglia", "Astrocyte"),
    ("bergmann glia", "Astrocyte"),
    ("endothelial", "Stromal cell"),
    ("pericyte", "Stromal cell"),
    ("fibroblast", "Stromal cell"),
    ("smooth muscle", "Stromal cell"),
    ("stromal", "Stromal cell"),
    ("vascular", "Stromal cell"),
    ("mural cell", "Stromal cell"),
]


def snap_to_fixed(label: str) -> Optional[str]:
    if label in FIXED_CELL_TYPES:
        return label
    label_lower = label.lower().strip()
    for kw, target in _KEYWORD_MAPPING_RULES:
        if kw in label_lower:
            return target
    return None


def enforce_fixed_cell_types_with_gpt(
    df: pd.DataFrame,
    col: str = "predicted_cell_type",
    api_config: Dict[str, str] = None,
) -> pd.DataFrame:
    df = df.copy()
    values = df[col].astype(str)
    valid_counts = values[values.isin(FIXED_CELL_TYPES)].value_counts()
    fallback = str(valid_counts.index[0]) if len(valid_counts) > 0 else "Astrocyte"
    rule_result = {}
    need_gpt = []
    for label in values.unique():
        result = snap_to_fixed(label)
        if result is not None:
            if result != label:
                log(f"[SnapLabel] message: '{label}' -> '{result}'")
            rule_result[label] = result
        else:
            need_gpt.append(label)
    gpt_result = {}
    if need_gpt:
        log(f"[SnapLabel] message message GPT message: {need_gpt}")
        if api_config and api_config.get("api_key") and OpenAI:
            client = OpenAI(
                api_key=api_config.get("api_key"),
                base_url=api_config.get("base_url"),
            )
            prompt = (
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                f"message {json.dumps(need_gpt, ensure_ascii=False)}\n\n"
                "Status update."
                "Status update."
            )
            try:
                resp = client.chat.completions.create(
                    model=api_config.get("model", "gpt-4o"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = (
                    resp.choices[0]
                    .message.content.strip()
                    .replace("```json", "")
                    .replace("```", "")
                )
                gpt_map = json.loads(content)
                for orig, mapped in gpt_map.items():
                    if mapped in FIXED_CELL_TYPES:
                        gpt_result[orig] = mapped
                        log(f"[SnapLabel] GPTmessage: '{orig}' -> '{mapped}'")
                    else:
                        log(
                            f"[SnapLabel] [WARN] GPTmessage '{mapped}' '{orig}' message: '{fallback}'"
                        )
                        gpt_result[orig] = fallback
            except Exception as e:
                log(f"[SnapLabel] [WARN] GPTmessage: {e} message: '{fallback}'")
        else:
            log(f"[SnapLabel] [WARN] GPTmessage message: '{fallback}'")

    def _apply(label):
        if label in rule_result:
            return rule_result[label]
        if label in gpt_result:
            return gpt_result[label]
        log(f"[SnapLabel] [WARN] '{label}' message message: '{fallback}'")
        return fallback

    df[col] = values.apply(_apply)
    return df


def import_scmulan(scmulan_root: Optional[str] = None):
    candidates: List[Path] = []
    if scmulan_root:
        candidates.append(Path(scmulan_root))
    candidates.append(Path(__file__).resolve().parent / "scMulan")
    candidates.append(Path("tools/scrna/scMulan"))
    import importlib

    def _purge_scmulan_from_sysmodules():
        for k in list(sys.modules.keys()):
            if k == "scMulan" or k.startswith("scMulan."):
                sys.modules.pop(k, None)

    def _add_path_for_candidate(p: Path) -> Optional[str]:
        if not p or not p.exists():
            return None
        if (p / "__init__.py").exists():
            return str(p.parent)
        if (p / "scMulan" / "__init__.py").exists() or (p / "scMulan.py").exists():
            return str(p)
        return str(p)

    last_err: Optional[Exception] = None
    for p in candidates:
        try_add = _add_path_for_candidate(p)
        if not try_add:
            continue
        if try_add not in sys.path:
            sys.path.insert(0, try_add)
        _purge_scmulan_from_sysmodules()
        try:
            scMulan_mod = importlib.import_module("scMulan")
            try:
                loc = getattr(scMulan_mod, "__file__", None)
            except Exception:
                loc = None
            try:
                from scMulan import GeneSymbolUniform

                log(f"[scMulan]   message scMulan + GeneSymbolUniform message {p} | file={loc} ")
                return scMulan_mod, GeneSymbolUniform
            except Exception as e1:
                try:
                    sub = importlib.import_module("scMulan.GeneSymbolUniform")
                    GeneSymbolUniform = getattr(sub, "GeneSymbolUniform")
                    log(f"[scMulan]   message scMulan + GeneSymbolUniform message ")
                    return scMulan_mod, GeneSymbolUniform
                except Exception as e2:
                    last_err = e2 if e2 else e1
                    log(f"[scMulan] [WARN] message {p} -> {e1} | {e2}")
        except Exception as e:
            last_err = e
            log(f"[scMulan] [WARN] message {p} -> {e}")
    try:
        _purge_scmulan_from_sysmodules()
        scMulan_mod = importlib.import_module("scMulan")
        from scMulan import GeneSymbolUniform

        log("Status update.")
        return scMulan_mod, GeneSymbolUniform
    except Exception as e:
        raise ImportError(
            f"message scMulan message {[str(x) for x in candidates]}\n"
            f"last_error={last_err or e}"
        )


def _get_compiled_sms_from_torch(torch_mod) -> List[int]:
    sms: List[int] = []
    try:
        flags = torch_mod._C._cuda_getArchFlags()
        if isinstance(flags, str):
            sms.extend([int(x) for x in re.findall(r"sm_(\d+)", flags)])
    except Exception:
        pass
    try:
        for a in torch_mod.cuda.get_arch_list():
            m = re.match(r"sm_(\d+)", str(a))
            if m:
                sms.append(int(m.group(1)))
    except Exception:
        pass
    return sorted(list(set(sms)))


def maybe_force_cpu_for_unsupported_gpu() -> bool:
    try:
        import torch
    except Exception:
        return False
    if not hasattr(torch, "cuda"):
        return False
    try:
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability(0)
        sm = int(cap[0] * 10 + cap[1])
    except Exception:
        return False
    compiled = _get_compiled_sms_from_torch(torch)
    if compiled and (sm not in compiled):
        log(f"[CUDA] GPU sm_{sm} message PyTorch message message CPU ")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
        except Exception:
            pass
        return True
    return False


HIGH_CONFIDENCE_MARKERS = {
    "AC-like": ["EGFR", "ASCL1", "DLL3", "MDK", "SOX2", "TNC", "HEY1", "TLX1", "HOPX", "S100B"],
    "MES-like": [
        "CHI3L1",
        "VIM",
        "CD44",
        "ANXA1",
        "NFKB1",
        "STAT3",
        "RELB",
        "TRADD",
        "FOSL2",
        "TIMP1",
    ],
    "NPC-like": [
        "CDK4",
        "SOX4",
        "DCX",
        "OLIG2",
        "DLL1",
        "NEUROD1",
        "EOMES",
        "SOX11",
        "STMN1",
        "CD24",
        "HES6",
        "TCF4",
    ],
    "OPC-like": ["PDGFRA", "CSPG4", "OLIG1", "BCAN", "PLP1", "PTPRZ1", "GPR17", "NKX2-2", "SOX10"],
}
IMMUNE_KEYWORDS = [
    "Microglia",
    "Macrophage",
    "T cell",
    "B cell",
    "Monocyte",
    "NK",
    "Natural Killer",
    "Neutrophil",
    "Dendritic",
    "Mast",
    "Endothelial",
    "Pericyte",
    "MDMs",
    "Stromal cell",
]


def load_api_config(config_path: str) -> Dict[str, str]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        log(f"[WARN] API message: {config_path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if "openai" in raw:
                config = raw["openai"]
                log("Status update.")
            else:
                config = raw
            if "base_url" in config and config["base_url"]:
                config["base_url"] = config["base_url"].rstrip("/").replace("/chat/completions", "")
            log(
                f"[API] message: {config.get('model', "Status update.")}, base_url: {config.get('base_url', "Status update.")}"
            )
            return config
    except Exception as e:
        log(f"[ERROR] message API message: {e}")
        return {}


def resolve_cluster_key(adata: sc.AnnData, requested: str) -> str:
    if requested in adata.obs:
        return requested
    try:
        best = adata.uns.get("best_cluster_key", None)
        if best and best in adata.obs:
            log(f"[WARN] message '{requested}' message '{best}'")
            return str(best)
    except Exception:
        pass
    for k in adata.obs.columns:
        if str(k).startswith("leiden"):
            log(f"[WARN] message '{requested}' message '{k}'")
            return str(k)
    raise KeyError(f"message cluster_key='{requested}' ")


def resolve_celltype_key(adata: sc.AnnData, requested: str) -> str:
    if requested in adata.obs:
        return requested
    for c in [
        "cell_type_from_scMulan",
        "cell_type",
        "cell_type_pred",
        "fine_cell_type",
        "predicted_cell_type",
        "celltype",
        "CellType",
    ]:
        if c in adata.obs:
            log(f"[WARN] message '{requested}' message '{c}'")
            return c
    raise KeyError(f"message celltype_key='{requested}' ")


def subsample_by_cluster(
    adata: sc.AnnData, cluster_key: str, max_cells_per_cluster: int
) -> sc.AnnData:
    if max_cells_per_cluster <= 0:
        return adata
    if cluster_key not in adata.obs:
        raise KeyError(f"message cluster message: {cluster_key}")
    groups = adata.obs[cluster_key].astype(str)
    keep_indices: List[int] = []
    rng = np.random.default_rng(0)
    for cl in sorted(groups.unique()):
        idx = np.where(groups == cl)[0]
        sel = rng.choice(idx, min(len(idx), max_cells_per_cluster), replace=False)
        keep_indices.extend(sel.tolist())
    return adata[keep_indices].copy()


def ensure_log1p_cp10k(adata: sc.AnnData, cap_max: float = 9.9999):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X
    try:
        import scipy.sparse as sp

        if sp.issparse(X):
            X = X.tocsr(copy=True)
            if X.data.size > 0:
                X.data = np.minimum(X.data, cap_max)
            adata.X = X
        else:
            adata.X = np.minimum(X, cap_max)
    except Exception:
        pass


def load_gene_positions(adata: sc.AnnData, local_file: str = None):
    cache_default = str(Path.home() / ".cache" / "hg38_gencode_v27.txt")
    legacy_default = "/root/autodl-tmp/hg38_gencode_v27.txt"
    candidates = []
    if local_file and os.path.exists(local_file):
        candidates.append(local_file)
    for c in [cache_default, legacy_default]:
        if os.path.exists(c):
            candidates.append(c)
    target_file = candidates[0] if candidates else cache_default
    if not os.path.exists(target_file):
        url = "https://data.broadinstitute.org/Trinity/CTAT/cnv/hg38_gencode_v27.txt"
        log(f"[InferCNV] message hg38_gencode_v27.txt -> {target_file}")
        try:
            Path(target_file).parent.mkdir(parents=True, exist_ok=True)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ctx) as u, open(target_file, "wb") as f:
                f.write(u.read())
            log("Status update.")
        except Exception as e:
            log(f"[ERROR] message: {e}")
            return False
    log(f"[InferCNV] message: {target_file}")
    try:
        try:
            gene_pos = pd.read_csv(
                target_file,
                sep="\t",
                header=None,
                index_col=0,
                names=["gene", "chromosome", "start", "end"],
                usecols=[0, 1, 2, 3],
                engine="python",
                on_bad_lines="skip",
            )
        except TypeError:
            gene_pos = pd.read_csv(
                target_file,
                sep="\t",
                header=None,
                index_col=0,
                names=["gene", "chromosome", "start", "end"],
                usecols=[0, 1, 2, 3],
                engine="python",
                error_bad_lines=False,
            )
        gene_pos = gene_pos[~gene_pos.index.duplicated(keep="first")]
        adata.var["chromosome"] = None
        adata.var["start"] = None
        adata.var["end"] = None
        common = adata.var_names.intersection(gene_pos.index)
        log(f"  - message: {len(common)} / {adata.n_vars}")
        if len(common) < 50:
            log("Status update.")
            return False
        adata.var.loc[common, "chromosome"] = gene_pos.loc[common, "chromosome"]
        adata.var.loc[common, "start"] = gene_pos.loc[common, "start"]
        adata.var.loc[common, "end"] = gene_pos.loc[common, "end"]
        return True
    except Exception as e:
        log(f"[ERROR] message: {e}")
        return False


def run_scmulan(
    scMulan_mod,
    GeneSymbolUniform_func,
    adata_path: Path,
    ckpt_path: Path,
    cluster_key: str,
    celltype_key: str,
    max_cells: int,
    use_smoothing: bool,
    n_process_cpu: int,
):
    log(f"[Step2] message h5ad {adata_path}")
    adata_raw = sc.read_h5ad(str(adata_path))
    cluster_key_resolved = resolve_cluster_key(adata_raw, cluster_key)
    log(f"[Step2] message (max={max_cells})...")
    adata = subsample_by_cluster(adata_raw, cluster_key_resolved, max_cells)
    log("[Step2] GeneSymbolUniform...")
    adata_proc = GeneSymbolUniform_func(
        adata, output_dir=str(adata_path.parent), output_prefix=adata_path.stem + ".sub"
    )
    ensure_log1p_cp10k(adata_proc, cap_max=9.9999)
    log("Status update.")
    try:
        scml = scMulan_mod.model_inference(str(ckpt_path), adata_proc)
    except AssertionError as e:
        raise RuntimeError(f"scMulan message {e}")
    try:
        import torch

        use_gpu = bool(
            getattr(torch, "cuda", None)
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 0
        )
    except Exception:
        use_gpu = False
    if use_gpu:
        log("Status update.")
        try:
            scml.get_cell_types_and_embds_for_adata(parallel=False)
        except Exception as e:
            if any(k in str(e) for k in ["no kernel image", "sm_120", "not compatible"]):
                raise RuntimeError(f"scMulan GPU message {e}")
            raise
    else:
        nproc = int(max(1, n_process_cpu))
        log(f"[Step2] CPU message n_process={nproc} ...")
        if nproc > 1:
            scml.get_cell_types_and_embds_for_adata(parallel=True, n_process=nproc)
        else:
            scml.get_cell_types_and_embds_for_adata(parallel=False)
    adata_mulan = scml.adata.copy()
    if use_smoothing:
        try:
            scMulan_mod.cell_type_smoothing(adata_mulan, threshold=0.1)
        except Exception:
            pass
    celltype_key_resolved = resolve_celltype_key(adata_mulan, celltype_key)
    if celltype_key_resolved != celltype_key:
        adata_mulan.obs[celltype_key] = adata_mulan.obs[celltype_key_resolved].astype(str)
    return adata_mulan, cluster_key_resolved, celltype_key


def aggregate_to_clusters(adata, cluster_key, celltype_key):
    clusters = adata.obs[cluster_key].astype(str)
    celltypes = adata.obs[celltype_key].astype(str)
    rows = []
    for cid in sorted(clusters.unique()):
        mask = clusters == cid
        sub_ct = celltypes[mask].dropna()
        if len(sub_ct) == 0:
            pred, conf, dist = "Unknown", 0.0, []
        else:
            vc = sub_ct.value_counts()
            pred = str(vc.idxmax())
            conf = float(vc.max() / vc.sum())
            dist = [{"label": str(k), "fraction": float(v / vc.sum())} for k, v in vc.items()]
        rows.append(
            {
                "cluster": cid,
                "predicted_cell_type": pred,
                "confidence": round(conf, 4),
                "votes_json": json.dumps(dist, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def get_reference_clusters(df, api_config, max_ref_fraction: float = 0.4) -> List[str]:
    candidates = set()
    mapping = dict(zip(df["cluster"], df["predicted_cell_type"]))
    if OpenAI and api_config.get("api_key"):
        client = OpenAI(api_key=api_config.get("api_key"), base_url=api_config.get("base_url"))
        prompt = (
            f"message {json.dumps(mapping, ensure_ascii=False)}\n"
            "Status update."
            "Microglia MDMs Stromal cell \n"
            "Status update."
            "Status update."
        )
        try:
            log("Status update.")
            resp = client.chat.completions.create(
                model=api_config.get("model", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = (
                resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
            )
            gpt_res = json.loads(content)
            candidates.update([str(c) for c in gpt_res])
            log(f"[Ref] GPT message Reference: {gpt_res}")
        except Exception as e:
            log(f"[WARN] GPT message Reference message ({e}) message ")
    if not candidates:
        log("Status update.")
        for cid, ctype in mapping.items():
            for kw in IMMUNE_KEYWORDS:
                if kw.lower() in str(ctype).lower():
                    candidates.add(str(cid))
                    log(f"  - Cluster {cid} ({ctype}) -> Reference message")
                    break
    total_clusters = len(df)
    max_ref = max(1, int(total_clusters * max_ref_fraction))
    if len(candidates) > max_ref:
        log(
            f"[Ref] [WARN] Reference message ({len(candidates)}) message cluster message "
            f"{max_ref_fraction*100:.0f}% ({max_ref} message) message confidence message "
        )
        conf_map = dict(zip(df["cluster"].astype(str), df["confidence"]))
        candidates_sorted = sorted(candidates, key=lambda c: conf_map.get(c, 0), reverse=True)
        candidates = set(candidates_sorted[:max_ref])
        log(f"[Ref] message Reference Clusters: {sorted(candidates)}")
    result = list(candidates)
    if not result:
        log("Status update.")
    else:
        log(f"[Ref] message Reference Clusters ({len(result)}message): {result}")
    return result


def ask_gpt_for_markers(api_config):
    gpt_markers = {}
    if api_config.get("api_key") and OpenAI:
        client = OpenAI(api_key=api_config.get("api_key"), base_url=api_config.get("base_url"))
        prompt = "Status update." "Status update." "Status update." "Status update."
        try:
            log("Status update.")
            resp = client.chat.completions.create(
                model=api_config.get("model", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            gpt_markers = json.loads(
                resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
            )
        except Exception as e:
            log(f"[WARN] GPT message Marker message: {e}")
    final_markers = {}
    for st in ["AC-like", "MES-like", "NPC-like", "OPC-like"]:
        merged = list(set(gpt_markers.get(st, [])) | set(HIGH_CONFIDENCE_MARKERS.get(st, [])))
        final_markers[st] = merged
    return final_markers


def run_infercnv_judge_and_subtype(
    adata, df_clusters, immune_clusters, cluster_key, api_config, gene_pos_file
):
    log("Status update.")
    if not cnv:
        log("Status update.")
        return df_clusters
    adata_cnv = subsample_by_cluster(adata, cluster_key, max_cells_per_cluster=300)
    if not immune_clusters:
        log("Status update.")
        return df_clusters
    adata_cnv.obs["cnv_status"] = "unknown"
    adata_cnv.obs.loc[
        adata_cnv.obs[cluster_key].astype(str).isin(immune_clusters), "cnv_status"
    ] = "reference"
    markers = ask_gpt_for_markers(api_config)
    log(f"\n[Subtype Markers]:\n{json.dumps(markers, ensure_ascii=False, indent=2)}\n")
    if not load_gene_positions(adata_cnv, gene_pos_file):
        log("Status update.")
        return df_clusters
    log("Status update.")
    try:
        cnv.tl.infercnv(
            adata_cnv,
            reference_key="cnv_status",
            reference_cat="reference",
            window_size=100,
            n_jobs=4,
        )
        log("Status update.")
        cnv.tl.pca(adata_cnv)
        cnv.pp.neighbors(adata_cnv)
        cnv.tl.leiden(adata_cnv)
        log("[InferCNV] CNV Score...")
        cnv.tl.cnv_score(adata_cnv)
    except Exception as e:
        log(f"[ERROR] InferCNV message: {e}")
        return df_clusters
    cluster_cnv_scores = adata_cnv.obs.groupby(cluster_key)["cnv_score"].mean()
    ref_cells = adata_cnv.obs[adata_cnv.obs["cnv_status"] == "reference"]
    baseline_score = ref_cells["cnv_score"].mean() if len(ref_cells) > 0 else 0.0
    log(f"[InferCNV] Baseline: {baseline_score:.5f}")
    df_clusters["cnv_score"] = df_clusters["cluster"].map(cluster_cnv_scores)
    threshold = max(baseline_score * 1.3, baseline_score + 0.015)
    log(f"[InferCNV] message: {threshold:.5f}")
    tumor_clusters = []
    is_tumor_map = {}
    for _, row in df_clusters.iterrows():
        cid = str(row["cluster"])
        score = row["cnv_score"] if not pd.isna(row["cnv_score"]) else 0.0
        if cid in immune_clusters:
            is_tumor_map[cid] = False
        elif score > threshold:
            is_tumor_map[cid] = True
            tumor_clusters.append(cid)
        else:
            is_tumor_map[cid] = False
    log(f"[InferCNV] message Cluster: {tumor_clusters}")
    subtype_map = {}
    if tumor_clusters:
        log("Status update.")
        tumor_mask = adata_cnv.obs[cluster_key].astype(str).isin(tumor_clusters)
        adata_tumor = adata_cnv[tumor_mask].copy()
        if adata_tumor.n_obs > 0:
            for subtype, genes in markers.items():
                valid = [g for g in genes if g in adata_tumor.var_names]
                if valid:
                    sc.tl.score_genes(adata_tumor, valid, score_name=f"score_{subtype}")
                else:
                    adata_tumor.obs[f"score_{subtype}"] = -1.0
            score_cols = [f"score_{st}" for st in markers.keys()]
            log("Status update.")
            mean_scores = adata_tumor.obs.groupby(cluster_key)[score_cols].mean()
            print(mean_scores.loc[mean_scores.index.astype(str).isin(tumor_clusters)])
            scores = adata_tumor.obs[score_cols].values
            subtypes_per_cell = [list(markers.keys())[i] for i in np.argmax(scores, axis=1)]
            adata_tumor.obs["predicted_subtype"] = subtypes_per_cell
            for cid in tumor_clusters:
                sub = adata_tumor[adata_tumor.obs[cluster_key].astype(str) == cid]
                if len(sub) > 0:
                    subtype_map[cid] = sub.obs["predicted_subtype"].value_counts().idxmax()
            log(f"\n[Subtype] message: {json.dumps(subtype_map, ensure_ascii=False)}\n")
    df_clusters["original_scMulan_type"] = df_clusters["predicted_cell_type"]
    df_clusters["is_tumor"] = df_clusters["cluster"].map(is_tumor_map)

    def get_final_label(row):
        cid = str(row["cluster"])
        if bool(row.get("is_tumor", False)):
            return subtype_map.get(cid, "AC-like")
        else:
            return row["predicted_cell_type"]

    df_clusters["predicted_cell_type"] = df_clusters.apply(get_final_label, axis=1)
    return df_clusters


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata_h5ad", required=True)
    ap.add_argument("--cluster_key", default="leiden")
    ap.add_argument("--celltype_key", default="cell_type_from_scMulan")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_cells_per_cluster", type=int, default=300)
    ap.add_argument("--api_config_file", type=str, default=None)
    ap.add_argument("--run_tumor_analysis", action="store_true")
    ap.add_argument("--gene_pos_file", type=str, default=None)
    ap.add_argument(
        "--scmulan_root", type=str, default=str(Path(__file__).resolve().parent / "scMulan")
    )
    ap.add_argument("--n_process", type=int, default=4)
    args, _ = ap.parse_known_args()
    forced_cpu = maybe_force_cpu_for_unsupported_gpu()
    if forced_cpu:
        log("Status update.")
    scMulan_mod, GeneSymbolUniform_func = import_scmulan(args.scmulan_root)
    api_config = load_api_config(args.api_config_file)
    adata_mulan, cluster_key_resolved, celltype_key_resolved = run_scmulan(
        scMulan_mod,
        GeneSymbolUniform_func,
        Path(args.adata_h5ad),
        Path(args.ckpt_path),
        args.cluster_key,
        args.celltype_key,
        args.max_cells_per_cluster,
        use_smoothing=False,
        n_process_cpu=args.n_process,
    )
    df_out = aggregate_to_clusters(adata_mulan, cluster_key_resolved, celltype_key_resolved)
    log("Status update.")
    df_out = enforce_fixed_cell_types_with_gpt(
        df_out, col="predicted_cell_type", api_config=api_config
    )
    log(f"[SnapLabel] GPTmessage:\n{df_out['predicted_cell_type'].value_counts().to_string()}")
    if args.run_tumor_analysis:
        immune_clusters = get_reference_clusters(df_out, api_config)
        df_out["is_reference"] = df_out["cluster"].astype(str).isin(immune_clusters)
        df_out = run_infercnv_judge_and_subtype(
            adata_mulan,
            df_out,
            immune_clusters,
            cluster_key_resolved,
            api_config,
            args.gene_pos_file,
        )
    log("Status update.")
    df_out = enforce_fixed_cell_types_with_gpt(
        df_out, col="predicted_cell_type", api_config=api_config
    )
    if "original_scMulan_type" in df_out.columns:
        df_out = enforce_fixed_cell_types_with_gpt(
            df_out, col="original_scMulan_type", api_config=api_config
        )
    log(f"[SnapLabel] message:\n{df_out['predicted_cell_type'].value_counts().to_string()}")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    log(f"[Done] message: {args.out_csv}")
    log(
        f"[Info] cluster_key_used={cluster_key_resolved}, celltype_key_used={celltype_key_resolved}"
    )


if __name__ == "__main__":
    main()
