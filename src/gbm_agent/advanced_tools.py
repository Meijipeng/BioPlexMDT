from __future__ import annotations
import base64
import csv
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
import requests
from bs4 import BeautifulSoup
from .prompts import (
    GIGATIME_VISION_SYSTEM_PROMPT,
    MRI_VISION_SYSTEM_PROMPT,
    gigatime_vision_user_prompt,
    mri_vision_user_prompt,
)

try:
    from .config import (
        client,
        GPT_MODEL,
        CHROMA_DB_DIR,
        CHROMA_COLLECTION_NAME,
        CHROMA_FACTS_COLLECTION_NAME,
        GOOGLE_API_KEY,
        GOOGLE_CSE_ID,
        PUBMED_API_KEY,
        embed_texts,
    )
except ImportError:
    print(" [Warning] config.py import failed. Using mock objects for testing.")
    client = None
    GPT_MODEL = "gemini-3.1-flash-image-preview"
    CHROMA_DB_DIR = Path("./chroma_db")
    CHROMA_COLLECTION_NAME = "guidelines"
    CHROMA_FACTS_COLLECTION_NAME = "facts"
    embed_texts = lambda x: [[0.0] * 1536] * len(x)
    GOOGLE_API_KEY = None
    PUBMED_API_KEY = None
    GOOGLE_CSE_ID = None
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "src" / "gbm_agent" / "data"
_DEFAULT_GIGATIME_SCRIPT = "tools/GigaTIME/scripts/custom_inference.py"
_DEFAULT_GIGATIME_CONDA_ENV = os.environ.get("BIOPLEX_GIGATIME_CONDA_ENV") or "gigatime_check_1"
_DEFAULT_RASPR_SCRIPT = "tools/raspr/run_case_all.sh"
_DEFAULT_WSL_DISTRO = os.environ.get("BIOPLEX_WSL_DISTRO") or "Ubuntu"
_DEFAULT_ROAM_ROOT = "tools/ROAM"
_DEFAULT_ROAM_CONDA_ENV = os.environ.get("BIOPLEX_ROAM_CONDA_ENV") or "roam"
_DEFAULT_SCRNA_TOOLS_DIR_WIN = os.environ.get("BIOPLEX_SCRNA_TOOLS_DIR") or "tools/scrna"
_DEFAULT_SCRNA_PIPELINE = "integrated_analysis_pipeline.py"
_DEFAULT_SCRNA_CONDA_ENV = os.environ.get("BIOPLEX_SCRNA_CONDA_ENV") or "base"
_DEFAULT_VISION_MODEL = os.environ.get("GBM_VISION_MODEL") or GPT_MODEL
try:
    from PIL import Image
except ImportError:
    Image = None
    print(" [Warning] Pillow not installed. llm_mri will fail.")


def _tool_result_base(tool_name: str, **extra: Any) -> Dict[str, Any]:
    base = {
        "ok": False,
        "tool_name": tool_name,
        "records": [],
        "formatted": "",
        "error": None,
    }
    base.update(extra)
    return base


def _run_subprocess(
    cmd_list: List[str], cwd: Optional[str] = None, timeout_min: int = 60
) -> subprocess.CompletedProcess:
    cmd_str = " ".join([str(x) for x in cmd_list])
    print(
        f" [Subprocess] Executing: {cmd_str [:200 ]}..."
        if len(cmd_str) > 200
        else f" [Subprocess] Executing: {cmd_str }"
    )
    if cwd:
        print(f" [Subprocess] CWD: {cwd }")
    is_wsl = len(cmd_list) > 0 and cmd_list[0] == "wsl"
    try:
        return subprocess.run(
            cmd_list,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=int(timeout_min) * 60,
            check=False,
        )
    except FileNotFoundError:
        if is_wsl:
            raise RuntimeError("Status update.")
        print(" [Subprocess] FileNotFoundError, retrying with shell=True...")
        cmd_quoted = " ".join(
            [
                f'"{x }"' if (" " in str(x) and not str(x).startswith('"')) else str(x)
                for x in cmd_list
            ]
        )
        return subprocess.run(
            cmd_quoted,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=int(timeout_min) * 60,
            check=False,
            shell=True,
        )


def _get_conda_exe() -> str:
    c = os.environ.get("CONDA_EXE")
    if c and shutil.which(c):
        return c
    c = shutil.which("conda")
    if c:
        return c
    candidates = [
        r"C:\ProgramData\miniconda3\Scripts\conda.exe",
        r"C:\ProgramData\Anaconda3\Scripts\conda.exe",
        os.path.expanduser(r"~\miniconda3\Scripts\conda.exe"),
        os.path.expanduser(r"~\anaconda3\Scripts\conda.exe"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "conda"


def _conda_run_supports_no_capture_output(conda_exe: str) -> bool:
    try:
        p = subprocess.run(
            [conda_exe, "run", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
            check=False,
        )
        help_txt = p.stdout or ""
        return ("--no-capture-output" in help_txt) or (
            "--no-capture-output" in help_txt.replace("\r", "")
        )
    except Exception:
        return False


class _StreamProcResult:
    def __init__(self, returncode: int, stdout_tail: str, total_lines: int) -> None:
        self.returncode = returncode
        self.stdout_tail = stdout_tail
        self.total_lines = total_lines


def _run_subprocess_stream(
    cmd_list: List[str],
    cwd: Optional[str] = None,
    timeout_min: int = 60,
    env: Optional[Dict[str, str]] = None,
    heartbeat_sec: int = 30,
    keep_last_lines: int = 400,
) -> _StreamProcResult:
    cmd_str = " ".join([str(x) for x in cmd_list])
    print(
        (
            f" [Subprocess(stream)] Executing: {cmd_str [:220 ]}..."
            if len(cmd_str) > 220
            else f" [Subprocess(stream)] Executing: {cmd_str }"
        ),
        flush=True,
    )
    if cwd:
        print(f" [Subprocess(stream)] CWD: {cwd }", flush=True)
    start = time.time()
    last_emit = start
    last_line_time = start
    tail_buf: List[str] = []
    total_lines = 0
    p = subprocess.Popen(
        cmd_list,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1,
        universal_newlines=True,
        env=env,
    )
    assert p.stdout is not None
    try:
        while True:
            if (time.time() - start) > (timeout_min * 60):
                try:
                    p.kill()
                except Exception:
                    pass
                raise TimeoutError(f"Subprocess timeout after {timeout_min } minutes")
            line = p.stdout.readline()
            if line:
                last_line_time = time.time()
                total_lines += 1
                s = line.rstrip("\n")
                print(s, flush=True)
                tail_buf.append(s)
                if len(tail_buf) > keep_last_lines:
                    tail_buf = tail_buf[-keep_last_lines:]
            else:
                if p.poll() is not None:
                    break
                now = time.time()
                if (now - last_emit) >= heartbeat_sec and (now - last_line_time) >= heartbeat_sec:
                    elapsed = now - start
                    print(
                        f" [stream] ...still running ({elapsed :.0f}s elapsed, no new output)...",
                        flush=True,
                    )
                    last_emit = now
                time.sleep(0.2)
        rc = int(p.wait())
        stdout_tail = "\n".join(tail_buf[-keep_last_lines:])
        return _StreamProcResult(returncode=rc, stdout_tail=stdout_tail, total_lines=total_lines)
    finally:
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass


def _extract_json_from_text(text: str) -> Dict[str, Any] | None:
    t = (text or "").strip()
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"```json\s*(\{.*:\})\s*```", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"(\{.*\})", t, flags=re.S)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    return None


def _normalize_to_uint8(arr):
    import numpy as np

    x = arr.astype("float32")
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = (x * 255.0).astype("uint8")
    return x


def _pil_to_data_url(img, max_side: int = 768) -> str:
    if not isinstance(img, Image.Image):
        raise TypeError("Status update.")
    img = img.convert("L")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64 }"


def _load_slices_from_input(input_path: str, max_images: int = 12):
    p = Path(input_path)
    if not p.exists():
        p_alt = _PROJECT_ROOT / input_path
        if p_alt.exists():
            p = p_alt
        else:
            raise FileNotFoundError(f"input_path message: {input_path }")
    if p.is_dir():
        print(f" [MRI] message message: {p .name }")
        mosaic = p / "mri_mosaic.png"
        if mosaic.exists():
            return _load_slices_from_input(str(mosaic), max_images)
        images = sorted([f for f in p.rglob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if images:
            print(f" [MRI] message {len (images )} message message ")
            return _load_slices_from_input(str(images[0]), max_images)
        print(f" [MRI] message DICOM...")
        try:
            import pydicom
        except Exception:
            raise RuntimeError("Status update.")
        dcm_files = sorted(list(p.rglob("*.dcm")))
        if not dcm_files:
            raise RuntimeError(f"message {p } message DICOM")
        import numpy as np

        n_pick = min(len(dcm_files), max_images)
        step = len(dcm_files) / max(n_pick, 1)
        selected = [dcm_files[int(i * step)] for i in range(n_pick)]
        data_urls = []
        for fp in selected:
            try:
                ds = pydicom.dcmread(str(fp), force=True)
                if hasattr(ds, "pixel_array"):
                    arr = ds.pixel_array
                    data_urls.append(_pil_to_data_url(Image.fromarray(_normalize_to_uint8(arr))))
            except Exception:
                pass
        return data_urls, {"kind": "dicom_folder", "count": len(data_urls)}
    suffix = p.suffix.lower()
    if suffix in [".nii", ".gz"]:
        try:
            import nibabel as nib
        except Exception:
            raise RuntimeError("Status update.")
        import numpy as np

        nii = nib.load(str(p))
        vol = nii.get_fdata()
        if vol.ndim < 3:
            raise RuntimeError("Status update.")
        z_dim = vol.shape[2]
        idxs = np.linspace(0, z_dim - 1, num=min(max_images, z_dim), dtype=int)
        data_urls = []
        for z in idxs:
            data_urls.append(_pil_to_data_url(Image.fromarray(_normalize_to_uint8(vol[:, :, z]))))
        return data_urls, {"kind": "nifti", "count": len(data_urls)}
    if not Image:
        raise RuntimeError("Pillow not installed")
    img = Image.open(p)
    return [_pil_to_data_url(img)], {"kind": "image", "count": 1}


def llm_mri(
    input_path: str = "",
    images: Optional[List[str]] = None,
    question: str = "",
    max_images: int = 12,
    model: str | None = None,
) -> Dict[str, Any]:
    if (not input_path or not str(input_path).strip()) and images:
        try:
            input_path = str(images[0])
        except Exception:
            pass
    print(f"\n [Tool: llm_mri] message")
    print(f" - Input: {input_path }")
    result = _tool_result_base("llm_mri", input_path=input_path)
    if not Image:
        result["error"] = "Pillow missing"
        result["formatted"] = "Error: Pillow library not installed."
        return result
    try:
        data_urls, _meta = _load_slices_from_input(input_path, max_images)
        print(f" [MRI] message {len (data_urls )} message/message")
        chosen_model = model or _DEFAULT_VISION_MODEL or GPT_MODEL
        system_text = MRI_VISION_SYSTEM_PROMPT
        user_text = mri_vision_user_prompt(question)
        content_parts = [{"type": "text", "text": system_text + "\n" + user_text}]
        for u in data_urls:
            content_parts.append({"type": "image_url", "image_url": {"url": u}})
        if not client:
            raise RuntimeError("OpenAI client not initialized (check config.py)")
        create_kwargs: Dict[str, Any] = {
            "model": chosen_model,
            "messages": [{"role": "user", "content": content_parts}],
        }
        create_kwargs["temperature"] = 0.2
        resp = client.chat.completions.create(**create_kwargs)
        raw = resp.choices[0].message.content or ""
        mri_struct = _extract_json_from_text(raw)
        if not mri_struct:
            mri_struct = {"raw_output": raw, "note": "JSON parsing failed"}
        findings = mri_struct.get("findings", [])
        impression = mri_struct.get("impression", [])
        md_lines = ["## LLM MRI Report"]
        md_lines.append(f"**Model:** {chosen_model }")
        if findings:
            md_lines.append("### Findings")
            md_lines.append(json.dumps(findings, indent=2, ensure_ascii=False))
        if impression:
            md_lines.append("### Impression")
            md_lines.append(json.dumps(impression, indent=2, ensure_ascii=False))
        if not findings and not impression:
            md_lines.append(raw)
        result["ok"] = True
        result["records"] = [{"mri_structured": mri_struct, "model_used": chosen_model}]
        result["formatted"] = "\n".join(md_lines)
        print("Status update.")
        return result
    except Exception as e:
        print(f" [Tool: llm_mri] message: {e }")
        result["formatted"] = f"MRI Error: {e }"
        result["error"] = str(e)
        return result


def search_local_guidelines(query: str) -> Dict[str, Any]:
    print(f"\n [Tool: search_local_guidelines] '{query }'")
    result = _tool_result_base("search_local_guidelines", query=query)
    try:
        chroma = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        col = chroma.get_collection(name=CHROMA_COLLECTION_NAME)
        emb = embed_texts([query])[0]
        res = col.query(query_embeddings=[emb], n_results=4)
        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        md = [f"## Local Guidelines: {query }"]
        records = []
        for i, (d, m) in enumerate(zip(docs, metas)):
            records.append({"content": d, "meta": m})
            md.append(f"{i +1 }. [{m .get ('source_type','Source')}] {d [:300 ]}...")
        result["ok"] = True
        result["records"] = records
        result["formatted"] = "\n\n".join(md)
        return result
    except Exception as e:
        print(f" Error: {e }")
        result["error"] = str(e)
        return result


def search_local_facts(query: str) -> Dict[str, Any]:
    print(f"\n [Tool: search_local_facts] '{query }'")
    result = _tool_result_base("search_local_facts", query=query)
    try:
        chroma = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        try:
            col = chroma.get_collection(name=CHROMA_FACTS_COLLECTION_NAME)
        except Exception:
            result["error"] = "Facts DB not built"
            return result
        emb = embed_texts([query])[0]
        res = col.query(query_embeddings=[emb], n_results=5)
        docs = res["documents"][0] if res.get("documents") else []
        result["ok"] = True
        result["records"] = [{"fact": d} for d in docs]
        result["formatted"] = f"## Local Facts: {query }\n" + "\n".join([f"- {d }" for d in docs])
        return result
    except Exception as e:
        print(f" Error: {e }")
        result["error"] = str(e)
        return result


def search_google(query: str) -> Dict[str, Any]:
    print(f"\n [Tool: search_google] '{query }'")
    result = _tool_result_base("search_google", query=query)
    if not GOOGLE_API_KEY:
        result["error"] = "Missing API Key"
        return result
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": 5}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        items = data.get("items", [])
        records = [
            {"title": i.get("title"), "link": i.get("link"), "snippet": i.get("snippet")}
            for i in items
        ]
        md = [f"## Google: {query }"] + [
            f"- [{r ['title']}]({r ['link']})\n {r ['snippet']}" for r in records
        ]
        result["ok"] = True
        result["records"] = records
        result["formatted"] = "\n\n".join(md)
        return result
    except Exception as e:
        print(f" Error: {e }")
        result["error"] = str(e)
        return result


def search_pubmed(query: str) -> Dict[str, Any]:
    print(f"\n [Tool: search_pubmed] '{query }'")
    result = _tool_result_base("search_pubmed", query=query)
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        r1 = requests.get(
            f"{base }/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": 5,
                "api_key": PUBMED_API_KEY,
            },
            timeout=10,
        )
        ids = r1.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            result["formatted"] = "No PubMed results."
            return result
        r2 = requests.get(
            f"{base }/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "api_key": PUBMED_API_KEY,
            },
            timeout=15,
        )
        root = ET.fromstring(r2.text)
        records = []
        md = [f"## PubMed: {query }"]
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = art.findtext(".//ArticleTitle")
            records.append({"pmid": pmid, "title": title})
            md.append(f"- PMID:{pmid } {title }")
        result["ok"] = True
        result["records"] = records
        result["formatted"] = "\n".join(md)
        return result
    except Exception as e:
        print(f" Error: {e }")
        result["error"] = str(e)
        return result


def read_webpage(url: str) -> Dict[str, Any]:
    print(f"\n [Tool: read_webpage] {url }")
    result = _tool_result_base("read_webpage", url=url)
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.content, "html.parser")
        text = soup.get_text(strip=True)[:2000]
        result["ok"] = True
        result["formatted"] = f"## Content of {url }\n{text }..."
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def _collect_gigatime_visualizations(
    output_dir: str, visualizations_dir: str = ""
) -> Tuple[str, List[str]]:
    base_dir = Path(_strip_quotes(output_dir or "")).expanduser()
    vis_dir = Path(_strip_quotes(visualizations_dir or "")).expanduser()
    if visualizations_dir:
        if not vis_dir.is_absolute():
            vis_dir = (Path.cwd() / vis_dir).resolve()
    else:
        if base_dir.exists():
            vis_dir = (base_dir / "visualizations").resolve()
        else:
            vis_dir = (Path(output_dir) / "visualizations").resolve()
    pngs: List[str] = []
    if vis_dir.exists() and vis_dir.is_dir():
        for p in vis_dir.rglob("Map_*.png"):
            pngs.append(str(p))
    pngs = sorted(pngs, key=lambda x: Path(x).name)
    return str(vis_dir), pngs


def _llm_vision_call_images(
    image_paths: List[str],
    question: str,
    model: Optional[str] = None,
    max_images: int = 12,
) -> Dict[str, Any]:
    if not Image:
        return {"ok": False, "error": "Pillow not installed"}
    if not client:
        return {"ok": False, "error": "OpenAI client not initialized (check config.py)"}
    chosen_model = model or _DEFAULT_VISION_MODEL or GPT_MODEL
    picked = image_paths[: max_images if max_images and max_images > 0 else 12]
    content_parts: List[Dict[str, Any]] = []
    system_text = GIGATIME_VISION_SYSTEM_PROMPT
    user_text = gigatime_vision_user_prompt(question)
    content_parts.append({"type": "text", "text": system_text + "\n" + user_text})
    for fp in picked:
        try:
            img = Image.open(fp).convert("RGB")
            w, h = img.size
            max_side = 1024
            scale = min(1.0, float(max_side) / float(max(w, h)))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{b64 }"
            content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        except Exception:
            continue
    create_kwargs: Dict[str, Any] = {
        "model": chosen_model,
        "messages": [{"role": "user", "content": content_parts}],
    }
    create_kwargs["temperature"] = 0.2
    resp = client.chat.completions.create(**create_kwargs)
    raw = resp.choices[0].message.content or ""
    parsed = _extract_json_from_text(raw) or {"raw_output": raw, "note": "JSON parsing failed"}
    return {
        "ok": True,
        "model_used": chosen_model,
        "image_count": len(picked),
        "images_used": picked,
        "analysis": parsed,
        "raw": raw,
    }


def llm_gigatime(
    output_dir: str = "",
    visualizations_dir: str = "",
    question: str = "",
    max_images: int = 12,
    model: str | None = None,
) -> Dict[str, Any]:
    print(f"\n [Tool: llm_gigatime] message", flush=True)
    print(f" - output_dir: {output_dir }", flush=True)
    print(f" - visualizations_dir: {visualizations_dir }", flush=True)
    result = _tool_result_base(
        "llm_gigatime",
        output_dir=output_dir,
        visualizations_dir=visualizations_dir,
        question=question,
        max_images=max_images,
    )
    try:
        vis_dir, maps = _collect_gigatime_visualizations(
            output_dir=output_dir, visualizations_dir=visualizations_dir
        )
        if not maps:
            result["error"] = f"No Map_*.png found under: {vis_dir }"
            result["formatted"] = f"[llm_gigatime] No Map_*.png found under: {vis_dir }"
            return result
        print(f" [llm_gigatime] Found {len (maps )} maps under: {vis_dir }", flush=True)
        call_res = _llm_vision_call_images(
            image_paths=maps,
            question=question,
            model=model,
            max_images=max_images,
        )
        if not call_res.get("ok"):
            result["error"] = call_res.get("error")
            result["formatted"] = f"[llm_gigatime] Error: {result ['error']}"
            return result
        analysis = call_res.get("analysis", {})
        chosen_model = call_res.get("model_used", model or _DEFAULT_VISION_MODEL or GPT_MODEL)
        md: List[str] = []
        md.append("## LLM-GigaTIME Interpretation")
        md.append(f"**Model:** {chosen_model }")
        md.append(f"**Visualizations Dir:** {vis_dir }")
        md.append(f"**Maps Used:** {call_res .get ('image_count')} / {len (maps )}")
        md.append("")
        md.append("### Structured Interpretation (JSON)")
        md.append(json.dumps(analysis, ensure_ascii=False, indent=2))
        result["ok"] = True
        result["records"] = [
            {
                "visualizations_dir": vis_dir,
                "map_paths": maps,
                "model_used": chosen_model,
                "analysis": analysis,
            }
        ]
        result["formatted"] = "\n".join(md)
        print("Status update.", flush=True)
        return result
    except Exception as e:
        print(f" [Tool: llm_gigatime] Error: {e }", flush=True)
        result["error"] = str(e)
        result["formatted"] = str(e)
        return result


def gigatime_infer(
    input_tiff: str,
    output_dir: str = "",
    conda_env: str = _DEFAULT_GIGATIME_CONDA_ENV,
    timeout_min: int = 60,
    run_llm_analysis: bool = True,
    llm_question: str = "",
    llm_max_images: int = 12,
    llm_model: str | None = None,
) -> Dict[str, Any]:
    print(f"\n [Tool: gigatime_infer] {input_tiff }", flush=True)
    result = _tool_result_base("gigatime_infer", input_tiff=input_tiff)
    try:
        conda_env = (conda_env or _DEFAULT_GIGATIME_CONDA_ENV).strip()
        tiff_path = Path(_resolve_project_path(input_tiff)).resolve()
        if not tiff_path.exists():
            tiff_alt = (_PROJECT_ROOT / input_tiff).resolve()
            if tiff_alt.exists():
                tiff_path = tiff_alt
            else:
                result["error"] = f"TIFF not found: {input_tiff }"
                return result
        script_cfg = _config_value(
            "BIOPLEX_GIGATIME_EXTERNAL_SCRIPT",
            default=_config_value("BIOPLEX_GIGATIME_SCRIPT", default=_DEFAULT_GIGATIME_SCRIPT),
        )
        script_fp = Path(_resolve_project_path(script_cfg))
        if not script_fp.exists():
            result["error"] = f"GigaTIME script not found: {script_fp }"
            return result
        if not output_dir:
            output_dir = str(
                _DATA_DIR / "gigatime_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        conda_exe = _get_conda_exe()
        cmd = [
            conda_exe,
            "run",
            "-n",
            conda_env,
            "python",
            "-u",
            str(script_fp),
            "--input_tiff",
            str(tiff_path),
            "--output_dir",
            str(output_dir),
        ]
        run_env = os.environ.copy()
        run_env["PYTHONUNBUFFERED"] = "1"
        print(f" [GigaTIME] conda_env={conda_env }", flush=True)
        print(f" [GigaTIME] script={script_fp }", flush=True)
        print(f" [GigaTIME] input_tiff={tiff_path }", flush=True)
        print(f" [GigaTIME] output_dir={output_dir }", flush=True)
        print(" [GigaTIME] ---- begin streaming logs ----", flush=True)
        t0 = time.time()
        proc_s = _run_subprocess_stream(
            cmd,
            cwd=str(script_fp.parent),
            timeout_min=timeout_min,
            env=run_env,
            heartbeat_sec=30,
            keep_last_lines=500,
        )
        elapsed = time.time() - t0
        print(" [GigaTIME] ---- end streaming logs ----", flush=True)
        gen_imgs = []
        for p in Path(output_dir).rglob("*.png"):
            if "tile_" not in p.name:
                gen_imgs.append(str(p))
        gen_imgs = sorted(gen_imgs)
        vis_dir, maps = _collect_gigatime_visualizations(
            output_dir=output_dir, visualizations_dir=""
        )
        map_names = [Path(x).name for x in maps[:30]]
        ok = (proc_s.returncode == 0) and (len(gen_imgs) > 0)
        tail = proc_s.stdout_tail or ""
        result["ok"] = bool(ok)
        rec: Dict[str, Any] = {
            "output_dir": output_dir,
            "generated_images": gen_imgs,
            "elapsed_sec": float(elapsed),
            "return_code": int(proc_s.returncode),
            "log_tail": tail[-2000:],
            "total_log_lines": int(proc_s.total_lines),
            "visualizations_dir": vis_dir if Path(vis_dir).exists() else None,
            "map_images": maps,
            "map_image_names_preview": map_names,
        }
        llm_block: Optional[Dict[str, Any]] = None
        if run_llm_analysis and ok and maps:
            try:
                print(" [GigaTIME][LLM] Starting llm_gigatime interpretation...", flush=True)
                llm_res = llm_gigatime(
                    output_dir=output_dir,
                    visualizations_dir=vis_dir,
                    question=llm_question,
                    max_images=llm_max_images,
                    model=llm_model,
                )
                llm_block = {
                    "ok": bool(llm_res.get("ok")),
                    "error": llm_res.get("error"),
                    "records": llm_res.get("records", []),
                    "formatted": llm_res.get("formatted", ""),
                }
                print(" [GigaTIME][LLM] Interpretation done.", flush=True)
            except Exception as e:
                llm_block = {"ok": False, "error": str(e)}
        if llm_block is not None:
            rec["llm_gigatime"] = llm_block
        result["records"] = [rec]
        img_list_str = "\n".join([f"- {Path (p ).name }" for p in gen_imgs[:5]])
        status = "Success" if result["ok"] else "Failed"
        formatted_parts: List[str] = []
        formatted_parts.append(f"## GigaTIME Result ({status })")
        formatted_parts.append(f"Output Dir: {output_dir }")
        formatted_parts.append(f"Elapsed: {elapsed :.2f}s")
        formatted_parts.append(f"ReturnCode: {proc_s .returncode }")
        formatted_parts.append(f"Visualizations Dir: {vis_dir }")
        formatted_parts.append(f"Map Images Found: {len (maps )}")
        if map_names:
            formatted_parts.append("Map Names Preview:")
            formatted_parts.append("\n".join([f"- {n }" for n in map_names[:10]]))
        formatted_parts.append("Generated Heatmaps (non-tile) Preview:")
        formatted_parts.append(img_list_str)
        if llm_block and llm_block.get("ok"):
            formatted_parts.append("\n--- LLM-GigaTIME Interpretation ---")
            formatted_parts.append(str(llm_block.get("formatted", ""))[:4000])
        elif llm_block and not llm_block.get("ok"):
            formatted_parts.append("\n--- LLM-GigaTIME Interpretation (Failed) ---")
            formatted_parts.append(str(llm_block.get("error", "")))
        formatted_parts.append("\nLog Tail:")
        formatted_parts.append(tail[-800:])
        result["formatted"] = "\n".join(formatted_parts)
        print("Status update.", flush=True)
        return result
    except Exception as e:
        print(f" [Tool: gigatime_infer] Error: {e }", flush=True)
        result["error"] = str(e)
        return result


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _config_value(*names: str, default: str = "") -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return default


def _is_probably_wsl_path(p: str) -> bool:
    p = (p or "").strip()
    return p.startswith("/mnt/") or p.startswith("/home/") or p.startswith("/")


def _resolve_project_path(path_text: str, default: str = "") -> str:
    raw = _strip_quotes(str(path_text or default or "")).strip()
    if not raw:
        return ""
    if _is_probably_wsl_path(raw) or raw.lower().startswith("wsl:"):
        return raw
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    try:
        return str(p.resolve(strict=False))
    except Exception:
        return str(p)


def _resolve_script_path(script_or_name: str, base_dir: str, default_name: str) -> str:
    raw = _strip_quotes(str(script_or_name or default_name or "")).strip()
    if not raw:
        return ""
    if Path(raw).is_absolute() or any(sep in raw for sep in ("/", "\\")):
        return _resolve_project_path(raw)
    return _resolve_project_path(str(Path(base_dir) / raw))


def _config_path_to_wsl(path_text: str, default: str = "", convert_windows_path: bool = True) -> str:
    resolved = _resolve_project_path(path_text, default=default)
    if convert_windows_path:
        return _win_to_wsl_path(resolved)
    return resolved


def _win_to_wsl_path(win_path: str) -> str:
    p = _strip_quotes(win_path)
    if not p:
        return p
    if _is_probably_wsl_path(p):
        return p
    p2 = p.replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p2)
    if not m:
        return p2
    drive = m.group(1).lower()
    rest = m.group(2)
    return f"/mnt/{drive }/{rest }"


def _infer_case_from_path(text: str) -> str:
    m = re.search(r"(MR\d{6,})", text or "", flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _sh_quote(s: str) -> str:
    s = s or ""
    return "'" + s.replace("'", r"'\''") + "'"


def _parse_raspr_paths_from_log(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    out_csv = None
    final_dir = None
    run_root = None
    m1 = re.search(r"RaSPr output:\s*(\S+)", text or "")
    if m1:
        out_csv = m1.group(1).strip()
    m2 = re.search(r"FINAL_DIR=(\S+)", text or "")
    if m2:
        final_dir = m2.group(1).strip()
    m3 = re.search(r"RUN_ROOT=(\S+)", text or "")
    if m3:
        run_root = m3.group(1).strip()
    return out_csv, final_dir, run_root


def _wsl_test_file_exists(wsl_distro: str, path_wsl: str, timeout_sec: int = 15) -> bool:
    bash_cmd = f"test -f {_sh_quote (path_wsl )} && echo OK || echo NO"
    args = ["wsl", "-d", wsl_distro, "bash", "-lc", bash_cmd]
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(timeout_sec),
            check=False,
        )
        out = (proc.stdout or b"").decode("utf-8", errors="ignore").strip()
        return out.endswith("OK")
    except Exception:
        return False


def _choose_existing_csv(wsl_distro: str, candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and _wsl_test_file_exists(wsl_distro, p):
            return p
    return None


def _wsl_cat_file_bytes(
    wsl_distro: str, path_wsl: str, timeout_sec: int = 30
) -> Tuple[int, bytes, str]:
    bash_cmd = (
        "python3 - <<'PY'\n"
        "import sys\n"
        f"p={path_wsl !r }\n"
        "with open(p,'rb') as f:\n"
        " sys.stdout.buffer.write(f.read())\n"
        "PY"
    )
    args = ["wsl", "-d", wsl_distro, "bash", "-lc", bash_cmd]
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(timeout_sec),
            check=False,
        )
        err_txt = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
        return int(proc.returncode), (proc.stdout or b""), err_txt
    except Exception as e:
        return 1, b"", f"{e }"


def _decode_csv_bytes(b: bytes) -> str:
    if not b:
        return ""
    for bom in (b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf"):
        idx = b.find(bom)
        if idx > 0 and idx < 4096:
            b = b[idx:]
            break
    if b.startswith((b"\xff\xfe", b"\xfe\xff")) or b.count(b"\x00") > 0:
        txt = b.decode("utf-16", errors="ignore")
    else:
        txt = b.decode("utf-8", errors="ignore")
    return (txt or "").replace("\x00", "")


def _parse_survival_from_csv_text(csv_text: str) -> Dict[str, Any]:
    txt = (csv_text or "").replace("\x00", "")
    warnings: List[str] = []
    cleaned_lines: List[str] = []
    for ln in txt.splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        if s.lower().startswith("wsl:"):
            warnings.append(s)
            continue
        if ("Status update." in s and "WSL" in s) or ("localhost" in s and "proxy" in s.lower()):
            warnings.append(s)
            continue
        cleaned_lines.append(ln)
    txt2 = "\n".join(cleaned_lines).strip()
    if not txt2:
        return {"raw": txt.splitlines()[:10], "warnings": warnings}
    rows = [r for r in csv.reader(io.StringIO(txt2)) if any((c or "").strip() for c in r)]
    if len(rows) < 2:
        return {"raw": rows[:5], "warnings": warnings}

    def is_header(r: List[str]) -> bool:
        if len(r) < 2:
            return False
        if any((c or "").strip().lower().startswith("wsl:") for c in r):
            return False
        return any(re.search(r"[A-Za-z]", (c or "")) for c in r)

    h_idx = None
    for i, r in enumerate(rows):
        if is_header(r) and i + 1 < len(rows):
            h_idx = i
            break
    if h_idx is None:
        header = [h.strip() for h in rows[0]]
        row = [c.strip() for c in rows[1]]
    else:
        header = [h.strip() for h in rows[h_idx]]
        row = [c.strip() for c in rows[h_idx + 1]]
    data: Dict[str, Any] = {}
    for i in range(min(len(header), len(row))):
        k = header[i]
        v = row[i]
        try:
            data[k] = float(v) if v != "" else v
        except Exception:
            data[k] = v
    keys = list(data.keys())
    pri = [k for k in keys if re.search(r"(survival|os|pfs|risk|score|month|time)", k, flags=re.I)]
    brief: Dict[str, Any] = {k: data.get(k) for k in pri[:12]}
    return {"fields": data, "highlights": brief, "warnings": warnings}


def _build_raspr_one_liner(
    case: str,
    dicom_dir_wsl: str,
    script_path_wsl: str,
    nnunet_results_wsl: str,
    nnunet_raw_wsl: str,
    nnunet_pre_wsl: str,
) -> str:
    q_case = _sh_quote(case)
    q_dicom = _sh_quote(dicom_dir_wsl)
    q_script = _sh_quote(script_path_wsl)
    q_res = _sh_quote(nnunet_results_wsl)
    q_raw = _sh_quote(nnunet_raw_wsl)
    q_pre = _sh_quote(nnunet_pre_wsl)
    parts: List[str] = []
    parts.append("set -e")
    parts.append('export PATH="$HOME/miniconda3/bin:$HOME/anaconda3/bin:/opt/conda/bin:$PATH"')
    parts.append(f"export nnUNet_results={q_res }")
    parts.append(f"export nnUNet_raw={q_raw }")
    parts.append(f"export nnUNet_preprocessed={q_pre }")
    parts.append('echo "[RaSPr][V4.5] ===== ENV ====="')
    parts.append('echo "[RaSPr] nnUNet_results=${nnUNet_results:-__EMPTY__}"')
    parts.append('echo "[RaSPr] nnUNet_raw=${nnUNet_raw:-__EMPTY__}"')
    parts.append('echo "[RaSPr] nnUNet_preprocessed=${nnUNet_preprocessed:-__EMPTY__}"')
    parts.append(f'echo "[RaSPr] dicom_dir={dicom_dir_wsl }"')
    parts.append(
        f"if [ ! -d {q_dicom } ]; then echo '[RaSPr][FATAL] DICOM dir not found:' {q_dicom }; exit 10; fi"
    )
    parts.append(
        f"if [ ! -f {q_script } ]; then echo '[RaSPr][FATAL] script not found:' {q_script }; exit 11; fi"
    )
    parts.append(f"ls -la {q_dicom } | head -n 10 || true")
    parts.append('echo "[RaSPr][V4.5] ===== RUN ====="')
    parts.append("set +e")
    parts.append(f"bash {q_script } --case {q_case } --dicom {q_dicom }")
    parts.append("rc=$?")
    parts.append("set -e")
    parts.append(
        'if [ "$rc" -eq 141 ]; then echo "[RaSPr][WARN] rc=141 (SIGPIPE from head under pipefail). Treat as success."; rc=0; fi'
    )
    parts.append("exit $rc")
    return "; ".join(parts)


def raspr_run(
    case: str = "",
    dicom_dir: str = "",
    output_dir: str = "",
    timeout_min: int = 120,
    script_path: str = "",
    wsl_distro: str = _DEFAULT_WSL_DISTRO,
    convert_windows_path_to_wsl: bool = True,
) -> Dict[str, Any]:
    dicom_in = _strip_quotes(str(dicom_dir or "")).strip()
    if not dicom_in:
        r = _tool_result_base("raspr_run")
        r["error"] = "dicom_dir is empty."
        r["formatted"] = "dicom_dir is empty."
        return r
    c = (case or "").strip() or _infer_case_from_path(dicom_in)
    if not c:
        r = _tool_result_base("raspr_run")
        r["error"] = "case is empty and cannot be inferred from dicom_dir."
        r["formatted"] = "case is empty and cannot be inferred from dicom_dir."
        return r
    script_cfg = (script_path or "").strip() or _config_value(
        "BIOPLEX_RASPR_EXTERNAL_SCRIPT",
        default=_config_value("BIOPLEX_RASPR_SCRIPT_PATH", default=_DEFAULT_RASPR_SCRIPT),
    )
    wsl_distro = (wsl_distro or _DEFAULT_WSL_DISTRO).strip()
    dicom_wsl = _win_to_wsl_path(dicom_in) if convert_windows_path_to_wsl else dicom_in
    dicom_wsl = dicom_wsl.replace("\\", "/")
    script_path_wsl = _config_path_to_wsl(script_cfg, convert_windows_path=convert_windows_path_to_wsl)
    nnunet_results_cfg = _config_value(
        "BIOPLEX_NNUNET_RESULTS", default=os.environ.get("NNUNET_RESULTS_WIN", "nnUNet_folders/results")
    )
    nnunet_results_wsl = _config_path_to_wsl(
        nnunet_results_cfg, convert_windows_path=convert_windows_path_to_wsl
    ).replace("\\", "/")
    nnunet_parent_wsl = posixpath.dirname(nnunet_results_wsl.rstrip("/")) or "."
    nnunet_raw_wsl = _config_path_to_wsl(
        _config_value("BIOPLEX_NNUNET_RAW", default=posixpath.join(nnunet_parent_wsl, "raw")),
        convert_windows_path=convert_windows_path_to_wsl,
    ).replace("\\", "/")
    nnunet_pre_wsl = _config_path_to_wsl(
        _config_value(
            "BIOPLEX_NNUNET_PREPROCESSED",
            default=posixpath.join(nnunet_parent_wsl, "preprocessed"),
        ),
        convert_windows_path=convert_windows_path_to_wsl,
    ).replace("\\", "/")
    win_dicom_count = None
    try:
        if os.path.isdir(dicom_in):
            win_dicom_count = len([x for x in os.listdir(dicom_in) if x.lower().endswith(".dcm")])
    except Exception:
        win_dicom_count = None
    print(f"\n [Tool: raspr_run] Case: {c }", flush=True)
    print(f" [RaSPr] DICOM message(WSL): {dicom_wsl }", flush=True)
    if win_dicom_count is not None:
        print(f" [RaSPr] Windowsmessage DICOM message: {win_dicom_count }", flush=True)
    print(f" [RaSPr] message (WSL): {nnunet_results_wsl }", flush=True)
    print(f" [RaSPr] script_path(WSL): {script_path_wsl }", flush=True)
    bash_one_liner = _build_raspr_one_liner(
        case=c,
        dicom_dir_wsl=dicom_wsl,
        script_path_wsl=script_path_wsl,
        nnunet_results_wsl=nnunet_results_wsl,
        nnunet_raw_wsl=nnunet_raw_wsl,
        nnunet_pre_wsl=nnunet_pre_wsl,
    )
    cmd = ["wsl", "-d", wsl_distro, "bash", "-lc", bash_one_liner]
    print(f" [Subprocess] Executing: wsl -d {wsl_distro } bash -lc <one-liner>", flush=True)
    result = _tool_result_base(
        "raspr_run",
        case=c,
        dicom_dir=dicom_wsl,
        output_dir=output_dir,
        script_path=script_path_wsl,
        wsl_distro=wsl_distro,
    )
    try:
        proc = _run_subprocess(cmd, timeout_min=timeout_min)
        rc = int(proc.returncode)
        merged = (proc.stdout or "").strip()
        out_csv_log, final_dir, run_root = _parse_raspr_paths_from_log(merged)
        runs_cfg = _config_value("BIOPLEX_RASPR_RUNS_ROOT", default="")
        if runs_cfg:
            fallback_runs_root = _config_path_to_wsl(
                runs_cfg, convert_windows_path=convert_windows_path_to_wsl
            ).replace("\\", "/")
        else:
            fallback_runs_root = posixpath.join(
                posixpath.dirname(script_path_wsl.rstrip("/")) or ".", "runs"
            )
        candidates: List[str] = []
        if out_csv_log:
            candidates.append(out_csv_log)
        if run_root:
            candidates.append(f"{run_root }/raspr/{c }_raspr.csv")
            candidates.append(f"{run_root }/raspr/{c }_raspr.csv".replace("//", "/"))
        if final_dir:
            candidates.append(f"{final_dir }/raspr/{c }_raspr.csv")
            candidates.append(f"{final_dir }/raspr/{c }_raspr.csv".replace("//", "/"))
        if not run_root:
            candidates.append(f"{fallback_runs_root }/{c }/raspr/{c }_raspr.csv")
            candidates.append(f"{fallback_runs_root }/{c }/final/raspr/{c }_raspr.csv")
        chosen_csv = _choose_existing_csv(wsl_distro, candidates)
        survival_pred: Dict[str, Any] = {"status": "Unknown"}
        if chosen_csv:
            rc2, b, wsl_err = _wsl_cat_file_bytes(wsl_distro, chosen_csv, timeout_sec=40)
            txt = _decode_csv_bytes(b) if (rc2 == 0 and b) else ""
            if txt:
                survival_pred = _parse_survival_from_csv_text(txt)
            else:
                survival_pred = {
                    "status": "Unknown",
                    "csv_path": chosen_csv,
                    "cat_error": f"rc={rc2 } bytes={len (b )}",
                }
            if wsl_err:
                survival_pred.setdefault("wsl_stderr", wsl_err)
        else:
            survival_pred = {
                "status": "Unknown",
                "csv_path": None,
                "note": "Could not locate raspr csv from candidates",
                "candidates": candidates[:10],
            }
        ok = (rc == 0) or (chosen_csv is not None) or (final_dir is not None)
        rec = {
            "predicted_survival": survival_pred,
            "final_dir": final_dir,
            "run_root": run_root,
            "raspr_csv_path": chosen_csv or out_csv_log,
            "return_code": rc,
            "log_tail": "\n".join(merged.splitlines()[-140:]),
        }
        formatted = "\n".join(
            [
                f"[RaSPr] case={c }",
                f"[RaSPr] dicom_dir={dicom_wsl }",
                f"[RaSPr] script_path={script_path }",
                f"[RaSPr] RUN_ROOT={run_root }" if run_root else "[RaSPr] RUN_ROOT=__UNKNOWN__",
                f"[RaSPr] FINAL_DIR={final_dir }" if final_dir else "[RaSPr] FINAL_DIR=__UNKNOWN__",
                (
                    f"[RaSPr] OUT_CSV(log)={out_csv_log }"
                    if out_csv_log
                    else "[RaSPr] OUT_CSV(log)=__UNKNOWN__"
                ),
                (
                    f"[RaSPr] OUT_CSV(chosen)={chosen_csv }"
                    if chosen_csv
                    else "[RaSPr] OUT_CSV(chosen)=__UNKNOWN__"
                ),
                "[RaSPr] SurvivalPrediction=" + json.dumps(survival_pred, ensure_ascii=False),
                "----- RaSPr Log Tail -----",
                rec["log_tail"],
            ]
        )
        result["ok"] = bool(ok)
        result["records"] = [rec]
        result["formatted"] = formatted
        if not result["ok"]:
            result["error"] = f"Process failed (rc={rc })"
        print(f" [Tool: raspr_run] message ({'Success'if result ['ok']else 'Fail'})", flush=True)
        return result
    except Exception as e:
        print(f" [Tool: raspr_run] Error: {e }", flush=True)
        result["error"] = str(e)
        result["formatted"] = str(e)
        return result


def _build_roam_one_liner(
    tiff_path_wsl: str,
    roam_root_wsl: str,
    assets_root_wsl: str,
    results_root_wsl: str,
    conda_env: str,
) -> str:
    q_tiff = _sh_quote(tiff_path_wsl)
    q_root = _sh_quote(roam_root_wsl)
    q_assets = _sh_quote(assets_root_wsl)
    q_results = _sh_quote(results_root_wsl)
    q_conda = _sh_quote(conda_env)
    parts: List[str] = []
    parts.append("set -e")
    parts.append('export PATH="$HOME/miniconda3/bin:$HOME/anaconda3/bin:/opt/conda/bin:$PATH"')
    parts.append("source ~/.bashrc >/dev/null 2>&1 || true")
    parts.append(
        'if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then source "$HOME/miniconda3/etc/profile.d/conda.sh"; fi'
    )
    parts.append(
        'if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then source "$HOME/anaconda3/etc/profile.d/conda.sh"; fi'
    )
    parts.append(
        'if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then source "/opt/conda/etc/profile.d/conda.sh"; fi'
    )
    parts.append(
        'if ! command -v conda >/dev/null 2>&1; then echo "[ROAM][FATAL] conda not found in PATH"; exit 12; fi'
    )
    parts.append(f"conda activate {q_conda }")
    parts.append(
        f"if [ ! -d {q_root } ]; then echo '[ROAM][FATAL] ROAM root not found:' {q_root }; exit 11; fi"
    )
    parts.append(f"cd {q_root }")
    parts.append(
        f"if [ ! -f {q_tiff } ]; then echo '[ROAM][FATAL] TIFF not found:' {q_tiff }; exit 10; fi"
    )
    parts.append(
        "if [ ! -f run_roam_infer.py ]; then echo '[ROAM][FATAL] run_roam_infer.py not found in ROAM root'; exit 13; fi"
    )
    parts.append(f"mkdir -p {q_results }")
    parts.append(
        "python run_roam_infer.py "
        f"--tiff {q_tiff } "
        f"--assets_root {q_assets } "
        f"--results_root {q_results } "
        "--pretrained_model ImageNet "
        "--device auto "
        "--test_dataset_name custom_single "
        "--force"
    )
    return "; ".join(parts)


def roam_infer(
    input_tiff: str,
    wsl_distro: str = _DEFAULT_WSL_DISTRO,
    timeout_min: int = 60,
    convert_windows_path_to_wsl: bool = True,
    roam_root: str = "",
    assets_root: str = "",
    results_root: str = "",
    conda_env: str = _DEFAULT_ROAM_CONDA_ENV,
) -> Dict[str, Any]:
    tiff_in = _strip_quotes(str(input_tiff or "")).strip()
    wsl_distro = (wsl_distro or "").strip() or _DEFAULT_WSL_DISTRO
    conda_env = (conda_env or "").strip() or _DEFAULT_ROAM_CONDA_ENV
    print(f"\n [Tool: roam_infer] {tiff_in }", flush=True)
    result = _tool_result_base("roam_infer", input_tiff=tiff_in, wsl_distro=wsl_distro)
    if not tiff_in:
        result["error"] = "input_tiff is empty."
        result["formatted"] = "input_tiff is empty."
        return result
    tiff_wsl = _win_to_wsl_path(tiff_in) if convert_windows_path_to_wsl else tiff_in
    tiff_wsl = (tiff_wsl or "").replace("\\", "/")
    root_cfg = (roam_root or "").strip() or _config_value(
        "BIOPLEX_ROAM_EXTERNAL_ROOT", default=_config_value("BIOPLEX_ROAM_ROOT", default=_DEFAULT_ROAM_ROOT)
    )
    root_wsl = _config_path_to_wsl(
        root_cfg, convert_windows_path=convert_windows_path_to_wsl
    ).replace("\\", "/")
    assets_cfg = (assets_root or "").strip() or _config_value(
        "BIOPLEX_ROAM_ASSETS", default=posixpath.join(root_wsl, "assets")
    )
    results_cfg = (results_root or "").strip() or _config_value(
        "BIOPLEX_ROAM_RESULTS", default=posixpath.join(root_wsl, "results")
    )
    assets_wsl = _config_path_to_wsl(
        assets_cfg, convert_windows_path=convert_windows_path_to_wsl
    ).replace("\\", "/")
    results_wsl = _config_path_to_wsl(
        results_cfg, convert_windows_path=convert_windows_path_to_wsl
    ).replace("\\", "/")
    print(f" [ROAM] TIFF message(WSL): {tiff_wsl }", flush=True)
    print(f" [ROAM] root(WSL): {root_wsl }", flush=True)
    bash_one_liner = _build_roam_one_liner(
        tiff_path_wsl=tiff_wsl,
        roam_root_wsl=root_wsl,
        assets_root_wsl=assets_wsl,
        results_root_wsl=results_wsl,
        conda_env=conda_env,
    )
    cmd = ["wsl", "-d", wsl_distro, "bash", "-lc", bash_one_liner]
    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    print(f" [ROAM] wsl_distro={wsl_distro }", flush=True)
    print(" [ROAM] ---- begin streaming logs ----", flush=True)
    try:
        t0 = time.time()
        proc_s = _run_subprocess_stream(
            cmd,
            cwd=None,
            timeout_min=timeout_min,
            env=run_env,
            heartbeat_sec=30,
            keep_last_lines=500,
        )
        elapsed = time.time() - t0
        print(" [ROAM] ---- end streaming logs ----", flush=True)
        merged_tail = proc_s.stdout_tail or ""
        pred = None
        for pat in [
            r"\bpred(?:icted)?(?:[_\s-]*(?:subtype|class|label))?\s*[:=]\s*(?P<value>[^\n\r]+)",
            r"\b(?:subtype|class|label)\s*[:=]\s*(?P<value>[^\n\r]+)",
        ]:
            m = re.search(pat, merged_tail or "", flags=re.IGNORECASE)
            if m:
                pred = re.sub(r"\s+", " ", m.group("value") or "").strip(" \"'")
                break
        ok = proc_s.returncode == 0
        rec = {
            "predicted_subtype_guess": pred,
            "return_code": int(proc_s.returncode),
            "elapsed_sec": float(elapsed),
            "tiff_wsl": tiff_wsl,
            "total_log_lines": int(proc_s.total_lines),
            "log_tail": "\n".join((merged_tail.splitlines()[-160:] if merged_tail else [])),
        }
        result["ok"] = bool(ok)
        result["records"] = [rec]
        result["formatted"] = "\n".join(
            [
                f"[ROAM] wsl_distro={wsl_distro }",
                f"[ROAM] tiff={tiff_wsl }",
                f"[ROAM] predicted_subtype_guess={pred }",
                f"[ROAM] return_code={proc_s .returncode }",
                f"[ROAM] elapsed_sec={elapsed :.2f}",
                "----- ROAM Log Tail -----",
                rec["log_tail"],
            ]
        )
        if not result["ok"]:
            result["error"] = f"Process failed (rc={proc_s .returncode })"
        print(f" [Tool: roam_infer] message ({'Success'if result ['ok']else 'Fail'})", flush=True)
        return result
    except Exception as e:
        print(f" [Tool: roam_infer] Error: {e }", flush=True)
        result["error"] = str(e)
        result["formatted"] = str(e)
        return result


def _safe_resolve_path(p: str) -> str:
    s = _strip_quotes(str(p or "")).strip()
    if not s:
        return s
    try:
        return _resolve_project_path(s)
    except Exception:
        return s
    return s


def _read_patient_cluster_csv(csv_path: str, sample_id: str) -> Optional[str]:
    try:
        p = Path(csv_path)
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            raw = p.read_text(encoding="utf-8-sig", errors="ignore")
        if not raw.strip():
            return None
        sample = raw[:4096]
        delim = "\t" if ("\t" in sample and sample.count("\t") >= sample.count(",")) else ","
        rows = [
            r
            for r in csv.reader(io.StringIO(raw), delimiter=delim)
            if any((c or "").strip() for c in r)
        ]
        if not rows:
            return None
        header = [c.strip() for c in rows[0]]
        has_header = any(
            h.lower() in ("sampleid", "sample_id", "patientcluster", "cluster") for h in header
        )

        def norm(s: str) -> str:
            return (s or "").strip().lower().replace(" ", "").replace("_", "")

        if has_header and len(rows) >= 2:
            h = [norm(x) for x in header]
            try:
                idx_sid = (
                    h.index("sampleid") if "sampleid" in h else h.index("sampleid".replace("_", ""))
                )
            except Exception:
                idx_sid = 0
            idx_cluster = None
            for cand in ("patientcluster", "cluster"):
                if cand in h:
                    idx_cluster = h.index(cand)
                    break
            if idx_cluster is None:
                idx_cluster = max(0, len(h) - 1)
            for r in rows[1:]:
                sid = (r[idx_sid] if idx_sid < len(r) else "").strip()
                if sid == sample_id:
                    clv = (r[idx_cluster] if idx_cluster < len(r) else "").strip()
                    if clv == "":
                        return None
                    m = re.search(r"(\d+)", clv)
                    if not m:
                        return None
                    return f"cluster{m .group (1 )}"
            return None
        for r in rows:
            if len(r) >= 2 and r[0].strip() == sample_id:
                m = re.search(r"(\d+)", r[1].strip())
                if m:
                    return f"cluster{m .group (1 )}"
        return None
    except Exception:
        return None


def scrna_pipeline_run(
    input_data: str,
    outdir: str,
    scmulan_ckpt: str,
    classifier_model: str,
    classifier_info: str,
    sample_id: str = "MySample",
    conda_env: str = _DEFAULT_SCRNA_CONDA_ENV,
    tools_dir: str = _DEFAULT_SCRNA_TOOLS_DIR_WIN,
    pipeline_script: str = _DEFAULT_SCRNA_PIPELINE,
    timeout_min: int = 240,
) -> Dict[str, Any]:
    print(f"\n [Tool: scrna_pipeline_run] sample_id={sample_id }", flush=True)
    result = _tool_result_base(
        "scrna_pipeline_run",
        input_data=input_data,
        outdir=outdir,
        scmulan_ckpt=scmulan_ckpt,
        classifier_model=classifier_model,
        classifier_info=classifier_info,
        sample_id=sample_id,
        conda_env=conda_env,
        tools_dir=tools_dir,
        pipeline_script=pipeline_script,
    )
    try:
        conda_env = (conda_env or _DEFAULT_SCRNA_CONDA_ENV).strip()
        tools_dir_cfg = (tools_dir or "").strip() or _config_value(
            "BIOPLEX_SCRNA_TOOLS_DIR", default=_DEFAULT_SCRNA_TOOLS_DIR_WIN
        )
        tools_dir_abs = _safe_resolve_path(tools_dir_cfg)
        pipeline_cfg = _config_value("BIOPLEX_SCRNA_EXTERNAL_PIPELINE", default=pipeline_script)
        pipeline_fp = _resolve_script_path(pipeline_cfg, tools_dir_abs, _DEFAULT_SCRNA_PIPELINE)
        input_data_abs = _safe_resolve_path(input_data)
        outdir_abs = _safe_resolve_path(outdir)
        scmulan_ckpt_abs = _safe_resolve_path(scmulan_ckpt)
        classifier_model_abs = _safe_resolve_path(classifier_model)
        classifier_info_abs = _safe_resolve_path(classifier_info)
        Path(outdir_abs).mkdir(parents=True, exist_ok=True)
        if not Path(pipeline_fp).exists():
            raise FileNotFoundError(f"Pipeline script not found: {pipeline_fp }")
        conda_exe = _get_conda_exe()
        supports_no_capture = _conda_run_supports_no_capture_output(conda_exe)
        run_env = os.environ.copy()
        run_env["PYTHONUNBUFFERED"] = "1"
        cmd = [conda_exe, "run"]
        if supports_no_capture:
            cmd += ["--no-capture-output"]
        cmd += [
            "-n",
            conda_env,
            "python",
            "-u",
            pipeline_fp,
            "--input_data",
            input_data_abs,
            "--outdir",
            outdir_abs,
            "--scmulan_ckpt",
            scmulan_ckpt_abs,
            "--classifier_model",
            classifier_model_abs,
            "--classifier_info",
            classifier_info_abs,
            "--sample_id",
            sample_id,
        ]
        print(f" [scRNA] tools_dir={tools_dir_abs }", flush=True)
        print(f" [scRNA] pipeline={pipeline_fp }", flush=True)
        print(f" [scRNA] conda_env={conda_env } (no-capture={supports_no_capture })", flush=True)
        print(f" [scRNA] outdir={outdir_abs }", flush=True)
        print(" [scRNA] ---- begin streaming logs ----", flush=True)
        t0 = time.time()
        proc_s = _run_subprocess_stream(
            cmd,
            cwd=str(Path(tools_dir_abs)),
            timeout_min=timeout_min,
            env=run_env,
            heartbeat_sec=30,
            keep_last_lines=500,
        )
        elapsed = time.time() - t0
        print(" [scRNA] ---- end streaming logs ----", flush=True)
        step1_h5ad = str(Path(outdir_abs) / "01_PreAnnotation" / "final_preannotation.h5ad")
        step3_h5ad = str(Path(outdir_abs) / "03_FinalResult" / "annotated_complete.h5ad")
        ok_files = Path(step1_h5ad).exists() and Path(step3_h5ad).exists()
        ok = (proc_s.returncode == 0) and ok_files
        tail = proc_s.stdout_tail or ""
        pred_guess = None
        m = re.search(
            r"(cluster\d+|Cluster\s*\d+|pred(::icted):\s*[:=]\s*\S+)", tail, flags=re.IGNORECASE
        )
        if m:
            pred_guess = m.group(1).strip()
        cluster_csv = str(
            Path(outdir_abs) / "04_PatientClassification" / "patient_cluster_labels.csv"
        )
        predicted_cluster = _read_patient_cluster_csv(cluster_csv, sample_id=sample_id)
        rec = {
            "return_code": int(proc_s.returncode),
            "elapsed_sec": float(elapsed),
            "outdir": outdir_abs,
            "final_h5ad": step3_h5ad if Path(step3_h5ad).exists() else None,
            "preannotation_h5ad": step1_h5ad if Path(step1_h5ad).exists() else None,
            "prediction_guess": pred_guess,
            "patient_cluster_csv": cluster_csv if Path(cluster_csv).exists() else None,
            "predicted_cluster": predicted_cluster,
            "total_log_lines": int(proc_s.total_lines),
            "log_tail": tail,
        }
        result["ok"] = bool(ok)
        result["records"] = [rec]
        if not ok:
            result["error"] = f"scrna pipeline failed (rc={proc_s .returncode }) or missing outputs"
        result["formatted"] = "\n".join(
            [
                f"[scRNA] conda_env={conda_env }",
                f"[scRNA] tools_dir={tools_dir_abs }",
                f"[scRNA] pipeline_script={pipeline_fp }",
                f"[scRNA] input_data={input_data_abs }",
                f"[scRNA] outdir={outdir_abs }",
                f"[scRNA] elapsed_sec={elapsed :.2f}",
                f"[scRNA] preannotation_h5ad={step1_h5ad }",
                f"[scRNA] final_h5ad={step3_h5ad }",
                f"[scRNA] patient_cluster_csv={cluster_csv }",
                f"[scRNA] predicted_cluster={predicted_cluster }",
                f"[scRNA] prediction_guess={pred_guess }",
                f"[scRNA] total_log_lines={proc_s .total_lines }",
                "----- scRNA Log Tail -----",
                tail,
            ]
        )
        print(f" [Tool: scrna_pipeline_run] message ({'Success'if ok else 'Fail'})", flush=True)
        return result
    except Exception as e:
        print(f" [Tool: scrna_pipeline_run] Error: {e }", flush=True)
        result["error"] = str(e)
        result["formatted"] = str(e)
        return result


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "llm_mri",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Status update."},
                    "images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Status update.",
                    },
                    "question": {"type": "string", "description": "Status update."},
                    "max_images": {
                        "type": "integer",
                        "description": "Status update.",
                        "default": 12,
                    },
                    "model": {"type": "string", "description": "Status update."},
                    "reason": {"type": "string", "description": "Status update."},
                },
                "required": ["input_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_guidelines",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_facts",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gigatime_infer",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_tiff": {"type": "string", "description": "Status update."},
                    "output_dir": {"type": "string", "description": "Status update."},
                    "conda_env": {"type": "string", "description": "Status update."},
                    "timeout_min": {
                        "type": "integer",
                        "description": "Status update.",
                        "default": 60,
                    },
                    "run_llm_analysis": {
                        "type": "boolean",
                        "description": "Status update.",
                        "default": True,
                    },
                    "llm_question": {"type": "string", "description": "Status update."},
                    "llm_max_images": {
                        "type": "integer",
                        "description": "Status update.",
                        "default": 12,
                    },
                    "llm_model": {"type": "string", "description": "Status update."},
                    "reason": {"type": "string"},
                },
                "required": ["input_tiff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "llm_gigatime",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_dir": {"type": "string", "description": "Status update."},
                    "visualizations_dir": {"type": "string", "description": "Status update."},
                    "question": {"type": "string", "description": "Status update."},
                    "max_images": {
                        "type": "integer",
                        "description": "Status update.",
                        "default": 12,
                    },
                    "model": {"type": "string", "description": "Status update."},
                    "reason": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "raspr_run",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {"type": "string", "description": "Status update."},
                    "dicom_dir": {"type": "string", "description": "Status update."},
                    "output_dir": {"type": "string", "description": "Status update."},
                    "timeout_min": {
                        "type": "integer",
                        "description": "Status update.",
                        "default": 120,
                    },
                    "script_path": {"type": "string", "description": "Status update."},
                    "wsl_distro": {"type": "string", "description": "Status update."},
                    "convert_windows_path_to_wsl": {
                        "type": "boolean",
                        "description": "Status update.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["dicom_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "roam_infer",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_tiff": {"type": "string", "description": "Status update."},
                    "wsl_distro": {"type": "string", "description": "Status update."},
                    "timeout_min": {
                        "type": "integer",
                        "description": "Status update.",
                        "default": 60,
                    },
                    "convert_windows_path_to_wsl": {
                        "type": "boolean",
                        "description": "Status update.",
                    },
                    "roam_root": {
                        "type": "string",
                        "description": "ROAM root path. Relative paths are resolved from the project root.",
                    },
                    "assets_root": {
                        "type": "string",
                        "description": "ROAM assets path. Relative paths are resolved from the project root.",
                    },
                    "results_root": {
                        "type": "string",
                        "description": "ROAM results path. Relative paths are resolved from the project root.",
                    },
                    "conda_env": {
                        "type": "string",
                        "description": "Conda environment name for ROAM.",
                        "default": _DEFAULT_ROAM_CONDA_ENV,
                    },
                    "reason": {"type": "string"},
                },
                "required": ["input_tiff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrna_pipeline_run",
            "description": "Status update.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_data": {"type": "string", "description": "Status update."},
                    "outdir": {"type": "string", "description": "Status update."},
                    "scmulan_ckpt": {"type": "string", "description": "Status update."},
                    "classifier_model": {"type": "string", "description": "Status update."},
                    "classifier_info": {"type": "string", "description": "Status update."},
                    "sample_id": {
                        "type": "string",
                        "description": "Status update.",
                        "default": "MySample",
                    },
                    "conda_env": {
                        "type": "string",
                        "description": "Status update.",
                        "default": _DEFAULT_SCRNA_CONDA_ENV,
                    },
                    "tools_dir": {
                        "type": "string",
                        "description": "Status update.",
                        "default": _DEFAULT_SCRNA_TOOLS_DIR_WIN,
                    },
                    "pipeline_script": {
                        "type": "string",
                        "description": "Status update.",
                        "default": _DEFAULT_SCRNA_PIPELINE,
                    },
                    "timeout_min": {
                        "type": "integer",
                        "description": "Status update.",
                        "default": 240,
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "input_data",
                    "outdir",
                    "scmulan_ckpt",
                    "classifier_model",
                    "classifier_info",
                ],
            },
        },
    },
]
