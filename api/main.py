from __future__ import annotations
import json
import queue
import re
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from gbm_agent.prompts import build_input_context_prompt

app = FastAPI(title="BioPlexMDT API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DATA_DIR = SRC_DIR / "gbm_agent" / "data"
WEB_DIR = ROOT / "web"
DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


def get_agent() -> Any:
    from gbm_agent.ultra_agent import UltraGBMAgent

    return UltraGBMAgent()


def normalize_model(model: str) -> Optional[str]:
    value = (model or "").strip()
    if not value:
        return None
    aliases = {
        "default": None,
        "gemini": DEFAULT_MODEL,
        "local": "local",
    }
    return aliases.get(value.lower(), value)


def to_data_url(path_str: str) -> str:
    try:
        path = Path(path_str).resolve()
        rel = path.relative_to(DATA_DIR.resolve())
        return "/data/" + rel.as_posix()
    except Exception:
        return ""


def collect_image_urls(path_str: str) -> List[str]:
    if not path_str:
        return []
    root = Path(path_str)
    if not root.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    high_quality_dir = root / "visualizations" / "high_quality_png"
    search_root = high_quality_dir if high_quality_dir.exists() else root
    excluded = {"merge_tiles", "channel_tiles", "temp_tiles", "predictions"}
    files = [
        p
        for p in search_root.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes and not excluded.intersection(set(p.parts))
    ]
    files.sort(key=lambda item: (0 if item.name.startswith("Overlay") else 1, item.name.lower()))
    urls = [to_data_url(str(path)) for path in files]
    return [url for url in urls if url]


def strip_heading(line: str) -> str:
    return re.sub(r"^[\s#*\- \d.  () \[\]]+", "", line or "").strip()


def split_answer_sections(answer: str) -> Dict[str, str]:
    text = (answer or "").strip()
    result = {"mdt": "", "patient": "", "professional": ""}
    if not text:
        return result
    patterns = [
        ("mdt", re.compile(r"(mdt\s+recommendation|multidisciplinary\s+recommendation)", re.I)),
        (
            "patient",
            re.compile(r"(patient[-\s]*friendly|patient\s+explanation|for\s+the\s+patient)", re.I),
        ),
        (
            "professional",
            re.compile(r"(professional\s+answer|clinical\s+answer|specialist\s+answer)", re.I),
        ),
    ]
    lines = text.splitlines()
    hits: List[Tuple[str, int]] = []
    for index, line in enumerate(lines):
        normalized = strip_heading(line)
        for key, pattern in patterns:
            if pattern.search(normalized):
                hits.append((key, index))
                break
    if hits:
        deduped: List[Tuple[str, int]] = []
        seen = set()
        for key, index in hits:
            marker = (key, index)
            if marker not in seen:
                deduped.append(marker)
                seen.add(marker)
        for pos, (key, start) in enumerate(deduped):
            end = deduped[pos + 1][1] if pos + 1 < len(deduped) else len(lines)
            chunk = "\n".join(lines[start:end]).strip()
            if chunk and not result[key]:
                result[key] = chunk
        used = set()
        for pos, (key, start) in enumerate(deduped):
            if key == "professional":
                continue
            end = deduped[pos + 1][1] if pos + 1 < len(deduped) else len(lines)
            used.update(range(start, end))
        if not result["professional"]:
            result["professional"] = "\n".join(
                line for i, line in enumerate(lines) if i not in used
            ).strip()
    if not result["professional"]:
        result["professional"] = text
    if not result["mdt"]:
        result["mdt"] = result["professional"]
    return result


def write_result_files(answer: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    sections = split_answer_sections(answer)
    output_dir = Path(
        meta.get("result_output_dir")
        or (DATA_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "mdt_file": output_dir / "mdt_recommendation.txt",
        "patient_file": output_dir / "patient_friendly.txt",
        "professional_file": output_dir / "professional_answers.txt",
    }
    files["mdt_file"].write_text(sections["mdt"], encoding="utf-8")
    files["patient_file"].write_text(sections["patient"], encoding="utf-8")
    files["professional_file"].write_text(sections["professional"], encoding="utf-8")
    return {
        "mdt_output": sections["mdt"],
        "patient_friendly_output": sections["patient"],
        "professional_answer_output": sections["professional"],
        "mdt_file_url": to_data_url(str(files["mdt_file"])),
        "patient_friendly_file_url": to_data_url(str(files["patient_file"])),
        "professional_answer_file_url": to_data_url(str(files["professional_file"])),
    }


def save_uploads(files: List[Any], out_dir: Optional[Path] = None) -> List[str]:
    if not files:
        return []
    upload_dir = DATA_DIR / "uploads"
    if out_dir is None:
        out_dir = upload_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for item in files:
        try:
            filename = Path(getattr(item, "filename", "file.bin")).name
            dest = out_dir / filename
            with dest.open("wb") as handle:
                shutil.copyfileobj(item.file, handle)
            saved.append(str(dest.resolve()))
        except Exception:
            continue
    return saved


def make_mosaic(image_paths: List[str], out_path: str, max_side: int = 900) -> str:
    paths = [p for p in image_paths if p][:4]
    if not paths:
        return ""
    try:
        from PIL import Image
    except Exception:
        return str(Path(paths[0]).resolve())
    try:
        cell = max_side // 2
        canvas = Image.new("RGB", (cell * 2, cell * 2), (255, 255, 255))
        for index, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            width, height = image.size
            scale = min(1.0, float(cell) / float(max(width, height)))
            resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
            row = index // 2
            col = index % 2
            x = col * cell + (cell - resized.size[0]) // 2
            y = row * cell + (cell - resized.size[1]) // 2
            canvas.paste(resized, (x, y))
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out, format="PNG")
        return str(out.resolve())
    except Exception:
        return str(Path(paths[0]).resolve())


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/parse-question")
async def api_parse_question(req: Request) -> Dict[str, Any]:
    from gbm_agent.question_parser import QuestionParser

    data = await req.json()
    question = str(data.get("question", "")).strip()
    model_name = normalize_model(str(data.get("model", "")).strip())
    struct = QuestionParser(model=model_name).parse(question) if question else {}
    return {"status": "ok", "model": model_name or DEFAULT_MODEL, "struct": struct}


def prepare_inputs(
    question: str,
    ts_dir: Path,
    *,
    mri_files: List[Any],
    tiff_files: List[Any],
    tiff_path_text: str,
    dicom_files: List[Any],
    dicom_dir_text: str,
    scrna_h5ad_file: Optional[Any],
    scrna_h5ad_path_text: str,
    tenx_files: List[Any],
    tenx_dir_text: str,
) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    saved_mri_paths = save_uploads(mri_files, out_dir=ts_dir) if mri_files else []
    mri_input_path = ""
    if saved_mri_paths:
        mri_input_path = make_mosaic(saved_mri_paths, str(ts_dir / "mri_mosaic.png"))
    meta.update(
        mri_input_path=mri_input_path,
        mri_saved_paths=saved_mri_paths,
        mri_mosaic_url=to_data_url(mri_input_path),
    )
    tiff_saved_paths = save_uploads(tiff_files, out_dir=ts_dir) if tiff_files else []
    tiff_input_path = str(Path(tiff_saved_paths[0]).resolve()) if tiff_saved_paths else ""
    if not tiff_input_path and tiff_path_text:
        tiff_input_path = tiff_path_text.strip()
    gigatime_output_dir = ""
    if tiff_input_path:
        gigatime_output_dir = str((DATA_DIR / "gigatime_outputs" / ts_dir.name).resolve())
        Path(gigatime_output_dir).mkdir(parents=True, exist_ok=True)
    meta.update(
        tiff_input_path=tiff_input_path,
        tiff_saved_paths=tiff_saved_paths,
        gigatime_output_dir=gigatime_output_dir,
    )
    dicom_saved_paths: List[str] = []
    dicom_dir = ""
    if dicom_files:
        dicom_out_dir = ts_dir / "dicom"
        dicom_out_dir.mkdir(parents=True, exist_ok=True)
        dicom_saved_paths = save_uploads(dicom_files, out_dir=dicom_out_dir)
        if dicom_saved_paths:
            dicom_dir = str(dicom_out_dir.resolve())
    if not dicom_dir and dicom_dir_text:
        dicom_dir = dicom_dir_text.strip()
    meta.update(dicom_dir=dicom_dir, dicom_saved_paths=dicom_saved_paths)
    scrna_h5ad_saved = ""
    scrna_h5ad_path = ""
    if scrna_h5ad_file is not None:
        scrna_dir = ts_dir / "scrna"
        scrna_dir.mkdir(parents=True, exist_ok=True)
        saved = save_uploads([scrna_h5ad_file], out_dir=scrna_dir)
        if saved:
            scrna_h5ad_saved = saved[0]
            scrna_h5ad_path = scrna_h5ad_saved
    if not scrna_h5ad_path and scrna_h5ad_path_text:
        scrna_h5ad_path = scrna_h5ad_path_text.strip()
    tenx_saved_paths: List[str] = []
    tenx_dir = ""
    if tenx_files:
        tenx_out_dir = ts_dir / "tenx"
        tenx_out_dir.mkdir(parents=True, exist_ok=True)
        tenx_saved_paths = save_uploads(tenx_files, out_dir=tenx_out_dir)
        if tenx_saved_paths:
            tenx_dir = str(tenx_out_dir.resolve())
    if not tenx_dir and tenx_dir_text:
        tenx_dir = tenx_dir_text.strip()
    scrna_output_dir = str((DATA_DIR / "scrna_outputs" / ts_dir.name).resolve())
    Path(scrna_output_dir).mkdir(parents=True, exist_ok=True)
    result_output_dir = str((DATA_DIR / "results" / ts_dir.name).resolve())
    Path(result_output_dir).mkdir(parents=True, exist_ok=True)
    meta.update(
        scrna_h5ad_path=scrna_h5ad_path,
        scrna_h5ad_saved=scrna_h5ad_saved,
        tenx_dir=tenx_dir,
        tenx_saved_paths=tenx_saved_paths,
        scrna_output_dir=scrna_output_dir,
        result_output_dir=result_output_dir,
    )
    context_prompt = build_input_context_prompt(meta)
    question_aug = question if not context_prompt else f"{question }\n\n{context_prompt }\n"
    return question_aug, meta


async def parse_request(req: Request) -> Tuple[str, str, Dict[str, Any]]:
    content_type = (req.headers.get("content-type") or "").lower()
    values: Dict[str, Any] = {
        "mri_files": [],
        "tiff_files": [],
        "tiff_path_text": "",
        "dicom_files": [],
        "dicom_dir_text": "",
        "scrna_h5ad_file": None,
        "scrna_h5ad_path_text": "",
        "tenx_files": [],
        "tenx_dir_text": "",
    }
    if "multipart/form-data" in content_type:
        form = await req.form()
        question = str(form.get("question", "")).strip()
        model = str(form.get("model", "")).strip()
        values["mri_files"] = list(form.getlist("mri_files")) if hasattr(form, "getlist") else []
        values["tiff_files"] = list(form.getlist("tiff_files")) if hasattr(form, "getlist") else []
        values["tiff_path_text"] = str(
            form.get("tiff_path", "") or form.get("tiff_input_path", "")
        ).strip()
        values["dicom_files"] = (
            list(form.getlist("dicom_files")) if hasattr(form, "getlist") else []
        )
        values["dicom_dir_text"] = str(form.get("dicom_dir", "")).strip()
        values["scrna_h5ad_file"] = form.get("scrna_h5ad_file", None)
        values["scrna_h5ad_path_text"] = str(form.get("scrna_h5ad_path", "")).strip()
        values["tenx_files"] = list(form.getlist("tenx_files")) if hasattr(form, "getlist") else []
        values["tenx_dir_text"] = str(form.get("tenx_dir", "")).strip()
        return question, model, values
    data = await req.json()
    question = str(data.get("question", "")).strip()
    model = str(data.get("model", "")).strip()
    values["dicom_dir_text"] = str(data.get("dicom_dir", "")).strip()
    values["tiff_path_text"] = str(
        data.get("tiff_path", "") or data.get("tiff_input_path", "")
    ).strip()
    values["scrna_h5ad_path_text"] = str(data.get("scrna_h5ad_path", "")).strip()
    values["tenx_dir_text"] = str(data.get("tenx_dir", "")).strip()
    return question, model, values


def final_payload(answer: str, trace: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    gigatime_output_urls = collect_image_urls(meta.get("gigatime_output_dir", ""))
    result_files = write_result_files(answer, meta)
    return {
        "status": "ok",
        "answer": answer,
        "trace": trace,
        **result_files,
        "mri_input_path": meta.get("mri_input_path", ""),
        "mri_saved_paths": meta.get("mri_saved_paths", []),
        "mri_mosaic_url": meta.get("mri_mosaic_url", ""),
        "mri_structured": None,
        "tiff_input_path": meta.get("tiff_input_path", ""),
        "tiff_saved_paths": meta.get("tiff_saved_paths", []),
        "gigatime_output_dir": meta.get("gigatime_output_dir", ""),
        "gigatime_structured": None,
        "gigatime_outputs": gigatime_output_urls,
        "gigatime_output_urls": gigatime_output_urls,
        "dicom_dir": meta.get("dicom_dir", ""),
        "dicom_saved_paths": meta.get("dicom_saved_paths", []),
        "scrna_h5ad_path": meta.get("scrna_h5ad_path", ""),
        "tenx_dir": meta.get("tenx_dir", ""),
        "scrna_output_dir": meta.get("scrna_output_dir", ""),
    }


@app.post("/api/ask")
async def api_ask(req: Request) -> Dict[str, Any]:
    try:
        question, model, values = await parse_request(req)
        if not question:
            return {"status": "error", "detail": "empty question"}
        ts_dir = DATA_DIR / "uploads" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ts_dir.mkdir(parents=True, exist_ok=True)
        question_aug, meta = prepare_inputs(question, ts_dir, **values)
        agent = get_agent()
        model_name = normalize_model(model)
        answer = agent.run(question_aug, model_name=model_name)
        return final_payload(answer, agent.get_trace(), meta)
    except Exception as exc:
        return {
            "status": "error",
            "detail": f"Backend Error: {exc }",
            "answer": f"Backend Error: {exc }",
            "trace": traceback.format_exc(),
        }


@app.post("/api/ask-stream")
async def api_ask_stream(req: Request) -> StreamingResponse:
    try:
        question, model, values = await parse_request(req)
        if not question:
            return StreamingResponse(
                iter([sse("error", {"detail": "empty question"})]),
                media_type="text/event-stream",
            )
        ts_dir = DATA_DIR / "uploads" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ts_dir.mkdir(parents=True, exist_ok=True)
        question_aug, meta = prepare_inputs(question, ts_dir, **values)
    except Exception as exc:
        return StreamingResponse(
            iter([sse("error", {"detail": f"request parse failed: {exc }"})]),
            media_type="text/event-stream",
        )
    events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
    done = threading.Event()
    events.put(("meta", meta))

    def push(event: str, payload: Any) -> None:
        events.put((event, payload))

    def worker() -> None:
        try:
            agent = get_agent()
            model_name = normalize_model(model)
            push("log", f"model={model_name or DEFAULT_MODEL }")
            answer = agent.run(
                question_aug,
                model_name=model_name,
                stream_callback=lambda line: push("log", line),
            )
            push("final", final_payload(answer, agent.get_trace(), meta))
        except Exception as exc:
            push("error", {"detail": f"Backend Error: {exc }", "trace": traceback.format_exc()})
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        last_ping = time.time()
        while not done.is_set() or not events.empty():
            try:
                event, payload = events.get(timeout=0.2)
                yield sse(event, payload)
            except queue.Empty:
                if time.time() - last_ping > 15:
                    last_ping = time.time()
                    yield b": ping\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def sse(event: str, payload: Any) -> bytes:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event }\ndata: {data }\n\n".encode("utf-8")


@app.post("/api/build-index")
def api_build_index() -> Dict[str, Any]:
    try:
        from gbm_agent.build_index import build_index

        build_index()
        return {"status": "completed", "detail": "text index rebuilt"}
    except Exception as exc:
        return {"status": "failed", "detail": str(exc)}


@app.post("/api/build-facts")
def api_build_facts() -> Dict[str, Any]:
    try:
        from gbm_agent.build_facts_index import main as build_facts_main

        build_facts_main()
        return {"status": "completed", "detail": "facts index rebuilt"}
    except Exception as exc:
        return {"status": "failed", "detail": str(exc)}


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
