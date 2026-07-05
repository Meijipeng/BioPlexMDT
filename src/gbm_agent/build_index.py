from __future__ import annotations
import json
import os
import re
import time
import random
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb
from tqdm import tqdm
from .config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    DATA_DIR,
    embed_texts as project_embed_texts,
)

TEXT_FIELD_CANDIDATES = ("content", "clean_text", "raw_text", "text", "overview", "title")
BAD_TEXT_VALUES = {"none", "null", "nan", ""}
TPM_LIMIT = 1_000_000
TPM_TARGET_RATIO = 0.92
TPM_TARGET = int(TPM_LIMIT * TPM_TARGET_RATIO)
CHARS_PER_TOKEN = 3.0
MAX_WORKERS = 3
MAX_BATCH_TOKENS_EST = 85_000
MAX_BATCH_CHARS = int(MAX_BATCH_TOKENS_EST * CHARS_PER_TOKEN)
CHUNK_MAX_CHARS = 22_000
CHUNK_OVERLAP = 200


def _pick_text(item: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    for k in TEXT_FIELD_CANDIDATES:
        v = item.get(k, None)
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        if s.lower() in BAD_TEXT_VALUES:
            continue
        return s, k
    return "", None


def load_raw_data(file_path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not file_path.exists():
        print(f" message: {file_path }")
        return []
    print(f" message: {file_path }")
    stats = {
        "num_lines": 0,
        "num_json_ok": 0,
        "num_json_error": 0,
        "num_kept": 0,
        "num_dropped_no_text": 0,
        "field_usage": {k: 0 for k in TEXT_FIELD_CANDIDATES},
    }
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stats["num_lines"] += 1
            s = line.strip()
            if not s:
                continue
            try:
                item = json.loads(s)
                stats["num_json_ok"] += 1
            except json.JSONDecodeError:
                stats["num_json_error"] += 1
                continue
            if not isinstance(item, dict):
                stats["num_dropped_no_text"] += 1
                continue
            text, used_field = _pick_text(item)
            if not text:
                stats["num_dropped_no_text"] += 1
                continue
            item["content"] = text
            if used_field:
                stats["field_usage"][used_field] += 1
            data.append(item)
            stats["num_kept"] += 1
    print(f" message {len (data )} message ")
    print("Status update.")
    print(f" - lines: {stats ['num_lines']}")
    print(f" - json_ok: {stats ['num_json_ok']}, json_error: {stats ['num_json_error']}")
    print(f" - kept: {stats ['num_kept']}, dropped_no_text: {stats ['num_dropped_no_text']}")
    print(" - field usage:")
    for k, v in stats["field_usage"].items():
        if v > 0:
            print(f" * {k }: {v }")
    return data


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def estimate_tokens_from_chars(char_count: int) -> int:
    return int(char_count / CHARS_PER_TOKEN)


def parse_retry_after_seconds(err_msg: str) -> float:
    m = re.search(r"try again in ([0-9]*\.:[0-9]+)s", err_msg)
    if m:
        return float(m.group(1))
    return 2.0


class TokenBucketTPM:
    def __init__(self, tpm_target: int):
        self.tpm_target = tpm_target
        self.lock = threading.Lock()
        self.events: List[Tuple[float, int]] = []

    def _prune(self, now: float):
        cutoff = now - 60.0
        while self.events and self.events[0][0] < cutoff:
            self.events.pop(0)

    def tokens_used_last_minute(self, now: float) -> int:
        self._prune(now)
        return sum(t for _, t in self.events)

    def acquire(self, tokens_needed: int):
        while True:
            now = time.time()
            with self.lock:
                used = self.tokens_used_last_minute(now)
                if used + tokens_needed <= self.tpm_target:
                    self.events.append((now, tokens_needed))
                    return
                if self.events:
                    oldest_t = self.events[0][0]
                    wait = max(0.05, (oldest_t + 60.0) - now)
                else:
                    wait = 0.2
            time.sleep(min(wait, 2.0))


tpm_bucket = TokenBucketTPM(TPM_TARGET)


def embed_texts_with_retry(texts: List[str], max_retries: int = 20) -> List[List[float]]:
    chars = sum(len(t) for t in texts)
    tokens_est = estimate_tokens_from_chars(chars)
    tpm_bucket.acquire(tokens_est)
    last_err = None
    for attempt in range(max_retries):
        try:
            return project_embed_texts(texts)
        except Exception as e:
            msg = str(e)
            last_err = e
            if (
                "Error code: 429" in msg
                or "rate_limit_exceeded" in msg
                or "Rate limit reached" in msg
            ):
                wait = parse_retry_after_seconds(msg)
                wait = wait + random.uniform(0.05, 0.35)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"embed_texts retry exceeded: {last_err }")


def build_index():
    raw_file = Path(DATA_DIR) / "raw" / "open_guidelines.jsonl"
    if not raw_file.exists():
        raw_file_alt = Path(DATA_DIR) / "open_guidelines.jsonl"
        if raw_file_alt.exists():
            raw_file = raw_file_alt
        else:
            print(f" message: {raw_file }")
            return
    docs = load_raw_data(raw_file)
    if not docs:
        print("Status update.")
        return
    print(f" message ChromaDB: {CHROMA_DB_DIR }")
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        print(f" message: {CHROMA_COLLECTION_NAME }")
    except ValueError:
        pass
    collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
    print(f" message: {CHROMA_COLLECTION_NAME }")
    added_chunks = 0
    failed_chunks = 0
    chroma_lock = threading.Lock()
    count_lock = threading.Lock()

    def add_to_chroma(docs_batch, metas_batch, ids_batch, embs):
        nonlocal added_chunks
        with chroma_lock:
            collection.add(
                documents=docs_batch,
                embeddings=embs,
                metadatas=metas_batch,
                ids=ids_batch,
            )
        with count_lock:
            added_chunks += len(docs_batch)

    def embed_with_split(docs_batch, metas_batch, ids_batch):
        nonlocal failed_chunks
        if not docs_batch:
            return
        try:
            embs = embed_texts_with_retry(docs_batch)
            if not embs:
                with count_lock:
                    failed_chunks += len(docs_batch)
                return
            add_to_chroma(docs_batch, metas_batch, ids_batch, embs)
            return
        except Exception as e:
            msg = str(e)
            if ("max_tokens_per_request" in msg) or ("max 300000 tokens per request" in msg):
                if len(docs_batch) == 1:
                    with count_lock:
                        failed_chunks += 1
                    print(f"\n Single batch too large even after split, skipped: {ids_batch [0 ]}")
                    return
                mid = len(docs_batch) // 2
                embed_with_split(docs_batch[:mid], metas_batch[:mid], ids_batch[:mid])
                embed_with_split(docs_batch[mid:], metas_batch[mid:], ids_batch[mid:])
                return
            if "maximum context length is 8192" in msg:
                if len(docs_batch) == 1:
                    t = docs_batch[0]
                    if len(t) <= 4000:
                        with count_lock:
                            failed_chunks += 1
                        print(f"\n Single chunk still too large, skipped: {ids_batch [0 ]}")
                        return
                    half = len(t) // 2
                    left = t[:half].strip()
                    right = t[half:].strip()
                    new_docs = [x for x in (left, right) if x]
                    new_metas = [metas_batch[0]] * len(new_docs)
                    new_ids = [f"{ids_batch [0 ]}_s{ix }" for ix in range(len(new_docs))]
                    embed_with_split(new_docs, new_metas, new_ids)
                    return
                mid = len(docs_batch) // 2
                embed_with_split(docs_batch[:mid], metas_batch[:mid], ids_batch[:mid])
                embed_with_split(docs_batch[mid:], metas_batch[mid:], ids_batch[mid:])
                return
            with count_lock:
                failed_chunks += len(docs_batch)
            print(f"\n Batch Error (async): {e }")

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pending = []
    batch_docs: List[str] = []
    batch_metas: List[Dict[str, Any]] = []
    batch_ids: List[str] = []
    batch_chars = 0

    def submit_batch():
        nonlocal batch_docs, batch_metas, batch_ids, batch_chars, pending
        if not batch_docs:
            return
        fut = executor.submit(embed_with_split, batch_docs, batch_metas, batch_ids)
        pending.append(fut)
        batch_docs, batch_metas, batch_ids, batch_chars = [], [], [], 0
        while len(pending) > MAX_WORKERS * 2:
            done = []
            for f in pending:
                if f.done():
                    done.append(f)
            for f in done:
                pending.remove(f)
            if not done:
                time.sleep(0.02)

    total = len(docs)
    pbar = tqdm(range(total), total=total, desc="Indexing")
    for i in pbar:
        item = docs[i]
        content = item.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        meta_base = {k: v for k, v in item.items() if k != "content"}
        for k, v in list(meta_base.items()):
            if isinstance(v, (list, dict)):
                meta_base[k] = json.dumps(v, ensure_ascii=False)
            if v is None:
                meta_base[k] = ""
        if len(content) <= CHUNK_MAX_CHARS:
            chunks = [content.strip()]
        else:
            chunks = chunk_text(content, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP)
        for j, chunk in enumerate(chunks):
            if not chunk:
                continue
            if batch_docs and (batch_chars + len(chunk) > MAX_BATCH_CHARS):
                submit_batch()
            doc_id = f"doc_{i }_chunk_{j }"
            meta = dict(meta_base)
            meta["chunk_id"] = j
            meta["chunk_count"] = len(chunks)
            batch_docs.append(chunk)
            batch_metas.append(meta)
            batch_ids.append(doc_id)
            batch_chars += len(chunk)
        if (i + 1) % 200 == 0:
            with count_lock:
                pbar.set_postfix(
                    {"added": added_chunks, "failed": failed_chunks, "pending": len(pending)}
                )
    submit_batch()
    for _ in as_completed(pending):
        pass
    executor.shutdown(wait=True)
    print("Status update.")
    print(f" - added_chunks: {added_chunks }")
    print(f" - failed_chunks: {failed_chunks }")
    print(f" - collection.count(): {collection .count ()}")
    print(f" message: {CHROMA_DB_DIR }")


def main():
    t0 = time.time()
    try:
        build_index()
    finally:
        print(f" message: {time .time ()-t0 :.2f} message")


if __name__ == "__main__":
    main()
