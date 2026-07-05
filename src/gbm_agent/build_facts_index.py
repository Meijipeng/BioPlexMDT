from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
import chromadb
from tqdm import tqdm
from .config import (
    client,
    GPT_MODEL,
    EMBED_MODEL,
    CHROMA_DB_DIR,
    CHROMA_FACTS_COLLECTION_NAME,
    PUBMED_JSONL,
    EMBED_BATCH_SIZE,
)
from .prompts import TRIAL_FACT_EXTRACTION_SYSTEM_PROMPT, trial_fact_user_prompt

logger = logging.getLogger(__name__)


@dataclass
class FactRecord:
    fact_id: str
    text: str
    metadata: Dict[str, Any]


def _llm_extract_trial_fact(pmid: str, title: str, abstract: str) -> Optional[Dict[str, Any]]:
    system_prompt = TRIAL_FACT_EXTRACTION_SYSTEM_PROMPT
    user_prompt = trial_fact_user_prompt(pmid, title, abstract)
    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content or "{}"
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("trial fact parser output is not dict")
        return data
    except Exception as e:
        logger.warning(" Trial fact extraction failed for PMID %s: %s", pmid, e)
        return None


def iter_fact_records(pubmed_jsonl: Path, stats: Dict[str, Any]) -> Iterator[FactRecord]:
    if not pubmed_jsonl.exists():
        logger.warning("Status update.", pubmed_jsonl)
        return
    with pubmed_jsonl.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Extracting trial facts from PubMed"):
            line = line.strip()
            if not line:
                continue
            stats["num_raw_records"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["num_parse_errors"] += 1
                continue
            pmid = str(rec.get("pmid") or rec.get("PMID") or "")
            if not pmid:
                stats["num_missing_pmid"] += 1
                continue
            title = rec.get("title") or rec.get("ArticleTitle") or ""
            abstract = rec.get("clean_text") or rec.get("abstract") or rec.get("Abstract") or ""
            if not title and not abstract:
                stats["num_no_title_abstract"] += 1
                continue
            data = _llm_extract_trial_fact(pmid, title, abstract)
            if not data:
                continue
            if not data.get("is_gbm_trial", False):
                stats["num_not_gbm"] += 1
                continue
            stats["num_gbm_trials"] += 1
            population = data.get("population")
            intervention = data.get("intervention")
            comparator = data.get("comparator")
            outcome = data.get("outcome")
            phase = data.get("phase")
            trial_type = data.get("trial_type")
            n_total = data.get("n_total")
            hr = data.get("hazard_ratio")
            p_value = data.get("p_value")
            year = data.get("year")
            summary_parts: List[str] = []
            summary_parts.append(f"PMID {pmid }: {title }")
            if population:
                summary_parts.append(f"Population: {population }")
            if intervention:
                if comparator:
                    summary_parts.append(
                        f"Intervention vs Comparator: {intervention } vs {comparator }"
                    )
                else:
                    summary_parts.append(f"Intervention: {intervention }")
            if outcome:
                summary_parts.append(f"Outcome: {outcome }")
            effect_summary = data.get("effect_summary")
            if effect_summary:
                summary_parts.append(f"Effect: {effect_summary }")
            elif hr is not None or p_value is not None:
                effect_text = f"HR={hr }" if hr is not None else ""
                if p_value is not None:
                    effect_text += f", p={p_value }"
                summary_parts.append(f"Effect: {effect_text }")
            text = "\n".join(summary_parts)
            meta: Dict[str, Any] = {
                "pmid": pmid,
                "title": title,
                "population": population,
                "intervention": intervention,
                "comparator": comparator,
                "outcome": outcome,
                "effect_summary": effect_summary,
                "phase": phase,
                "trial_type": trial_type,
                "n_total": n_total,
                "hazard_ratio": hr,
                "p_value": p_value,
                "year": year,
                "source_type": "TrialFact",
            }
            fact_id = f"trial_{pmid }"
            yield FactRecord(fact_id=fact_id, text=text, metadata=meta)


def _batch_embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        resp = client.embeddings.create(input=texts, model=EMBED_MODEL)
        return [d.embedding for d in resp.data]
    except Exception as e:
        logger.error(" Embedding Error in facts index: %s", e)
        return []


def main() -> None:
    stats: Dict[str, Any] = {
        "num_raw_records": 0,
        "num_parse_errors": 0,
        "num_missing_pmid": 0,
        "num_no_title_abstract": 0,
        "num_not_gbm": 0,
        "num_gbm_trials": 0,
        "num_facts_indexed": 0,
    }
    logger.info("==================================================")
    logger.info("Status update.")
    logger.info(" Collection: %s", CHROMA_FACTS_COLLECTION_NAME)
    logger.info(" Source PubMed JSONL: %s", PUBMED_JSONL)
    logger.info("==================================================")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        chroma_client.delete_collection(name=CHROMA_FACTS_COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=CHROMA_FACTS_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    batch_ids: List[str] = []
    batch_texts: List[str] = []
    batch_metas: List[Dict[str, Any]] = []
    for fact in iter_fact_records(PUBMED_JSONL, stats):
        batch_ids.append(fact.fact_id)
        batch_texts.append(fact.text)
        batch_metas.append(fact.metadata)
        if len(batch_ids) >= EMBED_BATCH_SIZE:
            embeddings = _batch_embed(batch_texts)
            if embeddings and len(embeddings) == len(batch_texts):
                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metas,
                    embeddings=embeddings,
                )
                stats["num_facts_indexed"] += len(batch_ids)
            batch_ids, batch_texts, batch_metas = [], [], []
    if batch_ids:
        embeddings = _batch_embed(batch_texts)
        if embeddings and len(embeddings) == len(batch_texts):
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metas,
                embeddings=embeddings,
            )
            stats["num_facts_indexed"] += len(batch_ids)
    stats["build_timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
    stats["embedding_model"] = EMBED_MODEL
    stats["collection_name"] = CHROMA_FACTS_COLLECTION_NAME
    logger.info("Status update.")
    for k, v in stats.items():
        logger.info(" %s: %s", k, v)
    manifest_path = Path(CHROMA_DB_DIR) / "facts_manifest.json"
    try:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info("Status update.", manifest_path)
    except Exception as e:
        logger.warning("Status update.", e)
    logger.info("Status update.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
