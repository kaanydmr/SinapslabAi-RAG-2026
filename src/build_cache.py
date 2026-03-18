"""
build_cache.py — Build two indexes from datasets:
  1. CAG Cache: question embeddings → instant answer lookup
  2. RAG Store: document chunks (answers/context) → retrieval context for LLM
"""

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

csv.field_size_limit(10 * 1024 * 1024)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def normalize_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ── Dataset Loaders ──────────────────────────────────────────────


def load_json_dataset(filepath: str) -> List[Dict[str, str]]:
    """Load main set.json → list of {query, answer}."""
    path = Path(filepath)
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for entry in data:
        q = normalize_text(entry.get("Soru", ""))
        a = normalize_text(entry.get("Cevap", ""))
        if q and a:
            records.append({"query": q, "answer": a, "source": "main_set.json"})
    logger.info(f"main set.json: {len(records)} QA çifti")
    return records


def load_csv_dataset(filepath: str) -> List[Dict[str, str]]:
    """Load turkish_law_dataset.csv → list of {query, answer, context}."""
    path = Path(filepath)
    if not path.exists():
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = normalize_text(row.get("soru", ""))
            a = normalize_text(row.get("cevap", ""))
            ctx = normalize_text(row.get("context", ""))
            src = row.get("kaynak", "")
            if q and a:
                records.append({
                    "query": q,
                    "answer": a,
                    "context": ctx,
                    "source": "turkish_law_dataset.csv",
                    "kaynak": src,
                })
    logger.info(f"turkish_law_dataset.csv: {len(records)} QA çifti")
    return records


# ── Build Functions ──────────────────────────────────────────────


def build_cag_index(records: List[Dict], model: SentenceTransformer, cache_dir: Path):
    """
    Build CAG cache: embed all QUESTIONS for fast semantic matching.
    Saves: cache_index.faiss + cache_metadata.json
    """
    queries = [r["query"] for r in records]
    logger.info(f"CAG: {len(queries)} soru embed ediliyor...")

    start = time.time()
    embeddings = model.encode(
        queries, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    logger.info(f"CAG embedding: {time.time() - start:.1f}s")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    metadata = []
    for r in records:
        metadata.append({
            "query": r["query"],
            "answer": r["answer"],
            "source": r["source"],
            "usage_count": 0,
        })

    faiss.write_index(index, str(cache_dir / "cache_index.faiss"))
    with open(cache_dir / "cache_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)

    logger.info(f"CAG indeksi kaydedildi: {index.ntotal} kayıt")


def build_rag_index(records: List[Dict], model: SentenceTransformer, rag_dir: Path):
    """
    Build RAG document store: chunk and embed ANSWERS + CONTEXT for retrieval.
    Saves: rag_index.faiss + rag_metadata.json
    """
    # Build document chunks from answers and context
    chunks = []
    for r in records:
        # Use context if available (richer legal text), otherwise use the answer
        content = r.get("context", "") or r["answer"]
        source_label = r.get("kaynak", r["source"])

        text_chunks = chunk_text(content)
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "text": chunk,
                "source": source_label,
                "original_query": r["query"],
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            })

    logger.info(f"RAG: {len(chunks)} belge parçası oluşturuldu")

    texts = [c["text"] for c in chunks]
    logger.info(f"RAG: {len(texts)} parça embed ediliyor...")

    start = time.time()
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    logger.info(f"RAG embedding: {time.time() - start:.1f}s")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    metadata = []
    for c in chunks:
        metadata.append({
            "text": c["text"],
            "source": c["source"],
            "original_query": c["original_query"],
        })

    faiss.write_index(index, str(rag_dir / "rag_index.faiss"))
    with open(rag_dir / "rag_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)

    logger.info(f"RAG indeksi kaydedildi: {index.ntotal} belge parçası")


# ── Main Build ───────────────────────────────────────────────────


def build_all(project_root: str, cache_dir: str, rag_dir: str):
    """Build both CAG cache and RAG document store."""
    root = Path(project_root)
    cache_path = Path(cache_dir)
    rag_path = Path(rag_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    rag_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RAG + CAG HİBRİT SİSTEM — İNDEKS OLUŞTURMA")
    logger.info("=" * 60)

    # Load datasets
    all_records = []
    json_path = root / "main set.json"
    if json_path.exists():
        all_records.extend(load_json_dataset(str(json_path)))
    csv_path = root / "turkish_law_dataset.csv"
    if csv_path.exists():
        all_records.extend(load_csv_dataset(str(csv_path)))

    if not all_records:
        logger.error("Hiç QA çifti bulunamadı!")
        return

    logger.info(f"Toplam {len(all_records)} QA çifti yüklendi")

    # Load model once, share between both builds
    logger.info(f"Model yükleniyor: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Build CAG (question embeddings)
    logger.info("-" * 40)
    logger.info("1/2 CAG Önbellek İndeksi")
    logger.info("-" * 40)
    build_cag_index(all_records, model, cache_path)

    # Build RAG (document chunks)
    logger.info("-" * 40)
    logger.info("2/2 RAG Belge İndeksi")
    logger.info("-" * 40)
    build_rag_index(all_records, model, rag_path)

    logger.info("=" * 60)
    logger.info("TAMAMLANDI — Her iki indeks de hazır")
    logger.info("=" * 60)
