"""
cache_search.py — Semantic cache with:
  - Turkish query normalization
  - LLM paraphrase expansion (multi-vector search)
  - Enriched metadata (confidence, timestamps, law_refs, TTL)
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.query_processor import normalize_turkish, expand_paraphrases

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_THRESHOLD = 0.85
DEFAULT_TTL_DAYS = 90


class SemanticCache:
    """FAISS-backed semantic cache with paraphrase expansion and TTL support."""

    def __init__(self, cache_dir: str, threshold: float = DEFAULT_THRESHOLD,
                 use_paraphrases: bool = False):
        self.cache_dir = Path(cache_dir)
        self.threshold = threshold
        self.use_paraphrases = use_paraphrases
        self.faiss_path = self.cache_dir / "cache_index.faiss"
        self.meta_path = self.cache_dir / "cache_metadata.json"

        # Load embedding model
        logger.info(f"Embedding modeli yükleniyor: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Load index + metadata
        self._load()

        # Cleanup expired on startup
        removed = self.cleanup_expired()
        if removed:
            logger.info(f"{removed} süresi dolmuş önbellek kaydı silindi")

    def _load(self):
        """Load FAISS index and metadata from disk."""
        if self.faiss_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata: List[Dict[str, Any]] = json.load(f)
            logger.info(f"Önbellek yüklendi: {self.index.ntotal} kayıt")
        else:
            logger.warning("Önbellek bulunamadı, boş oluşturuluyor")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

    def _save(self):
        """Save FAISS index and metadata to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.faiss_path))
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single text, returns 1D float32 array."""
        emb = self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True,
        )
        return emb[0].astype(np.float32)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts, returns 2D float32 array."""
        embs = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
        )
        return embs.astype(np.float32)

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired."""
        expires = entry.get("expires_at")
        if not expires:
            return False  # No expiry = never expires (dataset entries)
        try:
            return datetime.fromisoformat(expires) < datetime.now()
        except (ValueError, TypeError):
            return False

    def search(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search the cache with normalization + optional paraphrase expansion.

        Steps:
          1. Normalize the query (Turkish-aware)
          2. Optionally expand with LLM paraphrases
          3. Embed all variants
          4. Multi-vector search: take the best match across all variants
          5. Return if similarity >= threshold and not expired

        Returns dict with answer + metadata on hit, None on miss.
        """
        if self.index.ntotal == 0:
            logger.info("CACHE_MISS — önbellek boş")
            return None

        start = time.time()

        # Step 1: Normalize
        normalized = normalize_turkish(query)

        # Step 2: Expand (original + normalized + paraphrases)
        search_queries = [query, normalized]  # Always search with both

        if self.use_paraphrases:
            paraphrases = expand_paraphrases(query)
            for p in paraphrases:
                norm_p = normalize_turkish(p)
                if norm_p not in search_queries:
                    search_queries.append(norm_p)

        # Deduplicate
        search_queries = list(dict.fromkeys(search_queries))
        logger.info(f"Arama sorguları ({len(search_queries)}): {search_queries[:4]}")

        # Step 3: Embed all variants
        embeddings = self._embed_batch(search_queries)

        # Step 4: Multi-vector search — find the best match across all variants
        best_score = -1.0
        best_idx = -1
        best_variant = ""

        for i, emb in enumerate(embeddings):
            query_vec = emb.reshape(1, -1)
            scores, indices = self.index.search(query_vec, 1)
            score = float(scores[0][0])
            idx = int(indices[0][0])

            if score > best_score and 0 <= idx < len(self.metadata):
                best_score = score
                best_idx = idx
                best_variant = search_queries[i]

        search_time = time.time() - start

        if best_idx < 0:
            logger.info(f"CACHE_MISS — geçersiz indeks, süre: {search_time:.3f}s")
            return None

        entry = self.metadata[best_idx]

        # Check expiry
        if self._is_expired(entry):
            logger.info(f"CACHE_MISS — süresi dolmuş kayıt, süre: {search_time:.3f}s")
            return None

        if best_score >= self.threshold:
            # Cache hit — update usage
            now = datetime.now().isoformat()
            entry["usage_count"] = entry.get("usage_count", 0) + 1
            entry["last_used_at"] = now
            self._save()

            logger.info(
                f"CACHE_HIT — benzerlik: {best_score:.4f}, "
                f"varyant: '{best_variant[:50]}', "
                f"kullanım: {entry['usage_count']}, "
                f"süre: {search_time:.3f}s"
            )
            return {
                "answer": entry["answer"],
                "query": entry["query"],
                "source": entry.get("source", ""),
                "confidence": entry.get("confidence", 1.0),
                "law_refs": entry.get("law_refs", []),
                "usage_count": entry["usage_count"],
                "similarity": best_score,
                "matched_variant": best_variant,
                "search_time": search_time,
                "cache_hit": True,
            }
        else:
            logger.info(
                f"CACHE_MISS — en iyi: {best_score:.4f} "
                f"(eşik: {self.threshold}), süre: {search_time:.3f}s"
            )
            return None

    def add(self, query: str, answer: str, source: str = "rag",
            confidence: float = 0.8, law_refs: list = None,
            ttl_days: int = DEFAULT_TTL_DAYS):
        """
        Add a new QA pair to the cache with enriched metadata.

        Dataset entries: confidence=1.0, no expiry
        RAG entries: confidence=0.8, 90-day TTL
        """
        normalized = normalize_turkish(query)
        emb = self._embed(normalized).reshape(1, -1)

        now = datetime.now()
        self.index.add(emb)
        self.metadata.append({
            "query": query,
            "answer": answer,
            "source": source,
            "confidence": confidence,
            "law_refs": law_refs or [],
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=ttl_days)).isoformat() if source == "rag" else None,
            "usage_count": 0,
            "last_used_at": None,
        })
        self._save()

        logger.info(f"Önbelleğe eklendi — toplam: {self.index.ntotal}, güven: {confidence}")

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        if not self.metadata:
            return 0

        now = datetime.now()
        to_keep = []
        keep_indices = []

        for i, entry in enumerate(self.metadata):
            if not self._is_expired(entry):
                to_keep.append(entry)
                keep_indices.append(i)

        removed = len(self.metadata) - len(to_keep)
        if removed > 0:
            # Rebuild index with only non-expired entries
            # Get all embeddings from the existing index
            if keep_indices:
                vectors = np.array([
                    self.index.reconstruct(i) for i in keep_indices
                ])
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(vectors.astype(np.float32))
            else:
                self.index = faiss.IndexFlatIP(self.dimension)

            self.metadata = to_keep
            self._save()

        return removed

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = len(self.metadata)
        sources = {}
        total_usage = 0
        expired = 0
        for m in self.metadata:
            src = m.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            total_usage += m.get("usage_count", 0)
            if self._is_expired(m):
                expired += 1

        return {
            "total_entries": total,
            "active_entries": total - expired,
            "expired_entries": expired,
            "total_usage": total_usage,
            "sources": sources,
        }
