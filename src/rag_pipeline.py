"""
rag_pipeline.py — RAG fallback with its own document store.

Uses a SEPARATE FAISS index of chunked legal texts (answers + context).
This is different from the CAG cache which stores question embeddings.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

SYSTEM_PROMPT = """Sen bir Türk hukuku uzmanı yapay zeka asistanısın. Aşağıdaki kurallara kesinlikle uy:

1. Her zaman Türkçe cevap ver.
2. Cevaplarında mümkün olduğunca ilgili kanun maddelerini belirt.
3. Eğer kesin bir hukuki kaynak bulamıyorsan, bunu açıkça belirt.
4. Uydurma kanun maddeleri veya hukuki bilgi verme.
5. Verilen bağlam belgelerini kullanarak cevap ver.
6. Cevaplarını açık, anlaşılır ve profesyonel bir dilde yaz."""


def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class RAGPipeline:
    """RAG pipeline with its own separate document FAISS index."""

    def __init__(self, rag_dir: str, model: Optional[SentenceTransformer] = None):
        self.rag_dir = Path(rag_dir)
        self.faiss_path = self.rag_dir / "rag_index.faiss"
        self.meta_path = self.rag_dir / "rag_metadata.json"

        # Share model if provided, else load
        if model:
            self.model = model
        else:
            logger.info(f"RAG embedding modeli yükleniyor: {EMBED_MODEL}")
            self.model = SentenceTransformer(EMBED_MODEL)

        self.dimension = self.model.get_sentence_embedding_dimension()
        self._load()

    def _load(self):
        """Load RAG document index and metadata."""
        if self.faiss_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata: List[Dict[str, Any]] = json.load(f)
            logger.info(f"RAG belge indeksi yüklendi: {self.index.ntotal} parça")
        else:
            logger.warning("RAG indeksi bulunamadı!")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

    def _embed(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True,
        )
        return emb[0].astype(np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant document chunks from the RAG index.

        Returns list of {text, source, original_query, score}.
        """
        if self.index.ntotal == 0:
            return []

        query_emb = self._embed(normalize_text(query)).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                entry = self.metadata[idx]
                results.append({
                    "text": entry["text"],
                    "source": entry.get("source", ""),
                    "original_query": entry.get("original_query", ""),
                    "score": float(score),
                })

        return results

    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send retrieved context + query to LLM for answer generation."""
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get("source", "")
            text_preview = doc["text"][:500]
            context_parts.append(f"[Kaynak {i}: {source}]\n{text_preview}")

        context_text = "\n\n".join(context_parts)

        user_message = f"""Aşağıda, veritabanından getirilen ilgili hukuki belgeler bulunmaktadır:

{context_text}

---

Kullanıcının Sorusu: {query}

Talimat: Yukarıdaki hukuki belgeleri kullanarak soruyu Türkçe olarak kapsamlı şekilde cevapla. İlgili kanun maddelerini ve kaynaklarını belirt."""

        start = time.time()
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            answer = response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama hatası: {e}")
            answer = f"LLM yanıt üretemedi: {str(e)}"

        llm_latency = time.time() - start
        return {"answer": answer, "llm_latency": llm_latency}

    def refine_answer(self, query: str, cached_answer: str) -> Dict[str, Any]:
        """Send a short cached answer to LLM to expand/refine it."""
        user_message = f"""Aşağıda, sorunun sistem önbelleğindeki (cache) ham cevabı bulunmaktadır:

[Önbellek Cevabı]
{cached_answer}

---

Kullanıcının Sorusu: {query}

Talimat: Yukarıdaki önbellek cevabını temel alarak, soruyu Türkçe olarak daha kapsamlı, akıcı ve profesyonel bir dilde açıkla. Çelişen bir bilgi verme, sadece detaylandır ve düzenle."""

        start = time.time()
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            answer = response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama hatası: {e}")
            answer = f"Yeniden yazım hatası: {str(e)} (Orijinal: {cached_answer})"

        llm_latency = time.time() - start
        return {"answer": answer, "llm_latency": llm_latency}

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve relevant documents → generate answer.

        Returns {answer, context_docs, rag_latency, llm_latency}.
        """
        rag_start = time.time()

        # Retrieve from document store
        context_docs = self.retrieve(query, top_k)
        rag_latency = time.time() - rag_start

        logger.info(
            f"RAG: {len(context_docs)} belge parçası alındı "
            f"(en iyi skor: {context_docs[0]['score']:.3f})" if context_docs else "RAG: belge bulunamadı"
        )

        # Generate with LLM
        llm_result = self.generate(query, context_docs)

        return {
            "answer": llm_result["answer"],
            "context_docs": context_docs,
            "rag_latency": rag_latency,
            "llm_latency": llm_result["llm_latency"],
        }

    def _save(self):
        """Save RAG index and metadata to disk."""
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.faiss_path))
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def add_document(self, text: str, source: str = "upload",
                     chunk_size: int = 400, overlap: int = 80) -> int:
        """
        Add a new document to the RAG index.
        Text is chunked, embedded, and indexed.
        Returns number of chunks added.
        """
        # Chunk the text
        words = text.split()
        if len(words) <= chunk_size:
            chunks = [text]
        else:
            chunks = []
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunks.append(' '.join(words[start:end]))
                start += chunk_size - overlap

        # Embed all chunks
        embeddings = self.model.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True,
            batch_size=64,
        ).astype(np.float32)

        # Add to index
        self.index.add(embeddings)
        for chunk in chunks:
            self.metadata.append({
                "text": chunk,
                "source": source,
                "original_query": "",
            })

        self._save()
        logger.info(f"RAG'a eklendi: {len(chunks)} parça, kaynak: {source}")
        return len(chunks)

