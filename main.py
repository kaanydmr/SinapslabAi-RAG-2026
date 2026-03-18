"""
main.py — Legal RAG + CAG Hybrid System entry point.

Architecture:
    User Query
      → Turkish normalization (İ/I-aware, punctuation, stopwords)
      → LLM paraphrase expansion (3 alternative phrasings)
      → Multi-vector CAG Cache Search (FAISS)
        → similarity ≥ 0.85 → Return cached answer (fast)
        → else → RAG Pipeline (separate document index)
          → Retrieve legal text chunks → Qwen3:8B → Generate answer
          → Store new QA in CAG cache (with TTL)

Usage:
    python main.py --build       Build both CAG + RAG indexes
    python main.py               Interactive query mode
    python main.py --no-para     Interactive without paraphrase expansion
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
RAG_DIR = PROJECT_ROOT / "data"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("legal-hybrid")


def do_build():
    """Build both CAG cache and RAG document store from datasets."""
    from src.build_cache import build_all
    build_all(str(PROJECT_ROOT), str(CACHE_DIR), str(RAG_DIR))


def interactive_mode(use_paraphrases: bool = True):
    """Interactive query mode: CAG cache first, RAG fallback."""
    from src.cache_search import SemanticCache
    from src.rag_pipeline import RAGPipeline

    logger.info("=" * 60)
    logger.info("TÜRK HUKUK RAG + CAG HİBRİT SİSTEMİ")
    logger.info("Qwen3:8B via Ollama")
    logger.info("=" * 60)

    # Initialize CAG cache
    cache = SemanticCache(str(CACHE_DIR), use_paraphrases=use_paraphrases)
    cache_stats = cache.stats()

    # Initialize RAG pipeline (share embedding model)
    rag = RAGPipeline(str(RAG_DIR), model=cache.model)

    logger.info(
        f"CAG: {cache_stats['total_entries']} soru | "
        f"RAG: {rag.index.ntotal} belge parçası"
    )

    # UI
    print("\n" + "=" * 60)
    print("🇹🇷 Türk Hukuk RAG + CAG Yapay Zeka Asistanı")
    print("=" * 60)
    print(f"\n📦 Sistem Durumu:")
    print(f"   CAG Önbellek:   {cache_stats['total_entries']} soru")
    print(f"   RAG Belgeler:   {rag.index.ntotal} belge parçası")
    print(f"   Paraphrase:     {'Açık ✅' if use_paraphrases else 'Kapalı ❌'}")
    print(f"   Eşik değeri:    {cache.threshold}")
    print("\nÖrnek sorular:")
    print("  • Anayasa madde 1'e göre devlet şekli nedir?")
    print("  • Egemenlik kime aittir?")
    print("  • İşten çıkarılma hakları nelerdir?")
    print("\nKomutlar:")
    print("  stats  — Sistem istatistikleri")
    print("  q      — Çıkış")
    print()

    while True:
        try:
            query = input("📋 Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGüle güle!")
            break

        if not query:
            continue
        if query.lower() in ('q', 'quit', 'çık', 'exit'):
            print("\nGüle güle!")
            break
        if query.lower() in ('stats', 'cache', 'durum'):
            s = cache.stats()
            print(f"\n📊 Sistem Durumu:")
            print(f"   CAG toplam:      {s['total_entries']}")
            print(f"   CAG aktif:       {s['active_entries']}")
            print(f"   Süresi dolmuş:   {s['expired_entries']}")
            print(f"   Toplam kullanım: {s['total_usage']}")
            print(f"   Kaynaklar:       {s['sources']}")
            print(f"   RAG belge:       {rag.index.ntotal}\n")
            continue

        print()
        total_start = time.time()

        # ── STEP 1: CAG Cache Search ─────────────────────────────
        result = cache.search(query)

        if result:
            # === CACHE HIT ===
            total_time = time.time() - total_start
            print("⚡ CAG ÖNBELLEK SONUCU (Cache Hit)")
            print(f"   Benzerlik:    {result['similarity']:.4f}")
            print(f"   Güven:        {result['confidence']:.2f}")
            print(f"   Kullanım:     {result['usage_count']}")
            print(f"   Kaynak:       {result['source']}")
            if result.get('matched_variant') and result['matched_variant'] != query:
                print(f"   Eşleşen:      \"{result['matched_variant'][:60]}\"")
            if result.get('law_refs'):
                print(f"   Hukuki ref:   {result['law_refs']}")
            print(f"\n{'─' * 55}")
            print(result["answer"])
            print(f"{'─' * 55}")
            print(f"\n⏱  Süre: {total_time:.3f}s (önbellek)")

        else:
            # === CACHE MISS → RAG ===
            print("🔍 CAG önbellekte bulunamadı → RAG pipeline çalıştırılıyor...")

            # ── STEP 2: RAG Retrieval + LLM ──────────────────────
            rag_result = rag.answer(query, top_k=5)

            print(f"\n{'─' * 55}")
            print(rag_result["answer"])
            print(f"{'─' * 55}")

            # Show retrieved documents
            if rag_result["context_docs"]:
                print(f"\n📄 RAG Bağlam Belgeleri ({len(rag_result['context_docs'])} parça):")
                for i, doc in enumerate(rag_result["context_docs"][:3], 1):
                    src = doc.get("source", "?")
                    text_preview = doc["text"][:70] + "..."
                    print(f"   {i}. [{src}] {text_preview} (skor: {doc['score']:.3f})")

            # ── STEP 3: Store in CAG cache with TTL ──────────────
            cache.add(
                query=query,
                answer=rag_result["answer"],
                source="rag",
                confidence=0.8,
                ttl_days=90,
            )
            print(f"\n✅ Cevap CAG önbelleğe eklendi (toplam: {cache.index.ntotal}, TTL: 90 gün)")

            total_time = time.time() - total_start
            print(
                f"\n⏱  Süre — RAG: {rag_result['rag_latency']:.2f}s | "
                f"LLM: {rag_result['llm_latency']:.2f}s | "
                f"Toplam: {total_time:.2f}s"
            )

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Türk Hukuk RAG + CAG Hibrit Sistemi"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="CAG önbellek ve RAG indeksini oluştur",
    )
    parser.add_argument(
        "--no-para",
        action="store_true",
        help="Paraphrase genişletme olmadan çalıştır",
    )
    args = parser.parse_args()

    if args.build:
        do_build()
    else:
        interactive_mode(use_paraphrases=not args.no_para)


if __name__ == "__main__":
    main()
