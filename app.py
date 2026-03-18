"""
app.py — Flask web server for Legal RAG + CAG Hybrid System.

Routes:
    GET  /            → Serve the web UI
    POST /api/ask     → Query the system (CAG → RAG fallback)
    POST /api/upload  → Upload a document for RAG indexing
    GET  /api/stats   → System statistics
    POST /api/settings → Update settings (threshold, paraphrases)
"""

import logging
import os
import time
from pathlib import Path

from flask import Flask, request, jsonify, render_template

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
RAG_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['REFINE_CACHE'] = False  # Refine cache hits with LLM
app.config['KV_CACHE_CAG'] = False  # Prompt pre-filling KV Cache CAG

# Global instances (initialized on first request)
cache = None
rag = None
kv_inst = None


def get_components():
    """Lazy-load the CAG cache, RAG pipeline, and KV Cache pipeline."""
    global cache, rag, kv_inst
    if cache is None:
        from src.cache_search import SemanticCache
        from src.rag_pipeline import RAGPipeline
        from src.kv_cag_pipeline import KVCachePipeline
        cache = SemanticCache(str(CACHE_DIR), use_paraphrases=False)
        rag = RAGPipeline(str(RAG_DIR), model=cache.model)
        kv_inst = KVCachePipeline(model="qwen3:8b", max_pairs=100)
        logger.info("Components initialized.")
    return cache, rag, kv_inst


# ── Routes ───────────────────────────────────────────────────────


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask():
    """Query the system: CAG cache first, RAG fallback."""
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"error": "Soru boş olamaz"}), 400

    cache_inst, rag_inst, kv_pipeline = get_components()
    total_start = time.time()

    # Step 0: True KV Cache CAG (Prompt Pre-filling)
    if app.config.get('KV_CACHE_CAG'):
        kv_result = kv_pipeline.answer(query)
        if kv_result is not None:
            # Hit in pre-loaded KV Cache
            total_time = time.time() - total_start
            return jsonify({
                "type": "kv-cag",
                "answer": kv_result["answer"],
                "similarity": 1.0,  # Defined as exact context answer
                "confidence": 1.0,
                "source": kv_result["source"],
                "llm_latency": round(kv_result["latency"], 2),
                "total_time": round(total_time, 3),
            })

    # Step 1: CAG cache search
    result = cache_inst.search(query)

    if result:
        # Cache hit
        if app.config.get('REFINE_CACHE'):
            # Refine the raw cached answer with LLM
            refined = rag_inst.refine_answer(query, result["answer"])
            answer_text = refined["answer"]
            llm_latency = refined["llm_latency"]
            hit_type = "cache-refined"
        else:
            answer_text = result["answer"]
            llm_latency = 0.0
            hit_type = "cache"

        total_time = time.time() - total_start
        return jsonify({
            "type": hit_type,
            "answer": answer_text,
            "similarity": round(result["similarity"], 4),
            "confidence": round(result.get("confidence", 1.0), 2),
            "source": result.get("source", ""),
            "matched_query": result.get("query", ""),
            "usage_count": result.get("usage_count", 0),
            "matched_variant": result.get("matched_variant", ""),
            "law_refs": result.get("law_refs", []),
            "llm_latency": round(llm_latency, 2),
            "total_time": round(total_time, 3),
        })

    # Step 2: RAG fallback
    rag_result = rag_inst.answer(query, top_k=5)

    # Step 3: Store in CAG cache
    cache_inst.add(
        query=query,
        answer=rag_result["answer"],
        source="rag",
        confidence=0.8,
        ttl_days=90,
    )

    total_time = time.time() - total_start

    context_docs = []
    for doc in rag_result.get("context_docs", [])[:5]:
        context_docs.append({
            "text": doc["text"][:300],
            "source": doc.get("source", ""),
            "original_query": doc.get("original_query", ""),
            "score": round(doc.get("score", 0), 3),
        })

    return jsonify({
        "type": "rag",
        "answer": rag_result["answer"],
        "context_docs": context_docs,
        "rag_latency": round(rag_result.get("rag_latency", 0), 2),
        "llm_latency": round(rag_result.get("llm_latency", 0), 2),
        "total_time": round(total_time, 2),
    })


@app.route('/api/warmup_kv', methods=['POST'])
def warmup_kv():
    """Triggers the heavy initial encoding for the KV Cache."""
    _, _, kv_pipeline = get_components()
    success = kv_pipeline.warmup()
    if success:
        return jsonify({"success": True, "message": "KV Cache başarıyla ısıtıldı."})
    else:
        return jsonify({"error": "KV Cache ısıtma başarısız."}), 500


@app.route('/api/upload', methods=['POST'])
def upload():
    """Upload a document to the RAG index."""
    _, rag_inst, _ = get_components()

    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Dosya seçilmedi"}), 400

    # Read file content
    filename = file.filename
    ext = Path(filename).suffix.lower()

    try:
        if ext in ('.txt', '.md', '.csv'):
            content = file.read().decode('utf-8', errors='ignore')
        elif ext == '.json':
            import json
            data = json.loads(file.read().decode('utf-8'))
            if isinstance(data, list):
                # Array of objects — concatenate all text fields
                parts = []
                for item in data:
                    if isinstance(item, dict):
                        parts.append(' '.join(str(v) for v in item.values()))
                    else:
                        parts.append(str(item))
                content = '\n'.join(parts)
            elif isinstance(data, dict):
                content = ' '.join(str(v) for v in data.values())
            else:
                content = str(data)
        else:
            return jsonify({"error": f"Desteklenmeyen dosya türü: {ext}"}), 400

        if not content.strip():
            return jsonify({"error": "Dosya boş"}), 400

        # Save upload for reference
        save_path = UPLOAD_DIR / filename
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Add to RAG index
        chunks_added = rag_inst.add_document(content, source=filename)

        return jsonify({
            "success": True,
            "filename": filename,
            "chunks_added": chunks_added,
            "total_rag_docs": rag_inst.index.ntotal,
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics."""
    cache_inst, rag_inst, kv_pipeline = get_components()
    cache_stats = cache_inst.stats()

    return jsonify({
        "cache": {
            "total_entries": cache_stats["total_entries"],
            "active_entries": cache_stats["active_entries"],
            "expired_entries": cache_stats["expired_entries"],
            "total_usage": cache_stats["total_usage"],
            "sources": cache_stats["sources"],
        },
        "rag": {
            "total_chunks": rag_inst.index.ntotal,
        },
        "settings": {
            "threshold": cache_inst.threshold,
            "paraphrases": cache_inst.use_paraphrases,
            "refine_cache": app.config.get('REFINE_CACHE', False),
            "kv_cache_cag": app.config.get('KV_CACHE_CAG', False),
            "kv_warmed_up": kv_pipeline.is_warmed_up,
        },
    })


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update system settings."""
    cache_inst, _, _ = get_components()
    data = request.get_json()

    if 'threshold' in data:
        try:
            val = float(data['threshold'])
            if 0.5 <= val <= 1.0:
                cache_inst.threshold = val
        except (ValueError, TypeError):
            pass

    if 'paraphrases' in data:
        cache_inst.use_paraphrases = bool(data['paraphrases'])

    if 'refine_cache' in data:
        app.config['REFINE_CACHE'] = bool(data['refine_cache'])

    if 'kv_cache_cag' in data:
        app.config['KV_CACHE_CAG'] = bool(data['kv_cache_cag'])

    return jsonify({
        "threshold": cache_inst.threshold,
        "paraphrases": cache_inst.use_paraphrases,
        "refine_cache": app.config.get('REFINE_CACHE', False),
        "kv_cache_cag": app.config.get('KV_CACHE_CAG', False),
    })


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting Legal RAG + CAG Web Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
