import json
import logging
import time
from typing import Dict, Any, Optional
import ollama

logger = logging.getLogger(__name__)

class KVCachePipeline:
    """
    Implements a true KV Cache pre-computation CAG strategy.
    
    Loads a chunk of the dataset at initialization, sends it to Ollama
    with keep_alive to compute and hold the KV cache in RAM/VRAM.
    Subsequent queries use the exact same prefix to leverage the pre-computed state.
    """
    def __init__(self, data_path: str = "main set.json", model: str = "qwen3:8b", max_pairs: int = 100):
        self.model = model
        self.system_prompt = ""
        self.is_warmed_up = False
        
        # Load dataset
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Select a subset to fit within context window (e.g. 100 pairs ~4K-6K tokens depending on lengths)
            subset = data[:max_pairs]
            
            # Construct the massive knowledge prompt
            knowledge_text = "TÜRK HUKUKU BİLGİ BANKASI KAPSAMI:\n\n"
            for row in subset:
                knowledge_text += f"Soru: {row['Soru']}\nCevap: {row['Cevap']}\n---\n"
                
            self.system_prompt = f"""Sen bir Türk Hukuku uzmanısın. 
AŞAĞIDAKİ BİLGİ BANKASI SENİN TEK KAYNAĞINDIR.
Sadece bu listedeki bilgilerle cevap verebilirsin.

KURALLAR:
1. Soruya ait cevabı AŞAĞIDAKİ LİSTEDE (Bilgi Bankası) bulabiliyorsan, o cevabı kullanarak kısa ve öz cevap ver.
2. Eğer sorunun cevabı bu listede YÜZDE YÜZ AYNI ŞEKİLDE YOKSA, kesinlikle yorum yapma, uydurma ve SADECE ŞUNU YAZ: "BUNU BİLMİYORUM".
3. Başka hiçbir açıklama yapma.

BİLGİ BANKASI:
{knowledge_text}"""
            logger.info(f"KV Cache CAG: {len(subset)} adet Soru-Cevap ikilisi ile sistem komutu oluşturuldu.")
        except Exception as e:
            logger.error(f"KV CAG Veri Seti yüklenemedi: {e}")
            self.system_prompt = "Sen bir Türk Hukuk uzmanısın. (Veri seti bulunamadı)"

    def warmup(self):
        """
        Sends the massive prompt to Ollama to compute KV matrices.
        Uses keep_alive="2h" to keep the model and its KV cache in VRAM.
        """
        logger.info("KV Cache CAG: Model Isıtılıyor (VRAM KV Cache inşası başlıyor)...")
        start = time.time()
        try:
            ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Hazır mısın?"}
                ],
                options={"temperature": 0.0},
                keep_alive="2h"  # Keep loaded for 2 hours
            )
            self.is_warmed_up = True
            logger.info(f"KV Cache Isıtma Tamamlandı. Süre: {time.time() - start:.2f}s")
            return True
        except Exception as e:
            logger.error(f"KV Cache Isıtma Hatası: {e}")
            return False

    def answer(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Generates an answer using the pre-loaded KV Cache.
        Returns None if the LLM states it doesn't know (fallback signal).
        """
        if not self.is_warmed_up:
            # Lazy warmup if not done
            success = self.warmup()
            if not success:
                return None
                
        start = time.time()
        try:
            # We MUST use the identical system prompt so Ollama hits the KV cache
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ],
                options={"temperature": 0.0},
                keep_alive="2h"
            )
            ans = response["message"]["content"].strip()
            llm_latency = time.time() - start
            
            # Check for fallback
            ans_upper = ans.upper()
            if "BUNU BİLMİYORUM" in ans_upper or "BİLMİYORUM" in ans_upper or "BILMIYORUM" in ans_upper or "VERİ TABANINDA" in ans_upper or "HAFIZAMDA" in ans_upper:
                logger.info(f"KV CAG Fallback tetiklendi. Süre: {llm_latency:.2f}s (Cevap alınamadı)")
                return None
                
            return {
                "answer": ans,
                "latency": llm_latency,
                "source": "KV Cache (Önceden Yüklenmiş Bellek - Prompt Pre-fill)"
            }
        except Exception as e:
            logger.error(f"KV CAG Hatası: {e}")
            return None
