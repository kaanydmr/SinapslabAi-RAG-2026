"""
query_processor.py — Turkish query normalization + LLM paraphrase expansion.

Two key functions:
  1. normalize_turkish(query) → cleaned, lowercased, punctuation-free query
  2. expand_paraphrases(query) → list of alternative phrasings via LLM
"""

import re
import logging
from typing import List

import ollama

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen3:8b"

# ── Turkish Stopwords ────────────────────────────────────────────

TURKISH_STOPWORDS = {
    "bir", "ve", "ile", "için", "bu", "da", "de", "mi", "mı", "mu", "mü",
    "ne", "o", "şu", "ben", "sen", "biz", "siz", "ama", "fakat", "ancak",
    "çok", "daha", "en", "gibi", "kadar", "olan", "olarak", "veya", "ya",
    "hem", "ki", "ise", "bile", "sadece", "hep", "her", "hiç", "nasıl",
    "neden", "nere", "acaba",
}

# ── Turkish Lowercasing ──────────────────────────────────────────

# Turkish has special casing: İ→i, I→ı, Ş→ş, Ç→ç, Ğ→ğ, Ö→ö, Ü→ü
TURKISH_LOWER_MAP = str.maketrans({
    'İ': 'i',
    'I': 'ı',
    'Ş': 'ş',
    'Ç': 'ç',
    'Ğ': 'ğ',
    'Ö': 'ö',
    'Ü': 'ü',
})


def turkish_lower(text: str) -> str:
    """Lowercase text with Turkish-specific rules (İ→i, I→ı)."""
    # First apply Turkish-specific mappings
    text = text.translate(TURKISH_LOWER_MAP)
    # Then standard lowercase for remaining chars
    text = text.lower()
    return text


# ── Normalization ────────────────────────────────────────────────


def normalize_turkish(query: str) -> str:
    """
    Full Turkish query normalization:
      1. Remove control characters
      2. Turkish-aware lowercasing (İ→i, I→ı)
      3. Remove punctuation (keep alphanumeric + Turkish chars)
      4. Collapse whitespace
      5. Remove stopwords
    """
    # Remove control characters
    query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)

    # Turkish-aware lowercasing
    query = turkish_lower(query)

    # Remove punctuation but keep Turkish letters and digits
    query = re.sub(r'[^\wçğıöşü\s]', ' ', query)

    # Collapse whitespace
    query = re.sub(r'\s+', ' ', query).strip()

    # Remove stopwords
    words = query.split()
    words = [w for w in words if w not in TURKISH_STOPWORDS]

    return ' '.join(words)


# ── Paraphrase Expansion ────────────────────────────────────────

PARAPHRASE_PROMPT = """Sen bir Türk hukuku dil uzmanısın. Verilen soruyu 3 farklı şekilde yeniden yaz.

Kurallar:
- Aynı hukuki anlamı koru
- Her birini farklı kelimelerle ifade et (resmi/günlük/teknik)
- Sadece soruları yaz, başka açıklama ekleme
- Her satıra bir soru yaz
- Numaralandırma kullanma

Soru: {query}

3 alternatif:"""


def expand_paraphrases(query: str, max_retries: int = 2) -> List[str]:
    """
    Generate paraphrases of the query using the LLM.

    Returns a list of alternative phrasings (original + generated).
    On failure, returns just the original query.
    """
    paraphrases = [query]  # Always include original

    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": PARAPHRASE_PROMPT.format(query=query),
                }],
                options={
                    "temperature": 0.7,
                    "num_predict": 200,
                },
            )

            text = response["message"]["content"]

            # Parse lines — each non-empty line is a paraphrase
            lines = [line.strip() for line in text.strip().split('\n')]
            for line in lines:
                # Clean up numbering if present (1. 2. 3. or - or *)
                cleaned = re.sub(r'^[\d\.\-\*\)]+\s*', '', line).strip()
                if cleaned and len(cleaned) > 10 and cleaned != query:
                    paraphrases.append(cleaned)

            if len(paraphrases) > 1:
                logger.info(f"Paraphrase: {len(paraphrases) - 1} alternatif üretildi")
                return paraphrases[:4]  # Original + max 3 paraphrases

        except Exception as e:
            logger.warning(f"Paraphrase hatası (deneme {attempt + 1}): {e}")

    logger.info("Paraphrase üretilemedi, sadece orijinal sorgu kullanılacak")
    return paraphrases
