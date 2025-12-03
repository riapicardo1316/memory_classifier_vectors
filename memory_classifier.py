from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os
import re
import numpy as np
import argparse
import sys

STOPWORDS = set(ENGLISH_STOP_WORDS)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
embedder = SentenceTransformer("all-mpnet-base-v2")

memory_blocks = {
    "identity": [],
    "health_safety": [],
    "preferences": [],
    "goals_commitments": [],
    "emotional_state": [],
    "context": [],
    "appointment_timing": [],
    "medication": [],
}

expiration_log = []
contradiction_log = []

CATEGORIES = {
    "identity and personal background": [
        "I live here",
        "I work as",
        "I was born",
        "My family is",
        "My background is",
        "I grew up",
    ],
    "health and safety": [
        "I have anxiety",
        "I am allergic",
        "I have chronic illness",
        "I am going through therapy",
        "I have migraines",
        "I have insomnia",
    ],
    "preferences and interests": [
        "I love pizza",
        "I enjoy movies",
        "My favourite music is",
        "I like sports",
        "I am interested in art",
    ],
    "goals and commitments": [
        "I plan to",
        "My target is",
        "I want to achieve",
        "I am committed to",
        "I want to stop",
        "I want to quit",
    ],
    "emotional state": [
        "I feel sad",
        "I am stressed",
        "I am depressed",
        "I feel happy",
        "I am afraid",
        "I feel anxious",
        "I’m a little anxious ",
        "I’m feeling nervous",
    ],
    "situational context": [
        "I have a meeting",
        "I have an exam",
        "There is a deadline",
        "I have a flight",
        "I will travel soon",
        "I have an appointment",
    ],
    "appointment timing": [
        "my appointment is",
        "doctor appointment",
        "scheduled to meet doctor",
        "appointment moved",
        "appointment time",
        "follow-up appointment",
        "consultation scheduled"
    ],
    "medication and treatment": [
        "I take medication",
        "doctor prescribed",
        "I need to take pills",
        "I started antibiotics",
        "I need to take supplements",
        "painkiller dosage",
        "daily vitamins"
    ]
}

CATEGORY_EMBEDS = {
    label: embedder.encode(samples)
    for label, samples in CATEGORIES.items()
}

def compute_ttl(retention: str, future_level: int, label: str) -> int:
    """
    Returns TTL in 'turns' (an arbitrary unit that your decay() function decrements).
    Adjust these numbers to fit your demo evaluation timeframe.
    """
    if retention == "discard":
        return 0

    if retention == "long_term":
        return 60

    if "appointment" in label:
        if future_level == 1:
            return 7
        if future_level == 2:
            return 14
        if future_level == 3:
            return 30
        return 14

    if "medication" in label:
        return 60

    if future_level == 1:
        return 7
    if future_level == 2:
        return 14
    if future_level == 3:
        return 30

    if "preferences" in label:
        return 14
    if "goals" in label or "context" in label:
        return 14

    return 7

def extract_words(s: str) -> set:
    words = re.findall(r"[a-zA-Z']+", s.lower())
    return {w for w in words if w not in STOPWORDS}

def detect_future_level(sentence: str) -> int:
    s = sentence.lower()
    if re.search(r"\b(today|tomorrow|tonight|morning|evening|later)\b", s):
        return 1
    if re.search(r"\b(next week|this week|few days|this weekend)\b", s):
        return 2
    if re.search(r"\b(next month|next year|few months|this year|this summer)\b", s):
        return 3
    if re.search(r"\bin\s+\d+\s+(day|days|week|weeks)\b", s):
        return 2
    if re.search(r"\bin\s+\d+\s+(month|months|year|years)\b", s):
        return 3
    return 0

def detect_category(sentence: str):
    """
    Choose the best label by average cosine similarity between the sentence embedding
    and the sample bank for each category.
    """
    sent_emb = embedder.encode(sentence)
    scores = {}

    for label, bank in CATEGORY_EMBEDS.items():
        sims = cosine_similarity(sent_emb, bank)
        scores[label] = np.mean(sims)

    best_label = max(scores, key=scores.get)
    return best_label, round(float(scores[best_label]), 3)

def cosine_similarity(vec, bank):
  
    dot = np.dot(bank, vec)  
    norm_vec = np.linalg.norm(vec)
    norm_bank = np.linalg.norm(bank, axis=1)
    denom = norm_vec * norm_bank
    denom[denom == 0] = 1e-8
    return dot / denom

def retention(label: str, prob: float, sentence: str, future_level: int):
    """
    Returns (retention_decision, reason_string)
    """
    s = sentence.lower()

    if "medication" in label or "treatment" in label:
        return "long_term", "medication/treatment information is safety-critical"

    if "appointment" in label:
        if future_level > 0:
            return "short_term", "upcoming scheduled appointment (future detected)"
        return "discard", "appointment mention lacks a detected future time"
    
    if "health" in label or "identity" in label:
        return "long_term", "stable personal/health attribute"

    if "emotional" in label or "emotion" in label:
        if prob >= 0.5:
            return "long_term", "strong emotional intensity (high confidence)"
        return "short_term", "transient/mild emotional state (low-moderate confidence)"

    if "goals" in label or "commitments" in label:
        if ("quit" in s or "stop" in s or "I will" in s) and prob >= 0.45:
            return "long_term", "strong behavioral intention / commitment detected"
        return "short_term", "goal mentioned but no strong commitment signal"

    if "context" in label or "situational context" in label:
        if future_level > 0:
            return "short_term", "context tied to near-future event"
        return "discard", "transient context without future relevance"

    if "preferences" in label or "preference" in label:
        if prob >= 0.55:
            return "short_term", "preference with adequate confidence"
        return "discard", "weak preference signal (low confidence)"

    if prob >= 0.7:
        return "long_term", "high similarity confidence"
    if prob >= 0.45:
        return "short_term", "moderate similarity confidence"
    return "discard", "low similarity confidence"

def apply_contradiction(sentence, turn):
    new_words = extract_words(sentence)
    if not new_words:
        return

    for block, items in memory_blocks.items():
        for it in items:
            if it["expired"]:
                continue

            prev_words = it["_words"]
            if len(new_words.intersection(prev_words)) >= 3:
                it["expired"] = True
                contradiction_log.append({
                    "turn": turn,
                    "overridden": it["sentence"],
                    "block": block,
                    "by": sentence,
                })

def repetition(block, sentence):
    cw = extract_words(sentence)
    if not cw:
        return False

    for it in memory_blocks[block]:
        if it["expired"]:
            continue
        prev = it["_words"]
        inter = len(cw.intersection(prev))
        union = len(cw.union(prev))
        if union == 0:
            continue
        if inter >= 2 or (inter / union) > 0.4:
            return True
    return False

def decay(turn):
    for block, items in memory_blocks.items():
        for it in items:
            if it["expired"]:
                continue
            if it["ttl"] > 0:
                it["ttl"] -= 1
                if it["ttl"] <= 0:
                    it["expired"] = True
                    expiration_log.append({
                        "turn": turn,
                        "sentence": it["sentence"],
                        "block": block
                    })

def handle(sentence, turn):
    apply_contradiction(sentence, turn)

    category, score = detect_category(sentence)
    future_level = detect_future_level(sentence)
    ret, reason = retention(category, score, sentence, future_level)
    ttl_val = compute_ttl(ret, future_level, category)

    if "appointment" in category:
        block = "appointment_timing"
    elif "medication" in category:
        block = "medication"
    elif category.startswith("identity"):
        block = "identity"
    elif category.startswith("health"):
        block = "health_safety"
    elif "preferences" in category or "preference" in category:
        block = "preferences"
    elif "goals" in category or "commitments" in category:
        block = "goals_commitments"
    elif "emotional" in category or "emotion" in category:
        block = "emotional_state"
    else:
        block = "context"

    if block == "preferences" and ret != "long_term":
        if repetition(block, sentence):
            ret = "long_term"
            if score < 0.75:
                score = 0.75
            ttl_val = 60
            reason = "promotion to long_term due to repetition"

    entry = {
        "sentence": sentence.strip(),
        "turn": turn,
        "category": category,
        "confidence": score,
        "retention": ret,
        "reason": reason,
        "ttl": ttl_val,
        "expired": False,
        "_words": extract_words(sentence),
    }

    memory_blocks[block].append(entry)
    return entry

def main():
    parser = argparse.ArgumentParser(description="Run memory classifier on a conversation file")
    parser.add_argument(
        "filename",
        type=str,
        help="Name of the conversation text file (must exist in same directory)"
    )
    args = parser.parse_args()

    fname = args.filename

    if not os.path.exists(fname):
        print(f"ERROR: '{fname}' not found in current directory.")
        sys.exit(1)

    with open(fname, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    all_decisions = []

    for turn, sent in enumerate(lines):
        decay(turn)
        all_decisions.append(handle(sent, turn))

    print("\n=== DECISIONS ===\n")
    for d in all_decisions:
        print(
            f"{d['sentence']}\n"
            f" -> {d['category']} (sim={d['confidence']})\n"
            f" -> retention={d['retention']} (reason: {d['reason']})\n"
            f" -> ttl={d['ttl']}\n"
        )

    print("\n=== ACTIVE MEMORY ===")
    for block, items in memory_blocks.items():
        active = [i for i in items if not i["expired"] and (i["ttl"] > 0)]
        print(f"\n[{block.upper()}] {len(active)} items")
        for it in active:
            print(f" - {it['sentence']} (ttl={it['ttl']}, reason={it.get('reason')})")

    print("\n=== TTL EXPIRED ===")
    if not expiration_log:
        print("None")
    else:
        for e in expiration_log:
            print(f"Turn {e['turn']}: '{e['sentence']}' removed from {e['block']}")

    print("\n=== CONTRADICTIONS ===")
    if not contradiction_log:
        print("None")
    else:
        for c in contradiction_log:
            print(
                f"Turn {c['turn']}: '{c['overridden']}' overridden by '{c['by']}' in {c['block']}"
            )

    long_term_final = []
    short_term_final = []

    for block, items in memory_blocks.items():
        for it in items:
            if it["expired"]:
                continue

            if it["retention"] == "long_term" and it["ttl"] > 0:
                long_term_final.append((block, it))
            elif it["retention"] == "short_term" and it["ttl"] > 0:
                short_term_final.append((block, it))

    print("\n\n=== FINAL LONG-TERM MEMORY ===")
    if not long_term_final:
        print("None retained")
    else:
        for block, it in sorted(long_term_final, key=lambda x: x[1]['turn']):
            print(
                f"[{block.upper()}] {it['sentence']} "
                f"(turn={it['turn']}, ttl={it['ttl']}, sim={it['confidence']}, reason={it.get('reason')})"
            )

    print("\n=== FINAL SHORT-TERM MEMORY ===")
    if not short_term_final:
        print("None retained")
    else:
        for block, it in sorted(short_term_final, key=lambda x: x[1]['turn']):
            print(
                f"[{block.upper()}] {it['sentence']} "
                f"(turn={it['turn']}, ttl={it['ttl']}, sim={it['confidence']}, reason={it.get('reason')})"
            )

    print("\n=== CONTRADICTION-RESOLVED MEMORY ===")
    if not contradiction_log:
        print("None applied")
    else:
        seen = set()
        for c in contradiction_log:
            key = (c['overridden'], c['by'])
            if key in seen:
                continue
            seen.add(key)
            print(
                f"Overridden: '{c['overridden']}'\n"
                f"→ New version: '{c['by']}' (turn={c['turn']})\n"
            )

    total = len(lines)
    discarded = sum(
        1 for block, items in memory_blocks.items()
        for it in items if it.get("retention") == "discard"
    )
    inactive = sum(
        1 for block, items in memory_blocks.items()
        for it in items if (it.get("ttl", 0) <= 0 or it.get("expired", False))
    )

    print("\n=== STATS ===")
    print(f"Total processed sentences: {total}")
    print(f"Discarded by retention==\"discard\": {discarded} ({(discarded/total if total else 0):.2%})")
    print(f"Inactive (ttl<=0 or expired): {inactive} ({(inactive/total if total else 0):.2%})")

if __name__ == "__main__":
    main()
