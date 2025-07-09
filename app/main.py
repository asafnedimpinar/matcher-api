from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

app = FastAPI()

# Sabitler
WEIGHTS = np.array([14, 16, 24, 16, 17, 13])
RANGES = [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36)]
THRESHOLD = 20
K = 6
MAX_DISTANCE = np.sqrt(np.sum(WEIGHTS))  # normalize edilmiş max uzaklık

# -----------------------------
# Veri modelleri
# -----------------------------
class Sector(BaseModel):
    categoryId: int
    optionIds: List[int]

class Profile(BaseModel):
    userId: str
    organizationId: int
    profileTypeId: int  # 1: Investor, 2: Entrepreneur
    sectors: List[Sector]

# -----------------------------
# Ana Eşleştirme Endpoint'i
# -----------------------------
@app.post("/match")
def match_profiles(profiles: List[Profile]) -> List[Dict[str, Any]]:
    expanded_profiles = []

    # Her sektörü ayrı bir profil gibi yay
    for profile in profiles:
        for sector in profile.sectors:
            expanded_profiles.append({
                "userId": profile.userId,
                "organizationId": profile.organizationId,
                "profileTypeId": profile.profileTypeId,
                "categoryId": sector.categoryId,
                "optionIds": sector.optionIds
            })

    # Profilleri ayır
    entrepreneurs = [p for p in expanded_profiles if p["profileTypeId"] == 2]
    investors = [p for p in expanded_profiles if p["profileTypeId"] == 1]

    # Yatırımcı vektörlerini hazırla
    inv_data = []
    for inv in investors:
        vec = normalize_options(inv["optionIds"])
        weighted_vec = vec * np.sqrt(WEIGHTS)
        inv_data.append((inv["userId"], inv["organizationId"], inv["categoryId"], weighted_vec))

    matches = []

    for ent in entrepreneurs:
        ent_vec = normalize_options(ent["optionIds"]) * np.sqrt(WEIGHTS)
        ent_category = ent["categoryId"]

        # Aynı organizasyondaki yatırımcılar
        candidates = [d for d in inv_data if d[1] == ent["organizationId"]]
        if not candidates:
            continue

        distances = []
        for user_id, _, inv_category, inv_vec in candidates:
            dist = np.linalg.norm(ent_vec - inv_vec)
            distances.append((user_id, inv_category, dist))

        # En yakın K yatırımcı
        distances.sort(key=lambda x: x[2])
        top_k = distances[:min(K, len(distances))]

        for user_id, inv_category, dist in top_k:
            norm_dist = dist / MAX_DISTANCE
            similarity = 1 - norm_dist
            score = similarity * 100

            # Farklı kategori ise %20 ceza
            if ent_category != inv_category:
                score *= 0.8

            score = round(score, 2)

            if score >= THRESHOLD:
                matches.append({
                    "score": score,
                    "userId": user_id,
                    "userId2": ent["userId"]
                })

    # Aynı eşleşmeden sadece en yüksek puanı sakla
    unique_matches = {}
    for match in matches:
        key = (match["userId"], match["userId2"])
        if key not in unique_matches or match["score"] > unique_matches[key]["score"]:
            unique_matches[key] = match

    return sorted(unique_matches.values(), key=lambda x: -x["score"])

# -----------------------------
# Yardımcı Fonksiyonlar
# -----------------------------
def normalize_options(option_ids: List[int]) -> np.ndarray:
    """
    optionIds içindeki değerleri doğru aralıkla eşleştirerek normalize eder.
    Her aralık için yalnızca bir değer alınır; yoksa 0.0 kalır.
    """
    result = np.zeros(len(RANGES), dtype=float)

    for value in option_ids:
        for i, (min_val, max_val) in enumerate(RANGES):
            if min_val <= value <= max_val:
                result[i] = normalize(value, min_val, max_val)
                break

    return result

def normalize(value: int, min_val: int, max_val: int) -> float:
    return (value - min_val) / (max_val - min_val)
