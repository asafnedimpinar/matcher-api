from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

app = FastAPI()

WEIGHTS = np.array([14, 16, 24, 16, 17, 13])
RANGES = [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36)]
THRESHOLD = 60  # Eşik puanı
K = 6  # En iyi K eşleşme

MAX_DISTANCE = np.sqrt(np.sum(WEIGHTS))  # normalize edilmiş maksimum uzaklık

class Sector(BaseModel):
    categoryId: int
    optionIds: List[int]

class Profile(BaseModel):
    userId: str
    organizationId: int
    profileTypeId: int
    sectors: List[Sector]

@app.post("/match")
def match_profiles(profiles: List[Profile]) -> List[Dict[str, Any]]:
    expanded_profiles = []

    # Her sektörü ayrı profil gibi işliyoruz (userId aynı olabilir)
    for profile in profiles:
        for sector in profile.sectors:
            expanded_profiles.append({
                "userId": profile.userId,
                "organizationId": profile.organizationId,
                "profileTypeId": profile.profileTypeId,
                "categoryId": sector.categoryId,
                "optionIds": sector.optionIds
            })

    # Girişimcileri ve yatırımcıları ayır
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

        # Aynı organizasyondaki yatırımcıları filtrele
        candidates = [d for d in inv_data if d[1] == ent["organizationId"]]
        if not candidates:
            continue

        distances = []
        for user_id, _, category_id, inv_vec in candidates:
            dist = np.linalg.norm(ent_vec - inv_vec)
            distances.append((user_id, category_id, dist))

        distances.sort(key=lambda x: x[2])  # en yakınları seç
        top_k = distances[:min(K, len(distances))]

        for user_id, category_id, dist in top_k:
            norm_dist = dist / MAX_DISTANCE
            similarity = 1 - norm_dist
            score = similarity * 100

            # Kategori farkı varsa %20 düşür
            if ent["categoryId"] != category_id:
                score *= 0.8

            score = round(score, 2)

            if score >= THRESHOLD:
                matches.append({
                    "score": score,
                    "userId": user_id,
                    "userId2": ent["userId"]
                })

    matches.sort(key=lambda x: -x['score'])  # skora göre sırala
    return matches

def normalize_options(option_ids: List[int]) -> np.ndarray:
    return np.array([
        normalize(opt, rng[0], rng[1])
        for opt, rng in zip(option_ids, RANGES)
    ])

def normalize(value: int, min_val: int, max_val: int) -> float:
    return (value - min_val) / (max_val - min_val)
