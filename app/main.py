from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

app = FastAPI()

# Ağırlıklar ve normalize aralıklar
WEIGHTS = np.array([14, 16, 24, 16, 17, 13])
RANGES = [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36)]
THRESHOLD = 85  # Skor eşiği yüzde olarak
K = 3  # Her girişimci için en iyi K eşleşme sayısı

class Sector(BaseModel):
    categoryId: int
    optionIds: List[int]

class Profile(BaseModel):
    userId: str
    organizationId: int
    profileTypeId: int  # 1: yatırımcı, 2: girişimci
    sectors: List[Sector]

@app.post("/match")
def match_profiles(profiles: List[Profile]) -> List[Dict[str, Any]]:
    # Profil tiplerine göre ayır
    entrepreneurs = [p for p in profiles if p.profileTypeId == 2]
    investors = [p for p in profiles if p.profileTypeId == 1]

    # Yatırımcı temel vektörleri
    inv_data = []  # tuple(userId, organizationId, weighted_vec)
    for inv in investors:
        vec = normalize_options(inv.sectors[0].optionIds)
        weighted_vec = vec * np.sqrt(WEIGHTS)
        inv_data.append((inv.userId, inv.organizationId, weighted_vec))

    matches: List[Dict[str, Any]] = []
    for ent in entrepreneurs:
        # Aynı organizasyondaki yatırımcıları filtrele
        candidates = [d for d in inv_data if d[1] == ent.organizationId]
        if not candidates:
            continue

        # Girişimci vektörü
        ent_vec = normalize_options(ent.sectors[0].optionIds) * np.sqrt(WEIGHTS)

        # Mesafeleri hesapla
        distances = []  # tuple(userId, distance)
        for user_id, _, inv_vec in candidates:
            dist = np.linalg.norm(ent_vec - inv_vec)
            distances.append((user_id, dist))

        # En küçük K mesafeyi seç
        distances.sort(key=lambda x: x[1])
        top_k = distances[:min(K, len(distances))]

        # Eşleşmeleri ekle
        for user_id, dist in top_k:
            similarity = max(0.0, 1 - dist)
            score = round(similarity * 100, 2)
            if score >= THRESHOLD:
                matches.append({
                    "score": score,
                    "userId": user_id,   # Investor
                    "userId2": ent.userId  # Entrepreneur
                })

    matches.sort(key=lambda x: -x['score'])
    return matches


def normalize_options(option_ids: List[int]) -> np.ndarray:
    return np.array([
        normalize(opt, rng[0], rng[1])
        for opt, rng in zip(option_ids, RANGES)
    ])


def normalize(value: int, min_val: int, max_val: int) -> float:
    return (value - min_val) / (max_val - min_val)
```
