from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

# Ağırlıklar ve normalize aralıklar
WEIGHTS = np.array([14, 16, 24, 16, 17, 13])
RANGES = [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36)]
THRESHOLD = 85  # Skor eşiği yüzde olarak
K = 3  # Her girişimci için en iyi K eşleşme

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

    # Yatırımcı vektör matrisini oluştur ve ölçekle (weighted)
    inv_ids = []
    inv_orgs = []
    inv_vectors = []
    for inv in investors:
        inv_ids.append(inv.userId)
        inv_orgs.append(inv.organizationId)
        vec = normalize_options(inv.sectors[0].optionIds)
        # ağırlıklı uzaklık için sqrt(weights) ile ölçekle
        inv_vectors.append(vec * np.sqrt(WEIGHTS))
    inv_matrix = np.stack(inv_vectors)

    # NearestNeighbors modeli
    knn = NearestNeighbors(n_neighbors=K, metric='euclidean')
    knn.fit(inv_matrix)

    matches: List[Dict[str, Any]] = []
    for ent in entrepreneurs:
        # sadece aynı organizationId'ye bak
        # önce filtreleyip lokal matris kurabiliriz
        mask = [org == ent.organizationId for org in inv_orgs]
        if not any(mask):
            continue
        sub_ids = [inv_ids[i] for i, ok in enumerate(mask) if ok]
        sub_matrix = inv_matrix[mask]

        # girişimci vektörü
        ent_vec = normalize_options(ent.sectors[0].optionIds) * np.sqrt(WEIGHTS)

        # KNN sorgu
        distances, indices = NearestNeighbors(n_neighbors=min(K, len(sub_matrix)),
                                              metric='euclidean')\
                                  .fit(sub_matrix).kneighbors(ent_vec.reshape(1, -1))
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - dist
            score = round(similarity * 100, 2)
            if score >= THRESHOLD:
                matches.append({
                    "score": score,
                    "userId": sub_ids[idx],
                    "userId2": ent.userId
                })

    # en yüksek skorlu ilk K eşleşmeyi döndür
    matches.sort(key=lambda x: -x['score'])
    return matches


def normalize_options(option_ids: List[int]) -> np.ndarray:
    return np.array([
        normalize(opt, rng[0], rng[1])
        for opt, rng in zip(option_ids, RANGES)
    ])


def normalize(value: int, min_val: int, max_val: int) -> float:
    return (value - min_val) / (max_val - min_val)


def weighted_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # Bu fonksiyona gerek kalmadı, KNN metrikini doğrudan kullandık
    return np.sqrt(np.sum(((vec1 - vec2) ** 2) * WEIGHTS))
