from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()

# Ağırlıklar ve normalize aralıklar
WEIGHTS = np.array([14, 16, 24, 16, 17, 13])
RANGES = [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36)]
THRESHOLD = 0.85

class Sector(BaseModel):
    categoryId: int
    optionIds: List[int]

class Profile(BaseModel):
    userId: str
    organizationId: int
    profileTypeId: int  # 1: girişimci, 2: yatırımcı
    sectors: List[Sector]

@app.post("/match")
def match_profiles(profiles: List[Profile]):
    entrepreneurs = [p for p in profiles if p.profileTypeId == 2]
    investors = [p for p in profiles if p.profileTypeId == 1]

    matches = []

    for e in entrepreneurs:
        for i in investors:
            if e.organizationId != i.organizationId:
                continue  # Aynı organizasyon değilse geç

            # Normalize vektörleri hazırla
            e_vec = normalize_options(e.sectors[0].optionIds)
            i_vec = normalize_options(i.sectors[0].optionIds)

            score = round(1 - weighted_distance(e_vec, i_vec), 4)
            if score >= THRESHOLD:
                matches.append({
                    "score": score,
                    "userId": i.userId,   # Investor
                    "userId2": e.userId   # Entrepreneur
                })

    matches.sort(key=lambda x: -x["score"])
    return matches

def normalize_options(option_ids: List[int]) -> np.ndarray:
    return np.array([
        normalize(opt, rng[0], rng[1])
        for opt, rng in zip(option_ids, RANGES)
    ])

def normalize(value: int, min_val: int, max_val: int) -> float:
    return (value - min_val) / (max_val - min_val)

def weighted_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.sqrt(np.sum(((vec1 - vec2) ** 2) * WEIGHTS))
