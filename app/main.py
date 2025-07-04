from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

app = FastAPI()

# Ağırlıklar ve normalize aralıklar
WEIGHTS = np.array([14, 16, 24, 16, 17, 13])
RANGES = [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36)]
THRESHOLD = 0.85

class SectorDetails(BaseModel):
    investmentAmount: str
    numberOfStartups: str
    investmentType: str
    projectStage: str
    pastInvestmentAmount: str
    numberOfEmployees: str

class Sector(BaseModel):
    id: int
    details: SectorDetails

class OrganizationProfile(BaseModel):
    userID: int       # kullanıcı ID
    type: int         # 1: girişimci, 2: yatırımcı
    sectors: List[Sector]

class MatchRequest(BaseModel):
    profiles: List[OrganizationProfile]

def normalize(value, min_val, max_val):
    return (float(value) - min_val) / (max_val - min_val)

def get_normalized_vector(details: SectorDetails) -> np.ndarray:
    values = [
        float(details.investmentAmount),
        float(details.numberOfStartups),
        float(details.investmentType),
        float(details.projectStage),
        float(details.pastInvestmentAmount),
        float(details.numberOfEmployees),
    ]
    return np.array([normalize(val, r[0], r[1]) for val, r in zip(values, RANGES)])

def weighted_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.sqrt(np.sum(((vec1 - vec2) ** 2) * WEIGHTS))

@app.post("/match")
def match_profiles(request: MatchRequest):
    entrepreneurs = [p for p in request.profiles if p.type == 1]
    investors = [p for p in request.profiles if p.type == 2]

    matches = []

    for e in entrepreneurs:
        e_vec = get_normalized_vector(e.sectors[0].details)
        for i in investors:
            i_vec = get_normalized_vector(i.sectors[0].details)
            score = round(1 - weighted_distance(e_vec, i_vec), 4)

            if score >= THRESHOLD:
                matches.append({
                    "entrepreneurID": e.userID,
                    "investorID": i.userID,
                    "score": score
                })

    # Skora göre sırala, en yüksek skorlu eşleşmeler en üstte
    matches.sort(key=lambda x: -x["score"])

    return matches

