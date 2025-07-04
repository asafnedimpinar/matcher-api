from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

app = FastAPI()

WEIGHTS = np.array([14, 16, 24, 16, 17, 13])
RANGES = [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36)]
THRESHOLD = 0.85
MAX_MATCHES = 6

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
    userID: str  # "1" = girişimci, "2" = yatırımcı
    preference: int
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
    entrepreneurs = [p for p in request.profiles if p.userID == "1"]
    investors = [p for p in request.profiles if p.userID == "2"]

    entrepreneur_match_map: Dict[int, List[Dict[str, Any]]] = {}
    investor_match_map: Dict[int, List[Dict[str, Any]]] = {}

    for e in entrepreneurs:
        e_vec = get_normalized_vector(e.sectors[0].details)
        for i in investors:
            i_vec = get_normalized_vector(i.sectors[0].details)
            score = round(1 - weighted_distance(e_vec, i_vec), 4)

            if score >= THRESHOLD:
                entrepreneur_match_map.setdefault(id(e), []).append({
                    "id": id(i),
                    "compatibilityScore": score
                })
                investor_match_map.setdefault(id(i), []).append({
                    "id": id(e),
                    "compatibilityScore": score
                })

    entrepreneur_result = [
        {
            "id": id(e),
            "type": 1,
            "matches": entrepreneur_match_map.get(id(e), [])
        }
        for e in entrepreneurs
    ]

    investor_result = [
        {
            "id": id(i),
            "type": 2,
            "matches": investor_match_map.get(id(i), [])
        }
        for i in investors
    ]

    return {
        "entrepreneurMatches": entrepreneur_result,
        "investorMatches": investor_result
    }
