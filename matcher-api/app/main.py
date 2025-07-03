from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np

app = FastAPI()

WEIGHTS = np.array([14, 16, 24, 16, 17, 13])
RANGES = [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36)]
THRESHOLD = 0.85
MAX_MATCHES = 6

class User(BaseModel):
    id: int
    type: int
    investmentAmount: int
    numberOfStartups: int
    investmentType: int
    projectStage: int
    pastInvestmentAmount: int
    numberOfEmployees: int

class MatchRequest(BaseModel):
    users: List[User]

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def get_normalized_vector(user: dict) -> np.ndarray:
    values = [
        user["investmentAmount"],
        user["numberOfStartups"],
        user["investmentType"],
        user["projectStage"],
        user["pastInvestmentAmount"],
        user["numberOfEmployees"]
    ]
    return np.array([normalize(val, r[0], r[1]) for val, r in zip(values, RANGES)])

def weighted_distance(user1: dict, user2: dict) -> float:
    vec1 = get_normalized_vector(user1)
    vec2 = get_normalized_vector(user2)
    return np.sqrt(np.sum(((vec1 - vec2) ** 2) * WEIGHTS))

@app.post("/match")
def match_users(match_request: MatchRequest):
    users = [u.dict() for u in match_request.users]
    investors = [u for u in users if u["type"] == 1]
    startups = [u for u in users if u["type"] == 2]

    candidate_pairs = []
    for inv in investors:
        for st in startups:
            score = round(1 - weighted_distance(inv, st), 4)
            if score >= THRESHOLD:
                candidate_pairs.append({
                    "investor_id": inv["id"],
                    "startup_id": st["id"],
                    "score": score
                })

    candidate_pairs.sort(key=lambda x: -x["score"])
    matches: Dict[int, Dict] = {}
    match_counts: Dict[int, int] = {u["id"]: 0 for u in users}

    for pair in candidate_pairs:
        i_id = pair["investor_id"]
        s_id = pair["startup_id"]
        score = pair["score"]

        if match_counts[i_id] >= MAX_MATCHES:
            continue
        if match_counts[s_id] >= MAX_MATCHES:
            continue

        if i_id not in matches:
            matches[i_id] = {"id": i_id, "type": 1, "matches": []}
        matches[i_id]["matches"].append({"id": s_id, "compatibilityScore": score})
        match_counts[i_id] += 1

        if s_id not in matches:
            matches[s_id] = {"id": s_id, "type": 2, "matches": []}
        matches[s_id]["matches"].append({"id": i_id, "compatibilityScore": score})
        match_counts[s_id] += 1

    investor_matches = [m for m in matches.values() if m["type"] == 1]
    startup_matches = [m for m in matches.values() if m["type"] == 2]

    return {
        "investorMatches": investor_matches,
        "startupMatches": startup_matches
    }
