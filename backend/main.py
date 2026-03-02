"""
MLB Hockey Scanner - Over/Under Runs Prediction
Based on NHL scanner pattern
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import uvicorn

app = FastAPI(title="MLB Analytics API")

DATA_PATH = Path(__file__).parent / "data"

feature_cols = ['home_runs_scored', 'home_runs_allowed', 'home_era',
                'away_runs_scored', 'away_runs_allowed', 'away_era',
                'home_whip', 'away_whip', 'home_so', 'away_so']

df = None

def load_data():
    global df
    all_games = []
    
    for year in range(2018, 2026):
        batting_file = DATA_PATH / f"{year}_batting.csv"
        pitching_file = DATA_PATH / f"{year}_pitching.csv"
        
        if batting_file.exists() and pitching_file.exists():
            batting = pd.read_csv(batting_file)
            pitching = pd.read_csv(pitching_file)
            
            teams = batting['Tm'].tolist()
            
            print(f"Loaded {year}: {len(teams)} teams")
    
    print(f"Total data loaded")

class SimulationRequest(BaseModel):
    train_season_start: int
    train_season_end: int
    test_season_start: int
    test_season_end: int
    confidence_threshold: float = 70
    bet_amount: float = 100
    model_version: str = "v1"

class ScheduleRequest(BaseModel):
    days_ahead: int = 30
    threshold: float = 70
    model_version: str = "v1"

@app.on_event("startup")
async def startup():
    load_data()

@app.get("/")
async def root():
    return {"message": "MLB Analytics API v1.0"}

@app.get("/models")
async def get_models():
    return {"models": ["v1"]}

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return {
        "message": "MLB Backtesting - coming soon",
        "note": "Need game results data first"
    }

@app.post("/schedule")
async def schedule(request: ScheduleRequest):
    return {
        "games": [],
        "message": "MLB Schedule - coming soon"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
