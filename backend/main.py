from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from datetime import datetime

app = FastAPI(title="MLB Scanner API")

frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")

@app.get("/")
async def root():
    return FileResponse(frontend_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEAM_FULL_NAME = {
    'Arizona Diamondbacks': 'AZ', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU', 'Kansas City Royals': 'KC', 'Los Angeles Dodgers': 'LAD',
    'Washington Nationals': 'WSH', 'New York Mets': 'NYM', 'Oakland Athletics': 'ATH',
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD', 'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SF', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Minnesota Twins': 'MIN',
    'Philadelphia Phillies': 'PHI', 'Chicago White Sox': 'CWS', 'Miami Marlins': 'MIA',
    'New York Yankees': 'NYY', 'Milwaukee Brewers': 'MIL', 'Los Angeles Angels': 'LAA',
}

def load_batting_stats(year):
    try:
        df = pd.read_csv(f'../data/{year}_batting.csv')
        df.columns = df.columns.str.strip()
        team_col = df.columns[0]
        
        stats = {}
        for _, row in df.iterrows():
            full_name = row[team_col]
            team = TEAM_FULL_NAME.get(full_name)
            if team:
                stats[team] = {
                    'bat_r': row.get('R', 0),
                    'bat_h': row.get('H', 0),
                    'bat_hr': row.get('HR', 0),
                    'bat_bb': row.get('BB', 0),
                    'bat_so': row.get('SO', 0),
                    'bat_ba': row.get('BA', 0),
                    'bat_obp': row.get('OBP', 0),
                    'bat_slg': row.get('SLG', 0),
                    'bat_ops': row.get('OPS', 0),
                    'bat_g': row.get('G', 162),
                }
        return stats
    except:
        return {}

def load_pitching_stats(year):
    try:
        df = pd.read_csv(f'../data/{year}_pitching.csv')
        df.columns = df.columns.str.strip()
        team_col = df.columns[0]
        
        stats = {}
        for _, row in df.iterrows():
            full_name = row[team_col]
            team = TEAM_FULL_NAME.get(full_name)
            if team:
                stats[team] = {
                    'pit_era': row.get('ERA', 0),
                    'pit_w': row.get('W', 0),
                    'pit_l': row.get('L', 0),
                    'pit_ip': row.get('IP', 0),
                    'pit_h': row.get('H', 0),
                    'pit_r': row.get('R', 0),
                    'pit_hr': row.get('HR', 0),
                    'pit_bb': row.get('BB', 0),
                    'pit_so': row.get('SO', 0),
                    'pit_whip': row.get('WHIP', 0),
                    'pit_fip': row.get('FIP', 0),
                    'pit_g': row.get('G', 162),
                }
        return stats
    except:
        return {}

def load_results(year):
    try:
        return pd.read_csv(f'../data/{year}_results.csv')
    except:
        return pd.DataFrame()

def calculate_rolling_stats(results_df, team, date, window=10):
    team_games = results_df[(results_df['home_team'] == team) | (results_df['away_team'] == team)]
    team_games = team_games[team_games['date'] < date].tail(window)
    
    if len(team_games) == 0:
        return {
            'recent_runs_scored': 0,
            'recent_runs_allowed': 0,
            'recent_total': 0,
            'recent_wins': 0,
            'recent_games': 0
        }
    
    runs_scored = 0
    runs_allowed = 0
    wins = 0
    
    for _, game in team_games.iterrows():
        if game['home_team'] == team:
            runs_scored += game['home_score']
            runs_allowed += game['away_score']
            if game['home_score'] > game['away_score']:
                wins += 1
        else:
            runs_scored += game['away_score']
            runs_allowed += game['home_score']
            if game['away_score'] > game['home_score']:
                wins += 1
    
    return {
        'recent_runs_scored': runs_scored / len(team_games),
        'recent_runs_allowed': runs_allowed / len(team_games),
        'recent_total': (runs_scored + runs_allowed) / len(team_games),
        'recent_wins': wins / len(team_games),
        'recent_games': len(team_games)
    }

features = [
    'home_bat_ba', 'home_bat_ops', 'home_bat_obp', 'home_bat_slg', 'home_bat_r_g', 'home_bat_hr_g', 'home_bat_bb_g', 'home_bat_so_g',
    'away_bat_ba', 'away_bat_ops', 'away_bat_obp', 'away_bat_slg', 'away_bat_r_g', 'away_bat_hr_g', 'away_bat_bb_g', 'away_bat_so_g',
    'home_pit_era', 'home_pit_whip', 'home_pit_fip', 'home_pit_r_g', 'home_pit_so_g', 'home_pit_bb_g',
    'away_pit_era', 'away_pit_whip', 'away_pit_fip', 'away_pit_r_g', 'away_pit_so_g', 'away_pit_bb_g',
    'bat_avg_diff', 'ops_diff', 'obp_diff', 'slg_diff', 'era_diff', 'whip_diff', 'fip_diff',
    'expected_runs', 'expected_era', 'run_diff',
]

model = None

def train_model(train_years, over_under_line=7.5):
    global model
    
    train_years = list(range(train_years[0], train_years[1] + 1))
    
    all_games = []
    
    for year in train_years:
        results = load_results(year)
        if results.empty:
            continue
            
        batting = load_batting_stats(year)
        pitching = load_pitching_stats(year)
        
        for _, game in results.iterrows():
            home = game['home_team']
            away = game['away_team']
            
            if home not in batting or home not in pitching:
                continue
            if away not in batting or away not in pitching:
                continue
            
            home_bat = batting[home]
            away_bat = batting[away]
            home_pit = pitching[home]
            away_pit = pitching[away]
            
            game_data = {
                'is_over': 1 if game['total_runs'] > over_under_line else 0,
                'home_bat_ba': home_bat['bat_ba'],
                'home_bat_ops': home_bat['bat_ops'],
                'home_bat_obp': home_bat['bat_obp'],
                'home_bat_slg': home_bat['bat_slg'],
                'home_bat_r_g': home_bat['bat_r'] / home_bat['bat_g'],
                'home_bat_hr_g': home_bat['bat_hr'] / home_bat['bat_g'],
                'home_bat_bb_g': home_bat['bat_bb'] / home_bat['bat_g'],
                'home_bat_so_g': home_bat['bat_so'] / home_bat['bat_g'],
                'away_bat_ba': away_bat['bat_ba'],
                'away_bat_ops': away_bat['bat_ops'],
                'away_bat_obp': away_bat['bat_obp'],
                'away_bat_slg': away_bat['bat_slg'],
                'away_bat_r_g': away_bat['bat_r'] / away_bat['bat_g'],
                'away_bat_hr_g': away_bat['bat_hr'] / away_bat['bat_g'],
                'away_bat_bb_g': away_bat['bat_bb'] / away_bat['bat_g'],
                'away_bat_so_g': away_bat['bat_so'] / away_bat['bat_g'],
                'home_pit_era': home_pit['pit_era'],
                'home_pit_whip': home_pit['pit_whip'],
                'home_pit_fip': home_pit['pit_fip'],
                'home_pit_r_g': home_pit['pit_r'] / home_pit['pit_g'],
                'home_pit_so_g': home_pit['pit_so'] / home_pit['pit_g'],
                'home_pit_bb_g': home_pit['pit_bb'] / home_pit['pit_g'],
                'away_pit_era': away_pit['pit_era'],
                'away_pit_whip': away_pit['pit_whip'],
                'away_pit_fip': away_pit['pit_fip'],
                'away_pit_r_g': away_pit['pit_r'] / away_pit['pit_g'],
                'away_pit_so_g': away_pit['pit_so'] / away_pit['pit_g'],
                'away_pit_bb_g': away_pit['pit_bb'] / away_pit['pit_g'],
                'bat_avg_diff': home_bat['bat_ba'] - away_bat['bat_ba'],
                'ops_diff': home_bat['bat_ops'] - away_bat['bat_ops'],
                'obp_diff': home_bat['bat_obp'] - away_bat['bat_obp'],
                'slg_diff': home_bat['bat_slg'] - away_bat['bat_slg'],
                'era_diff': home_pit['pit_era'] - away_pit['pit_era'],
                'whip_diff': home_pit['pit_whip'] - away_pit['pit_whip'],
                'fip_diff': home_pit['pit_fip'] - away_pit['pit_fip'],
                'expected_runs': (home_bat['bat_r'] / home_bat['bat_g'] + away_bat['bat_r'] / away_bat['bat_g']) / 2,
                'expected_era': (home_pit['pit_era'] + away_pit['pit_era']) / 2,
                'run_diff': (home_bat['bat_r'] / home_bat['bat_g'] + away_pit['pit_r'] / away_pit['pit_g']) / 2 - (away_bat['bat_r'] / away_bat['bat_g'] + home_pit['pit_r'] / home_pit['pit_g']) / 2,
            }
            
            all_games.append(game_data)
    
    df = pd.DataFrame(all_games)
    
    X_train = df[features]
    y_train = df['is_over']
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    return model

class SimulationRequest(BaseModel):
    train_season_start: int = 2018
    train_season_end: int = 2024
    test_season_start: int = 2025
    test_season_end: int = 2025
    over_under_line: float = 6.5
    confidence_threshold: float = 50.0

class GamePredict(BaseModel):
    date: str
    home_team: str
    away_team: str

class PredictRequest(BaseModel):
    games: list[GamePredict]
    stats_season: int = 2025
    over_under_line: float = 6.5
    threshold: float = 70.0

@app.get("/")
def root():
    return {"message": "MLB Scanner API - Over/Under Predictions"}

@app.get("/teams")
def get_teams():
    return {"teams": list(TEAM_FULL_NAME.values())}

@app.get("/seasons")
def get_seasons():
    return {"seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]}

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    global model
    
    over_under_line = request.over_under_line
    
    model = train_model([request.train_season_start, request.train_season_end], over_under_line)
    
    test_years = list(range(request.test_season_start, request.test_season_end + 1))
    
    test_games = []
    for year in test_years:
        results = load_results(year)
        if results.empty:
            continue
        
        results = results[results['date'] >= f'{year}-04-01']
            
        batting = load_batting_stats(year)
        pitching = load_pitching_stats(year)
        
        for _, game in results.iterrows():
            home = game['home_team']
            away = game['away_team']
            
            if home not in batting or home not in pitching:
                continue
            if away not in batting or away not in pitching:
                continue
            
            home_bat = batting[home]
            away_bat = batting[away]
            home_pit = pitching[home]
            away_pit = pitching[away]
            
            game_data = {
                'date': game['date'],
                'home_team': home,
                'away_team': away,
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'total_runs': game['total_runs'],
                'is_over': 1 if game['total_runs'] > over_under_line else 0,
                'home_bat_ba': home_bat['bat_ba'],
                'home_bat_ops': home_bat['bat_ops'],
                'home_bat_obp': home_bat['bat_obp'],
                'home_bat_slg': home_bat['bat_slg'],
                'home_bat_r_g': home_bat['bat_r'] / home_bat['bat_g'],
                'home_bat_hr_g': home_bat['bat_hr'] / home_bat['bat_g'],
                'home_bat_bb_g': home_bat['bat_bb'] / home_bat['bat_g'],
                'home_bat_so_g': home_bat['bat_so'] / home_bat['bat_g'],
                'away_bat_ba': away_bat['bat_ba'],
                'away_bat_ops': away_bat['bat_ops'],
                'away_bat_obp': away_bat['bat_obp'],
                'away_bat_slg': away_bat['bat_slg'],
                'away_bat_r_g': away_bat['bat_r'] / away_bat['bat_g'],
                'away_bat_hr_g': away_bat['bat_hr'] / away_bat['bat_g'],
                'away_bat_bb_g': away_bat['bat_bb'] / away_bat['bat_g'],
                'away_bat_so_g': away_bat['bat_so'] / away_bat['bat_g'],
                'home_pit_era': home_pit['pit_era'],
                'home_pit_whip': home_pit['pit_whip'],
                'home_pit_fip': home_pit['pit_fip'],
                'home_pit_r_g': home_pit['pit_r'] / home_pit['pit_g'],
                'home_pit_so_g': home_pit['pit_so'] / home_pit['pit_g'],
                'home_pit_bb_g': home_pit['pit_bb'] / home_pit['pit_g'],
                'away_pit_era': away_pit['pit_era'],
                'away_pit_whip': away_pit['pit_whip'],
                'away_pit_fip': away_pit['pit_fip'],
                'away_pit_r_g': away_pit['pit_r'] / away_pit['pit_g'],
                'away_pit_so_g': away_pit['pit_so'] / away_pit['pit_g'],
                'away_pit_bb_g': away_pit['pit_bb'] / away_pit['pit_g'],
                'bat_avg_diff': home_bat['bat_ba'] - away_bat['bat_ba'],
                'ops_diff': home_bat['bat_ops'] - away_bat['bat_ops'],
                'obp_diff': home_bat['bat_obp'] - away_bat['bat_obp'],
                'slg_diff': home_bat['bat_slg'] - away_bat['bat_slg'],
                'era_diff': home_pit['pit_era'] - away_pit['pit_era'],
                'whip_diff': home_pit['pit_whip'] - away_pit['pit_whip'],
                'fip_diff': home_pit['pit_fip'] - away_pit['pit_fip'],
                'expected_runs': (home_bat['bat_r']/home_bat['bat_g'] + away_bat['bat_r']/away_bat['bat_g']) / 2,
                'expected_era': (home_pit['pit_era'] + away_pit['pit_era']) / 2,
                'run_diff': (home_bat['bat_r']/home_bat['bat_g'] + away_pit['pit_r']/away_pit['pit_g'])/2 - (away_bat['bat_r']/away_bat['bat_g'] + home_pit['pit_r']/home_pit['pit_g'])/2,
            }
            
            test_games.append(game_data)
    
    if not test_games:
        raise HTTPException(status_code=400, detail="Brak danych testowych")
    
    test_df = pd.DataFrame(test_games)
    
    X_test = test_df[features]
    prob = model.predict_proba(X_test)[:, 1]
    
    test_df['predicted_prob'] = prob
    
    threshold = request.confidence_threshold / 100
    
    qualifying = test_df[test_df['predicted_prob'] >= threshold].copy()
    
    if len(qualifying) == 0:
        return {
            "total_profit": 0.0,
            "roi_percent": 0.0,
            "hit_rate": 0.0,
            "total_bets": 0,
            "hits": 0,
            "games": []
        }
    
    hits = int(qualifying['is_over'].sum())
    total_bets = len(qualifying)
    hit_rate = hits / total_bets
    
    profit = int(hits - (total_bets - hits))
    roi = profit / total_bets * 100
    
    games = []
    for _, g in qualifying.iterrows():
        games.append({
            "date": g['date'],
            "home_team": g['home_team'],
            "away_team": g['away_team'],
            "home_score": int(g['home_score']),
            "away_score": int(g['away_score']),
            "total_runs": int(g['total_runs']),
            "over_line": over_under_line,
            "actual_over": bool(g['is_over']),
            "predicted_prob": round(g['predicted_prob'] * 100, 1)
        })
    
    return {
        "total_profit": profit,
        "roi_percent": round(roi, 1),
        "hit_rate": round(hit_rate * 100, 1),
        "total_bets": total_bets,
        "hits": hits,
        "over_under_line": over_under_line,
        "confidence_threshold": request.confidence_threshold,
        "games": games
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    batting = load_batting_stats(request.stats_season)
    pitching = load_pitching_stats(request.stats_season)
    
    if not batting or not pitching:
        raise HTTPException(status_code=400, detail=f"Brak statystyk dla sezonu {request.stats_season}")
    
    train_years = list(range(2018, request.stats_season + 1))
    model = train_model(train_years, request.over_under_line)
    
    predict_games = []
    for game in request.games:
        home, away = game.home_team, game.away_team
        
        if home not in batting or away not in batting or home not in pitching or away not in pitching:
            continue
        
        hb, ab = batting[home], batting[away]
        hp, ap = pitching[home], pitching[away]
        
        game_data = {
            'home_bat_ba': hb['bat_ba'],
            'home_bat_ops': hb['bat_ops'],
            'home_bat_obp': hb['bat_obp'],
            'home_bat_slg': hb['bat_slg'],
            'home_bat_r_g': hb['bat_r'] / hb['bat_g'],
            'home_bat_hr_g': hb['bat_hr'] / hb['bat_g'],
            'home_bat_bb_g': hb['bat_bb'] / hb['bat_g'],
            'home_bat_so_g': hb['bat_so'] / hb['bat_g'],
            'away_bat_ba': ab['bat_ba'],
            'away_bat_ops': ab['bat_ops'],
            'away_bat_obp': ab['bat_obp'],
            'away_bat_slg': ab['bat_slg'],
            'away_bat_r_g': ab['bat_r'] / ab['bat_g'],
            'away_bat_hr_g': ab['bat_hr'] / ab['bat_g'],
            'away_bat_bb_g': ab['bat_bb'] / ab['bat_g'],
            'away_bat_so_g': ab['bat_so'] / ab['bat_g'],
            'home_pit_era': hp['pit_era'],
            'home_pit_whip': hp['pit_whip'],
            'home_pit_fip': hp['pit_fip'],
            'home_pit_r_g': hp['pit_r'] / hp['pit_g'],
            'home_pit_so_g': hp['pit_so'] / hp['pit_g'],
            'home_pit_bb_g': hp['pit_bb'] / hp['pit_g'],
            'away_pit_era': ap['pit_era'],
            'away_pit_whip': ap['pit_whip'],
            'away_pit_fip': ap['pit_fip'],
            'away_pit_r_g': ap['pit_r'] / ap['pit_g'],
            'away_pit_so_g': ap['pit_so'] / ap['pit_g'],
            'away_pit_bb_g': ap['pit_bb'] / ap['pit_g'],
            'bat_avg_diff': hb['bat_ba'] - ab['bat_ba'],
            'ops_diff': hb['bat_ops'] - ab['bat_ops'],
            'obp_diff': hb['bat_obp'] - ab['bat_obp'],
            'slg_diff': hb['bat_slg'] - ab['bat_slg'],
            'era_diff': hp['pit_era'] - ap['pit_era'],
            'whip_diff': hp['pit_whip'] - ap['pit_whip'],
            'fip_diff': hp['pit_fip'] - ap['pit_fip'],
            'expected_runs': (hb['bat_r']/hb['bat_g'] + ab['bat_r']/ab['bat_g']) / 2,
            'expected_era': (hp['pit_era'] + ap['pit_era']) / 2,
            'run_diff': (hb['bat_r']/hb['bat_g'] + ap['pit_r']/ap['pit_g'])/2 - (ab['bat_r']/ab['bat_g'] + hp['pit_r']/hp['pit_g'])/2,
        }
        game_data['date'] = game.date
        game_data['home_team'] = home
        game_data['away_team'] = away
        predict_games.append(game_data)
    
    if not predict_games:
        raise HTTPException(status_code=400, detail="Brak pasujących drużyn")
    
    pred_df = pd.DataFrame(predict_games)
    X_pred = pred_df[features]
    probs = model.predict_proba(X_pred)[:, 1]
    
    threshold = request.threshold / 100
    
    results = []
    qualifying_count = 0
    for i, game in enumerate(predict_games):
        prob = float(probs[i])
        if prob >= threshold:
            qualifying_count += 1
            results.append({
                "date": game['date'],
                "home_team": game['home_team'],
                "away_team": game['away_team'],
                "probability": round(prob * 100, 1),
                "bet": "OVER " + str(request.over_under_line)
            })
    
    return {
        "total_games": len(request.games),
        "qualifying_games": qualifying_count,
        "threshold": request.threshold,
        "over_under_line": request.over_under_line,
        "stats_season": request.stats_season,
        "predictions": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
