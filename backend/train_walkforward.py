#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import os
import time

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

def load_batting(year):
    try:
        df = pd.read_csv(f'../data/{year}_batting.csv')
        df.columns = df.columns.str.strip()
        team_col = df.columns[0]
        stats = {}
        for _, row in df.iterrows():
            team = TEAM_FULL_NAME.get(row[team_col])
            if team:
                stats[team] = {
                    'bat_r': row.get('R', 0), 'bat_hr': row.get('HR', 0),
                    'bat_ba': row.get('BA', 0), 'bat_ops': row.get('OPS', 0),
                    'bat_g': row.get('G', 162)
                }
        return stats
    except: return {}

def load_pitching(year):
    try:
        df = pd.read_csv(f'../data/{year}_pitching.csv')
        df.columns = df.columns.str.strip()
        team_col = df.columns[0]
        stats = {}
        for _, row in df.iterrows():
            team = TEAM_FULL_NAME.get(row[team_col])
            if team:
                stats[team] = {
                    'pit_era': row.get('ERA', 0), 'pit_whip': row.get('WHIP', 0),
                    'pit_r': row.get('R', 0), 'pit_g': row.get('G', 162)
                }
        return stats
    except: return {}

def load_results(year):
    try: return pd.read_csv(f'../data/{year}_results.csv')
    except: return pd.DataFrame()

_rolling_cache = {}

def prepare_data(years, ou=6.5):
    all_games = []
    for year in years:
        results = load_results(year)
        if results.empty: continue
        batting = load_batting(year)
        pitching = load_pitching(year)
        for _, game in results.iterrows():
            home, away = game['home_team'], game['away_team']
            if home not in batting or away not in batting or home not in pitching or away not in pitching: continue
            hb, ab = batting[home], batting[away]
            hp, ap = pitching[home], pitching[away]
            all_games.append({
                'year': year, 'date': game['date'], 'home_team': home, 'away_team': away, 
                'home_score': game['home_score'], 'away_score': game['away_score'], 'total_runs': game['total_runs'],
                'is_over': 1 if game['total_runs'] > ou else 0,
                'home_bat_ba': hb['bat_ba'], 'home_bat_ops': hb['bat_ops'], 'home_bat_r_g': hb['bat_r']/hb['bat_g'], 'home_bat_hr_g': hb['bat_hr']/hb['bat_g'],
                'away_bat_ba': ab['bat_ba'], 'away_bat_ops': ab['bat_ops'], 'away_bat_r_g': ab['bat_r']/ab['bat_g'], 'away_bat_hr_g': ab['bat_hr']/ab['bat_g'],
                'home_pit_era': hp['pit_era'], 'home_pit_whip': hp['pit_whip'], 'home_pit_r_g': hp['pit_r']/hp['pit_g'],
                'away_pit_era': ap['pit_era'], 'away_pit_whip': ap['pit_whip'], 'away_pit_r_g': ap['pit_r']/ap['pit_g'],
                'bat_avg_diff': hb['bat_ba']-ab['bat_ba'], 'ops_diff': hb['bat_ops']-ab['bat_ops'],
                'era_diff': hp['pit_era']-ap['pit_era'], 'whip_diff': hp['pit_whip']-ap['pit_whip'],
                'expected_runs': (hb['bat_r']/hb['bat_g']+ab['bat_r']/ab['bat_g'])/2,
                'expected_era': (hp['pit_era']+ap['pit_era'])/2,
            })
    return pd.DataFrame(all_games)

FEATURES = ['home_bat_ba', 'home_bat_ops', 'home_bat_r_g', 'home_bat_hr_g', 
            'away_bat_ba', 'away_bat_ops', 'away_bat_r_g', 'away_bat_hr_g', 
            'home_pit_era', 'home_pit_whip', 'home_pit_r_g', 
            'away_pit_era', 'away_pit_whip', 'away_pit_r_g', 
            'bat_avg_diff', 'ops_diff', 'era_diff', 'whip_diff', 
            'expected_runs', 'expected_era']

OVER_UNDER_LINE = 6.5
ODDS = 1.5
STAKE = 100
WIN_PROFIT = int(STAKE * ODDS - STAKE)

print(f"=== MLB Walk-Forward Validation | Over {OVER_UNDER_LINE} | Kurs: {ODDS} | Stawka: {STAKE} ===\n")
break_even = 1 / ODDS
print(f"Break-even: {break_even:.1%}\n")

ALL_YEARS = list(range(2018, 2026))
WALK_FORWARD_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

results_summary = []

for test_year in WALK_FORWARD_YEARS:
    train_years = [y for y in ALL_YEARS if y < test_year]
    
    if len(train_years) < 2:
        continue
    
    train_df = prepare_data(train_years, OVER_UNDER_LINE)
    test_df = prepare_data([test_year], OVER_UNDER_LINE)
    
    if len(train_df) < 100 or len(test_df) < 10:
        continue
    
    X_train, y_train = train_df[FEATURES], train_df['is_over']
    X_test, y_test = test_df[FEATURES], test_df['is_over']
    
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    prob = model.predict_proba(X_test)[:, 1]
    
    print(f"=== {test_year} | Train: {len(train_df)} | Test: {len(test_df)} | OVER rate: {y_test.mean():.1%} ===")
    
    best_threshold = None
    best_roi = -float('inf')
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        bet_mask = prob >= threshold
        bet_count = bet_mask.sum()
        if bet_count < 5:
            continue
        
        hits = int(y_test.values[bet_mask].sum())
        miss = bet_count - hits
        profit = hits * WIN_PROFIT - miss * STAKE
        roi = profit / (bet_count * STAKE) * 100
        
        if roi > best_roi:
            best_roi = roi
            best_threshold = threshold
    
    if best_threshold:
        bet_mask = prob >= best_threshold
        bet_count = bet_mask.sum()
        hits = int(y_test.values[bet_mask].sum())
        miss = bet_count - hits
        profit = hits * WIN_PROFIT - miss * STAKE
        roi = profit / (bet_count * STAKE) * 100
        
        print(f"  Best: {best_threshold:.0%} → {bet_count} zak | {hits} HIT ({hits/bet_count:.1%}) | Zysk: {profit} PLN | ROI: {roi:+.1f}%")
        
        results_summary.append({
            'year': test_year,
            'games': len(test_df),
            'over_rate': y_test.mean(),
            'threshold': best_threshold,
            'bets': bet_count,
            'hits': hits,
            'hit_rate': hits/bet_count,
            'profit': profit,
            'roi': roi
        })
    else:
        print(f"  Brak wystarczających zakładów")
        results_summary.append({
            'year': test_year, 'games': len(test_df), 'over_rate': y_test.mean(),
            'threshold': None, 'bets': 0, 'hits': 0, 'hit_rate': 0, 'profit': 0, 'roi': 0
        })

print("\n" + "="*60)
print("=== PODSUMOWANIE WSZYSTKICH LAT ===")
print("="*60)

df_summary = pd.DataFrame(results_summary)
print(df_summary.to_string(index=False))

total_profit = df_summary['profit'].sum()
total_bets = df_summary['bets'].sum()
total_hits = df_summary['hits'].sum()

if total_bets > 0:
    total_roi = total_profit / (total_bets * STAKE) * 100
    print(f"\nŁĄCZNE: {total_bets} zakładów | {total_hits} trafień ({total_hits/total_bets:.1%})")
    print(f"ZYSK: {total_profit} PLN | ROI: {total_roi:+.1f}%")

    if total_roi > 0:
        print("\n✅ MODEL JEST PROFITOWALNY!")
    else:
        print("\n❌ MODEL NIE JEST PROFITOWALNY")
