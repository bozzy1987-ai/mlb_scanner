#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import os
import sys

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
            team = TEAM_FULL_NAME.get(row[team_col])
            if team:
                stats[team] = {'bat_r': row.get('R', 0), 'bat_hr': row.get('HR', 0), 'bat_ba': row.get('BA', 0), 'bat_ops': row.get('OPS', 0), 'bat_g': row.get('G', 162)}
        return stats
    except: return {}

def load_pitching_stats(year):
    try:
        df = pd.read_csv(f'../data/{year}_pitching.csv')
        df.columns = df.columns.str.strip()
        team_col = df.columns[0]
        stats = {}
        for _, row in df.iterrows():
            team = TEAM_FULL_NAME.get(row[team_col])
            if team:
                stats[team] = {'pit_era': row.get('ERA', 0), 'pit_whip': row.get('WHIP', 0), 'pit_r': row.get('R', 0), 'pit_g': row.get('G', 162)}
        return stats
    except: return {}

def load_results(year):
    try: return pd.read_csv(f'../data/{year}_results.csv')
    except: return pd.DataFrame()

def calc_rolling(results_df, team, date, window=10):
    team_games = results_df[(results_df['home_team'] == team) | (results_df['away_team'] == team)]
    team_games = team_games[team_games['date'] < date].tail(window)
    if len(team_games) == 0:
        return 0, 0, 0
    rs = ra = 0
    for _, g in team_games.iterrows():
        if g['home_team'] == team:
            rs += g['home_score']
            ra += g['away_score']
        else:
            rs += g['away_score']
            ra += g['home_score']
    return rs/len(team_games), ra/len(team_games), (rs+ra)/len(team_games)

def prepare_data(years, ou=7.5):
    all_games = []
    for year in years:
        results = load_results(year)
        if results.empty: continue
        batting = load_batting_stats(year)
        pitching = load_pitching_stats(year)
        for _, game in results.iterrows():
            home, away = game['home_team'], game['away_team']
            if home not in batting or away not in batting or home not in pitching or away not in pitching: continue
            hb, ab = batting[home], batting[away]
            hp, ap = pitching[home], pitching[away]
            rh, ra, rt = calc_rolling(results, home, game['date'])
            rh2, ra2, rt2 = calc_rolling(results, away, game['date'])
            all_games.append({
                'date': game['date'], 'home_team': home, 'away_team': away, 
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
                'home_recent_runs_scored': rh, 'home_recent_runs_allowed': ra,
                'away_recent_runs_scored': rh2, 'away_recent_runs_allowed': ra2,
                'home_recent_total': rt, 'away_recent_total': rt2,
            })
    return pd.DataFrame(all_games)

features = ['home_bat_ba', 'home_bat_ops', 'home_bat_r_g', 'home_bat_hr_g', 'away_bat_ba', 'away_bat_ops', 'away_bat_r_g', 'away_bat_hr_g', 'home_pit_era', 'home_pit_whip', 'home_pit_r_g', 'away_pit_era', 'away_pit_whip', 'away_pit_r_g', 'bat_avg_diff', 'ops_diff', 'era_diff', 'whip_diff', 'expected_runs', 'expected_era', 'home_recent_runs_scored', 'home_recent_runs_allowed', 'away_recent_runs_scored', 'away_recent_runs_allowed', 'home_recent_total', 'away_recent_total']

over_under_line = 7.5
odds = 1.65
win_profit = 65

print(f'=== MLB Over {over_under_line} | Kurs {odds} | Stawka 100 | Wygrana: 165 | Czysty: {win_profit} PLN ===')
print(f'Break-even: 60.6%\n')

train_years = [2018, 2019, 2021, 2022, 2023, 2024]
test_years = [2025]

print("Przygotowuje dane treningowe...")
train_df = prepare_data(train_years, over_under_line)
print(f"Train: {len(train_df)} meczow")

print("Przygotowuje dane testowe...")
test_df = prepare_data(test_years, over_under_line)
print(f"Test: {len(test_df)} meczow")

print(f"OVER rate (train): {train_df['is_over'].mean():.1%}")
print(f"OVER rate (test): {test_df['is_over'].mean():.1%}\n")

X_train, y_train = train_df[features], train_df['is_over']
X_test, y_test = test_df[features], test_df['is_over']

print("Trenowanie modelu...")
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
print(f"Model Accuracy: {accuracy_score(y_test, pred):.1%}\n")

print(f"=== ZYSK Z BETTINGU (Kurs {odds}, Stawka 100, Czysty={win_profit}) ===")
for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    bet_mask = prob >= threshold
    bet_count = bet_mask.sum()
    if bet_count == 0: continue
    hits = int(y_test.values[bet_mask].sum())
    miss = bet_count - hits
    profit = hits * win_profit - miss * 100
    roi = profit / (bet_count * 100) * 100
    print(f"Threshold {threshold:.0%}: {bet_count:4d} zak. | Traf: {hits:4d} ({hits/bet_count:.1%}) | Zysk: {profit:7.0f} PLN | ROI: {roi:+.1f}%")

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/mlb_model_75.pkl')
print(f"\nModel zapisany: model/mlb_model_75.pkl")
