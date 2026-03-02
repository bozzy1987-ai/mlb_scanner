#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
from datetime import datetime

TEAMS = [
    'LAA', 'AZ', 'BAL', 'BOS', 'CHC', 'CIN', 'CLE', 'COL', 'DET', 'HOU',
    'KC', 'LAD', 'WSH', 'NYM', 'ATH', 'PIT', 'SD', 'SEA', 'SF', 'STL',
    'TB', 'TEX', 'TOR', 'MIN', 'PHI', 'ATL', 'CWS', 'MIA', 'NYY', 'MIL'
]

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
    'Arizona Diamondbacks': 'AZ'
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
    except Exception as e:
        print(f"Error loading batting {year}: {e}")
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
    except Exception as e:
        print(f"Error loading pitching {year}: {e}")
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

def prepare_data(years, over_under_line=7.5):
    all_games = []
    
    for year in years:
        print(f"Przygotowuję dane dla {year}...")
        
        results = load_results(year)
        if results.empty:
            print(f"  Brak wyników dla {year}")
            continue
            
        batting = load_batting_stats(year)
        pitching = load_pitching_stats(year)
        
        results['year'] = year
        
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
            
            recent_home = calculate_rolling_stats(results, home, game['date'])
            recent_away = calculate_rolling_stats(results, away, game['date'])
            
            game_data = {
                'date': game['date'],
                'year': year,
                'home_team': home,
                'away_team': away,
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'total_runs': game['total_runs'],
                'over_line': over_under_line,
                'is_over': 1 if game['total_runs'] > over_under_line else 0,
                
                'home_bat_ba': home_bat['bat_ba'],
                'home_bat_ops': home_bat['bat_ops'],
                'home_bat_r_g': home_bat['bat_r'] / home_bat['bat_g'],
                'home_bat_hr_g': home_bat['bat_hr'] / home_bat['bat_g'],
                
                'away_bat_ba': away_bat['bat_ba'],
                'away_bat_ops': away_bat['bat_ops'],
                'away_bat_r_g': away_bat['bat_r'] / away_bat['bat_g'],
                'away_bat_hr_g': away_bat['bat_hr'] / away_bat['bat_g'],
                
                'home_pit_era': home_pit['pit_era'],
                'home_pit_whip': home_pit['pit_whip'],
                'home_pit_r_g': home_pit['pit_r'] / home_pit['pit_g'],
                
                'away_pit_era': away_pit['pit_era'],
                'away_pit_whip': away_pit['pit_whip'],
                'away_pit_r_g': away_pit['pit_r'] / away_pit['pit_g'],
                
                'bat_avg_diff': home_bat['bat_ba'] - away_bat['bat_ba'],
                'ops_diff': home_bat['bat_ops'] - away_bat['bat_ops'],
                'era_diff': home_pit['pit_era'] - away_pit['pit_era'],
                'whip_diff': home_pit['pit_whip'] - away_pit['pit_whip'],
                
                'expected_runs': (home_bat['bat_r'] / home_bat['bat_g'] + away_bat['bat_r'] / away_bat['bat_g']) / 2,
                'expected_era': (home_pit['pit_era'] + away_pit['pit_era']) / 2,
                
                'home_recent_runs_scored': recent_home['recent_runs_scored'],
                'home_recent_runs_allowed': recent_home['recent_runs_allowed'],
                'away_recent_runs_scored': recent_away['recent_runs_scored'],
                'away_recent_runs_allowed': recent_away['recent_runs_allowed'],
                'home_recent_total': recent_home['recent_total'],
                'away_recent_total': recent_away['recent_total'],
            }
            
            all_games.append(game_data)
    
    return pd.DataFrame(all_games)

def main():
    over_under_line = 7.5
    
    print(f"=== MLB Over/Under Model (linia {over_under_line}) ===\n")
    
    train_years = [2018, 2019, 2021, 2022, 2023, 2024]
    test_years = [2025]
    
    print("Przygotowuję dane treningowe...")
    train_df = prepare_data(train_years, over_under_line)
    print(f" Dane treningowe: {len(train_df)} meczów\n")
    
    print("Przygotowuję dane testowe...")
    test_df = prepare_data(test_years, over_under_line)
    print(f" Dane testowe: {len(test_df)} meczów\n")
    
    features = [
        'home_bat_ba', 'home_bat_ops', 'home_bat_r_g', 'home_bat_hr_g',
        'away_bat_ba', 'away_bat_ops', 'away_bat_r_g', 'away_bat_hr_g',
        'home_pit_era', 'home_pit_whip', 'home_pit_r_g',
        'away_pit_era', 'away_pit_whip', 'away_pit_r_g',
        'bat_avg_diff', 'ops_diff', 'era_diff', 'whip_diff',
        'expected_runs', 'expected_era',
        'home_recent_runs_scored', 'home_recent_runs_allowed',
        'away_recent_runs_scored', 'away_recent_runs_allowed',
        'home_recent_total', 'away_recent_total',
    ]
    
    X_train = train_df[features]
    y_train = train_df['is_over']
    
    X_test = test_df[features]
    y_test = test_df['is_over']
    
    print(f"Target distribution (train): {y_train.mean():.2%} OVER {over_under_line}")
    print(f"Target distribution (test): {y_test.mean():.2%} OVER {over_under_line}\n")
    
    print("Trenowanie modelu XGBoost...")
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
    
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    
    print("\n=== WYNIKI MODELU ===")
    print(f"Accuracy: {accuracy_score(y_test, pred):.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, pred, target_names=[f'UNDER {over_under_line}', f'OVER {over_under_line}']))
    
    print("\n=== TESTY SYMULACJI BETTINGU ===")
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        bet_mask = prob >= threshold
        bet_count = bet_mask.sum()
        if bet_count == 0:
            continue
        
        actual_results = y_test.values[bet_mask]
        hits = (actual_results == 1).sum()
        hit_rate = hits / bet_count
        
        profit = hits - (bet_count - hits)
        roi = profit / bet_count * 100
        
        print(f"Threshold {threshold:.0%}: {bet_count} zakładów, {hits} trafionych ({hit_rate:.1%}), ROI: {roi:+.1f}%")
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, f'model/mlb_model_{int(over_under_line*10)}.pkl')
    print(f"\nModel zapisany do model/mlb_model_{int(over_under_line*10)}.pkl")
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 FEATURES ===")
    print(feature_importance.head(10).to_string(index=False))
    
    test_df['predicted_prob'] = prob
    test_df.to_csv('../data/test_results.csv', index=False)
    print("\nWyniki testowe zapisane do ../data/test_results.csv")

if __name__ == '__main__':
    main()
