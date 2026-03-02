import requests
import csv
from datetime import datetime, timedelta

TEAM_MAP = {
    108: "LAA", 109: "AZ", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL"
}

def get_schedule(year, month, day):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={year}-{month:02d}-{day:02d}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return []
    data = resp.json()
    games = []
    for date_game in data.get("dates", []):
        for game in date_game.get("games", []):
            if game["status"]["abstractGameState"] != "Final":
                continue
            game_date = date_game["date"]
            teams = game["teams"]
            away = teams.get("away", {})
            home = teams.get("home", {})
            away_id = away.get("team", {}).get("id")
            home_id = home.get("team", {}).get("id")
            away_score = away.get("score", 0)
            home_score = home.get("score", 0)
            if away_id in TEAM_MAP and home_id in TEAM_MAP:
                games.append({
                    "date": game_date,
                    "home_team": TEAM_MAP[home_id],
                    "away_team": TEAM_MAP[away_id],
                    "home_score": home_score,
                    "away_score": away_score,
                    "total_runs": home_score + away_score
                })
    return games

all_games = []
start_date = datetime(2024, 3, 1)
end_date = datetime(2024, 11, 1)
current = start_date

while current <= end_date:
    print(f"Pobieram {current.strftime('%Y-%m-%d')}...")
    games = get_schedule(current.year, current.month, current.day)
    all_games.extend(games)
    current += timedelta(days=1)

print(f"\nŁącznie pobrano {len(all_games)} meczów")

with open("../data/2024_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "home_team", "away_team", "home_score", "away_score", "total_runs"])
    writer.writeheader()
    for g in sorted(all_games, key=lambda x: x["date"]):
        writer.writerow(g)

print("Zapisano do data/2024_results.csv")
