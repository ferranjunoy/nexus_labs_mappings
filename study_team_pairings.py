import pandas as pd
import json

df = pd.read_csv("Team pairings.csv")

pillars_list = list(df["PILLAR"])
teams_list = list(df["TEAM"])
teams = set(df["TEAM"])
team_pillars = {
    team: [] for team in teams
}

for idx, team in enumerate(teams_list):
    team_pillars[team].append(pillars_list[idx])

with open("team_pillars.json", "w") as file:
    json.dump(team_pillars, file)