import requests
from api import BetsAPIClient
from dotenv import load_dotenv
import os
from datetime import datetime






#'10048705', 'Esoccer GT Leagues - 12 mins play' ;'10047781', 'Esoccer Battle - 8 mins play'

#testar pegar evento 171732570 mais tarde   171790606
load_dotenv()

api = os.getenv("API_KEY")



apiclient = BetsAPIClient(api_key=api)


ids = apiclient.getUpcoming(leagues=apiclient.leagues_ids)




odds = apiclient.filtraOdds(ids=ids)
print(odds)


#erro = apiclient.get_odds(FI=171948551)
#print(erro)

'''
import json

data = {
    "success": 1,
    "results": {
        "h2h": [
            {
                "id": "199439",
                "sport_id": "1",
                "league": {
                    "id": "849",
                    "name": "China Super League",
                    "cc": "cn"
                },
                "home": {
                    "id": "10121",
                    "name": "Liaoning Hongyun",
                    "image_id": "41429",
                    "cc": "cn"
                },
                "away": {
                    "id": "43806",
                    "name": "Guangzhou R&F",
                    "image_id": "3375",
                    "cc": "cn"
                },
                "time": "1466924400",
                "ss": "3-1",
                "time_status": "3"
            },
            {
                "id": "199440",
                "sport_id": "1",
                "league": {
                    "id": "849",
                    "name": "China Super League",
                    "cc": "cn"
                },
                "home": {
                    "id": "43806",
                    "name": "Guangzhou R&F",
                    "image_id": "3375",
                    "cc": "cn"
                },
                "away": {
                    "id": "10121",
                    "name": "Liaoning Hongyun",
                    "image_id": "41429",
                    "cc": "cn"
                },
                "time": "1467934400",
                "ss": "2-2",
                "time_status": "3"
            }
        ]
    }
}

# Dicionário para armazenar estatísticas dos times por ID
team_stats = {}

# Variáveis para calcular a média geral de gols por partida
total_goals = 0
num_matches = 0

# Percorre as partidas
for match in data["results"]["h2h"]:
    if "ss" in match and match["ss"]:  # Verifica se há placar disponível
        home_team_id = match["home"]["id"]
        home_team_name = match["home"]["name"]
        away_team_id = match["away"]["id"]
        away_team_name = match["away"]["name"]

        home_goals, away_goals = map(int, match["ss"].split("-"))

        # Soma os gols para a média geral
        total_goals += home_goals + away_goals
        num_matches += 1

        # Atualiza os dados do time da casa
        if home_team_id not in team_stats:
            team_stats[home_team_id] = {"name": home_team_name, "total_gols": 0, "num_jogos": 0}
        team_stats[home_team_id]["total_gols"] += home_goals
        team_stats[home_team_id]["num_jogos"] += 1

        # Atualiza os dados do time visitante
        if away_team_id not in team_stats:
            team_stats[away_team_id] = {"name": away_team_name, "total_gols": 0, "num_jogos": 0}
        team_stats[away_team_id]["total_gols"] += away_goals
        team_stats[away_team_id]["num_jogos"] += 1

# Calcula a média de gols por partida
media_total_gols = total_goals / num_matches if num_matches > 0 else 0

# Calcula a média de gols por time
media_por_time = {
    stats["name"]: stats["total_gols"] / stats["num_jogos"]
    for stats in team_stats.values()
}

# Exibir os resultados
print(f"Média total de gols por partida: {media_total_gols:.2f}\n")
print("Média de gols por time:")
for team, media in media_por_time.items():
    print(f"{team}: {media:.2f} gols por jogo")
'''


'''
for a in res:
    timestamp = a['time']
    dt = datetime.fromtimestamp(int(timestamp))
    print(dt)
    '''






'''
for id in res:
    odds = apiclient.get_odds(FI=id)['results'][0]['goals']['sp']['goals_over_under']
    print(odds)


res1 = apiclient.get_fifa_matches(league_id=10047781)



for id in res1:
    odds = apiclient.get_odds(FI=id)['results'][0]['goals']['sp']['goals_over_under']
    print(odds)

'''



