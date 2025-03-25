import requests
from typing import Dict, Any, List
from datetime import datetime
from math import ceil

# Obter a data atual e formatar como YYYYMMDD
data_atual = datetime.now().strftime('%Y%m%d')

# upcoming(ids jogos) -> filtraOdds(id, odds) -> getHist(historico) -> calculaMedia(team_stats(mediaTotal, statsDosTimes)) -> encontraLinhasDesreguladas


class BetsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.b365api.com/v1/bet365/'
        self.base_url_v3 = 'https://api.b365api.com/v3/bet365/'
        self.base_url_event = 'https://api.b365api.com/v1/'
        self._leagues_ids = [10048705, 10047781]


    @property
    def leagues_ids(self):
        return self._leagues_ids


    def getUpcoming(self, sport_id: int = 1, leagues: List[Any] = [], day: str = data_atual) -> List[Any]:
        '''pega jogos futuros'''

        results = []
        for league in leagues:
            pages = self.pages(league_id=league)
            for page in range(1, pages+1):
                res = self.get_fifa_matches(league_id = league, page = page)
                results.extend(res)
        return results

        
        
    
    def pages(self, sport_id: int = 1,league_id: int = 10048705, day: str = data_atual) -> int:
        '''pega o numero de paginas da requisiçao'''

        url = f'{self.base_url}upcoming'
        params = {
            'token': self.api_key,
            'sport_id': sport_id,
            'league_id': league_id,
            'day': day,
        }
        response = requests.get(url, params=params)
        resp = response.json()
        
        res = ceil(resp['pager']['total'] / resp['pager']['per_page'])

        if response.status_code == 200:
            return res
        else:
            raise Exception(f"Erro ao obter dados da API: {response.status_code}")
        
    

    def get_fifa_matches(self, sport_id: int = 1,league_id: int = 10048705, day: str = data_atual, page: int = 1) -> Dict[str, Any]:
        """
        pega jogos de FIFA 8 e 12 minutos da BetsAPI.
        """
        url = f'{self.base_url}upcoming'
        params = {
            'token': self.api_key,
            'sport_id': sport_id,
            'league_id': league_id,
            'day': day,
            'page': page
            
        }
        response = requests.get(url, params=params)
        resp = response.json()
        
        res = [x['id'] for x in resp['results'] if (x.get('ss') is None) or x.get('time_status') == 0]

        if response.status_code == 200:
            return res
        else:
            raise Exception(f"Erro ao obter dados da API: {response.status_code}")
        
    def getInplayEvents(self, sport_id: int = 1, league_id: int = 10048705) -> Dict[str,Any]:
        '''pega jogos ao vivo'''

        url = f'{self.base_url}inplay_filter'
        params = {
            'token': self.api_key,
            'sport_id': sport_id,
            'league_id': league_id,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao obter inplay_eventos da API: {response.status_code}")
        

    def filtraOdds(self, ids: List[Any] = []):
        done = 0
        RED = "\033[31m"
        RESET = "\033[0m"
        print(ids)
        games = {}

        for id in ids:
            try:
                odds_data = self.get_odds(FI=id)['results'][0]
                
                # Verifica se 'goals' existe antes de acessar
                if 'goals' in odds_data and 'goals_over_under' in odds_data['goals']['sp']:
                    odds = odds_data['goals']['sp']['goals_over_under']
                
                # Se 'goals_over_under' não estiver em 'goals', verifica em 'main'
                elif 'main' in odds_data and 'goals_over_under' in odds_data['main']['sp']:
                    odds = odds_data['main']['sp']['goals_over_under']
                
                else:
                    raise KeyError("goals_over_under não encontrado em 'goals' nem em 'main'.")

                games[id] = odds
                print(f'{RED}{id}{RESET}', odds)
                done += 1

            except KeyError:
                print(f'KeyError, rever filtragem de odds para o evento {id}')

        print(len(ids) - done)
        return games

        
    def getEvent(self, event_id: int = 1) -> Dict[str, Any]:

        '''retorna dados de um evento'''

        url = f'{self.base_url}result'

        params = {
            'event_id': event_id
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao obter odds da API: {response.status_code}")

    def get_odds(self, FI: int) -> Dict[str, Any]:
        """
        pega as odds para um evento específico.
        """
        url = f'{self.base_url_v3}prematch'
        params = {
            'token': self.api_key,
            'FI': FI
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao obter odds da API: {response.status_code}")
        
    
    def getHist(self, event_id: int = 1, qty: int = 20) -> Dict[str, Any]:

        """
        pega o historico de h2h
        """
        url = f'{self.base_url_event}event/history'
        params = {
            'token': self.api_key,
            'event_id': event_id,
            'qty': qty
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao obter odds da API: {response.status_code}")
        


    def calculaMediaDeGols(historico: Dict[Any, Any] = {}):
        # Dicionário para armazenar estatísticas dos times por ID
        team_stats = {}

        # Variáveis para calcular a média geral de gols por partida
        total_goals = 0
        num_matches = 0

        # Percorre as partidas
        for match in historico["results"]["h2h"]:
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

        team_stats['media_total_gols'] = media_total_gols
        
        return team_stats

    def encontraLinhasDesreguladas(self, team_stats: Dict[Any, Any], odds_filtradas: Dict[Any, Any]):
        linhas_desreguladas = []

        media_total_gols = team_stats.get("media_total_gols", 2.5)  # Padrão caso não tenha

        for game_id, odds_data in odds_filtradas.items():
            if "odds" not in odds_data:
                continue  # Pula se não houver odds

            # Extrai as equipes envolvidas
            home_team = odds_data.get("home_team")
            away_team = odds_data.get("away_team")

            # Obtém a média de gols dos times (se não encontrado, assume a média total)
            media_home = team_stats.get(home_team, {}).get("total_gols", media_total_gols) / team_stats.get(home_team, {}).get("num_jogos", 1)
            media_away = team_stats.get(away_team, {}).get("total_gols", media_total_gols) / team_stats.get(away_team, {}).get("num_jogos", 1)

            # Média esperada do jogo
            media_esperada = (media_home + media_away) / 2

            # Verifica odds disponíveis
            for odd in odds_data["odds"]:
                linha = float(odd["name"])  # Ex: 2.5 gols
                over_odds = float(odd["odds"]) if odd["header"] == "Over" else None
                under_odds = float(odd["odds"]) if odd["header"] == "Under" else None

                # Critério de desregulação: Diferença significativa entre média esperada e linha oferecida
                if media_esperada > linha + 0.5 and over_odds > 1.80:
                    linhas_desreguladas.append({"game_id": game_id, "tipo": "Over", "linha": linha, "odds": over_odds, "media_esperada": media_esperada})
                elif media_esperada < linha - 0.5 and under_odds > 1.80:
                    linhas_desreguladas.append({"game_id": game_id, "tipo": "Under", "linha": linha, "odds": under_odds, "media_esperada": media_esperada})

        return linhas_desreguladas


        
        
    
    
        
    

