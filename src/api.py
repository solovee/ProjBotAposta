import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import pandas as pd



# Obter a data atual e formatar como YYYYMMDD
def data_atual():

    data_atual = datetime.now().strftime('%Y%m%d')
    return data_atual

# upcoming(ids jogos) -> filtraOdds(id, odds) -> getHist(historico) -> calculaMedia(team_stats(mediaTotal, statsDosTimes)) -> encontraLinhasDesreguladas

def dia_anterior():
    """Retorna o dia anterior ao atual no formato YYYYMMDD."""
    ontem = datetime.now() - timedelta(days=1)
    return ontem.strftime("%Y%m%d")


class BetsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.b365api.com/v1/bet365/'
        self.base_url_v3 = 'https://api.b365api.com/v3/bet365/'
        self.base_url_event = 'https://api.b365api.com/v1/'
        self._leagues_ids = [10048705, 10047781]
        self.chamadas = 0


    @property
    def leagues_ids(self):
        return self._leagues_ids


    def get_odds_with_retry(self, FI, max_retries=3, base_delay=2):
        for attempt in range(max_retries):
            try:
                response = self.get_odds(FI=FI)
                if response:
                    return response
            except Exception as e:
                print(f"Tentativa {attempt + 1}/{max_retries} falhou: {e}")
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))  # Atraso exponencial
        raise Exception(f"Falha após {max_retries} tentativas ao obter odds para {FI}")


    #DATES
    
    

    def ultimos_90_dias(data_str: str):
        """Retorna uma lista com todos os dias dos últimos 90 dias anteriores à data fornecida no formato YYYYMMDD."""
        data = datetime.strptime(data_str, "%Y%m%d")  # Converte a string para objeto datetime
        dias = [(data - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, 91)]  # Lista dos últimos 90 dias
        
        return dias

    


    

    


    #OLDS
    def getAllOlds(self, sport_id: int = 1, leagues: List[int] = [], day: str = dia_anterior()) -> List[Any]:
        """Pega jogos antigos."""

        results = []
        di = []

        for league in leagues:
            pages = self.pagesOld(league_id=league, day=day)
            if league == 10048705 or league == "10048705":
                liga = 12
            elif league == 10047781 or league == "10047781":
                liga = 8
            

            for page in range(1, pages + 1):
                res = self.get_old_matches(league_id=league, page=page, day=day)
                r = res.get('results', [])  # Evita erro se 'results' não estiver presente
                for registro in r:
                    if str(registro.get('time_status')) != '3':
                        continue  # Evita erro se 'results' não estiver presente
                    ss = registro.get('ss')  # Pega o valor de 'ss', se existir, ou None
                    
                    if ss and len(ss) >= 3:  # Verifica se 'ss' não é None e tem pelo menos 3 caracteres
                        home_goals = int(ss.split('-')[0])  # Pega os gols do time da casa
                        away_goals = int(ss.split('-')[1])  # Pega os gols do time visitante
                    else:
                        home_goals = away_goals = tot_goals = None  # Define como None caso 'ss' seja inválido

                    dic = {
                        'id': registro['id'],
                        'home': registro['home']['id'],
                        'away': registro['away']['id'],
                        'league': liga,
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'tot_goals': (home_goals + away_goals) if home_goals is not None and away_goals is not None else None,
                        
                    }
                    di.append(dic)

                

                if res:
                    for a in r:
                        results.append(a['id'])

        #print(results)
        return results, di

    

    def pagesOld(self, sport_id: int = 1, league_id: int = 10048705, day: str = dia_anterior()) -> int:
        """Pega o número de páginas da requisição."""
        
        url = f'{self.base_url}upcoming'
        params = {
            'token': self.api_key,
            'sport_id': sport_id,
            'league_id': league_id,
            'day': day
        }
        response = requests.get(url, params=params)
        self.chamadas += 1

        if response.status_code != 200:
            raise Exception(f"Erro ao obter dados da API: {response.status_code} - {response.text}")

        try:
            resp = response.json()
            #print(f"Resposta da API para pagesOld: {resp['pager']}")  # Debug

            pager = resp.get('pager', {})
            total = pager.get('total', 0)
            per_page = pager.get('per_page', 1)  # Evita divisão por zero

            pages = ceil(total / per_page)
            #print(f"Total: {total}, Por página: {per_page}, Páginas calculadas: {pages}")  # Debug

            return pages
        except (ValueError, KeyError) as e:
            raise Exception(f"Erro ao processar resposta da API: {e}")

        
    
    
    
    def get_old_matches(self, sport_id: int = 1, league_id: int = 10048705, day: str = dia_anterior(), page: int = 1) -> Dict[str, Any]:
        """
        Pega jogos de FIFA 8 e 12 minutos da BetsAPI.
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
        self.chamadas += 1

        if response.status_code != 200:
            raise Exception(f"Erro ao obter dados da API: {response.status_code} - {response.text}")

        try:
            resp = response.json()
            return resp
        except ValueError:
            raise Exception("Erro ao processar resposta JSON da API")
        

    def filtraOddsOlds(self, ids: List[Any] = []):
        '''pega odds de eventos e devolve filtradas'''
        RED = "\033[31m"
        RESET = "\033[0m"
        
        game = {}  # Dicionário final
        done = 0
        
        def process_id(event_id):
            try:
                response = self.get_odds_with_retry(event_id)
                
                if not response or "results" not in response or not response["results"]:
                    #print(f"{RED}Erro: Dados ausentes para o evento {event_id}{RESET}")
                    return event_id, None

                odds_data = response["results"][0]
                odds = None
                
                if "goals" in odds_data and "sp" in odds_data["goals"] and "goals_over_under" in odds_data["goals"]["sp"]:
                    odds = odds_data["goals"]["sp"]["goals_over_under"]
                elif "main" in odds_data and "sp" in odds_data["main"] and "goals_over_under" in odds_data["main"]["sp"]:
                    odds = odds_data["main"]["sp"]["goals_over_under"]
                
                if not odds or "odds" not in odds or len(odds["odds"]) < 2:
                    #print(f"{RED}Odds não encontradas para o evento {event_id}{RESET}")
                    return event_id, None
                
                games = {
                    "odd_over": odds["odds"][0]["odds"],
                    "odd_under": odds["odds"][1]["odds"],
                    "goals_odd": odds["odds"][0]["name"]
                }
                
                
                return event_id, games
            except KeyError:
                print(f"{RED}KeyError: Estrutura inesperada para o evento {event_id}{RESET}")
                return event_id, None

        with ThreadPoolExecutor() as executor:
            future_to_id = {executor.submit(process_id, id): id for id in ids}
            for future in as_completed(future_to_id):
                event_id, result = future.result()
                if result is not None:
                    game[event_id] = result
                    done += 1
        
        #print(f"Total de eventos processados: {done}/{len(ids)}")
        return game
    
    def precisa_pegar_dia_seguinte():
        agora = datetime.now()
        return agora.hour >= 21

    
    def getUpcoming(self, sport_id: int = 1, leagues: List[Any] = [], day: str = data_atual()) -> List[Any]:
        results = []
        ts_list = []
        times = []
        times_id = [] 
        league_durations = [] # Agora será uma lista de tuplas (home_id, away_id)
        for league in leagues:
            if league == 10048705 or league == '10048705':
                league_duration = 12
            elif league == 10047781 or league == '10047781':
                league_duration = 8
            
            page = 1
            while True:
                res, ts, tm, ti, total_pages = self.get_fifa_matches_with_total(league_id=league, page=page, day=day)
                
                results.extend(res)
                ts_list.extend(ts)
                times.extend(tm)
                times_id.extend(ti) 
                league_durations.append(league_duration)
                
                 # Já são tuplas individuais por jogo
                if page >= total_pages:
                    break
                page += 1

        return results, ts_list, times, times_id, league_durations
    
    def getUpcoming_check(self, sport_id: int = 1, leagues: List[Any] = [], day: str = data_atual()) -> List[Any]:
        results = []
        ts_list = []
        times = []
        times_id = []  # Agora será uma lista de tuplas (home_id, away_id)
        for league in leagues:
            page = 1
            while True:
                res, ts, tm, ti, total_pages = self.get_fifa_matches_with_total_check(league_id=league, page=page, day=day)
                
                results.extend(res)
                ts_list.extend(ts)
                times.extend(tm)
                times_id.extend(ti)  # Já são tuplas individuais por jogo
                if page >= total_pages:
                    break
                page += 1

        return results, ts_list, times, times_id


# LIVE

    def get_fifa_matches_with_total(self, sport_id: int = 1, league_id: int = 10048705, day: str = data_atual(), page: int = 1):
        url = f'{self.base_url}upcoming'
        params = {
            'token': self.api_key,
            'sport_id': sport_id,
            'league_id': league_id,
            'day': day,
            'page': page
        }
        response = requests.get(url, params=params)
        self.chamadas += 1
        resp = response.json()
        pager = resp.get('pager', {})
        total_pages = ceil(pager.get('total', 0) / pager.get('per_page', 1))

        res = []
        ts = []
        times = []
        times_id = []  # Lista de tuplas (home_id, away_id)

        for x in resp['results']:
            if (x.get('time_status') == 0) or (int(x.get('time',0)) > time.time()):
                res.append(x['id'])
                ts.append(x['time'])
                times.append((x['home']['name'], x['away']['name']))
                times_id.append((x['home']['id'], x['away']['id']))  # Coleta diretamente o par de IDs

        return res, ts, times, times_id, total_pages
    
    def get_fifa_matches_with_total_check(self, sport_id: int = 1, league_id: int = 10048705, day: str = data_atual(), page: int = 1):
        url = f'{self.base_url}upcoming'
        params = {
            'token': self.api_key,
            'sport_id': sport_id,
            'league_id': league_id,
            'day': day,
            'page': page
        }
        response = requests.get(url, params=params)
        self.chamadas += 1
        resp = response.json()
        pager = resp.get('pager', {})
        total_pages = ceil(pager.get('total', 0) / pager.get('per_page', 1))

        res = []
        ts = []
        times = []
        times_id = []  # Lista de tuplas (home_id, away_id)

        for x in resp['results']:
            if (x.get('time_status') == 3):
                res.append(x['id'])
                ts.append(x['time'])
                times.append((x['home']['name'], x['away']['name']))
                times_id.append((x['home']['id'], x['away']['id']))  # Coleta diretamente o par de IDs

        return res, ts, times, times_id, total_pages

    
    def pages(self, sport_id: int = 1,league_id: int = 10048705, day: str = data_atual()) -> int:
        '''pega o numero de paginas da requisiçao'''

        url = f'{self.base_url}upcoming'
        params = {
            'token': self.api_key,
            'sport_id': sport_id,
            'league_id': league_id,
            'day': day,
        }
        response = requests.get(url, params=params)
        self.chamadas += 1
        resp = response.json()
        c = resp['pager']['total']
        print(f'tot: {c}')
        
        res = ceil(resp['pager']['total'] / resp['pager']['per_page'])

        if response.status_code == 200:
            return res
        else:
            raise Exception(f"Erro ao obter dados da API: {response.status_code}")

    
    

    def get_fifa_matches(self, sport_id: int = 1,league_id: int = 10048705, day: str = data_atual(), page: int = 1) -> Dict[str, Any]:
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
        self.chamadas += 1

        resp = response.json()
        with open('a.txt','w') as f:
            f.write(str(resp))
        
        res = [x['id'] for x in resp['results'] if (x.get('ss') is None) or x.get('time_status') == 0]
        ts = [x['time'] for x in resp['results'] if (x.get('ss') is None) or x.get('time_status') == 0]
        times = [
            (x['home']['name'], x['away']['name'])
            for x in resp['results']
            if (x.get('ss') is None) or (x.get('time_status') == 0)
        ]

        
        if response.status_code == 200:
            return res, ts, times
        else:
            raise Exception(f"Erro ao obter dados da API: {response.status_code}")
        
    

    def filtraOdds(self, ids: List[Any] = []):
        done = 0
        RED = "\033[31m"
        RESET = "\033[0m"
        
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
                #print(f'{RED}{id}{RESET}', odds)
                done += 1

            except KeyError:
                print(f'KeyError, rever filtragem de odds para o evento {id}')
                

        #print(len(ids) - done)
        return games
    def filtraOddsNovo1(self, ids: List[Any] = []):
        games = {}
        
        for id in ids:
            try:
                odds_data = self.get_odds(FI=id)['results'][0]
                game_data = {
                    "goals_over_under": [],
                    "asian_handicap": [],
                    "goal_line": [],
                    "double_chance": [],
                    "draw_no_bet": []
                }

                # --- Asian Handicap --- (Mantendo exatamente como estava)
                if 'asian_lines' in odds_data and 'asian_handicap' in odds_data['asian_lines']['sp']:
                    for odd in odds_data['asian_lines']['sp']['asian_handicap']['odds']:
                        game_data["asian_handicap"].append({
                            "handicap": odd["handicap"],
                            "team": odd["header"],
                            "odds": odd["odds"]
                        })
                
                # --- Goal Line --- (Mantendo exatamente como estava)
                if 'asian_lines' in odds_data and 'goal_line' in odds_data['asian_lines']['sp']:
                    for odd in odds_data['asian_lines']['sp']['goal_line']['odds']:
                        game_data["goal_line"].append({
                            "handicap": odd.get("name", "N/A"),
                            "type": odd["header"],
                            "odds": odd["odds"]
                        })

                # --- Goals Over/Under --- (Mantendo exatamente como estava)
                if 'goals' in odds_data and 'goals_over_under' in odds_data['goals']['sp']:
                    for odd in odds_data['goals']['sp']['goals_over_under']['odds']:
                        game_data["goals_over_under"].append({
                            "handicap": odd.get("name", odd.get("handicap", "N/A")),
                            "type": odd["header"],
                            "odds": odd["odds"]
                        })
                elif 'main' in odds_data and 'goals_over_under' in odds_data['main']['sp']:
                    for odd in odds_data['main']['sp']['goals_over_under']['odds']:
                        game_data["goals_over_under"].append({
                            "handicap": odd.get("name", odd.get("handicap", "N/A")),
                            "type": odd["header"],
                            "odds": odd["odds"]
                        })

                # --- Double Chance --- (Mantendo exatamente como estava)
                if 'main' in odds_data and 'double_chance' in odds_data['main']['sp']:
                    for odd in odds_data['main']['sp']['double_chance']['odds']:
                        game_data["double_chance"].append({
                            "type": odd["name"],
                            "odds": odd["odds"]
                        })
                
                # --- Draw No Bet --- (Mantendo exatamente como estava)
                if 'main' in odds_data and 'draw_no_bet' in odds_data['main']['sp']:
                    for odd in odds_data['main']['sp']['draw_no_bet']['odds']:
                        game_data["draw_no_bet"].append({
                            "team": odd["name"],
                            "odds": odd["odds"]
                        })

                games[id] = game_data
                
            except Exception as e:
                print(f'Erro ao processar evento {id}: {str(e)}')
                games[id] = {  # Mantém estrutura vazia
                    "goals_over_under": [],
                    "asian_handicap": [],
                    "goal_line": [],
                    "double_chance": [],
                    "draw_no_bet": []
                }
        
        return games
    def filtraOddsNovo(self, ids: List[Any] = []):
        done = 0
        RED = "\033[31m"
        RESET = "\033[0m"
        #print(ids)
        games = {}

        for id in ids:
            try:
                odds_data = self.get_odds(FI=id)['results'][0]
                game_data = {
                    "goals_over_under": [],
                    "asian_handicap": [],
                    "goal_line": [],
                    "double_chance": [],
                    "draw_no_bet": []
                }

                # --- Asian Handicap ---
                if 'asian_lines' in odds_data and 'asian_handicap' in odds_data['asian_lines']['sp']:
                    for odd in odds_data['asian_lines']['sp']['asian_handicap']['odds']:
                        game_data["asian_handicap"].append({
                            "handicap": odd["handicap"],
                            "team": odd["header"],
                            "odds": odd["odds"]
                        })
            

                # --- Goal Line ---
                if 'asian_lines' in odds_data and 'goal_line' in odds_data['asian_lines']['sp']:
                    for odd in odds_data['asian_lines']['sp']['goal_line']['odds']:
                        game_data["goal_line"].append({
                            "handicap": odd.get("name", "N/A"),
                            "type": odd["header"],
                            "odds": odd["odds"]
                        })

                # --- Goals Over/Under ---
                if 'goals' in odds_data and 'goals_over_under' in odds_data['goals']['sp']:
                    for odd in odds_data['goals']['sp']['goals_over_under']['odds']:
                        game_data["goals_over_under"].append({
                            "handicap": odd.get("name", odd.get("handicap", "N/A")),
                            "type": odd["header"],
                            "odds": odd["odds"]
                        })
                elif 'main' in odds_data and 'goals_over_under' in odds_data['main']['sp']:
                    for odd in odds_data['main']['sp']['goals_over_under']['odds']:
                        game_data["goals_over_under"].append({
                            "handicap": odd.get("name", odd.get("handicap", "N/A")),
                            "type": odd["header"],
                            "odds": odd["odds"]
                        })

                # --- Double Chance ---
                if 'main' in odds_data and 'double_chance' in odds_data['main']['sp']:
                    for odd in odds_data['main']['sp']['double_chance']['odds']:
                        game_data["double_chance"].append({
                            "type": odd["name"],
                            "odds": odd["odds"]
                        })
                        
                # --- Draw No Bet ---
                if 'main' in odds_data and 'draw_no_bet' in odds_data['main']['sp']:
                    for odd in odds_data['main']['sp']['draw_no_bet']['odds']:
                        game_data["draw_no_bet"].append({
                            "team": odd["name"],
                            "odds": odd["odds"]
                        })

                # Remove empty lists
                game_data = {k: v for k, v in game_data.items() if v}
                
                games[id] = game_data
                #print(f'{RED}{id}{RESET}', game_data)
                done += 1

            except KeyError as e:
                print(f'KeyError ({e}), rever filtragem de odds para o evento {id}')
            except Exception as e:
                print(f'Erro inesperado ao processar evento {id}: {e}')

        #print(f"Eventos processados: {done}/{len(ids)}")
        return games


        
    def getEvent(self, event_id: int = 1) -> Dict[str, Any]:

        '''retorna dados de um evento'''

        url = f'{self.base_url}result'

        params = {
            'event_id': event_id
        }
        response = requests.get(url, params=params)
        self.chamadas += 1
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
        self.chamadas += 1
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
    
    def transform_betting_data(self, odds_data):
        """Transforma os dados de odds em um DataFrame estruturado."""
        rows = []
        
        for match_id, odds in odds_data.items():
            row = {'id': match_id}
            
            # Goals Over/Under
            ou_markets = odds.get('goals_over_under', [])
            if ou_markets:
                ou_dict = {item['type']: item for item in ou_markets if item['handicap'] == '2.5'}
                if 'Over' in ou_dict and 'Under' in ou_dict:
                    row['goals_over_under'] = '2.5'
                    row['odd_goals_over1'] = ou_dict['Over']['odds']
                    row['odd_goals_under1'] = ou_dict['Under']['odds']
            
            # Asian Handicap
            for i, ah in enumerate(odds.get('asian_handicap', []), 1):
                row[f'asian_handicap{i}'] = ah['handicap']
                row[f'team_ah{i}'] = ah['team']
                row[f'odds_ah{i}'] = ah['odds']
            
            # Goal Line
            for i, gl in enumerate(odds.get('goal_line', []), 1):
                row[f'goal_line{i}'] = gl['handicap']
                row[f'type_gl{i}'] = 1 if gl['type'] == 'Over' else 2
                row[f'odds_gl{i}'] = gl['odds']
            
            # Double Chance
            for i, dc in enumerate(odds.get('double_chance', []), 1):
                row[f'double_chance{i}'] = dc['type']
                row[f'odds_dc{i}'] = dc['odds']
            
            # Draw No Bet
            for i, dnb in enumerate(odds.get('draw_no_bet', []), 1):
                row[f'draw_no_bet_team{i}'] = dnb['team']
                row[f'odds_dnb{i}'] = dnb['odds']
            
            rows.append(row)
        
        return pd.DataFrame(rows)
        

