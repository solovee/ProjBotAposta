import requests
from typing import Dict, Any
from datetime import datetime

# Obter a data atual e formatar como YYYYMMDD
data_atual = datetime.now().strftime('%Y%m%d')


class BetsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.b365api.com/v1/bet365/'
        self.base_url_betsapi = 'https://api.b365api.com/v1/'
        self.base_url_v3 = 'https://api.b365api.com/v3/bet365/'

    def getHistory(self, event_id: int = 1, qty: int = 10) -> Dict[str, Any]:
        """
        pega o hisórico de partidas
        """
        url = f'{self.base_url_betsapi}event/history'

        params = {
            'token': self.api_key,
            'event_id' : event_id
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao obter dados da API: {response.status_code}")

    def get_inplay_games(self, sport_id: int = 1, league_id: int = 10048705) -> Dict[str, Any]:

        url = f'{self.base_url}inplay_filter'

        params = {
            'sport_id': sport_id,
            'league_id': league_id
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao obter dados da API: {response.status_code}")
        


    def get_fifa_matches(self, sport_id: int = 1,league_id: int = 10048705, day: str = data_atual) -> Dict[str, Any]:
        """
        pega jogos de FIFA 8 e 12 minutos da BetsAPI.
        """
        url = f'{self.base_url}upcoming'
        params = {
            'token': self.api_key,
            'sport_id': sport_id,
            'league_id': league_id,
            'day': day
            #precisa de filtragem
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro ao obter dados da API: {response.status_code}")
        
    def getEvent(self, event_id: int = 1):

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
    
    def resultados(self, event_id: int=1):
        url = f'{self.base_url}result'

        params = {
            'event_id': event_id,
            'token': self.api_key
        }
        

        '''
        ligas_esoccer = [liga for liga in ligas['results'] if 'Esoccer' in liga['name']]
eventos_esoccer = []

for liga in ligas_esoccer:
    params = {
        'token': api_key,
        'league_id': liga['id']
    }
    response = requests.get('https://api.betsapi.com/v1/events/upcoming', params=params)
    eventos = response.json()
    eventos_esoccer.extend(eventos['results'])
'''
