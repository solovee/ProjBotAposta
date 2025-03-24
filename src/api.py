import requests
from typing import Dict, Any, List
from datetime import datetime
from math import ceil

# Obter a data atual e formatar como YYYYMMDD
data_atual = datetime.now().strftime('%Y%m%d')


class BetsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.b365api.com/v1/bet365/'
        self.base_url_v3 = 'https://api.b365api.com/v3/bet365/'
        self._leagues_ids = [10048705, 10047781]


    @property
    def leagues_ids(self):
        return self._leagues_ids


    def getUpcoming(self, sport_id: int = 1, leagues: List[Any] = [], day: str = data_atual) -> List[Any]:
        results = []
        for league in leagues:
            pages = self.pages(league_id=league)
            for page in range(1, pages+1):
                res = self.get_fifa_matches(league_id = league, page = page)
                results.extend(res)
        return results

        

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
        
    
    def pages(self, sport_id: int = 1,league_id: int = 10048705, day: str = data_atual) -> int:
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
            raise Exception(f"Erro ao obter odds da API: {response.status_code}")
        
    def getEvent(self, event_id: int = 1) -> Dict[str, Any]:

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
    
    

