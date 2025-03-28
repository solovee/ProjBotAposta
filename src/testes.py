import pandas as pd
import requests
from api import BetsAPIClient
from dotenv import load_dotenv
import os
from datetime import datetime




#'10048705', 'Esoccer GT Leagues - 12 mins play' ;'10047781', 'Esoccer Battle - 8 mins play'

#testar pegar evento 171732570 mais tarde   171790606  172006772 9723272 172006783
load_dotenv()

api = os.getenv("API_KEY")



apiclient = BetsAPIClient(api_key=api)


ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=20250326)

r = apiclient.filtraOddsOlds(ids=ids)



# Lista para armazenar os dicionários mesclados
resultado = []

for dados in dicio:
    event_id = dados.get('id')  # Obtém o ID do evento
    if event_id in r:  # Verifica se há odds para esse ID
        merged = {**dados, **r[event_id]}  # Mescla os dicionários
        resultado.append(merged)  # Adiciona à lista de resultados
    else:
        resultado.append(dados)  # Caso não tenha odds, mantém os dados originais

df = pd.DataFrame(resultado)
df.isnull().sum()


