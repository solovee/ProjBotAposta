
'''
import pandas as pd
import requests
from api import BetsAPIClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta




#'10048705', 'Esoccer GT Leagues - 12 mins play' ;'10047781', 'Esoccer Battle - 8 mins play'

#testar pegar evento 171732570 mais tarde   171790606  172006772 9723272 172006783
load_dotenv()

api = os.getenv("API_KEY")



apiclient = BetsAPIClient(api_key=api)


ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=20250326)

r = apiclient.filtraOddsOlds(ids=ids)
def dia_anterior():
        """Retorna o dia anterior ao atual no formato YYYYMMDD."""
        ontem = datetime.now() - timedelta(days=1)
        return ontem.strftime("%Y%m%d")
    

def ultimos_90_dias(data_str: str):
    """Retorna uma lista com todos os dias dos últimos 90 dias anteriores à data fornecida no formato YYYYMMDD."""
    data = datetime.strptime(data_str, "%Y%m%d")  # Converte a string para objeto datetime
    dias = [(data - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, 91)]  # Lista dos últimos 90 dias
        
    return dias


# Lista para armazenar os dicionários mesclados
resultado = []

for dados in dicio:
    event_id = dados.get('id')  # Obtém o ID do evento
    if event_id in r:  # Verifica se há odds para esse ID
        merged = {**dados, **r[event_id]}  # Mescla os dicionários
        resultado.append(merged)  # Adiciona à lista de resultados
    else:
        resultado.append(dados)  # Caso não tenha odds, mantém os dados originais
'''


import pandas as pd
import requests

import time
from api import BetsAPIClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()
api = os.getenv("API_KEY")

apiclient = BetsAPIClient(api_key=api)

def dia_anterior():
    """Retorna o dia anterior ao atual no formato YYYYMMDD."""
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

def ultimos_90_dias(data_str: str):
    """Retorna uma lista com todos os dias dos últimos 90 dias anteriores à data fornecida no formato YYYYMMDD."""
    data = datetime.strptime(data_str, "%Y%m%d")
    return [(data - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, 91)]

def obter_dados():
    """Obtém os dados das partidas e odds e retorna um DataFrame consolidado."""
    ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia_anterior())
    odds = apiclient.filtraOddsOlds(ids=ids)
    
    resultado = []
    for dados in dicio:
        event_id = dados.get('id')
        merged = {**dados, **odds.get(event_id, {})}  # Mescla os dicionários, mantendo os dados originais se não houver odds
        resultado.append(merged)
    
    return pd.DataFrame(resultado)

def atualizar_dados():
    """Atualiza os DataFrames com os dados mais recentes e salva em CSV."""
    df_novos = obter_dados()
    df_novos.to_csv("dados_ultimos_90_dias.csv", index=False)
    
    # Carregar os dados acumulados (se existirem)
    try:
        df_acumulado = pd.read_csv("dados_acumulados.csv")
        df_acumulado = pd.concat([df_acumulado, df_novos], ignore_index=True)
    except FileNotFoundError:
        df_acumulado = df_novos
    
    df_acumulado.to_csv("dados_acumulados.csv", index=False)
    print(f"Dados atualizados em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(df_acumulado.shape)
    print(df_novos.shape)


atualizar_dados()
print()

