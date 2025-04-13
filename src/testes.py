
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



def media_gols_confronto(df, home_id, away_id):
    # Filtrar confrontos entre os dois times, independentemente de quem foi mandante
    confrontos = df[((df['home'] == home_id) & (df['away'] == away_id)) |
                    ((df['home'] == away_id) & (df['away'] == home_id))]

    # Verificar se há confrontos
    if confrontos.empty:
        return "Nenhum confronto encontrado."
    total_gols = confrontos['tot_goals'].sum()
    total_jogos = confrontos.shape[0]
    media_gols = confrontos['tot_goals'].mean()

    # Exibir informações
    print(f"Total de gols: {total_gols}")
    print(f"Total de jogos: {total_jogos}")
    print(f"Média de gols do confronto: {media_gols}")
    # Calcular e retornar a média de gols do confronto
    return confrontos['tot_goals'].mean()



from collections import defaultdict


df = pd.DataFrame(df_temp)

# Dicionário para contar os confrontos entre os times
match_count = defaultdict(int)

# Percorre o DataFrame e contabiliza os confrontos
for _, row in df.iterrows():
    match = tuple(sorted([row['home'], row['away']]))  # Ordena para evitar duplicatas
    match_count[match] += 1

# Filtra os confrontos que ocorreram mais de uma vez
repeated_matches = {match: count for match, count in match_count.items() if count > 1}

# Exibe os confrontos repetidos
for (team1, team2), count in repeated_matches.items():
    print(f"Times {team1} e {team2} se enfrentaram {count} vezes.")


    


id_especifico = 11230361  # Defina o ID desejado

# Filtrar os jogos onde o time é home ou away
df_filtrado = df_temp[(df_temp['home'] == id_especifico) | (df_temp['away'] == id_especifico)]

# Criar uma nova coluna 'gols_time' baseada na posição do time
df_filtrado = df_filtrado.assign(
    gols_time=df_filtrado.apply(
        lambda row: row['home_goals'] if row['home'] == id_especifico else row['away_goals'], axis=1
    )
)

# Pegar os últimos 5 jogos do time
df_ultimos_5 = df_filtrado.tail(10)

print(df_ultimos_5[['home', 'away', 'gols_time']])  # Exibir só as colunas principais

'''
'''
import time
import os
import pandas as pd
from datetime import datetime, timedelta
import requests
from api import BetsAPIClient
from dotenv import load_dotenv

load_dotenv()

api = os.getenv("API_KEY")
apiclient = BetsAPIClient(api_key=api)
CSV_FILE = "resultados_novo.csv"

def transform_betting_data(odds_data):
    rows = []
    for match_id, odds in odds_data.items():
        row = {'id': match_id}

        ou_markets = odds.get('goals_over_under', [])
        if ou_markets:
            ou_dict = {item['type']: item for item in ou_markets if item['handicap'] == '2.5'}
            if 'Over' in ou_dict and 'Under' in ou_dict:
                row['goals_over_under'] = '2.5'
                row['odd_goals_over1'] = ou_dict['Over']['odds']
                row['odd_goals_under1'] = ou_dict['Under']['odds']

        for i, ah in enumerate(odds.get('asian_handicap', []), 1):
            row[f'asian_handicap{i}'] = ah['handicap']
            row[f'team_ah{i}'] = ah['team']
            row[f'odds_ah{i}'] = ah['odds']

        for i, gl in enumerate(odds.get('goal_line', []), 1):
            row[f'goal_line{i}'] = gl['handicap']
            row[f'type_gl{i}'] = 1 if gl['type'] == 'Over' else 2
            row[f'odds_gl{i}'] = gl['odds']

        for i, dc in enumerate(odds.get('double_chance', []), 1):
            row[f'double_chance{i}'] = dc['type']
            row[f'odds_dc{i}'] = dc['odds']

        for i, dnb in enumerate(odds.get('draw_no_bet', []), 1):
            row[f'draw_no_bet_team{i}'] = dnb['team']
            row[f'odds_dnb{i}'] = dnb['odds']

        rows.append(row)

    return pd.DataFrame(rows)

def dia_anterior():
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

def processar_dia_anterior():
    dia = dia_anterior()
    print(f"🔄 Processando jogos do dia {dia}")

    try:
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=int(dia))
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        df_odds = transform_betting_data(odds_data)

        novos_dados = []
        for dados_evento in dicio:
            event_id = dados_evento.get('id')
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')

            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
            else:
                merged = {**dados_evento, "event_day": dia}

            novos_dados.append(merged)

        if novos_dados:
            df_novo = pd.DataFrame(novos_dados)
            colunas_ordenadas = ['id', 'event_day'] + [col for col in df_novo.columns if col not in ['id', 'event_day']]
            df_novo = df_novo[colunas_ordenadas]

            if os.path.exists(CSV_FILE):
                df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})

                # Adiciona os dados novos
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)

                # Ordena cronologicamente
                df_final = df_final.sort_values(by="event_day").reset_index(drop=True)

                # Remove o primeiro (mais antigo) dia
                primeiro_dia = df_final["event_day"].min()
                df_final = df_final[df_final["event_day"] != primeiro_dia]
            else:
                df_final = df_novo

            df_final.to_csv(CSV_FILE, index=False)
            print(f"✅ Dados atualizados com sucesso! Dia {primeiro_dia} removido, dia {dia} adicionado.")
        else:
            print(f"⚠️ Nenhum dado encontrado para o dia {dia}")

    except Exception as e:
        print(f"❌ Erro ao processar dia {dia}: {e}")

import pandas as pd
import requests
from api import BetsAPIClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import numpy as np
import NN
import tensorflow as tf
#'10048705', 'Esoccer GT Leagues - 12 mins play' ;'10047781', 'Esoccer Battle - 8 mins play'

#testar pegar evento 171732570 mais tarde   171790606  172006772 9723272 172006783
load_dotenv()

api = os.getenv("API_KEY")



apiclient = BetsAPIClient(api_key=api)

ups = apiclient.getUpcoming(leagues=apiclient.leagues_ids, day='20250409')
print(ups)
'''
import pandas as pd
import requests
from api import BetsAPIClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import numpy as np
import main

#'10048705', 'Esoccer GT Leagues - 12 mins play' ;'10047781', 'Esoccer Battle - 8 mins play'

#testar pegar evento 171732570 mais tarde   171790606  172006772 9723272 172006783
load_dotenv()

api = os.getenv("API_KEY")



apiclient = BetsAPIClient(api_key=api)
lista_th = main.criaTodasNNs()