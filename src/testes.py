import time
import os
import pandas as pd
from datetime import datetime, timedelta
import requests
from api import BetsAPIClient
from dotenv import load_dotenv
import NN
import main

load_dotenv()

api = os.getenv("API_KEY")
apiclient = BetsAPIClient(api_key=api)
'''
CSV_FILE = "resultados_60_a.csv"

def transform_betting_data(odds_data):
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

def dia_anterior():
    """Retorna o dia anterior ao atual no formato YYYYMMDD."""
    ontem = datetime.now() - timedelta(days=1)
    return ontem.strftime("%Y%m%d")

def ultimos_60_dias(data_str: str):
    """Retorna uma lista com todos os dias dos últimos 60 dias anteriores à data fornecida no formato YYYYMMDD."""
    data = datetime.strptime(data_str, "%Y%m%d")
    return [(data - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, 61)]

# Carregar dias já processados
if os.path.exists(CSV_FILE):
    df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
    dias_processados = set(df_existente["event_day"].unique())
else:
    dias_processados = set()


# Configuração: pular os N dias mais recentes (ex: já processados)
DIAS_JA_PROCESSADOS = 0  # você pode alterar isso

# Obter os últimos 60 dias
todos_os_dias = ultimos_60_dias(dia_anterior())

# Pegar apenas os dias que ainda não foram processados (os mais antigos)
dias_todos = todos_os_dias[DIAS_JA_PROCESSADOS:]


while dias_processados != set(dias_todos):
    dias_pendentes = [dia for dia in dias_todos if dia not in dias_processados][:6]
    
    if not dias_pendentes:
        print("✅ Todos os dias já foram processados!")
        break

    print(f"🔄 Processando os dias: {dias_pendentes}")
    novos_dados = []

    for dia in dias_pendentes:
        try:
            ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
            odds_data = apiclient.filtraOddsNovo(ids=ids)
            
            # Transformar os dados de odds
            df_odds = transform_betting_data(odds_data)
            
            # Juntar com dados do evento
            for dados_evento in dicio:
                event_id = dados_evento.get('id')
                odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')
                
                if odds_transformadas:
                    merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
                else:
                    # Criar estrutura vazia se não houver odds
                    merged = {**dados_evento, "event_day": dia}
                
                novos_dados.append(merged)

            dias_processados.add(dia)

        except Exception as e:
            print(f"❌ Erro ao processar dia {dia}: {e}")

    # Salvar os novos dados
    if novos_dados:
        df_novo = pd.DataFrame(novos_dados)
        
        # Reordenar colunas para consistência
        colunas_ordenadas = ['id', 'event_day'] + [col for col in df_novo.columns if col not in ['id', 'event_day']]
        df_novo = df_novo[colunas_ordenadas]
        
        if not os.path.exists(CSV_FILE):
            df_novo.to_csv(CSV_FILE, index=False)
        else:
            df_novo.to_csv(CSV_FILE, mode='a', header=False, index=False)

    print(f"✅ Total de dias processados: {len(dias_processados)}")
    print("⏳ Aguardando 1 hora para a próxima execução...")
    time.sleep(3600)
    '''
import time
from datetime import datetime, timedelta
import tensorflow as tf
from api import BetsAPIClient, dia_anterior
import pandas as pd
from dotenv import load_dotenv
import os
import threading
import NN
import telegramBot as tb
import logging
import random
import ast
load_dotenv()

api = os.getenv("API_KEY")
chat_id = int(os.getenv("CHAT_ID"))



apiclient = BetsAPIClient(api_key=api)
CSV_FILE = r"C:\Users\Leoso\Downloads\projBotAposta\resultados_60_ofc.csv"
def processar_dois_dias_anteriores():
    dias = ["20250414", "20250413"]
    novos_dados = []

    for dia in dias:
        print(f"🔄 Processando jogos do dia {dia}")

        try:
            ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
            odds_data = apiclient.filtraOddsNovo(ids=ids)
            df_odds = apiclient.transform_betting_data(odds_data)

            for dados_evento in dicio:
                event_id = dados_evento.get('id')
                odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')

                if odds_transformadas:
                    merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
                else:
                    merged = {**dados_evento, "event_day": dia}

                novos_dados.append(merged)

        except Exception as e:
            print(f"❌ Erro ao processar dia {dia}: {e}")

    if novos_dados:
        df_novo = pd.DataFrame(novos_dados)
        colunas_ordenadas = ['id', 'event_day'] + [col for col in df_novo.columns if col not in ['id', 'event_day']]
        df_novo = df_novo[colunas_ordenadas]

        if os.path.exists(CSV_FILE):
            df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
            # Adiciona os dados novos no início
            df_final = pd.concat([df_novo, df_existente], ignore_index=True)
        else:
            df_final = df_novo

        df_final.to_csv(CSV_FILE, index=False)
        print(f"✅ Dados dos dias {dias} adicionados ao início com sucesso!")
    else:
        print("⚠️ Nenhum dado foi coletado nos dois dias anteriores.")
    import pandas as pd

def remover_dois_dias_mais_antigos(CSV_FILE):
    try:
        # Lê o CSV e converte a coluna 'event_day' para o tipo string para garantir que as datas fiquem corretas
        df = pd.read_csv(CSV_FILE, dtype={"event_day": str})

        # Encontra os dois dias mais antigos (mínimos)
        primeiro_dia = df["event_day"].min()
        segundo_dia = df[df["event_day"] != primeiro_dia]["event_day"].min()

        # Filtra os registros removendo os dois dias mais antigos
        df_final = df[~df["event_day"].isin([primeiro_dia, segundo_dia])]

        # Salva o DataFrame de volta no CSV
        df_final.to_csv(CSV_FILE, index=False)

        print(f"✅ Dois dias mais antigos ({primeiro_dia} e {segundo_dia}) removidos com sucesso. CSV atualizado.")

    except Exception as e:
        print(f"❌ Erro ao remover os dois dias mais antigos: {e}")

remover_dois_dias_mais_antigos(CSV_FILE)


