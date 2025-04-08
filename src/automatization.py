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

CSV_FILE = "resultados.csv"

def dia_anterior():
    """Retorna o dia anterior ao atual no formato YYYYMMDD."""
    ontem = datetime.now() - timedelta(days=1)
    return ontem.strftime("%Y%m%d")

def ultimos_60_dias(data_str: str):
    """Retorna uma lista com todos os dias dos últimos 60 dias anteriores à data fornecida no formato YYYYMMDD."""
    data = datetime.strptime(data_str, "%Y%m%d")
    return [(data - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, 61)]  # Últimos 60 dias

# Carregar dias já processados do CSV (se existir)
if os.path.exists(CSV_FILE):
    df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})  # Carregar arquivo existente
    dias_processados = set(df_existente["event_day"].unique())  # Obter os dias já coletados
else:
    dias_processados = set()

# Lista de dias a serem processados
dias_todos = ultimos_60_dias(dia_anterior())

while dias_processados != set(dias_todos):  # Enquanto houver dias para processar
    dias_pendentes = [dia for dia in dias_todos if dia not in dias_processados][:5]  # Pega 5 dias por vez
    
    if not dias_pendentes:
        print("✅ Todos os dias já foram processados!")
        break

    print(f"🔄 Processando os dias: {dias_pendentes}")
    novos_dados = []

    for dia in dias_pendentes:
        try:
            ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=int(dia))
            r = apiclient.filtraOddsOlds(ids=ids)

            # Identificar todas as chaves de odds disponíveis nos eventos processados
            chaves_odds_padrao = set()
            for odds in r.values():
                chaves_odds_padrao.update(odds.keys())  # Adiciona todas as chaves únicas ao conjunto
            
            for dados in dicio:
                event_id = dados.get('id')

                if event_id in r:
                    merged = {**dados, **r[event_id], "event_day": dia}
                else:
                    # Criar estrutura com todas as odds possíveis setadas como None
                    odds_padrao = {chave: None for chave in chaves_odds_padrao}
                    merged = {**dados, **odds_padrao, "event_day": dia}

                novos_dados.append(merged)

            dias_processados.add(dia)  # Marca o dia como processado

        except Exception as e:
            print(f"❌ Erro ao processar dia {dia}: {e}")

    # Salvar os novos dados no CSV
    if novos_dados:
        df_novo = pd.DataFrame(novos_dados)
        
        if not os.path.exists(CSV_FILE):
            df_novo.to_csv(CSV_FILE, index=False)  # Cria um novo arquivo
        else:
            df_novo.to_csv(CSV_FILE, mode="a", header=False, index=False)  # Adiciona ao existente

    print(f"✅ Total de dias processados: {len(dias_processados)}")
    
    # Espera 1 hora antes de processar os próximos 5 dias
    print("⏳ Aguardando 1 hora para a próxima execução...")
    time.sleep(3600)


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
            row[f'type_gl{i}'] = 1 if gl['type'] == 'Over' else 'Under'
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

dias_todos = ultimos_60_dias(dia_anterior())

while dias_processados != set(dias_todos):
    dias_pendentes = [dia for dia in dias_todos if dia not in dias_processados][:5]
    
    if not dias_pendentes:
        print("✅ Todos os dias já foram processados!")
        break

    print(f"🔄 Processando os dias: {dias_pendentes}")
    novos_dados = []

    for dia in dias_pendentes:
        try:
            ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=int(dia))
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