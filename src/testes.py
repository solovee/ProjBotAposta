



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
CSV_FILE = "teste_pegar_odds_ontem.csv"

# Lista de colunas esperadas no CSV final, mesmo que estejam vazias
COLUNAS_PADRAO = [
    'id', 'event_day', 'home', 'away', 'home_goals', 'away_goals', 'tot_goals',
    'goals_over_under', 'odd_goals_over1', 'odd_goals_under1',
    'asian_handicap1', 'team_ah1', 'odds_ah1',
    'asian_handicap2', 'team_ah2', 'odds_ah2',
    'goal_line1', 'type_gl1', 'odds_gl1',
    'goal_line2', 'type_gl2', 'odds_gl2',
    'double_chance1', 'odds_dc1',
    'double_chance2', 'odds_dc2',
    'double_chance3', 'odds_dc3',
    'draw_no_bet_team1', 'odds_dnb1',
    'draw_no_bet_team2', 'odds_dnb2',
]


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


# Carregar dias já processados
if os.path.exists(CSV_FILE):
    df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
    dias_processados = set(df_existente["event_day"].unique())
else:
    dias_processados = set()


# Pegar apenas o dia anterior
dia_anterior_str = dia_anterior()

# Verificar se o dia anterior já foi processado
if dia_anterior_str in dias_processados:
    print(f"✅ O dia {dia_anterior_str} já foi processado.")
else:
    try:
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia_anterior_str)
        odds_data = apiclient.filtraOddsNovo(ids=ids)

        # Transformar os dados de odds
        df_odds = transform_betting_data(odds_data)
        
        # Juntar com dados do evento
        novos_dados = []
        for dados_evento in dicio:
            event_id = dados_evento.get('id')
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')
            
            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0], "event_day": dia_anterior_str}
            else:
                # Criar estrutura vazia se não houver odds
                merged = {**dados_evento, "event_day": dia_anterior_str}
            
            novos_dados.append(merged)

        # Salvar os novos dados
        if novos_dados:
            df_novo = pd.DataFrame(novos_dados)
            
            # Reordenar colunas para consistência
            for col in COLUNAS_PADRAO:
                if col not in df_novo.columns:
                    df_novo[col] = None

            df_novo = df_novo[COLUNAS_PADRAO]

            if not os.path.exists(CSV_FILE):
                df_novo.to_csv(CSV_FILE, index=False)
            else:
                df_novo.to_csv(CSV_FILE, mode='a', header=False, index=False)

        print(f"✅ Dados do dia {dia_anterior_str} processados e salvos com sucesso.")

    except Exception as e:
        print(f"❌ Erro ao processar o dia {dia_anterior_str}: {e}")

