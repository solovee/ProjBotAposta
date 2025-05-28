



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
'''
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
print('oi')
# Verificar se o dia anterior já foi processado
if dia_anterior_str in dias_processados:
    print(f"✅ O dia {dia_anterior_str} já foi processado.")
else:
    try:
        print('entrou')
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia_anterior_str)
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        print('ei')
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
CSV_FILE = "resultados_60_ofc.csv"

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

# Dias que devem ser substituídos no CSV
DIAS_ESPECIFICOS = ["20250417", "20250416", "20250415"]

novos_dados = []

for dia in DIAS_ESPECIFICOS:
    try:
        print(f"🔄 Reprocessando o dia: {dia}")
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        
        df_odds = transform_betting_data(odds_data)
        
        for dados_evento in dicio:
            event_id = dados_evento.get('id')
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')
            
            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
            else:
                merged = {**dados_evento, "event_day": dia}
            
            novos_dados.append(merged)

    except Exception as e:
        print(f"❌ Erro ao processar o dia {dia}: {e}")

if novos_dados:
    df_novo = pd.DataFrame(novos_dados)

    # Garantir que todas as colunas existam e estejam na ordem correta
    for col in COLUNAS_PADRAO:
        if col not in df_novo.columns:
            df_novo[col] = None
    df_novo = df_novo[COLUNAS_PADRAO]

    # Carregar o CSV atual e remover os dias específicos
    if os.path.exists(CSV_FILE):
        df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
        df_existente = df_existente[~df_existente['event_day'].isin(DIAS_ESPECIFICOS)]
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    # Salvar o novo CSV substituindo o antigo
    df_final.to_csv(CSV_FILE, index=False)
    print("✅ CSV atualizado com os novos dados dos dias específicos.")
else:
    print("⚠️ Nenhum dado novo foi processado.")
'''
'''
import pandas as pd

CSV_FILE = "resultados_60_ofc.csv"

# Carregar o CSV
df = pd.read_csv(CSV_FILE, dtype={"event_day": str})

# Ordenar por 'event_day' do mais recente para o mais antigo
df = df.sort_values(by="event_day", ascending=False)

# Salvar de volta no mesmo arquivo
df.to_csv(CSV_FILE, index=False)

print("✅ CSV ordenado com sucesso por event_day (do mais recente para o mais antigo).")

import time
import os
import pandas as pd
from datetime import datetime
from api import BetsAPIClient
from dotenv import load_dotenv

load_dotenv()

api = os.getenv("API_KEY")
apiclient = BetsAPIClient(api_key=api)
CSV_FILE = "resultados_60_ofc.csv"

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

DIAS_ADICIONAR = ['20250421', '20250422', '20250423']


def transform_betting_data(odds_data):
    rows = []
    for match_id, odds in odds_data.items():
        row = {'id': match_id}
        
        # Over/Under
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


# Carrega CSV existente
df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str}) if os.path.exists(CSV_FILE) else pd.DataFrame(columns=COLUNAS_PADRAO)

# Remove os 3 dias mais antigos
dias_unicos_ordenados = sorted(df_existente['event_day'].unique())
dias_remover = dias_unicos_ordenados[:3]
df_existente = df_existente[~df_existente['event_day'].isin(dias_remover)]
print(f"🗑️ Dias removidos: {dias_remover}")

# Processa os 3 dias mais recentes
novos_dados = []

for dia in DIAS_ADICIONAR:
    try:
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        df_odds = transform_betting_data(odds_data)

        for dados_evento in dicio:
            event_id = dados_evento.get('id')
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')
            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
            else:
                merged = {**dados_evento, "event_day": dia}
            novos_dados.append(merged)

        print(f"✅ Dia {dia} processado com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao processar dia {dia}: {e}")

# Adiciona ao DataFrame e salva
if novos_dados:
    df_novos = pd.DataFrame(novos_dados)

    for col in COLUNAS_PADRAO:
        if col not in df_novos.columns:
            df_novos[col] = None
    df_novos = df_novos[COLUNAS_PADRAO]

    df_final = pd.concat([df_existente, df_novos], ignore_index=True)
    df_final = df_final.sort_values(by="event_day", ascending=False)
    df_final.to_csv(CSV_FILE, index=False)
    print(f"💾 CSV atualizado com os novos dias: {DIAS_ADICIONAR}")
else:
    print("⚠️ Nenhum dado novo foi adicionado.")

'''
'''
import os
import pandas as pd
from datetime import datetime
from api import BetsAPIClient
from dotenv import load_dotenv

load_dotenv()

api = os.getenv("API_KEY")
apiclient = BetsAPIClient(api_key=api)
CSV_FILE = "resultados_60.csv"

# Colunas esperadas
COLUNAS_PADRAO = [
    'id', 'event_day', 'home', 'away','league', 'home_goals', 'away_goals', 'tot_goals',
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

DIAS_ALVO = ["20250505", "20250506"]

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

# Carregar CSV existente
if os.path.exists(CSV_FILE):
    df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
else:
    df_existente = pd.DataFrame(columns=COLUNAS_PADRAO)

# Remover os dois dias mais antigos


# Processar os dois dias desejados
novos_dados = []
for dia in DIAS_ALVO:
    try:
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        df_odds = transform_betting_data(odds_data)
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

# Salvar os novos dados no DataFrame
if novos_dados:
    df_novo = pd.DataFrame(novos_dados)
    for col in COLUNAS_PADRAO:
        if col not in df_novo.columns:
            df_novo[col] = None
    df_novo = df_novo[COLUNAS_PADRAO]
    df_existente = pd.concat([df_existente, df_novo], ignore_index=True)

# Reordenar do mais recente para o mais antigo
df_existente = df_existente.sort_values(by="event_day", ascending=False)

# Salvar no arquivo
df_existente.to_csv(CSV_FILE, index=False)
print("✅ CSV atualizado com os dias 20250505 e 20250506. Dois dias mais antigos removidos.")
'''
import main
import NN
import pandas as pd
import dqndc


'''
a = pd.read_csv('df_temp_preprocessado.csv')
NN.NN_double_chance(a)


a = pd.read_csv('df_temp_preprocessado_teste.csv')
NN.ql_dc(a)
NN.ql_dnb(a)
NN.ql_h(a)
NN.ql_gl(a)
main.main()
'''
import pandas as pd



main.checa()