
'''import time
import os
import pandas as pd
from datetime import datetime, timedelta
import requests
from api import BetsAPIClient
from dotenv import load_dotenv

load_dotenv()

api = os.getenv("API_KEY")
apiclient = BetsAPIClient(api_key=api)
CSV_FILE = "resultados_60.csv"

# Lista de colunas esperadas no CSV final, mesmo que estejam vazias
COLUNAS_PADRAO = [
    'id', 'event_day', 'home', 'away','league','time', 'home_goals', 'away_goals', 'tot_goals',
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

def ultimos_40_dias(data_str: str):
    """Retorna uma lista com todos os dias dos últimos 40 dias anteriores à data fornecida no formato YYYYMMDD."""
    data = datetime.strptime(data_str, "%Y%m%d")
    return [(data - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, 41)]

# Carregar dias já processados
if os.path.exists(CSV_FILE):
    df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
    dias_processados = set(df_existente["event_day"].unique())
else:
    dias_processados = set()


# Configuração: pular os N dias mais recentes (ex: já processados)
DIAS_JA_PROCESSADOS = 0  # você pode alterar isso

# Obter os últimos 40 dias
todos_os_dias = ultimos_40_dias(dia_anterior())

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
        
        # Garantir que todas as colunas existam e estejam na ordem correta
        for col in COLUNAS_PADRAO:
            if col not in df_novo.columns:
                df_novo[col] = None
        df_novo = df_novo[COLUNAS_PADRAO]
        
        # Ordenar por event_day
        df_novo["time"] = df_novo["time"].astype(int)
        df_novo = df_novo.sort_values(by="time", ascending=False)

        if not os.path.exists(CSV_FILE):
            df_novo.to_csv(CSV_FILE, index=False)  # Cria um novo arquivo
        else:
            df_novo.to_csv(CSV_FILE, mode="a", header=False, index=False)  # Adiciona ao existente

    print(f"✅ Total de dias processados: {len(dias_processados)}")
    print("⏳ Aguardando 1 hora para a próxima execução...")
    time.sleep(3600)
    ]'''
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
CSV_FILE = "resultados_60.csv"
import main
import pandas as pd
import os
from datetime import datetime, timedelta
import time

import time
import os
import pandas as pd
from datetime import datetime, timedelta

def gerar_lista_dias_anteriores(ultimos_n: int, referencia: str = None) -> list[str]:
    """
    Gera os últimos `ultimos_n` dias no formato 'yyyymmdd' a partir da data de referência.
    Se `referencia` for None, usa a data atual como base e pega os dias anteriores (1..n).
    """
    if referencia:
        base = datetime.strptime(referencia, "%Y%m%d")
    else:
        base = datetime.utcnow()
    dias = []
    for i in range(1, ultimos_n + 1):
        d = base - timedelta(days=i)
        dias.append(d.strftime("%Y%m%d"))
    return dias

def _esperar_proximo_intervalo_horario():
    now = datetime.utcnow()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    segundos = (next_hour - now).total_seconds()
    print(f"⏳ Limite de 6 dias por hora atingido. Aguardando {int(segundos)}s até {next_hour.strftime('%Y-%m-%d %H:%M:%S')} UTC...")
    time.sleep(segundos)

def processar_ultimos_dias(n: int):
    """
    Processa os últimos `n` dias anteriores (sem incluir o dia corrente), em lotes de 6 dias por hora.
    Mantém o padrão de colunas, converte 'time' para int, ordena por event_day desc e remove duplicatas por 'id'
    mantendo a primeira aparição (mais recente).
    """
    COLUNAS_PADRAO = [
        'id', 'event_day', 'home', 'away','league','time', 'home_goals', 'away_goals', 'tot_goals',
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

    dias = gerar_lista_dias_anteriores(n)
    print(f"🔄 Processando jogos dos últimos {n} dias: {', '.join(dias)}")

    acumulado = []
    lote_size = 6
    for i in range(0, len(dias), lote_size):
        lote = dias[i : i + lote_size]
        print(f"\n🗂 Iniciando lote de dias: {', '.join(lote)}")
        for dia in lote:
            try:
                print(f"📅 Processando dia {dia}...")
                print("🔎 Buscando IDs e dicionário de eventos...")
                ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
                print(f"✔️ {len(ids)} eventos encontrados para {dia}.")

                print("📊 Filtrando e transformando odds...")
                odds_data = apiclient.filtraOddsNovo(ids=ids)
                df_odds = apiclient.transform_betting_data(odds_data)

                for dados_evento in dicio:
                    event_id = dados_evento.get('id')
                    odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')
                    if odds_transformadas:
                        merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
                    else:
                        merged = {**dados_evento, "event_day": dia}
                    acumulado.append(merged)
                print(f"🧩 {len(dicio)} eventos do dia {dia} preparados.")
            except Exception as e:
                print(f"❌ Erro ao processar o dia {dia}: {type(e).__name__}: {e} (pulando)")

        # se ainda há dias depois deste lote, respeita limite e espera até próxima hora
        if i + lote_size < len(dias):
            _esperar_proximo_intervalo_horario()

    if not acumulado:
        print(f"⚠️ Nenhum dado agregado para os últimos {n} dias.")
        return

    df_novo = pd.DataFrame(acumulado)

    # Garantir padrão de colunas e preencher ausentes
    colunas_adicionadas = []
    for coluna in COLUNAS_PADRAO:
        if coluna not in df_novo.columns:
            df_novo[coluna] = None
            colunas_adicionadas.append(coluna)
    if colunas_adicionadas:
        print(f"➕ Colunas adicionadas automaticamente: {', '.join(colunas_adicionadas)}")

    df_novo = df_novo[COLUNAS_PADRAO]

    # Mesclar com CSV existente
    if os.path.exists(CSV_FILE):
        print("📂 CSV existente encontrado, mesclando dados...")
        df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        print("📄 Nenhum CSV encontrado, criando novo arquivo...")
        df_final = df_novo

    # Garantir time como int
    if "time" in df_final.columns:
        df_final["time"] = pd.to_numeric(df_final["time"], errors="coerce").fillna(0).astype(int)
    else:
        df_final["time"] = 0

    # Ordenar: event_day mais recente em cima e time decrescente
    df_final = df_final.sort_values(by=["event_day", "time"], ascending=[False, False]).reset_index(drop=True)

    # Remover duplicatas por 'id', mantendo a primeira (mais recente)
    if "id" in df_final.columns:
        antes = len(df_final)
        df_final = df_final.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
        removidas = antes - len(df_final)
        print(f"🧹 Removidas {removidas} duplicatas com base em 'id'.")
    else:
        print("⚠️ Coluna 'id' inexistente; não foi possível deduplicar.")

    # Salvar e limpeza extra
    df_final.to_csv(CSV_FILE, index=False)
    try:
        main.remover_duplicatas()
    except Exception as e:
        print(f"⚠️ Falha ao chamar remover_duplicatas(): {type(e).__name__}: {e}")

    primeiro_dia = df_final["event_day"].min() if "event_day" in df_final.columns else "N/A"
    print(f"✅ Dados atualizados com sucesso. Primeiro dia presente: {primeiro_dia}.")

processar_ultimos_dias(1)
