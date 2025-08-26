import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from api import BetsAPIClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import logging
import pickle


import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split



logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'resultados_60.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

#'10048705', 'Esoccer GT Leagues - 12 mins play' ;'10047781', 'Esoccer Battle - 8 mins play'
#testar pegar evento 171732570 mais tarde   171790606  172006772 9723272 172006783
load_dotenv()
api = os.getenv("API_KEY")
apiclient = BetsAPIClient(api_key=api)

def dia_anterior():
        """Retorna o dia anterior ao atual no formato YYYYMMDD."""
        ontem = datetime.now() - timedelta(days=1)
        return ontem.strftime("%Y%m%d")

# Carregar o DataFrame
df_temp = pd.read_csv(CSV_FILE)

# Funções auxiliares para carregar/salvar modelos e scalers
def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, f'{model_name}.keras')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def load_scaler(scaler_name):
    scaler_path = os.path.join(MODELS_DIR, f'{scaler_name}.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_model(model, model_name):
    model_path = os.path.join(MODELS_DIR, f'{model_name}.keras')
    model.save(model_path)

def save_scaler(scaler, scaler_name):
    scaler_path = os.path.join(MODELS_DIR, f'{scaler_name}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

def preProcessGeneral(df=df_temp):
    '''realiza o preprocessamento geral de jogos passados'''
    df = preProcessEstatisticasGerais(df)
    df = preProcessOverUnder(df)
    df = preProcessHandicap(df)
    df = preProcessGoalLine(df)
    df = preProcessDoubleChance(df)
    df = preProcessDrawNoBet(df)
    return df


def preProcessGeneral_x(df):
    '''realiza o preprocessamento necessario para jogos futuros (a ser previstos)'''
    df = preProcessEstatisticasGerais_X(df)
    df = preProcessHandicap_X(df)
    df = preProcessGoalLine_X(df)
    return df



def preProcessEstatisticasGerais(df):
    '''calcula estatisticas do time individual e do confronto h2h de jogos antigos'''
    # Atualiza a chamada para passar também a data do jogo
    df[['media_goals_home','media_goals_sofridos_home', 'media_victories_home', 'media_goals_away','media_goals_sofridos_away', 'media_victories_away']] = df.apply(
        lambda row: estatisticas_ultimos_5(row['home'], row['away']),
        axis=1
    )

    # Aplica o cálculo de médias H2H (já está certo)
    medias = df.apply(
        lambda row: calcular_medias_h2h(row['home'], row['away'], row.name),
        axis=1
    )

    df['h2h_mean'] = medias.apply(lambda x: x['h2h_mean'])
    df['home_h2h_mean'] = medias.apply(lambda x: x['home_h2h_mean'])
    df['away_h2h_mean'] = medias.apply(lambda x: x['away_h2h_mean'])
    df['home_h2h_win_rate'] = medias.apply(lambda x: x['home_h2h_win_rate'])
    df['away_h2h_win_rate'] = medias.apply(lambda x: x['away_h2h_win_rate'])
    df['h2h_total_games'] = medias.apply(lambda x: x['h2h_total_games'])
    return df





def preProcessEstatisticasGerais_X(df):
    '''calcula estatisticas do time individual e do confronto h2h de jogos futuros, a serem previstos'''
    logger.info("🧮 Iniciando preProcessEstatisticasGerais_X")
    try:

        # Calcular as estatísticas das últimas 5 partidas
        df[['media_goals_home', 'media_goals_sofridos_home','media_victories_home', 'media_goals_away','media_goals_sofridos_away', 'media_victories_away']] = df.apply(
            lambda row: estatisticas_ultimos_5(int(row['home']), int(row['away'])),
            axis=1,
            result_type='expand'
        )
        logger.debug("📊 Estatísticas últimas 5 partidas calculadas com sucesso")

        # Calcular as estatísticas H2H
        medias = df.apply(
            lambda row: calcular_medias_h2h_X(int(row['home']), int(row['away'])),
            axis=1
        )
        df['h2h_mean'] = medias.apply(lambda x: x['h2h_mean'])
        df['home_h2h_mean'] = medias.apply(lambda x: x['home_h2h_mean'])
        df['away_h2h_mean'] = medias.apply(lambda x: x['away_h2h_mean'])
        df['home_h2h_win_rate'] = medias.apply(lambda x: x['home_h2h_win_rate'])
        df['away_h2h_win_rate'] = medias.apply(lambda x: x['away_h2h_win_rate'])
        df['h2h_total_games'] = medias.apply(lambda x: x['h2h_total_games'])
        

        

        logger.debug("📊 Estatísticas H2H calculadas")
        return df

    except Exception as e:
        logger.exception("❌ Erro em preProcessEstatisticasGerais_X")
        return df



def preProcessOverUnder(df=df_temp):
    '''parte de PreProcessgeneral'''
    df['res_goals_over_under'] = df['tot_goals'] > df['goals_over_under'].astype(float)
    return df




def preProcessHandicap(df=df_temp):
    '''parte de PreProcessgeneral'''
    # Aplicar a transformação para ambas as colunas
    df[['asian_handicap1_1', 'asian_handicap1_2']] = df['asian_handicap1'].apply(lambda x: pd.Series(split_handicap(x)))
    df[['asian_handicap2_1', 'asian_handicap2_2']] = df['asian_handicap2'].apply(lambda x: pd.Series(split_handicap(x)))
    df['diff_goals'] = df['home_goals'] - df['away_goals']
    # Aplicando a função ao DataFrame
    
    df['classificacao_ah1'] = df.apply(
        lambda row: classify_asian_handicap(row['team_ah1'], row['asian_handicap1_1'], row['asian_handicap1_2'],row['diff_goals']), axis=1
    )
    df['classificacao_ah2'] = df.apply(
        lambda row: classify_asian_handicap(row['team_ah2'], row['asian_handicap2_1'], row['asian_handicap2_2'], row['diff_goals']), axis=1
    )
    
    df['ah1_positivo'] = df['classificacao_ah1'] == 'positivo'
    df['ah1_negativo'] = df['classificacao_ah1'] == 'negativo'
    df['ah1_reembolso'] = df['classificacao_ah1'] == 'reembolso'
    df['ah1_indefinido'] = df['classificacao_ah1'] == 'indefinido'

    df['ah2_positivo'] = df['classificacao_ah2'] == 'positivo'
    df['ah2_negativo'] = df['classificacao_ah2'] == 'negativo'
    df['ah2_reembolso'] = df['classificacao_ah2'] == 'reembolso'
    df['ah2_indefinido'] = df['classificacao_ah2'] == 'indefinido'
    return df

def preProcessHandicap_i(df=df_temp):
    '''utilizado no main.py para fazer a checagem de resultados'''
    # Aplicar a transformação para ambas as colunas
    df[['asian_handicap1_1', 'asian_handicap1_2']] = df['asian_handicap1'].apply(lambda x: pd.Series(split_handicap(x)))
    df[['asian_handicap2_1', 'asian_handicap2_2']] = df['asian_handicap2'].apply(lambda x: pd.Series(split_handicap(x)))
    df['diff_goals'] = df['home_goals'] - df['away_goals']
    # Aplicando a função ao DataFrame
    
    df['classificacao_ah1'] = df.apply(
        lambda row: classify_asian_handicap_i(row['team_ah1'], row['asian_handicap1_1'], row['asian_handicap1_2'],row['diff_goals']), axis=1
    )
    df['classificacao_ah2'] = df.apply(
        lambda row: classify_asian_handicap_i(row['team_ah2'], row['asian_handicap2_1'], row['asian_handicap2_2'], row['diff_goals']), axis=1
    )
    
    df['ah1_positivo'] = df['classificacao_ah1'] == 'positivo'
    df['ah1_negativo'] = df['classificacao_ah1'] == 'negativo'
    df['ah1_reembolso'] = df['classificacao_ah1'] == 'reembolso'
    df['ah1_indefinido'] = df['classificacao_ah1'] == 'indefinido'
    df['ah1_meio_ganho'] = df['classificacao_ah1'] == 'meio_ganho'
    df['ah1_meia_perda'] = df['classificacao_ah1'] == 'meia_perda'

    df['ah2_positivo'] = df['classificacao_ah2'] == 'positivo'
    df['ah2_negativo'] = df['classificacao_ah2'] == 'negativo'
    df['ah2_reembolso'] = df['classificacao_ah2'] == 'reembolso'
    df['ah2_indefinido'] = df['classificacao_ah2'] == 'indefinido'
    df['ah2_meio_ganho'] = df['classificacao_ah2'] == 'meio_ganho'
    df['ah2_meia_perda'] = df['classificacao_ah2'] == 'meia_perda'

    return df


def preProcessHandicap_X(df):
    '''faz parte de preprocessgeneral_X'''
    logger.info("🧮 Iniciando preProcessHandicap_X")
    try:
        required_cols = ['asian_handicap1', 'asian_handicap2']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Colunas ausentes em preProcessHandicap_X: {missing_cols}")
            return df

        df[['asian_handicap1_1', 'asian_handicap1_2']] = df['asian_handicap1'].apply(lambda x: pd.Series(split_handicap(x)))
        df[['asian_handicap2_1', 'asian_handicap2_2']] = df['asian_handicap2'].apply(lambda x: pd.Series(split_handicap(x)))

        logger.debug("📊 Colunas de handicap divididas com sucesso")
        return df

    except Exception as e:
        logger.exception("❌ Erro em preProcessHandicap_X")
        return df



def preProcessGoalLine(df=df_temp):
    '''faz parte de preprocessgeneral'''
    # Para goal_line1
    df[['goal_line1_1', 'goal_line1_2']] = df['goal_line1'].apply(lambda x: pd.Series(split_goal_line(x)))
    # Para goal_line2
    df[['goal_line2_1', 'goal_line2_2']] = df['goal_line2'].apply(lambda x: pd.Series(split_goal_line(x)))
    # Aplicando ao DataFrame
    df['classificacao_gl1'] = df.apply(
        lambda row: classify_goal_line(row['type_gl1'], row['goal_line1_1'], row['goal_line1_2'], row['tot_goals']),
        axis=1
    )
    df['classificacao_gl2'] = df.apply(
        lambda row: classify_goal_line(row['type_gl2'], row['goal_line2_1'], row['goal_line2_2'], row['tot_goals']),
        axis=1
    )
    df = pd.get_dummies(df, columns=['classificacao_gl1'], prefix='gl1')
    df = pd.get_dummies(df, columns=['classificacao_gl2'], prefix='gl2')
    return df

def preProcessGoalLine_i(df=df_temp):
    '''utilizado no main.py para fazer a checagem de resultados'''
    # Separando os valores compostos de goal line
    df[['goal_line1_1', 'goal_line1_2']] = df['goal_line1'].apply(lambda x: pd.Series(split_goal_line(x)))
    df[['goal_line2_1', 'goal_line2_2']] = df['goal_line2'].apply(lambda x: pd.Series(split_goal_line(x)))

    # Classificando cada linha
    df['classificacao_gl1'] = df.apply(
        lambda row: classify_goal_line_i(row['type_gl1'], row['goal_line1_1'], row['goal_line1_2'], row['tot_goals']),
        axis=1
    )
    df['classificacao_gl2'] = df.apply(
        lambda row: classify_goal_line_i(row['type_gl2'], row['goal_line2_1'], row['goal_line2_2'], row['tot_goals']),
        axis=1
    )

    # Criando colunas booleanas
    for prefix in ['gl1', 'gl2']:
        df[f'{prefix}_positivo'] = df[f'classificacao_{prefix}'] == 'positivo'
        df[f'{prefix}_negativo'] = df[f'classificacao_{prefix}'] == 'negativo'
        df[f'{prefix}_reembolso'] = df[f'classificacao_{prefix}'] == 'reembolso'
        df[f'{prefix}_indefinido'] = df[f'classificacao_{prefix}'] == 'indefinido'
        df[f'{prefix}_meio_ganho'] = df[f'classificacao_{prefix}'] == 'meio_ganho'
        df[f'{prefix}_meia_perda'] = df[f'classificacao_{prefix}'] == 'meia_perda'

    return df



def preProcessGoalLine_X(df):
    '''faz parte de preprocessgeneral_X'''
    logger.info("🧮 Iniciando preProcessGoalLine_X")
    try:
        required_cols = ['goal_line1', 'goal_line2']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Colunas ausentes em preProcessGoalLine_X: {missing_cols}")
            return df

        df[['goal_line1_1', 'goal_line1_2']] = df['goal_line1'].apply(lambda x: pd.Series(split_goal_line(x)))
        df[['goal_line2_1', 'goal_line2_2']] = df['goal_line2'].apply(lambda x: pd.Series(split_goal_line(x)))

        logger.debug("📊 Colunas de goal line divididas com sucesso")
        return df

    except Exception as e:
        logger.exception("❌ Erro em preProcessGoalLine_X")
        return df



def preProcessDoubleChance(df=df_temp):
    '''faz parte de preprocessgeneral'''
    df = calcular_resultado_double_chance(df)
    df = calcular_resultado_double_chance_ind(df)
    return df

def preProcessDrawNoBet(df=df_temp):
    '''faz parte de preprocessgeneral'''
    df['res_draw_no_bet1'] = df.apply(
    lambda row: classify_draw_no_bet(row['draw_no_bet_team1'], row['home_goals'], row['away_goals']), axis=1
    )
    df['res_draw_no_bet2'] = df.apply(
        lambda row: classify_draw_no_bet(row['draw_no_bet_team2'], row['home_goals'], row['away_goals']), axis=1
    )
    df = pd.get_dummies(df, columns=['res_draw_no_bet1'], prefix='dnb1')
    df = pd.get_dummies(df, columns=['res_draw_no_bet2'], prefix='dnb2')
    return df

def preProcessDrawNoBet_i(df=df_temp):
    '''utilizado no main.py para checagem de resultados'''
    # Classificação das apostas DNB
    df['res_draw_no_bet1'] = df.apply(
        lambda row: classify_draw_no_bet(row['draw_no_bet_team1'], row['home_goals'], row['away_goals']), axis=1
    )
    df['res_draw_no_bet2'] = df.apply(
        lambda row: classify_draw_no_bet(row['draw_no_bet_team2'], row['home_goals'], row['away_goals']), axis=1
    )

    # Criando colunas booleanas explícitas para DNB1 e DNB2
    for prefix in ['dnb1', 'dnb2']:
        df[f'{prefix}_ganha'] = df[f'res_draw_no_bet{prefix[-1]}'] == 'ganha'
        df[f'{prefix}_perde'] = df[f'res_draw_no_bet{prefix[-1]}'] == 'perde'
        df[f'{prefix}_reembolso'] = df[f'res_draw_no_bet{prefix[-1]}'] == 'reembolso'
        df[f'{prefix}_indefinido'] = df[f'res_draw_no_bet{prefix[-1]}'] == 'indefinido'

    return df








'''funçoes de normalização e divisão'''

def normalizacao_and_split(X, y):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_standardized,y, random_state=42, test_size=0.2)
    return x_train, x_test, y_train, y_test

def normalizacao(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized
    
def split(X_standardized, y):
    x_train, x_test, y_train, y_test = train_test_split(X_standardized,y, random_state=42, test_size=0.2)
    return x_train, x_test, y_train, y_test










def estatisticas_ultimos_5(home_team, away_team):
    '''pega estatisticas dos ultimos 5 jogos dos times (sem confronto direto)'''
    try:
        # Filtra os 8 jogos mais recentes do home_team, seja como mandante ou visitante
        df_home = df_temp[(df_temp['home'] == home_team) | (df_temp['away'] == home_team)].head(5)

        if not df_home.empty:
            # Gols marcados pelo home_team em cada jogo
            df_home['gols_home_team'] = df_home.apply(
                lambda row: row['home_goals'] if row['home'] == home_team else row['away_goals'], axis=1
            )

            # Gols sofridos pelo home_team em cada jogo
            df_home['gols_sofridos_home_team'] = df_home.apply(
                lambda row: row['away_goals'] if row['home'] == home_team else row['home_goals'], axis=1
            )

            # Verifica se o home_team venceu o jogo
            df_home['vitoria_home_team'] = df_home.apply(
                lambda row: (
                    row['home_goals'] > row['away_goals'] if row['home'] == home_team
                    else row['away_goals'] > row['home_goals']
                ), axis=1
            )

            # Calcula as médias com base nas colunas criadas
            media_gols_home = df_home['gols_home_team'].mean()
            media_gols_sofridos_home = df_home['gols_sofridos_home_team'].mean()
            vitorias_home = df_home['vitoria_home_team'].mean()
        else:
            media_gols_home = np.nan
            media_gols_sofridos_home = np.nan
            vitorias_home = np.nan

        # Filtra os 8 jogos mais recentes do away_team, seja como mandante ou visitante
        df_away = df_temp[(df_temp['away'] == away_team) | (df_temp['home'] == away_team)].head(5)

        if not df_away.empty:
            # Calcula os gols marcados pelo away_team em cada jogo (independente de ser mandante ou visitante)
            df_away['gols_away_team'] = df_away.apply(
                lambda row: row['away_goals'] if row['away'] == away_team else row['home_goals'], axis=1
            )

            # Calcula os gols sofridos pelo away_team em cada jogo
            df_away['gols_sofridos_away_team'] = df_away.apply(
                lambda row: row['home_goals'] if row['away'] == away_team else row['away_goals'], axis=1
            )

            # Calcula se o away_team venceu em cada jogo
            df_away['vitoria_away_team'] = df_away.apply(
                lambda row: (
                    row['away_goals'] > row['home_goals'] if row['away'] == away_team
                    else row['home_goals'] > row['away_goals']
                ), axis=1
            )

            # Agora calcula as médias com base nas novas colunas
            media_gols_away = df_away['gols_away_team'].mean()
            media_gols_sofridos_away = df_away['gols_sofridos_away_team'].mean()
            vitorias_away = df_away['vitoria_away_team'].mean()
        else:
            media_gols_away = np.nan
            media_gols_sofridos_away = np.nan
            vitorias_away = np.nan

        return pd.Series({
            'media_goals_home': media_gols_home,
            'media_goals_sofridos_home': media_gols_sofridos_home,
            'media_victories_home': vitorias_home,
            'media_goals_away': media_gols_away,
            'media_goals_sofridos_away': media_gols_sofridos_away,
            'media_victories_away': vitorias_away
        })

    except Exception as e:
        print(f"❌ Erro em estatisticas_ultimos_5 para {home_team} x {away_team}: {e}")
        return pd.Series({
            'media_goals_home': np.nan,
            'media_goals_sofridos_home': np.nan,
            'media_victories_home': np.nan,
            'media_goals_away': np.nan,
            'media_goals_sofridos_away': np.nan,
            'media_victories_away': np.nan
        })


def calcular_medias_h2h(home_id, away_id, index):
    '''calcula as estatisticas de confronto direto para jogos antigos'''
    """
    Calcula seis estatísticas de confronto direto com base na data do jogo:
    - h2h_mean: média de gols totais nos confrontos anteriores
    - home_h2h_mean: média de gols marcados pelo time atual como mandante
    - away_h2h_mean: média de gols marcados pelo time atual como visitante
    - home_h2h_win_rate: média de vitórias do time mandante nos confrontos
    - away_h2h_win_rate: média de vitórias do time visitante nos confrontos
    - h2h_total_games: total de confrontos encontrados (máximo 10)
    """
    df = df_temp.copy()

    
    confrontos = df[((df['home'] == home_id) & (df['away'] == away_id))|
                    ((df['home'] == away_id) & (df['away'] == home_id))]

    # Filtrar apenas confrontos ocorridos antes da data do jogo atual
    confrontos_passados = confrontos.loc[index + 1:].head(10)

    # Total de confrontos encontrados
    total_confrontos = len(confrontos_passados)

    if confrontos_passados.empty:
        return {
            'h2h_mean': None,
            'home_h2h_mean': None,
            'away_h2h_mean': None,
            'home_h2h_win_rate': None,
            'away_h2h_win_rate': None,
            'h2h_total_games': 0
        }
    

    # Média de gols totais por confronto
    h2h_mean = confrontos_passados['tot_goals'].mean()

    # Ajustar médias por lado (quem está sendo analisado como home/away no jogo atual)
    home_goals = []
    away_goals = []
    home_wins = []
    away_wins = []
    
    for _, row in confrontos_passados.iterrows():
        if row['home'] == home_id:
            # Time atual (home_id) jogou como mandante neste confronto passado
            home_goals.append(row['home_goals'])
            away_goals.append(row['away_goals'])
            home_wins.append(1 if row['home_goals'] > row['away_goals'] else 0)
            away_wins.append(1 if row['away_goals'] > row['home_goals'] else 0)
        else:
            # Time atual (home_id) jogou como visitante neste confronto passado
            home_goals.append(row['away_goals'])
            away_goals.append(row['home_goals'])
            home_wins.append(1 if row['away_goals'] > row['home_goals'] else 0)
            away_wins.append(1 if row['home_goals'] > row['away_goals'] else 0)

    return {
        'h2h_mean': h2h_mean,
        'home_h2h_mean': np.mean(home_goals),
        'away_h2h_mean': np.mean(away_goals),
        'home_h2h_win_rate': np.mean(home_wins),
        'away_h2h_win_rate': np.mean(away_wins),
        'h2h_total_games': total_confrontos
    }

def calcular_medias_h2h_X(home_id, away_id):
    '''calcula as estatisticas de confronto direto para jogos antigos'''

    """
    Calcula seis estatísticas de confronto direto, considerando apenas os últimos 10 confrontos
    anteriores à data do jogo:
    - h2h_mean: média de gols totais (home_goals + away_goals) nos confrontos anteriores
    - home_h2h_mean: média de gols marcados pelo time da casa (home_id) nos confrontos
    - away_h2h_mean: média de gols marcados pelo time visitante (away_id) nos confrontos
    - home_h2h_win_rate: média de vitórias do time mandante nos confrontos
    - away_h2h_win_rate: média de vitórias do time visitante nos confrontos
    - h2h_total_games: total de confrontos encontrados (máximo 10)
    """
    df = df_temp.copy()

    # Filtrar todos os confrontos entre os times
    
    confrontos = df[((df['home'] == home_id) & (df['away'] == away_id))|
                    ((df['home'] == away_id) & (df['away'] == home_id))]

    if confrontos.empty:
        return {
            'h2h_mean': None, 
            'home_h2h_mean': None, 
            'away_h2h_mean': None,
            'home_h2h_win_rate': None,
            'away_h2h_win_rate': None,
            'h2h_total_games': 0
        }

    # Filtrar apenas os confrontos anteriores à data do jogo
    confrontos_passados = confrontos.head(10)
    
    # Total de confrontos encontrados
    total_confrontos = len(confrontos_passados)

    if confrontos_passados.empty:
        return {
            'h2h_mean': None, 
            'home_h2h_mean': None, 
            'away_h2h_mean': None,
            'home_h2h_win_rate': None,
            'away_h2h_win_rate': None,
            'h2h_total_games': 0
        }

    # Calcular média geral de gols totais
    h2h_mean = confrontos_passados['tot_goals'].mean()

    # Calcular média de gols específicos por time e vitórias
    home_goals = []
    away_goals = []
    home_wins = []
    away_wins = []
    
    for _, row in confrontos_passados.iterrows():
        if row['home'] == home_id:
            # home_id jogou como mandante neste confronto
            home_goals.append(row['home_goals'])
            away_goals.append(row['away_goals'])
            home_wins.append(1 if row['home_goals'] > row['away_goals'] else 0)
            away_wins.append(1 if row['away_goals'] > row['home_goals'] else 0)
        else:
            # home_id jogou como visitante neste confronto
            home_goals.append(row['away_goals'])
            away_goals.append(row['home_goals'])
            home_wins.append(1 if row['away_goals'] > row['home_goals'] else 0)
            away_wins.append(1 if row['home_goals'] > row['away_goals'] else 0)

    home_h2h_mean = np.mean(home_goals) if home_goals else None
    away_h2h_mean = np.mean(away_goals) if away_goals else None
    home_h2h_win_rate = np.mean(home_wins) if home_wins else None
    away_h2h_win_rate = np.mean(away_wins) if away_wins else None

    return {
        'h2h_mean': h2h_mean,
        'home_h2h_mean': home_h2h_mean,
        'away_h2h_mean': away_h2h_mean,
        'home_h2h_win_rate': home_h2h_win_rate,
        'away_h2h_win_rate': away_h2h_win_rate,
        'h2h_total_games': total_confrontos
    }

#HANDICAP

def split_handicap(value):
    """Separa handicaps compostos em duas colunas. Se for simples, duplica o valor."""
    if pd.isna(value):  # Tratar valores nulos
        return np.array([np.nan, np.nan])
    value = str(value).strip()  # Garantir que seja string
    if ',' in value:  # Handicap composto (ex: "-0.75,-1")
        colon = value.find(',')
        c1 = float(value[:colon])
        c2 = float(value[colon+1:])
        return np.array([c1, c2])
    else:  # Handicap simples, duplicar (ex: "-1" → [-1, -1])
        c = float(value)
        return np.array([c, c])



def classify_asian_handicap(team, ah1, ah2, diff_goals):
    """
    Classifica uma aposta em handicap asiático para uma equipe, incluindo 'indefinido' para valores NaN.
    
    Args:
        team: 1.0 para time da casa, 2.0 para visitante
        ah1: primeiro valor do handicap asiático
        ah2: segundo valor do handicap asiático
        diff_goals: diferença de gols (home_goals - away_goals)
        
    Returns:
        'positiva', 'negativa', 'reembolso' ou 'indefinido'
    """
    team, ah1, ah2, diff_goals = float(team), float(ah1), float(ah2), float(diff_goals)
    # Verifica se algum valor é NaN
    if pd.isna(team) or pd.isna(ah1) or pd.isna(ah2) or pd.isna(diff_goals):
        return 'indefinido'
    # Ajusta a diferença de gols conforme o time (inverte para visitante)
    adjusted_diff = diff_goals if team == 1.0 else -diff_goals
    # Verifica se é handicap simples
    if ah1 == ah2:
        resultado = adjusted_diff + ah1
        if resultado > 0:
            return 'positivo'
        elif resultado < 0:
            return 'negativo'
        else:
            return 'reembolso'
    else:
        # Handicap composto
        res1 = adjusted_diff + ah1
        res2 = adjusted_diff + ah2
        
        if res1 > 0 and res2 > 0:
            return 'positivo'
        elif res1 < 0 and res2 < 0:
            return 'negativo'
        else:
            return 'reembolso'



def classify_asian_handicap_i(team, ah1, ah2, diff_goals):
    """
    Classifica uma aposta em handicap asiático para uma equipe, incluindo 'indefinido' para valores NaN.

    Args:
        team: 1.0 para time da casa, 2.0 para visitante
        ah1: primeiro valor do handicap asiático
        ah2: segundo valor do handicap asiático
        diff_goals: diferença de gols (home_goals - away_goals)

    Returns:
        'positivo', 'negativo', 'reembolso', 'meio ganho', 'meia perda', 'meia' ou 'indefinido'
    """
    import pandas as pd

    team, ah1, ah2, diff_goals = float(team), float(ah1), float(ah2), float(diff_goals)

    if pd.isna(team) or pd.isna(ah1) or pd.isna(ah2) or pd.isna(diff_goals):
        return 'indefinido'

    adjusted_diff = diff_goals if team == 1.0 else -diff_goals

    if ah1 == ah2:
        resultado = adjusted_diff + ah1
        if resultado > 0:
            return 'positivo'
        elif resultado < 0:
            return 'negativo'
        else:
            return 'reembolso'
    else:
        res1 = adjusted_diff + ah1
        res2 = adjusted_diff + ah2

        if res1 > 0 and res2 > 0:
            return 'positivo'
        elif res1 < 0 and res2 < 0:
            return 'negativo'
        elif res1 == 0 and res2 == 0:
            return 'reembolso'
        elif (res1 > 0 and res2 == 0) or (res2 > 0 and res1 == 0):
            return 'meio_ganho'
        elif (res1 < 0 and res2 == 0) or (res2 < 0 and res1 == 0):
            return 'meia_perda'
        elif (res1 > 0 and res2 < 0) or (res2 > 0 and res1 < 0):
            return 'reembolso'
        else:
            return 'indefinido'  # fallback para casos extremos

#GOAL_LINE

def split_goal_line(value):
    """Separa handicaps compostos da goal_line em duas colunas. Se for simples, duplica o valor."""
    if pd.isna(value):  # Tratar valores nulos
        return np.array([np.nan, np.nan])
    
    value = str(value).strip()  # Garantir que seja string
    if ',' in value:  # Handicap composto (ex: "1.5,2")
        parts = value.split(',')
        return np.array([float(parts[0]), float(parts[1])])
    else:  # Handicap simples (ex: "1.5" → [1.5, 1.5])
        c = float(value)
        return np.array([c, c])



def classify_goal_line(team_gl, gl1, gl2, tot_goals):
    """
    Classifica uma aposta em Goal Line em:
    - 'positivo': aposta vencedora
    - 'negativo': aposta perdedora
    - 'reembolso': empate exato no handicap
    - 'indefinido': dados inválidos
    
    Args:
        team_gl: 1 para Over, 2 para Under
        gl1: primeiro valor do handicap (ex: 1.5)
        gl2: segundo valor do handicap (para handicaps compostos)
        tot_goals: total de gols do jogo (home_goals + away_goals)
    """
    team_gl, gl1, gl2, tot_goals = float(team_gl), float(gl1), float(gl2),float(tot_goals)
    # Verificação de valores nulos
    if pd.isna(team_gl) or pd.isna(gl1) or pd.isna(tot_goals):
        return 'indefinido'
    
    # Handicap simples (quando gl1 == gl2)
    if gl1 == gl2:
        if tot_goals > gl1:
            return 'positivo' if team_gl == 1 else 'negativo'
        elif tot_goals < gl1:
            return 'negativo' if team_gl == 1 else 'positivo'
        else:
            return 'reembolso'
    
    # Handicap composto
    else:
        if team_gl == 1:  # Aposta em Over
            if tot_goals > gl2:
                return 'positivo'  # Ganho total
            elif tot_goals < gl1:
                return 'negativo'  # Perda total
            else:
                return 'reembolso'  # Meio ganho/meio reembolso
        else:  # Aposta em Under (team_gl == 2)
            if tot_goals < gl1:
                return 'positivo'  # Ganho total
            elif tot_goals > gl2:
                return 'negativo'  # Perda total
            else:
                return 'reembolso'  # Meio ganho/meio reembolso


def classify_goal_line_i(team_gl, gl1, gl2, tot_goals):
    """
    Classifica uma aposta em Goal Line asiático para Over (1) ou Under (2), com suporte a handicaps compostos.

    Args:
        team_gl: 1 para Over, 2 para Under
        gl1: primeira linha do goal line (float)
        gl2: segunda linha do goal line (float)
        tot_goals: total de gols no jogo (home_goals + away_goals)

    Returns:
        'positivo', 'negativo', 'meio ganho', 'meia perda', 'reembolso', ou 'indefinido'
    """
    try:
        team_gl, gl1, gl2, tot_goals = float(team_gl), float(gl1), float(gl2), float(tot_goals)
    except (TypeError, ValueError):
        return 'indefinido'

    if pd.isna(team_gl) or pd.isna(gl1) or pd.isna(gl2) or pd.isna(tot_goals):
        return 'indefinido'

    # Handicap simples
    if gl1 == gl2:
        diff = tot_goals - gl1 if team_gl == 1 else gl1 - tot_goals
        if diff > 0:
            return 'positivo'
        elif diff < 0:
            return 'negativo'
        else:
            return 'reembolso'

    # Handicap composto
    else:
        res1 = tot_goals - gl1 if team_gl == 1 else gl1 - tot_goals
        res2 = tot_goals - gl2 if team_gl == 1 else gl2 - tot_goals

        if res1 > 0 and res2 > 0:
            return 'positivo'
        elif res1 < 0 and res2 < 0:
            return 'negativo'
        elif (res1 == 0 and res2 > 0) or (res2 == 0 and res1 > 0):
            return 'meio_ganho'
        elif (res1 == 0 and res2 < 0) or (res2 == 0 and res1 < 0):
            return 'meia_perda'
        else:
            return 'reembolso'

#DOUBLE_CHANCE
def calcular_resultado_double_chance(df):
    # Double Chance 1: vitória do time da casa ou empate
    df['res_double_chance1'] = ((df['home_goals'] > df['away_goals']) | (df['home_goals'] == df['away_goals'])).astype(int)

    # Double Chance 2: vitória do time visitante ou empate
    df['res_double_chance2'] = ((df['away_goals'] > df['home_goals']) | (df['home_goals'] == df['away_goals'])).astype(int)

    # Double Chance 3: vitória de qualquer time (não pode empatar)
    df['res_double_chance3'] = ((df['home_goals'] != df['away_goals'])).astype(int)
    return df
def calcular_resultado_double_chance_ind(df):
    def get_result(row):
        if row['home_goals'] > row['away_goals']:
            return 1, 0, 0
        elif row['home_goals'] < row['away_goals']:
            return 0, 1, 0
        else:
            return 0, 0, 1

    df[['res_game_home', 'res_game_away', 'res_game_empate']] = df.apply(get_result, axis=1, result_type='expand')
    return df



#criar nn 1, 2 e 3


#DRAW_NO_BET
def classify_draw_no_bet(team, home_goals, away_goals):
    """
    Classifica o resultado de uma aposta Draw No Bet com base no time escolhido e no resultado do jogo.
    
    Args:
        team: 1.0 para time da casa, 2.0 para visitante
        home_goals: gols do time da casa
        away_goals: gols do time visitante
        
    Returns:
        'ganha', 'perde', 'reembolso' ou 'indefinido'
    """
    team, home_goals, away_goals = float(team), float(home_goals), float(away_goals)
    if pd.isna(team) or pd.isna(home_goals) or pd.isna(away_goals):
        return 'indefinido'
    
    diff_goals = home_goals - away_goals
    
    if team == 1.0:  # aposta no time da casa
        if diff_goals > 0:
            return 'ganha'
        elif diff_goals < 0:
            return 'perde'
        else:
            return 'reembolso'
    
    elif team == 2.0:  # aposta no visitante
        if diff_goals < 0:
            return 'ganha'
        elif diff_goals > 0:
            return 'perde'
        else:
            return 'reembolso'
    
    return 'indefinido'






