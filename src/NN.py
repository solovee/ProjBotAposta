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
import xgboost as xgb
import joblib
from sklearn.linear_model import LogisticRegression
from autogluon.tabular import TabularPredictor
import pandas as pd
import random
from datetime import datetime
import tempfile
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Importando funções de discretização e a classe Q-Learning
from qlearning import (
    discretizar_goals, discretizar_vitorias, discretizar_odds, 
    discretizar_goal_diff, discretizar_league, q_learning_dc,q_learning_gl,
    QLearningDoubleChance, QLearningGoalLine,QLearningDrawNoBet, QLearningHandicap,q_learning_h
)


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

def criaNNs():
    '''cria todos os modelos autogluon'''
    df = df_temp.copy()
    df = preProcessGeneral(df)
    df.to_csv('df_temp_preprocessado.csv', index=False)
    z_over_under_positivo, z_over_under_negativo = NN_over_under(df)
    z_handicap = NN_handicap(df)
    z_goal_line = NN_goal_line(df)
    z_double_chance = NN_double_chance(df)
    z_draw_no_bet = NN_draw_no_bet(df)
    lista = [z_over_under_positivo, z_over_under_negativo, z_handicap, z_goal_line, z_double_chance , z_draw_no_bet]
    return lista


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


#criar nn 1 e 2



def encontrar_melhor_z_binario_positivo(y_true, y_pred_probs, min_percent=0.85):
    thresholds_pos = np.arange(0.5, 1.01, 0.025)
    # Total de previsões positivas (com qualquer confiança)
    total_pred_positivas = np.sum(y_pred_probs >= 0.5)
    melhor_z = None
    melhor_acc = 0
    melhor_n = 0

    for z in thresholds_pos:
        # Previsões com probabilidade ≥ z (alta confiança na classe positiva)
        mask = (y_pred_probs >= z)
        mask = mask.ravel()
        n = np.sum(mask)
        correct = np.sum(y_true[mask] == 1)
        acc = correct / n if n > 0 else 0

        # Verifica se atende ao critério de mínimo de 5% das positivas
        if n >= total_pred_positivas * min_percent:
            if acc > melhor_acc:
                melhor_acc = acc
                melhor_z = z
                melhor_n = n
    

    return melhor_z



def encontrar_melhor_z_binario_negativo(y_true, y_pred_probs, min_percent=0.9):

    thresholds_neg = np.arange(0.5, -0.01, -0.025)
    # Total de previsões negativas (com qualquer confiança)
    total_pred_negativas = np.sum(y_pred_probs < 0.5)

    melhor_z = None
    melhor_acc = 0
    melhor_n = 0

    for z in thresholds_neg:
        # Previsões com probabilidade ≤ z (alta confiança na classe negativa)
        mask = (y_pred_probs <= z)
        mask = mask.ravel()  # Alternativa a flatten
        n = np.sum(mask)
        correct = np.sum(y_true[mask] == 0)
        acc = correct / n if n > 0 else 0

        # Verifica se atende ao critério de mínimo de 5% das negativas
        if n >= total_pred_negativas * min_percent:
            if acc > melhor_acc:
                melhor_acc = acc
                melhor_z = z
                melhor_n = n

    return melhor_z

def encontrar_melhor_z_softmax_positivo(y_test, y_pred_probs, min_percent=0.1):

    thresholds=np.arange(0.40, 1.01, 0.025)

    classe_positiva = 1

    # Classes reais e previstas
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Total de previsões classificadas como positivas (classe t)
    total_pred_positivas = np.sum(y_pred_classes == classe_positiva)

    melhor_z = None
    melhor_acc = 0
    melhor_n = 0

    for z in thresholds:
        # Máscara para previsões positivas com confiança ≥ z
        mask = (y_pred_classes == classe_positiva) & (y_pred_probs[:, classe_positiva] >= z)
        n = np.sum(mask)
        correct = np.sum(mask & (y_true_classes == classe_positiva))
        acc = correct / n if n > 0 else 0

        # Verifica se atende o critério de mínimo de 5%
        if n >= total_pred_positivas * min_percent:
            if acc > melhor_acc:
                melhor_acc = acc
                melhor_z = z
                melhor_n = n

    return melhor_z

def prepNNOver_under_X(df=df_temp):
    '''prepara os jogos da linha para o autogluon'''
    required_columns = ['home','away','times','league','odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNOver_under_X': {', '.join(missing_cols)}")
        return None, None
    
    
    df_temporario = df[required_columns].copy()
    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNoverunder*")
        return None, None
    z = df_temporario[['times','odd_goals_over1', 'odd_goals_under1', 'league']].copy()

    X = df_temporario[['odd_goals_over1', 'odd_goals_under1','media_goals_home','media_goals_away' ,'h2h_mean', 'league']]

    try:
        with open('scaler_over_under.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_standardized = scaler.transform(X)
        X = pd.DataFrame(X_standardized, columns=['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean', 'league']).reset_index(drop=True)
    except Exception as e:
        print('Erro over_under')
        return None, None
    print("Colunas de X (over/under):", X.columns.tolist())
    return X, z

def prepNNHandicap_X(df=df_temp):
    '''prepara os jogos da linha para o autogluon'''
    required_columns = ['home','away','times','media_goals_home', 'media_goals_away','home_h2h_mean', 'away_h2h_mean',
                        'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 
                        'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2','league','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home']
 
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNHandicap_X': {', '.join(missing_cols)}")
        return None, None
    df_temporario = df[required_columns].copy()
    df_temporario['odds_ah1'] = pd.to_numeric(df_temporario['odds_ah1'], errors='coerce')
    df_temporario['odds_ah2'] = pd.to_numeric(df_temporario['odds_ah2'], errors='coerce')

    df_temporario['favorite_by_odds'] = df_temporario['odds_ah1'] < df_temporario['odds_ah2']
    df_temporario['odds_ratio'] = df_temporario['odds_ah1'] / df_temporario['odds_ah2']
    
    df_temporario = preparar_df_handicaps_X(df_temporario)
    df_temporario['goals_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['h2h_diff'] = df_temporario['home_h2h_mean'] - df_temporario['away_h2h_mean']
    df_temporario['home_ratio'] = df_temporario['media_goals_home'] - df_temporario['media_goals_sofridos_home']
    df_temporario['away_ratio'] = df_temporario['media_goals_away'] - df_temporario['media_goals_sofridos_away']
    df_temporario['vic_ratio'] = df_temporario['home_h2h_win_rate'] - df_temporario['away_h2h_win_rate']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']

    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNhandicap*")
        return None, None
    z = df_temporario[['times','team_ah','asian_handicap_1', 'asian_handicap_2', 'odds', 'league']]
    z = z.sort_values('team_ah').reset_index(drop=True)
    df_temporario = df_temporario.sort_values('team_ah').reset_index(drop=True)


    df_temporario['team_ah'] = df_temporario['team_ah'].astype(float)

    df_temporario = pd.get_dummies(df_temporario, columns=['team_ah'], prefix='team_ah')
    X = df_temporario[['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds', 'league','goals_diff', 'h2h_diff', 'favorite_by_odds', 'odds_ratio','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home','home_ratio','away_ratio','vic_ratio','victory_diff']]
    try:
        '''
        with open('scaler_handicap.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_standardized = scaler.transform(X)
        X = pd.DataFrame(X_standardized, columns=['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds', 'league']).reset_index(drop=True)
        '''
        type_df = df_temporario[['team_ah_1.0','team_ah_2.0']].reset_index(drop=True)
    except Exception as e:
        print('Erro handicap')
        return None, None
    X_final = pd.concat([X, type_df], axis=1)
    print("Colunas de X (handicap):", X_final.columns.tolist())
    return X_final, z

def prepNNHandicap_X_conj(df=df_temp):
    '''prepara os jogos da linha para o autogluon'''
    required_columns = ['media_goals_home', 'media_goals_away','home_h2h_mean', 'away_h2h_mean',
                        'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 
                        'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2','league','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'conj handicap': {', '.join(missing_cols)}")
        return None, None
    df_temporario = df[required_columns].copy()
    df_temporario['odds_ah1'] = pd.to_numeric(df_temporario['odds_ah1'], errors='coerce')
    df_temporario['odds_ah2'] = pd.to_numeric(df_temporario['odds_ah2'], errors='coerce')

    df_temporario['favorite_by_odds'] = df_temporario['odds_ah1'] < df_temporario['odds_ah2']
    df_temporario['odds_ratio'] = df_temporario['odds_ah1'] / df_temporario['odds_ah2']
    
    
    df_temporario['goals_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['h2h_diff'] = df_temporario['home_h2h_mean'] - df_temporario['away_h2h_mean']
    df_temporario['home_ratio'] = df_temporario['media_goals_home'] - df_temporario['media_goals_sofridos_home']
    df_temporario['away_ratio'] = df_temporario['media_goals_away'] - df_temporario['media_goals_sofridos_away']
    df_temporario['vic_ratio'] = df_temporario['home_h2h_win_rate'] - df_temporario['away_h2h_win_rate']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']

    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNhandicap*")
        return None, None
    
    
    

    
    X = df_temporario[['media_goals_home', 'media_goals_away','home_h2h_mean', 'away_h2h_mean',
                        'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 
                        'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2','league','odds_ratio','favorite_by_odds','goals_diff','h2h_diff','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home','home_ratio','away_ratio','vic_ratio','victory_diff']]
   
        
    
 
    return X

def prepNNGoal_line_X_conj(df=df_temp):
    '''prepara os jogos da linha para o autogluon'''
    required_columns = ['home','away','times','h2h_mean' ,'media_goals_home' ,'media_goals_away',
                        'goal_line1_1','goal_line1_2','type_gl1','odds_gl1', 'odds_gl2',
                        'goal_line2_1','goal_line2_2','type_gl2', 'league','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away']
  
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNGoal_line_X': {', '.join(missing_cols)}")
        return None, None
    df_temporario = df[required_columns].copy()
    df_temporario['odds_gl1'] = pd.to_numeric(df_temporario['odds_gl1'], errors='coerce')
    df_temporario['odds_gl2'] = pd.to_numeric(df_temporario['odds_gl2'], errors='coerce')

    df_temporario['prob_gl1'] = 1 / df_temporario['odds_gl1']
    df_temporario['prob_gl2'] = 1 / df_temporario['odds_gl2']
    soma = df_temporario['prob_gl1'] + df_temporario['prob_gl2']
    df_temporario['prob_gl1'] = df_temporario['prob_gl1'].div(soma, axis=0)
    df_temporario['prob_gl2'] = df_temporario['prob_gl2'].div(soma, axis=0)
    
    
    df_temporario['split_line'] = (df_temporario['goal_line1_1'] != df_temporario['goal_line1_2']).astype(int)
    df_temporario['goals_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['goals_ratio'] = ((df_temporario['media_goals_home'] + df_temporario['media_goals_sofridos_home']) + (df_temporario['media_goals_away'] + df_temporario['media_goals_sofridos_away']))/2

 
    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNgl*")
        return None, None

    
    X = df_temporario[['h2h_mean' ,'media_goals_home' ,'media_goals_away',
                        'goal_line1_1','goal_line1_2','odds_gl1', 'odds_gl2',
                        'league','prob_gl1','prob_gl2','split_line','goals_diff','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away','goals_ratio']].copy()

    
  
    return X

def prepNNGoal_line_X(df=df_temp):
    '''prepara os jogos da linha para o autogluon'''
    required_columns = ['home','away','times','h2h_mean' ,'media_goals_home' ,'media_goals_away',
                        'goal_line1_1','goal_line1_2','type_gl1','odds_gl1', 'odds_gl2',
                        'goal_line2_1','goal_line2_2','type_gl2', 'league','media_goals_sofridos_home', 'h2h_total_games', 'media_goals_sofridos_away']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNGoal_line_X': {', '.join(missing_cols)}")
        return None, None
    df_temporario = df[required_columns].copy()
    df_temporario['odds_gl1'] = pd.to_numeric(df_temporario['odds_gl1'], errors='coerce')
    df_temporario['odds_gl2'] = pd.to_numeric(df_temporario['odds_gl2'], errors='coerce')

    df_temporario['prob_gl1'] = 1 / df_temporario['odds_gl1']
    df_temporario['prob_gl2'] = 1 / df_temporario['odds_gl2']
    soma = df_temporario['prob_gl1'] + df_temporario['prob_gl2']
    df_temporario['prob_gl1'] = df_temporario['prob_gl1'].div(soma, axis=0)
    df_temporario['prob_gl2'] = df_temporario['prob_gl2'].div(soma, axis=0)
    
    df_temporario = preparar_df_goallines_X(df_temporario)
    df_temporario['split_line'] = (df_temporario['goal_line_1'] != df_temporario['goal_line_2']).astype(int)
    df_temporario['goals_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['goals_ratio'] = ((df_temporario['media_goals_home'] + df_temporario['media_goals_sofridos_home']) + (df_temporario['media_goals_away'] + df_temporario['media_goals_sofridos_away']))/2

    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNgl*")
        return None, None
    z = df_temporario[['times','goal_line_1', 'goal_line_2','type_gl', 'odds_gl', 'league']].copy()
    df_temporario['type_gl'] = df_temporario['type_gl'].astype(float)
    df_temporario = pd.get_dummies(df_temporario, columns=['type_gl'], prefix='type_gl')
    X = df_temporario[['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2', 'league','prob','split_line','goals_diff','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away','goals_ratio']].copy()
    try:
        '''
        with open('scaler_goal_line.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_standardized = scaler.transform(X)
        X = pd.DataFrame(X_standardized, columns=['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2', 'league']).reset_index(drop=True)
        '''
        type_df = df_temporario[['type_gl_1.0', 'type_gl_2.0']].reset_index(drop=True)
    except Exception as e:
        print('Erro goal_line')
        return None, None
    X_final = pd.concat([X, type_df], axis=1)
    print("Colunas de X (goal_line):", X_final.columns.tolist())
    return X_final, z

def prepNNDouble_chance_X(df=df_temp):
    '''prepara os jogos da linha para o autogluon'''
    required_columns = ['home','away','times','media_goals_home','media_goals_away','media_victories_home', 'league','media_victories_away', 
                        'home_h2h_mean', 'away_h2h_mean', 'double_chance1','odds_dc1', 
                        'double_chance2', 'odds_dc2', 'double_chance3', 'odds_dc3','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNDouble_chance_X': {', '.join(missing_cols)}")
        return None, None

    df_temporario = df[required_columns].copy()
    df_temporario['odds_dc1'] = pd.to_numeric(df_temporario['odds_dc1'], errors='coerce')
    df_temporario['odds_dc2'] = pd.to_numeric(df_temporario['odds_dc2'], errors='coerce')
    df_temporario['odds_dc3'] = pd.to_numeric(df_temporario['odds_dc3'], errors='coerce')

    df_temporario['prob_dc1'] = 1 / df_temporario['odds_dc1']
    df_temporario['prob_dc2'] = 1 / df_temporario['odds_dc2']
    df_temporario['prob_dc3'] = 1 / df_temporario['odds_dc3']
    total = df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']].sum(axis=1)
    
    df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']] = df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']].div(total, axis=0)

    df_temporario = preparar_df_double_chance_X(df_temporario)

    df_temporario['goal_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']
    df_temporario['h2h_diff'] = df_temporario['home_h2h_mean'] - df_temporario['away_h2h_mean']
    df_temporario['vic_ratio'] = df_temporario['home_h2h_win_rate'] - df_temporario['away_h2h_win_rate']
    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNdc*")
        return None, None
    z = df_temporario[['times','double_chance', 'odds']].copy()
    
    
    df_temporario = pd.get_dummies(df_temporario, columns=['double_chance'], prefix='double_chance_type')
    X = df_temporario[['media_goals_home', 'media_goals_away', 'media_victories_home', 'media_victories_away','league',
                       'home_h2h_mean', 'away_h2h_mean','prob', 'odds', 'goal_diff', 'victory_diff', 'h2h_diff','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games','vic_ratio']].copy()

    try:
        '''
        with open('scaler_double_chance.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_standardized = scaler.transform(X)
        X = pd.DataFrame(X_standardized, columns=['media_goals_home', 'media_goals_away', 'media_victories_home','media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds']).reset_index(drop=True)
        '''
        type_df = df_temporario[['double_chance_type_1', 'double_chance_type_2','double_chance_type_3']].reset_index(drop=True)
    except Exception as e:
        print('Erro double_chance')
        return None, None
    X_final = pd.concat([X, type_df], axis=1)
    print("Colunas de X (doublechance):", X_final.columns.tolist())
    return X_final, z

def prepNNDouble_chance_X_conj(df=df_temp):
    '''prepara os jogos da linha para o autogluon'''
    required_columns = ['home','away','times','media_goals_home','media_goals_away','media_victories_home', 'media_victories_away', 'league',
                        'home_h2h_mean', 'away_h2h_mean', 'double_chance1','odds_dc1', 
                        'double_chance2', 'odds_dc2', 'double_chance3', 'odds_dc3','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNDouble_chance_X': {', '.join(missing_cols)}")
        return None, None

    df_temporario = df[required_columns].copy()
    df_temporario['odds_dc1'] = pd.to_numeric(df_temporario['odds_dc1'], errors='coerce')
    df_temporario['odds_dc2'] = pd.to_numeric(df_temporario['odds_dc2'], errors='coerce')
    df_temporario['odds_dc3'] = pd.to_numeric(df_temporario['odds_dc3'], errors='coerce')

    df_temporario['prob_dc1'] = 1 / df_temporario['odds_dc1']
    df_temporario['prob_dc2'] = 1 / df_temporario['odds_dc2']
    df_temporario['prob_dc3'] = 1 / df_temporario['odds_dc3']
    total = df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']].sum(axis=1)
    
    df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']] = df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']].div(total, axis=0)

    

    df_temporario['goal_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']
    df_temporario['h2h_diff'] = df_temporario['home_h2h_mean'] - df_temporario['away_h2h_mean']
    df_temporario['vic_ratio'] = df_temporario['home_h2h_win_rate'] - df_temporario['away_h2h_win_rate']
    
    
    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNdc*")
        return None, None
    
    
    X = df_temporario[['media_goals_home','media_goals_away','media_victories_home', 'media_victories_away', 'league',
                        'home_h2h_mean', 'away_h2h_mean', 'double_chance1','odds_dc1', 
                        'double_chance2', 'odds_dc2', 'double_chance3', 'odds_dc3','prob_dc1','prob_dc2','prob_dc3','goal_diff','victory_diff','h2h_diff','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games','vic_ratio']].copy()
  
        

   
    
    
    return X

def prepNNDraw_no_bet_X(df=df_temp):
    '''prepara os jogos da linha para o autogluon'''
    required_columns = ['home','away','times', 'media_goals_home', 'media_goals_away', 'media_victories_home','media_victories_away',
                        'home_h2h_mean','away_h2h_mean', 'draw_no_bet_team1', 'odds_dnb1', 'draw_no_bet_team2', 'odds_dnb2','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNDraw_no_bet_X': {', '.join(missing_cols)}")
        return None, None

    df_temporario = df[required_columns].copy()

    df_temporario['odds_dnb1'] = pd.to_numeric(df_temporario['odds_dnb1'], errors='coerce')
    df_temporario['odds_dnb2'] = pd.to_numeric(df_temporario['odds_dnb2'], errors='coerce')

    df_temporario['prob_odds_dnb1'] = 1 / df_temporario['odds_dnb1']
    df_temporario['prob_odds_dnb2'] = 1 / df_temporario['odds_dnb2']
    tot = df_temporario['prob_odds_dnb1'] + df_temporario['prob_odds_dnb2']
    df_temporario['prob_odds_dnb1'] = df_temporario['prob_odds_dnb1'].div(tot, axis=0)
    df_temporario['prob_odds_dnb2'] = df_temporario['prob_odds_dnb2'].div(tot, axis=0)

    df_temporario = preparar_df_draw_no_bet_X(df_temporario)

    df_temporario['goal_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['team_strength_home'] = df_temporario['media_victories_home'] / df_temporario['media_goals_home']
    df_temporario['team_strength_away'] = df_temporario['media_victories_away'] / df_temporario['media_goals_away']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']
    df_temporario['vic_ratio'] = df_temporario['home_h2h_win_rate'] - df_temporario['away_h2h_win_rate']


    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNdnb*")
        return None, None
    z = df_temporario[['times','draw_no_bet_team', 'odds']].copy()
    df_temporario['draw_no_bet_team'] = df_temporario['draw_no_bet_team'].astype(int)
    #df_temporario = pd.get_dummies(df_temporario, columns=['draw_no_bet_team'], prefix='draw_no_bet_type')
    X = df_temporario[['media_goals_home', 'media_goals_away', 'media_victories_home','media_victories_away','home_h2h_mean', 'away_h2h_mean', 'odds', 'prob_odds','goal_diff','team_strength_home','team_strength_away','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games', 'draw_no_bet_team','victory_diff','vic_ratio']].copy()
    
    
    print("Colunas de X (draw_no_bet):", X.columns.tolist())
    return X, z


def prepNNDraw_no_bet_X_conj(df=df_temp):
    '''prepara os jogos da linha para o autogluon'''
    required_columns = ['times', 'media_goals_home', 'media_goals_away', 'media_victories_home','media_victories_away',
                        'home_h2h_mean','away_h2h_mean', 'draw_no_bet_team1', 'odds_dnb1', 'draw_no_bet_team2', 'odds_dnb2','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNDraw_no_bet_X': {', '.join(missing_cols)}")
        return None, None

    df_temporario = df[required_columns].copy()

    df_temporario['odds_dnb1'] = pd.to_numeric(df_temporario['odds_dnb1'], errors='coerce')
    df_temporario['odds_dnb2'] = pd.to_numeric(df_temporario['odds_dnb2'], errors='coerce')

    df_temporario['prob_odds_dnb1'] = 1 / df_temporario['odds_dnb1']
    df_temporario['prob_odds_dnb2'] = 1 / df_temporario['odds_dnb2']
    tot = df_temporario['prob_odds_dnb1'] + df_temporario['prob_odds_dnb2']
    df_temporario['prob_odds_dnb1'] = df_temporario['prob_odds_dnb1'].div(tot, axis=0)
    df_temporario['prob_odds_dnb2'] = df_temporario['prob_odds_dnb2'].div(tot, axis=0)

    

    df_temporario['goal_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['team_strength_home'] = df_temporario['media_victories_home'] / df_temporario['media_goals_home']
    df_temporario['team_strength_away'] = df_temporario['media_victories_away'] / df_temporario['media_goals_away']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']
    df_temporario['vic_ratio'] = df_temporario['home_h2h_win_rate'] - df_temporario['away_h2h_win_rate']
    


    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNdnb*")
        return None, None
    
  
    
    X = df_temporario[['media_goals_home', 'media_goals_away', 'media_victories_home','media_victories_away',
                        'home_h2h_mean','away_h2h_mean', 'draw_no_bet_team1', 'odds_dnb1', 'draw_no_bet_team2', 'odds_dnb2','prob_odds_dnb1','prob_odds_dnb2','goal_diff','team_strength_home','team_strength_away','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games','victory_diff','vic_ratio']].copy()

    
    
    return X




#NN over_under

def NN_over_under(df):
    '''cria o autogluon para a linha'''
    df_temporario = df[['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean','res_goals_over_under', 'league']].copy()
    df_temporario.dropna(inplace=True)
    X = df_temporario[['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean', 'league']]
    print("Colunas de X (Over/Under):", X.columns.tolist())
    
    y = df_temporario['res_goals_over_under']
    scaler_over_under = StandardScaler()
    X_standardized = scaler_over_under.fit_transform(X)
    with open('scaler_over_under.pkl', 'wb') as f:
        pickle.dump(scaler_over_under, f)
    x_train, x_test, y_train, y_test = split(X_standardized,y)
    
    model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(x_train, y_train)
    y_pred = model_xgb.predict(x_test)
    print("Acurácia OU xgb:", accuracy_score(y_test, y_pred))
    joblib.dump(model_xgb, 'model_xgb_over_under.pkl')



    model_over_under = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária
    ])
    model_over_under.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    hist_bin = model_over_under.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))

    y_pred_probs = model_over_under.predict(x_test)
    
    melhor_z_positivo = encontrar_melhor_z_binario_positivo(y_test, y_pred_probs)
    melhor_z_negativo = encontrar_melhor_z_binario_negativo(y_test, y_pred_probs)
 
    model_over_under.save("model_binario_over_under.keras")  # Salva em formato nativo do Keras
    y_pred_probs_nn = model_over_under.predict(x_train).flatten()
    y_pred_probs_xgb = model_xgb.predict(x_train).flatten()
    X_meta = np.column_stack((y_pred_probs_nn, y_pred_probs_xgb))
    
    meta_model = LogisticRegression()  # ou XGBoost para um meta-modelo mais forte
    meta_model.fit(X_meta, y_train)
    y_pred_meta = meta_model.predict(X_meta)
    joblib.dump(meta_model, 'meta_model_over_under.pkl')
    print("Acurácia OU meta:", accuracy_score(y_train, y_pred_meta))

    return melhor_z_positivo, melhor_z_negativo


#junta handicaps
def preparar_df_handicaps(df):
    '''utilizado para individualizar a aposta em apostas individuais separadas'''

    # Seleciona e renomeia o df_temporario1
    df1 = df[['home','away','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
              'asian_handicap1_1', 'asian_handicap1_2','team_ah1', 'odds_ah1',
              'ah1_indefinido', 'ah1_negativo', 'ah1_positivo', 'ah1_reembolso', 'league','favorite_by_odds','odds_ratio','media_goals_sofridos_home', 'media_goals_sofridos_away', 'media_victories_home','media_victories_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']].copy()
    df1.columns = ['home','away','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
                   'asian_handicap_1', 'asian_handicap_2','team_ah', 'odds',
                   'indefinido', 'negativo', 'positivo', 'reembolso', 'league','favorite_by_odds','odds_ratio','media_goals_sofridos_home', 'media_goals_sofridos_away', 'media_victories_home','media_victories_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']

    # Seleciona e renomeia o df_temporario2
    df2 = df[['home','away','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
              'asian_handicap2_1', 'asian_handicap2_2','team_ah2', 'odds_ah2',
              'ah2_indefinido', 'ah2_negativo', 'ah2_positivo', 'ah2_reembolso', 'league','favorite_by_odds','odds_ratio','media_goals_sofridos_home', 'media_goals_sofridos_away', 'media_victories_home','media_victories_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']].copy()
    df2.columns = ['home','away','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
                   'asian_handicap_1', 'asian_handicap_2','team_ah', 'odds',
                   'indefinido', 'negativo', 'positivo', 'reembolso', 'league','favorite_by_odds','odds_ratio','media_goals_sofridos_home', 'media_goals_sofridos_away', 'media_victories_home','media_victories_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']

    # Concatena os dois dataframes
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final



def preparar_df_handicaps_X(df):
    '''utilizado para individualizar a aposta em apostas individuais separadas a ser prevista'''
    

    # Seleciona e renomeia o df_temporario1
    df1 = df[['home','away','times','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
              'asian_handicap1_1', 'asian_handicap1_2','team_ah1', 'odds_ah1', 'league', 'favorite_by_odds', 'odds_ratio','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home']].copy()
    df1.columns = ['home','away','times','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
                   'asian_handicap_1', 'asian_handicap_2','team_ah', 'odds', 'league', 'favorite_by_odds', 'odds_ratio','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home']

    # Seleciona e renomeia o df_temporario2
    df2 = df[['home','away','times','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
              'asian_handicap2_1', 'asian_handicap2_2','team_ah2', 'odds_ah2', 'league', 'favorite_by_odds', 'odds_ratio','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home']].copy()
    df2.columns = ['home','away','times','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
                   'asian_handicap_1', 'asian_handicap_2','team_ah', 'odds', 'league', 'favorite_by_odds', 'odds_ratio','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home']

    # Concatena os dois dataframes
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final



def NN_handicap(df=df_temp):
    '''cria modelo autogluon para a linha'''
    # Seleção inicial das colunas
    df_temporario = df[['home','away','media_goals_home','media_goals_sofridos_home', 'media_goals_away','media_goals_sofridos_away','media_victories_home','media_victories_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games', 'home_h2h_mean', 'away_h2h_mean',
                        'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 
                        'ah1_indefinido','ah1_negativo', 'ah1_positivo','ah1_reembolso', 
                        'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2', 
                        'ah2_indefinido','ah2_negativo', 'ah2_positivo','ah2_reembolso', 'league']].copy()

    #CONJ
    
    df_conj = df_temporario.copy()
    df_conj['favorite_by_odds'] = df_conj['odds_ah1'] < df_conj['odds_ah2']
    df_conj['odds_ratio'] = df_conj['odds_ah1'] / df_conj['odds_ah2']
    df_conj['goals_diff'] = df_conj['media_goals_home'] - df_conj['media_goals_away']
    df_conj['h2h_diff'] = df_conj['home_h2h_mean'] - df_conj['away_h2h_mean']
    df_conj['home_ratio'] = df_conj['media_goals_home'] - df_conj['media_goals_sofridos_home']
    df_conj['away_ratio'] = df_conj['media_goals_away'] - df_conj['media_goals_sofridos_away']
    df_conj['vic_ratio'] = df_conj['home_h2h_win_rate'] - df_conj['away_h2h_win_rate']
    df_conj['victory_diff'] = df_conj['media_victories_home'] - df_conj['media_victories_away']

    
    def transformar_resultado(row):
        if row['ah1_positivo'] == 1:
            return 0
        elif row['ah2_positivo'] == 1:
            return 1
        else:
            return None
        

    df_conj['resultado'] = df_conj.apply(transformar_resultado, axis=1)
    df_conj = df_conj[df_conj['resultado'].notna()].copy()

    df_conj = df_conj[['media_goals_home', 'media_goals_away','home_h2h_mean', 'away_h2h_mean','asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2', 'league','favorite_by_odds','odds_ratio','goals_diff','h2h_diff','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home','home_ratio','away_ratio','vic_ratio','victory_diff','resultado']].copy()
    

    train_conj, test_conj = train_test_split(df_conj, test_size=0.1, random_state=42)

    # Diretório temporário
    temp_dir = tempfile.mkdtemp()

    # Treinamento com AutoGluon
    predictor = TabularPredictor(label='resultado', path=temp_dir, problem_type='binary').fit(
        train_conj,
        presets='best_quality',
        time_limit=1000
    )

    # Leaderboard e melhor modelo
    leaderboard = predictor.leaderboard(train_conj, silent=True)
    try:
        best_model_name = predictor.model_best
        best_model_score = leaderboard.loc[leaderboard['model'] == best_model_name, 'score_val'].values[0]
    except:
        best_model_name = leaderboard.loc[leaderboard['score_val'].idxmax(), 'model']
        best_model_score = leaderboard.loc[leaderboard['score_val'].idxmax(), 'score_val']

    # Salvar o modelo final
    final_model_path = "autogluon_handicap_model_conj"
    shutil.move(temp_dir, final_model_path)
    predictor = TabularPredictor.load(final_model_path)

    # Predição no conjunto de teste
    y_pred = predictor.predict(test_conj.drop(columns=['resultado']), model=best_model_name)

    # Avaliação das previsões
    
    from sklearn.metrics import precision_score

# Garantir que ambos estão como strings
    y_true = test_conj['resultado']


    print(f"Precisão: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nMelhor modelo para handicap: {best_model_name}")
    print(f"Acurácia no treino (validação interna): {best_model_score:.4f}")
    with open('autogluon_handicap_model_leaderboard_conj.txt', 'w+') as f:
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"Acurácia (validação interna): {best_model_score:.4f}\n")
        f.write("\nMétricas de Avaliação no conjunto de teste (handicap):\n")
        f.write(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precisão (macro): {precision_score(y_true, y_pred):.4f}\n")
        f.write(f"Recall (macro): {recall_score(y_true, y_pred):.4f}\n")
        f.write(f"F1-Score (macro): {f1_score(y_true, y_pred):.4f}\n")
    
    
    
    # Pré-processamento

    df_temporario['favorite_by_odds'] = df_temporario['odds_ah1'] < df_temporario['odds_ah2']
    df_temporario['odds_ratio'] = df_temporario['odds_ah1'] / df_temporario['odds_ah2']
    df_temporario = preparar_df_handicaps(df_temporario)
    df_temporario = pd.get_dummies(df_temporario, columns=['team_ah'], prefix='team_ah')
    df_temporario = df_temporario[df_temporario['indefinido'] == False]
    df_temporario['goals_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['h2h_diff'] = df_temporario['home_h2h_mean'] - df_temporario['away_h2h_mean']

    
    




    # Tratamento de nulos e tipos problemáticos
    df_temporario['positivo'] = pd.to_numeric(df_temporario['positivo'], errors='coerce')
    df_temporario.dropna(subset=['positivo'], inplace=True)
    df_temporario['home_ratio'] = df_temporario['media_goals_home'] - df_temporario['media_goals_sofridos_home']
    df_temporario['away_ratio'] = df_temporario['media_goals_away'] - df_temporario['media_goals_sofridos_away']
    df_temporario['vic_ratio'] = df_temporario['home_h2h_win_rate'] - df_temporario['away_h2h_win_rate']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']
    # Feature set e target
    X = df_temporario[['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
                       'asian_handicap_1', 'asian_handicap_2', 'odds', 'league','goals_diff','h2h_diff','favorite_by_odds','odds_ratio','media_goals_sofridos_home', 'media_goals_sofridos_away', 'media_victories_home','media_victories_away','home_h2h_win_rate','home_ratio','away_ratio','vic_ratio','victory_diff',
       'away_h2h_win_rate','h2h_total_games']].copy()
    [['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds', 'league','goals_diff', 'h2h_diff', 'favorite_by_odds', 'odds_ratio','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home']]

    # Adiciona colunas dummies com segurança
    for col in ['team_ah_1.0', 'team_ah_2.0']:
        if col not in df_temporario.columns:
            df_temporario[col] = 0
        X[col] = df_temporario[col]

    y = df_temporario['positivo'].astype(int).reset_index(drop=True)
    X.reset_index(drop=True, inplace=True)

    print("Colunas de X (handicap):", X.columns.tolist())

    if y.nunique() < 2:
        print("Variável target com menos de 2 classes. Retornando None.")
        return None

    # Split treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    df_ag_train = X_train.copy()
    df_ag_train['target'] = y_train

    df_ag_test = X_test.copy()
    df_ag_test['target'] = y_test

    # Diretório temporário para o modelo
    temp_dir = tempfile.mkdtemp()

    # Treinamento com AutoGluon
    predictor = TabularPredictor(label='target', path=temp_dir, problem_type='binary').fit(
        df_ag_train,
        presets='best_quality',
        time_limit=1000
    )

    # Leaderboard e seleção do melhor modelo com try/except
    leaderboard = predictor.leaderboard(df_ag_train, silent=True)
    try:
        best_model_name = predictor.model_best
        best_model_score = leaderboard.loc[leaderboard['model'] == best_model_name, 'score_val'].values[0]
    except:
        best_model_name = leaderboard.loc[leaderboard['score_val'].idxmax(), 'model']
        best_model_score = leaderboard.loc[leaderboard['score_val'].idxmax(), 'score_val']

    print(f"\nMelhor modelo para Handicap: {best_model_name}")
    print(f"Acurácia no treino (validação interna): {best_model_score:.4f}")
    

    # Salvar modelo final
    predictor_path = "autogluon_handicap_model"
    shutil.move(temp_dir, predictor_path)

    # Recarrega o modelo salvo
    predictor = TabularPredictor.load(predictor_path)

    # Predição no conjunto de teste
    y_pred = predictor.predict(df_ag_test.drop(columns=['target']), model=best_model_name)

    # Avaliação
    y_true = df_ag_test['target']
    print("\nMétricas de Avaliação no conjunto de teste (Handicap):")
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
    with open('autogluon_handicap_model_leaderboard.txt', 'w+') as f:
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"Acurácia: {best_model_score:.4f}\n")
        f.write("\nMétricas de Avaliação no conjunto de teste (Handicap):")
        f.write(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")

    return 0.7



    # 5. Previsões
    
    

#junta goal_lines
def preparar_df_goallines(df):
    '''individualiza uma aposta inteira em apostas individuais'''
    # Seleciona e renomeia as colunas relacionadas à goal line 1
    df1 = df[['home','away','h2h_mean', 'media_goals_home', 'media_goals_away',
              'goal_line1_1', 'goal_line1_2', 'type_gl1','odds_gl1',
              'gl1_indefinido', 'gl1_negativo', 'gl1_positivo', 'gl1_reembolso', 'league', 'prob_gl1','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away']].copy()
    df1.columns = ['home','away','h2h_mean', 'media_goals_home', 'media_goals_away',
                   'goal_line_1', 'goal_line_2', 'type_gl','odds_gl',
                   'indefinido', 'negativo', 'positivo', 'reembolso', 'league', 'prob','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away']

    # Seleciona e renomeia as colunas relacionadas à goal line 2
    df2 = df[['home','away','h2h_mean', 'media_goals_home', 'media_goals_away',
              'goal_line2_1', 'goal_line2_2', 'type_gl2','odds_gl2',
              'gl2_indefinido', 'gl2_negativo', 'gl2_positivo', 'gl2_reembolso', 'league', 'prob_gl2','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away']].copy()
    df2.columns = ['home','away','h2h_mean', 'media_goals_home', 'media_goals_away',
                   'goal_line_1', 'goal_line_2', 'type_gl','odds_gl',
                   'indefinido', 'negativo', 'positivo', 'reembolso', 'league', 'prob','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away']

    # Concatena os dois dataframes
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final


#junta goal_lines
def preparar_df_goallines_X(df):
    '''individualiza uma aposta inteira em apostas individuais a ser prevista'''
    # Seleciona e renomeia as colunas relacionadas à goal line 1
    df1 = df[['home','away','times','h2h_mean', 'media_goals_home', 'media_goals_away',
              'goal_line1_1', 'goal_line1_2', 'type_gl1','odds_gl1', 'league', 'prob_gl1','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away']].copy()
    df1.columns = ['home','away','times','h2h_mean', 'media_goals_home', 'media_goals_away',
                   'goal_line_1', 'goal_line_2', 'type_gl','odds_gl', 'league', 'prob','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away']

    # Seleciona e renomeia as colunas relacionadas à goal line 2
    df2 = df[['home','away','times','h2h_mean', 'media_goals_home', 'media_goals_away',
              'goal_line2_1', 'goal_line2_2', 'type_gl2','odds_gl2', 'league', 'prob_gl2','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away']].copy()
    df2.columns = ['home','away','times','h2h_mean', 'media_goals_home', 'media_goals_away',
                   'goal_line_1', 'goal_line_2', 'type_gl','odds_gl', 'league', 'prob','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away']

    # Concatena os dois dataframes
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final




def NN_goal_line(df=df_temp):
    '''cria o autogluon para a linha'''
    # Pré-processamento do dataframe
    
    df_temporario = df[['home', 'away', 'h2h_mean', 'media_goals_home','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away', 'media_goals_away',
                        'goal_line1_1', 'goal_line1_2', 'type_gl1', 'odds_gl1', 'odds_gl2',
                        'goal_line2_1', 'goal_line2_2', 'type_gl2', 'gl1_indefinido', 'gl1_negativo',
                        'gl1_positivo', 'gl1_reembolso', 'gl2_indefinido', 'gl2_negativo', 'gl2_positivo',
                        'gl2_reembolso', 'league']].copy()
    
    df_temporario['prob_gl1'] = 1 / df_temporario['odds_gl1']
    df_temporario['prob_gl2'] = 1 / df_temporario['odds_gl2']
    soma = df_temporario['prob_gl1'] + df_temporario['prob_gl2']
    df_temporario['prob_gl1'] /= soma
    df_temporario['prob_gl2'] /= soma

    #CONJ
    df_conj = df_temporario.copy()
    df_conj['goals_diff'] = df_conj['media_goals_home'] - df_conj['media_goals_away']
    df_conj['goals_ratio'] = ((df_conj['media_goals_home'] + df_conj['media_goals_sofridos_home']) + (df_conj['media_goals_away'] + df_conj['media_goals_sofridos_away']))/2
    df_conj.dropna(inplace=True)
    df_conj = df_conj[['h2h_mean', 'media_goals_home', 'media_goals_away','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away',
                        'goal_line1_1', 'goal_line1_2', 'odds_gl1', 'odds_gl2',
                        'league', 'prob_gl1','prob_gl2','goals_diff','gl1_positivo','gl2_positivo','goals_ratio']].copy()
    df_conj['split_line'] = (df_conj['goal_line1_1'] != df_conj['goal_line1_2']).astype(int)

    def transformar_target(row):
        if (row['gl1_positivo'] == 1):
            return 0
        elif (row['gl2_positivo'] == 1):
            return 1
        else:
            return None
        

    df_conj['resultado'] = df_conj.apply(transformar_target, axis=1)
    df_conj = df_conj[df_conj['resultado'].notna()]  # Remove os casos de reembolso
    
    df_conj.drop(columns=['gl1_positivo','gl2_positivo'], inplace=True)
    '''
    # **** PASSO 1: TREINAR O MODELO Q-LEARNING ****
    print("\n=== Treinando modelo Q-Learning para Goal Line ===")
    
    
    
    # Dividir em conjunto de treino e teste para o Q-Learning
    train_q, test_q = train_test_split(df_conj, test_size=0.05, random_state=42)
    
    # Inicializar e treinar o agente Q-Learning
    agent = QLearningGoalLine(alpha=0.1, gamma=0.6, epsilon=0.1)
    agent.train(train_q, num_episodes=1000)  # Ajuste o número de episódios conforme necessário
    
    # Avaliar o modelo Q-Learning
    q_evaluation = agent.evaluate(test_q)
    print(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
    print(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
    print(f"unidades: {q_evaluation['uni']}")
    
    print("\nAcurácia por tipo de gl (Q-Learning):")
    for action, accuracy in q_evaluation['accuracy_by_action'].items():
        dc_type = {0: "OVER", 1: "UNDER"}
        correct = q_evaluation['results_by_action'][action]['correct']
        total = q_evaluation['results_by_action'][action]['total']
        print(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
        
    with open('q_learning_gl.txt','w+') as f:
        f.write(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
        f.write(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
        f.write(f"unidades: {q_evaluation['uni']}")
        
        f.write("\nAcurácia por tipo de gl (Q-Learning):")
        for action, accuracy in q_evaluation['accuracy_by_action'].items():
            dc_type = {0: "OVER", 1: "UNDER"}
            correct = q_evaluation['results_by_action'][action]['correct']
            total = q_evaluation['results_by_action'][action]['total']
            f.write(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    # Salvar o modelo Q-Learning
    agent.save_model('q_learning_gl_model_final.pkl')
    '''
    train_conj, test_conj = train_test_split(df_conj, test_size=0.1, random_state=42)
    train_conj['resultado'] = train_conj['resultado'].astype(int)
    test_conj['resultado'] = test_conj['resultado'].astype(int)


    # Diretório temporário
    temp_dir = tempfile.mkdtemp()
    
    # Treinamento com AutoGluon
    predictor = TabularPredictor(label='resultado', path=temp_dir, problem_type='binary').fit(
        train_conj,
        presets='best_quality',
        time_limit=1000
    )

    # Leaderboard e melhor modelo
    leaderboard = predictor.leaderboard(train_conj, silent=True)
    try:
        best_model_name = predictor.model_best
        best_model_score = leaderboard.loc[leaderboard['model'] == best_model_name, 'score_val'].values[0]
    except:
        best_model_name = leaderboard.loc[leaderboard['score_val'].idxmax(), 'model']
        best_model_score = leaderboard.loc[leaderboard['score_val'].idxmax(), 'score_val']

    

    # Salvar modelo final
    predictor_path = "autogluon_goal_line_model_conj"
    shutil.move(temp_dir, predictor_path)

    # Recarrega o modelo salvo
    predictor = TabularPredictor.load(predictor_path)
    

    

    # Predição no conjunto de teste
    try:

        y_pred = predictor.predict(test_conj.drop(columns=['resultado']), model=best_model_name)
    except:
        y_pred = predictor.predict(test_conj.drop(columns=['resultado']), model=predictor.model_best)
    
        # Normalização dos valores previstos
    if y_pred.dtype == object or y_pred.dtype == str:
        y_pred = y_pred.str.lower()  # garantir lowercase para evitar 'Over' ou 'UNDER'
        y_pred = y_pred.map({'over': 0, 'under': 1})
    elif set(y_pred.unique()).issubset({0, 1}):
        pass  # já está no formato esperado
    else:
        raise ValueError(f"y_pred contém valores inesperados: {y_pred.unique()}")


    # Avaliação das previsões
    y_true = test_conj['resultado']
    
    print(f"Precisão: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nMelhor modelo para Goal line: {best_model_name}")
    print(f"Acurácia no treino (validação interna): {best_model_score:.4f}")
    with open('autogluon_goal_line_model_leaderboard_conj.txt', 'w+') as f:
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"Acurácia (validação interna): {best_model_score:.4f}\n")
        f.write("\nMétricas de Avaliação no conjunto de teste (goal_line):\n")
        f.write(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precisão (macro): {precision_score(y_true, y_pred):.4f}\n")
        f.write(f"Recall (macro): {recall_score(y_true, y_pred):.4f}\n")
        f.write(f"F1-Score (macro): {f1_score(y_true, y_pred):.4f}\n")

    

    df_temporario = preparar_df_goallines(df_temporario)
    df_temporario['split_line'] = (df_temporario['goal_line_1'] != df_temporario['goal_line_2']).astype(int)
    df_temporario['goals_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['goals_ratio'] = ((df_temporario['media_goals_home'] + df_temporario['media_goals_sofridos_home']) + (df_temporario['media_goals_away'] + df_temporario['media_goals_sofridos_away']))/2

    df_temporario = pd.get_dummies(df_temporario, columns=['type_gl'], prefix='type_gl')
    df_temporario = df_temporario[df_temporario['indefinido'] == False]
    df_temporario['positivo'] = pd.to_numeric(df_temporario['positivo'], errors='coerce')

    df_temporario.dropna(inplace=True)
    df_temporario = df_temporario[df_temporario['positivo'].notna()]

    # Definição de X e y
    X = df_temporario[['h2h_mean', 'media_goals_home', 'media_goals_away', 'odds_gl',
                    'goal_line_1', 'goal_line_2', 'league','prob','split_line','goals_diff','media_goals_sofridos_home','h2h_total_games', 'media_goals_sofridos_away','goals_ratio']].copy().reset_index(drop=True)


    type_df = df_temporario[['type_gl_1.0', 'type_gl_2.0']].reset_index(drop=True)
    X_final = pd.concat([X, type_df], axis=1)

    y = df_temporario['positivo'].astype(int).reset_index(drop=True)

    print("Colunas de X (goal_line):", X_final.columns.tolist())

    if y.nunique() < 2:
        print("Variável target com menos de 2 classes. Retornando None.")
        return None

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.1, random_state=42)

 

    df_ag_train = X_train.copy()
    df_ag_train['target'] = y_train

    df_ag_test = X_test.copy()
    df_ag_test['target'] = y_test

 

    # Diretório temporário
    temp_dir = tempfile.mkdtemp()

    # Treinamento com AutoGluon usando sample_weight
    predictor = TabularPredictor(
        label='target', 
        path=temp_dir, 
        problem_type='binary'
    ).fit(
        df_ag_train,
        presets='best_quality',
        time_limit=1000
    )

    # Leaderboard e melhor modelo
    leaderboard = predictor.leaderboard(df_ag_train, silent=True)
    try:
        best_model_name = predictor.model_best
        best_model_score = leaderboard.loc[leaderboard['model'] == best_model_name, 'score_val'].values[0]
    except:
        best_model_name = leaderboard.loc[leaderboard['score_val'].idxmax(), 'model']
        best_model_score = leaderboard.loc[leaderboard['score_val'].idxmax(), 'score_val']

    # Salvar modelo final
    predictor_path = "autogluon_goal_line_model"
    shutil.move(temp_dir, predictor_path)

    # Recarrega o modelo salvo
    predictor = TabularPredictor.load(predictor_path)

    # Predição no conjunto de teste
    y_pred = predictor.predict(df_ag_test.drop(columns=['target']), model=best_model_name)

    # Avaliação das previsões
    y_true = df_ag_test['target']

    print(f"Precisão: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nMelhor modelo para Goal Line: {best_model_name}")
    print(f"Acurácia no treino (validação interna): {best_model_score:.4f}")



    with open('autogluon_goal_line_model_leaderboard.txt', 'w+') as f:
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"Acurácia: {best_model_score:.4f}\n")
        f.write("\nMétricas de Avaliação no conjunto de teste (Goal Line):\n")
        f.write(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precisão: {precision_score(y_true, y_pred):.4f}\n")
        f.write(f"Recall: {recall_score(y_true, y_pred):.4f}\n")
        f.write(f"F1-Score: {f1_score(y_true, y_pred):.4f}\n")

    return 0.5



#juntar double_chances
def preparar_df_double_chance(df):
    '''individualiza uma aposta inteira em apostas individuais'''
    colunas_comuns = ['home','away','league','media_goals_home', 'media_goals_away', 'media_victories_home',
                      'media_victories_away', 'home_h2h_mean', 'away_h2h_mean','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']
    
    # Cria df para cada linha de double chance
    df1 = df[colunas_comuns + ['odds_dc1', 'prob_dc1', 'res_double_chance1']].copy()
    df1['double_chance'] = 1
    df1.rename(columns={'odds_dc1': 'odds', 'prob_dc1': 'prob', 'res_double_chance1': 'resultado'}, inplace=True)

    df2 = df[colunas_comuns + ['odds_dc2', 'prob_dc2', 'res_double_chance2']].copy()
    df2['double_chance'] = 2
    df2.rename(columns={'odds_dc2': 'odds', 'prob_dc2': 'prob', 'res_double_chance2': 'resultado'}, inplace=True)

    df3 = df[colunas_comuns + ['odds_dc3', 'prob_dc3', 'res_double_chance3']].copy()
    df3['double_chance'] = 3
    df3.rename(columns={'odds_dc3': 'odds', 'prob_dc3': 'prob', 'res_double_chance3': 'resultado'}, inplace=True)

    # Concatena os três em um só
    df_final = pd.concat([df1, df2, df3], ignore_index=True)

    return df_final



def preparar_df_double_chance_X(df):
    '''individualiza uma aposta inteira em apostas individuais a serem previstas'''
    colunas_comuns = ['home','away','times','media_goals_home', 'media_goals_away', 'media_victories_home',
                      'media_victories_away', 'league','home_h2h_mean', 'away_h2h_mean','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']
    
    # Cria df para cada linha de double chance
    df1 = df[colunas_comuns + ['odds_dc1', 'prob_dc1']].copy()
    df1['double_chance'] = 1
    df1.rename(columns={'odds_dc1': 'odds', 'prob_dc1': 'prob'}, inplace=True)

    df2 = df[colunas_comuns + ['odds_dc2', 'prob_dc2']].copy()
    df2['double_chance'] = 2
    df2.rename(columns={'odds_dc2': 'odds', 'prob_dc2': 'prob'}, inplace=True)

    df3 = df[colunas_comuns + ['odds_dc3', 'prob_dc3']].copy()
    df3['double_chance'] = 3
    df3.rename(columns={'odds_dc3': 'odds', 'prob_dc3': 'prob'}, inplace=True)

    # Concatena os três em um só
    df_final = pd.concat([df1, df2, df3], ignore_index=True)

    return df_final



def NN_double_chance(df):
    """
    Treinamento de modelo para Double Chance incluindo Q-Learning junto com AutoGluon
    """
    # Pré-processamento do dataframe
    df_temporario = df[['home', 'away', 'media_goals_home', 'media_goals_away', 'league', 'media_victories_home',
                        'media_victories_away','media_goals_sofridos_home','media_goals_sofridos_away' ,'home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games', 'home_h2h_mean', 'away_h2h_mean', 'double_chance1',
                        'odds_dc1', 'double_chance2', 'odds_dc2', 'double_chance3', 'odds_dc3',
                        'res_double_chance1', 'res_double_chance2', 'res_double_chance3',
                        'res_game_home', 'res_game_away', 'res_game_empate']].copy()
    
    df_temporario['prob_dc1'] = 1 / df_temporario['odds_dc1']
    df_temporario['prob_dc2'] = 1 / df_temporario['odds_dc2']
    df_temporario['prob_dc3'] = 1 / df_temporario['odds_dc3']
    total = df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']].sum(axis=1)
    
    df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']] = df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']].div(total, axis=0)

    # Adicionar cálculos de diferenças
    df_temporario['goal_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']
    df_temporario['h2h_diff'] = df_temporario['home_h2h_mean'] - df_temporario['away_h2h_mean']
    df_temporario['vic_ratio'] = df_temporario['home_h2h_win_rate'] - df_temporario['away_h2h_win_rate']
    df_temporario.dropna(inplace=True)
    '''
    # **** PASSO 1: TREINAR O MODELO Q-LEARNING ****
    print("\n=== Treinando modelo Q-Learning para Double Chance ===")
    
    # Preparar o DataFrame para Q-Learning (isso já calcula as diferenças)
    df_q = preparar_df_para_q_learning(df_temporario)
    
    

    

   
    # Dividir em conjunto de treino e teste para o Q-Learning
    train_q, test_q = train_test_split(df_q, test_size=0.1, random_state=42)
    
    # Inicializar e treinar o agente Q-Learning
    agent = QLearningDoubleChanсe()
    agent.train(train_q, num_episodes=500)  # Ajuste o número de episódios conforme necessário
    
    # Avaliar o modelo Q-Learning
    q_evaluation = agent.evaluate(test_q)
    print(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
    print(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
    
    
    print("\nAcurácia por tipo de Double Chance (Q-Learning):")
    for action, accuracy in q_evaluation['accuracy_by_action'].items():
        dc_type = {0: "DC1 (Casa ou Empate)", 1: "DC2 (Fora ou Empate)", 2: "DC3 (Casa ou Fora)"}
        correct = q_evaluation['results_by_action'][action]['correct']
        total = q_evaluation['results_by_action'][action]['total']
        print(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    with open('q_learning_dc.txt','w+') as f:
        f.write(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
        f.write(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
        f.write(f"unidades: {q_evaluation['uni']}")
        
        f.write("\nAcurácia por tipo de Double Chance (Q-Learning):")
        for action, accuracy in q_evaluation['accuracy_by_action'].items():
            dc_type = {0: "DC1 (Casa ou Empate)", 1: "DC2 (Fora ou Empate)", 2: "DC3 (Casa ou Fora)"}
            correct = q_evaluation['results_by_action'][action]['correct']
            total = q_evaluation['results_by_action'][action]['total']
            f.write(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    # Salvar o modelo Q-Learning
    agent.save_model('q_learning_dc_model_final.pkl')
    '''
    # **** PASSO 2: TREINAR O MODELO AUTOGLUON CONJUNTO ****
    print("\n=== Treinando modelo AutoGluon conjunto ===")
    
    df_conj = df_temporario.copy()
    
    def transformar_target(row):
        if row['res_game_home'] == 1:
            return 0
        elif row['res_game_away'] == 1:
            return 1
        else:
            return None
    
    df_conj['resultado'] = df_conj.apply(transformar_target, axis=1)
    df_conj = df_conj[df_conj['resultado'].notna()].copy()
    df_modelo = df_conj[['media_goals_home', 'media_goals_away', 'league', 'media_victories_home',
                     'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'double_chance1',
                     'odds_dc1', 'double_chance2', 'odds_dc2', 'double_chance3', 'odds_dc3',
                     'goal_diff', 'victory_diff', 'h2h_diff', 'media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games','vic_ratio','resultado']].copy()
   
    train_conj, test_conj = train_test_split(df_modelo, test_size=0.1, random_state=42)

    # Diretório temporário
    temp_dir = tempfile.mkdtemp()

    # Treinamento com AutoGluon
    predictor = TabularPredictor(label='resultado', path=temp_dir, problem_type='binary').fit(
        train_conj,
        presets='best_quality',
        time_limit=1000
    )

    # Leaderboard e melhor modelo
    leaderboard = predictor.leaderboard(train_conj, silent=True)
    try:
        best_model_name = predictor.model_best
        best_model_score = leaderboard.loc[leaderboard['model'] == best_model_name, 'score_val'].values[0]
    except:
        best_model_name = leaderboard.loc[leaderboard['score_val'].idxmax(), 'model']
        best_model_score = leaderboard.loc[leaderboard['score_val'].idxmax(), 'score_val']

    # Salvar o modelo final
    final_model_path = "autogluon_double_chance_model_conj"
    shutil.move(temp_dir, final_model_path)
    predictor = TabularPredictor.load(final_model_path)

    # Predição no conjunto de teste
    y_pred = predictor.predict(test_conj.drop(columns=['resultado']), model=best_model_name)

    # Avaliação das previsões
    y_true = test_conj['resultado']
    
    print(f"Precisão (AutoGluon conjunto): {precision_score(y_true, y_pred):.4f}")
    print(f"Recall (AutoGluon conjunto): {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score (AutoGluon conjunto): {f1_score(y_true, y_pred):.4f}")

    print("\nMatriz de Confusão (AutoGluon conjunto):")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nMelhor modelo para Double Chance (AutoGluon conjunto): {best_model_name}")
    print(f"Acurácia no treino (validação interna): {best_model_score:.4f}")
    
    with open('autogluon_double_chance_model_leaderboard_conj.txt', 'w+') as f:
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"Acurácia (validação interna): {best_model_score:.4f}\n")
        f.write("\nMétricas de Avaliação no conjunto de teste (Double Chance):\n")
        f.write(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precisão (macro): {precision_score(y_true, y_pred):.4f}\n")
        f.write(f"Recall (macro): {recall_score(y_true, y_pred):.4f}\n")
        f.write(f"F1-Score (macro): {f1_score(y_true, y_pred):.4f}\n")

    # **** PASSO 3: TREINAR O MODELO AUTOGLUON ESPECÍFICO ****
    print("\n=== Treinando modelo AutoGluon específico ===")

    # IMPORTANTE: Fazer uma cópia do df_temporario que ainda tem goal_diff, victory_diff e h2h_diff
    df_especifico = df_temporario.copy()
    
    # Agora aplicar preparar_df_double_chance em uma cópia separada
    df_especifico = preparar_df_double_chance(df_especifico)
    
    # Criar as colunas de diferença novamente se elas foram removidas
    if 'goal_diff' not in df_especifico.columns:
        df_especifico['goal_diff'] = df_especifico['media_goals_home'] - df_especifico['media_goals_away']
    
    if 'victory_diff' not in df_especifico.columns:
        df_especifico['victory_diff'] = df_especifico['media_victories_home'] - df_especifico['media_victories_away']
    
    if 'h2h_diff' not in df_especifico.columns:
        df_especifico['h2h_diff'] = df_especifico['home_h2h_mean'] - df_especifico['away_h2h_mean']

    df_especifico['vic_ratio'] = df_especifico['home_h2h_win_rate'] - df_especifico['away_h2h_win_rate']
    
    df_especifico = pd.get_dummies(df_especifico, columns=['double_chance'], prefix='double_chance_type')
    df_especifico['resultado'] = pd.to_numeric(df_especifico['resultado'], errors='coerce')
    df_especifico.dropna(inplace=True)
    df_especifico = df_especifico[df_especifico['resultado'].notna()]

    # Verifique se todas as colunas necessárias estão presentes
    required_cols = ['media_goals_home', 'media_goals_away', 'league', 'media_victories_home', 'media_victories_away',
                    'home_h2h_mean', 'away_h2h_mean', 'prob', 'odds', 'goal_diff', 'victory_diff', 'h2h_diff','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games','vic_ratio']
    
    for col in required_cols:
        if col not in df_especifico.columns:
            print(f"Coluna '{col}' ausente em df_especifico. Colunas disponíveis: {df_especifico.columns.tolist()}")
            return None
    
    # Definição de X e y
    X = df_especifico[required_cols].copy().reset_index(drop=True)
    
    # Verifique se as colunas de tipo estão presentes
    type_cols = ['double_chance_type_1', 'double_chance_type_2', 'double_chance_type_3']
    missing_type_cols = [col for col in type_cols if col not in df_especifico.columns]
    
    if missing_type_cols:
        print(f"Colunas ausentes após get_dummies: {missing_type_cols}")
        print(f"Colunas disponíveis: {df_especifico.columns.tolist()}")
        # Crie colunas vazias para os tipos ausentes
        for col in missing_type_cols:
            df_especifico[col] = 0
    
    type_df = df_especifico[['double_chance_type_1', 'double_chance_type_2', 'double_chance_type_3']].reset_index(drop=True)
    X_final = pd.concat([X, type_df], axis=1)

    y = df_especifico['resultado'].astype(int).reset_index(drop=True)

    if y.nunique() < 2:
        print("Variável target com menos de 2 classes. Retornando None.")
        return None

    print("Colunas de X (double chance):", X_final.columns.tolist())

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.1, random_state=42)

    df_ag_train = X_train.copy()
    df_ag_train['target'] = y_train

    df_ag_test = X_test.copy()
    df_ag_test['target'] = y_test

    # Diretório temporário
    temp_dir = tempfile.mkdtemp()

    # Treinamento com AutoGluon
    predictor = TabularPredictor(label='target', path=temp_dir, problem_type='binary').fit(
        df_ag_train,
        presets='best_quality',
        time_limit=1000
    )

    # Leaderboard e melhor modelo
    leaderboard = predictor.leaderboard(df_ag_train, silent=True)
    try:
        best_model_name = predictor.model_best
        best_model_score = leaderboard.loc[leaderboard['model'] == best_model_name, 'score_val'].values[0]
    except:
        best_model_name = leaderboard.loc[leaderboard['score_val'].idxmax(), 'model']
        best_model_score = leaderboard.loc[leaderboard['score_val'].idxmax(), 'score_val']

    # Salvar o modelo final
    final_model_path = "autogluon_double_chance_model"
    shutil.move(temp_dir, final_model_path)

    # Recarregar modelo salvo
    predictor = TabularPredictor.load(final_model_path)

    # Predição no conjunto de teste
    y_pred = predictor.predict(df_ag_test.drop(columns=['target']), model=best_model_name)

    # Avaliação das previsões
    y_true = df_ag_test['target']
    
    print(f"Precisão (AutoGluon específico): {precision_score(y_true, y_pred):.4f}")
    print(f"Recall (AutoGluon específico): {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score (AutoGluon específico): {f1_score(y_true, y_pred):.4f}")
    print("\nMatriz de Confusão (AutoGluon específico):")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nMelhor modelo para Double Chance (AutoGluon específico): {best_model_name}")
    print(f"Acurácia no treino (validação interna): {best_model_score:.4f}")
    
    with open('autogluon_double_chance_model_leaderboard.txt', 'w+') as f:
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"Acurácia: {best_model_score:.4f}\n")
        f.write("\nMétricas de Avaliação no conjunto de teste (Double Chance):")
        f.write(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    
    return 0.6


#junta draw_no_bet
def preparar_df_draw_no_bet(df):
    '''individualiza uma aposta inteira em apostas individuais'''
    # Seleciona e renomeia o lado 1
    df1 = df[['home','away', 'media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team1', 'odds_dnb1', 'dnb1_indefinido', 'dnb1_perde', 'dnb1_ganha', 'dnb1_reembolso','prob_odds_dnb1','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']].copy()

    df1.columns = ['home','away', 'media_goals_home', 'media_goals_away',
                   'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
                   'draw_no_bet_team', 'odds', 'indefinido', 'perde', 'ganha', 'reembolso','prob_odds','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']

    # Seleciona e renomeia o lado 2
    df2 = df[['home','away', 'media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team2', 'odds_dnb2', 'dnb2_indefinido', 'dnb2_perde', 'dnb2_ganha', 'dnb2_reembolso','prob_odds_dnb2','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']].copy()

    df2.columns = ['home','away', 'media_goals_home', 'media_goals_away',
                   'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
                   'draw_no_bet_team', 'odds', 'indefinido', 'perde', 'ganha', 'reembolso','prob_odds','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']

    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final

def preparar_df_draw_no_bet_X(df):
    '''individualiza uma aposta inteira em apostas individuais a serem previstas'''
    # Seleciona e renomeia o lado 1
    df1 = df[['home','away','times', 'media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team1', 'odds_dnb1', 'prob_odds_dnb1','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']].copy()

    df1.columns = ['home','away','times', 'media_goals_home', 'media_goals_away',
                   'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
                   'draw_no_bet_team', 'odds', 'prob_odds','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']

    # Seleciona e renomeia o lado 2
    df2 = df[['home','away', 'times','media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team2', 'odds_dnb2', 'prob_odds_dnb2','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']].copy()

    df2.columns = ['home','away','times', 'media_goals_home', 'media_goals_away',
                   'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
                   'draw_no_bet_team', 'odds', 'prob_odds','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']
   

    # Concatena os dois lados
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final



def NN_draw_no_bet(df):
    '''cria modelo autogluon para a linha'''
    # Pré-processamento
    df['prob_odds_dnb1'] = 1 / df['odds_dnb1']
    df['prob_odds_dnb2'] = 1 / df['odds_dnb2']
    tot = df['prob_odds_dnb1'] + df['prob_odds_dnb2']
    df['prob_odds_dnb1'] = df['prob_odds_dnb1'] / tot
    df['prob_odds_dnb2'] = df['prob_odds_dnb2'] / tot

    df_conj = df[['media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team1', 'odds_dnb1', 'dnb1_ganha', 'prob_odds_dnb1','draw_no_bet_team2', 'odds_dnb2','dnb2_ganha', 'prob_odds_dnb2','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games']].copy()
    df_conj['goal_diff'] = df_conj['media_goals_home'] - df_conj['media_goals_away']
    df_conj['team_strength_home'] = df_conj['media_victories_home'] / df_conj['media_goals_home']
    df_conj['team_strength_away'] = df_conj['media_victories_away'] / df_conj['media_goals_away']
    df_conj['victory_diff'] = df_conj['media_victories_home'] - df_conj['media_victories_away']
    df_conj['vic_ratio'] = df_conj['home_h2h_win_rate'] - df_conj['away_h2h_win_rate']
    df_conj.dropna(inplace=True)

    def transformar_res(row):
        if row['dnb1_ganha'] == 1:
            return 0
        elif row['dnb2_ganha'] == 1:
            return 1
        else:
            return None
        

    df_conj['resultado'] = df_conj.apply(transformar_res, axis=1)
    df_conj = df_conj[df_conj['resultado'].notna()].copy()
    '''
    # **** PASSO 1: TREINAR O MODELO Q-LEARNING ****
    print("\n=== Treinando modelo Q-Learning para Double Chance ===")
    
    # Dividir em conjunto de treino e teste para o Q-Learning
    train_q, test_q = train_test_split(df_conj, test_size=0.1, random_state=42)
    
    # Inicializar e treinar o agente Q-Learning
    agent = QLearningDrawNoBet(alpha=0.1, gamma=0.6, epsilon=0.1)
    agent.train(train_q, num_episodes=500)  # Ajuste o número de episódios conforme necessário
    
    # Avaliar o modelo Q-Learning
    q_evaluation = agent.evaluate(test_q)
    print(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
    print(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
    print(f"unidades: {q_evaluation['uni']}")
    
    print("\nAcurácia por tipo de DNB (Q-Learning):")
    for action, accuracy in q_evaluation['accuracy_by_action'].items():
        dc_type = {0: "vitoria time home", 1: "vitoria time away"}
        correct = q_evaluation['results_by_action'][action]['correct']
        total = q_evaluation['results_by_action'][action]['total']
        print(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    with open('q_learning_dc.txt','w+') as f:
        f.write(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
        f.write(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
        f.write(f"unidades: {q_evaluation['uni']}")
        
        f.write("\nAcurácia por tipo de DNB (Q-Learning):")
        for action, accuracy in q_evaluation['accuracy_by_action'].items():
            dc_type = {0: "vitoria time home", 1: "vitoria time away"}
            correct = q_evaluation['results_by_action'][action]['correct']
            total = q_evaluation['results_by_action'][action]['total']
            f.write(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    # Salvar o modelo Q-Learning
    agent.save_model('q_learning_dnb_model_final.pkl')
    '''
    df_conj = df_conj[['media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team1', 'odds_dnb1', 'prob_odds_dnb1','draw_no_bet_team2', 'odds_dnb2', 'prob_odds_dnb2', 'team_strength_home','team_strength_away','media_goals_sofridos_home','media_goals_sofridos_away','home_h2h_win_rate',
       'away_h2h_win_rate','h2h_total_games','victory_diff','vic_ratio','resultado']].copy()

    

    train_conj, test_conj = train_test_split(df_conj, test_size=0.1, random_state=42)

    

   

    # Diretório temporário
    temp_dir = tempfile.mkdtemp()

    # Treinamento com AutoGluon
    predictor = TabularPredictor(label='resultado', path=temp_dir, problem_type='binary').fit(
        train_conj,
        presets='best_quality',
        time_limit=1000
    )

    # Leaderboard e melhor modelo
    leaderboard = predictor.leaderboard(train_conj, silent=True)
    try:
        best_model_name = predictor.model_best
        best_model_score = leaderboard.loc[leaderboard['model'] == best_model_name, 'score_val'].values[0]
    except:
        best_model_name = leaderboard.loc[leaderboard['score_val'].idxmax(), 'model']
        best_model_score = leaderboard.loc[leaderboard['score_val'].idxmax(), 'score_val']

    # Salvar o modelo final
    final_model_path = "autogluon_draw_no_bet_model_conj"
    shutil.move(temp_dir, final_model_path)
    predictor = TabularPredictor.load(final_model_path)

    # Predição no conjunto de teste
    y_pred = predictor.predict(test_conj.drop(columns=['resultado']), model=best_model_name)

    # Avaliação das previsões
    
    from sklearn.metrics import precision_score

# Garantir que ambos estão como strings
    y_true = test_conj['resultado']



    
    print(f"Precisão: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nMelhor modelo para draw no bet: {best_model_name}")
    print(f"Acurácia no treino (validação interna): {best_model_score:.4f}")
    with open('autogluon_draw_no_bet_model_leaderboard_conj.txt', 'w+') as f:
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"Acurácia (validação interna): {best_model_score:.4f}\n")
        f.write("\nMétricas de Avaliação no conjunto de teste (dnb):\n")
        f.write(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precisão (macro): {precision_score(y_true, y_pred):.4f}\n")
        f.write(f"Recall (macro): {recall_score(y_true, y_pred):.4f}\n")
        f.write(f"F1-Score (macro): {f1_score(y_true, y_pred):.4f}\n")


    df_temporario = preparar_df_draw_no_bet(df)
    df_temporario.drop(columns=['indefinido', 'perde', 'reembolso','home','away'], inplace=True)
    df_temporario['goal_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['team_strength_home'] = df_temporario['media_victories_home'] / df_temporario['media_goals_home']
    df_temporario['team_strength_away'] = df_temporario['media_victories_away'] / df_temporario['media_goals_away']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']
    df_temporario['vic_ratio'] = df_temporario['home_h2h_win_rate'] - df_temporario['away_h2h_win_rate']


    if df_temporario.empty:
        print("DataFrame temporário vazio. Retornando None.")
        return None

    df_temporario.dropna(inplace=True)
    df_temporario['draw_no_bet_team'] = df_temporario['draw_no_bet_team'].astype(int)

    if 'ganha' not in df_temporario.columns:
        print("Coluna 'ganha' não encontrada no DataFrame. Retornando None.")
        return None

    df_temporario = df_temporario[df_temporario['ganha'].notna()]
    df_temporario['ganha'] = df_temporario['ganha'].astype(int)
    y_binario = df_temporario['ganha'].reset_index(drop=True)

    if y_binario.isna().any():
        print("Ainda existem valores NaN em y_binario. Retornando None.")
        return None

    X_final = df_temporario.drop(columns=['ganha']).reset_index(drop=True)

    if len(X_final) != len(y_binario):
        print(f"Tamanhos diferentes: X_final={len(X_final)}, y_binario={len(y_binario)}. Retornando None.")
        return None

    if X_final.isna().sum().sum() > 0:
        print("Ainda existem valores NaN em X_final. Retornando None.")
        return None

    if y_binario.nunique() < 2:
        print("Variável target tem menos de 2 classes. Retornando None.")
        return None

    print("Colunas de X (draw no bet):", X_final.columns.tolist())

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_binario, test_size=0.1, random_state=42)

    df_ag_train = X_train.copy()
    df_ag_train['target'] = y_train

    df_ag_test = X_test.copy()
    df_ag_test['target'] = y_test

    # Diretório temporário
    temp_dir = tempfile.mkdtemp()

    # Treinamento
    predictor = TabularPredictor(label='target', path=temp_dir, problem_type='binary').fit(
        df_ag_train,
        presets='best_quality',
        time_limit=1000
    )

    # Leaderboard
    # Leaderboard
    leaderboard = predictor.leaderboard(df_ag_train, silent=True)
    try:
        best_model_name = predictor.model_best  # <-- CORREÇÃO AQUI
        best_model_score = leaderboard.loc[leaderboard['model'] == best_model_name, 'score_val'].values[0]
    except:
        best_model_name = leaderboard.loc[leaderboard['score_val'].idxmax(), 'model']
        best_model_score = leaderboard.loc[leaderboard['score_val'].idxmax(), 'score_val']

    

    # Salvar modelo final
    final_model_path = "autogluon_draw_no_bet_model"
    shutil.move(temp_dir, final_model_path)

    # Recarregar
    predictor = TabularPredictor.load(final_model_path)

    # Predição no conjunto de teste com o melhor modelo
    y_pred = predictor.predict(df_ag_test.drop(columns=['target']), model=best_model_name)   

    # Avaliação
    y_true = df_ag_test['target']
    
    print(f"Precisão: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nMelhor modelo para Draw No Bet: {best_model_name}")
    print(f"Acurácia no treino (validação interna): {best_model_score:.4f}")
    with open('autogluon_draw_no_bet_model_leaderboard.txt', 'w+') as f:
        f.write(f"Melhor modelo: {best_model_name}\n")
        f.write(f"Acurácia: {best_model_score:.4f}\n")
        f.write("\nMétricas de Avaliação no conjunto de teste (Draw No Bet):")
        f.write(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")

    return 0.7


def ql_gl(df=df_temp):
    '''cria q learning para goalline'''
    # Pré-processamento do dataframe
  
    '''
    df_temporario['prob_gl1'] = 1 / df_temporario['odds_gl1']
    df_temporario['prob_gl2'] = 1 / df_temporario['odds_gl2']
    soma = df_temporario['prob_gl1'] + df_temporario['prob_gl2']
    df_temporario['prob_gl1'] /= soma
    df_temporario['prob_gl2'] /= soma
    '''
    
    #CONJ

    df_conj = df.copy()
    
   

    
    df_conj = df_conj[['goal_line1_1', 'goal_line1_2','league','odds_gl1','odds_gl2','media_goals_home','media_goals_sofridos_home','media_goals_away','media_goals_sofridos_away','h2h_mean','h2h_total_games','gl1_positivo','gl2_positivo']]
    df_conj.dropna(inplace=True)
    def transformar_target(row):
        if (row['gl1_positivo'] == 1):
            return 0
        elif (row['gl2_positivo'] == 1):
            return 1
        else:
            return None

    df_conj['resultado'] = df_conj.apply(transformar_target, axis=1)
    df_conj = df_conj[df_conj['resultado'].notna()]  # Remove os casos de reembolso
    df_conj.drop(columns=['gl1_positivo','gl2_positivo'], inplace=True)

    #  PASSO 1: TREINAR O MODELO Q-LEARNING 
    print("\n=== Treinando modelo Q-Learning para Goal Line ===")

    # Dividir em conjunto de treino e teste para o Q-Learning
    train_q, test_q = train_test_split(df_conj, test_size=0.05, random_state=42)

    # Inicializar e treinar o agente Q-Learning
    agent = QLearningGoalLine()
    agent.train(train_q)  # Ajuste o número de episódios conforme necessário

    # Avaliar o modelo Q-Learning
    q_evaluation = agent.evaluate(test_q)
    print(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
    print(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
    print(f"unidades: {q_evaluation['uni']}")

    print("\nAcurácia por tipo de gl (Q-Learning):")
    for action, accuracy in q_evaluation['accuracy_by_action'].items():
        dc_type = {0: "OVER", 1: "UNDER"}
        correct = q_evaluation['results_by_action'][action]['correct']
        total = q_evaluation['results_by_action'][action]['total']
        print(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")

    with open('q_learning_gl.txt','w+') as f:
        f.write(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
        f.write(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
        f.write(f"unidades: {q_evaluation['uni']}")

        f.write("\nAcurácia por tipo de gl (Q-Learning):")
        for action, accuracy in q_evaluation['accuracy_by_action'].items():
            dc_type = {0: "OVER", 1: "UNDER"}
            correct = q_evaluation['results_by_action'][action]['correct']
            total = q_evaluation['results_by_action'][action]['total']
            f.write(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")

    # Salvar o modelo Q-Learning
    agent.save_model('q_learning_gl_model_final.pkl')



def ql_dc(df=df_temp):
    '''cria q learning para double_chance'''
    """
    Treinamento de modelo para Double Chance incluindo Q-Learning junto com AutoGluon
    """
    # Pré-processamento do dataframe
    df_temporario = df.copy()
    '''
    df_temporario['prob_dc1'] = 1 / df_temporario['odds_dc1']
    df_temporario['prob_dc2'] = 1 / df_temporario['odds_dc2']
    df_temporario['prob_dc3'] = 1 / df_temporario['odds_dc3']
    total = df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']].sum(axis=1)
    
    df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']] = df_temporario[['prob_dc1', 'prob_dc2', 'prob_dc3']].div(total, axis=0)
    '''
    # Adicionar cálculos de diferenças
    df_temporario['goal_diff'] = df_temporario['media_goals_home'] - df_temporario['media_goals_away']
    df_temporario['victory_diff'] = df_temporario['media_victories_home'] - df_temporario['media_victories_away']
    df_temporario['h2h_diff'] = df_temporario['home_h2h_mean'] - df_temporario['away_h2h_mean']
    df_temporario.dropna(inplace=True)
    
    # **** PASSO 1: TREINAR O MODELO Q-LEARNING ****
    print("\n=== Treinando modelo Q-Learning para Double Chance ===")
    
    # Preparar o DataFrame para Q-Learning (isso já calcula as diferenças)
  
    df_conj = df_temporario.copy()
   

    
    df_conj['goals_ratio_home'] = df_conj['media_goals_home'] - df_conj['media_goals_sofridos_home']
    df_conj['goals_ratio_away'] = df_conj['media_goals_away'] - df_conj['media_goals_sofridos_away']
    df_conj['vic_ratio'] = df_conj['home_h2h_win_rate'] - df_conj['away_h2h_win_rate']

    df_conj = df_conj[['league',  
         'odds_dc1', 
        'odds_dc2', 
         'odds_dc3',
        'goal_diff', 'victory_diff', 'h2h_diff','goals_ratio_home','goals_ratio_away','vic_ratio','h2h_total_games','home_h2h_win_rate','away_h2h_win_rate',
        'res_double_chance1', 'res_double_chance2', 'res_double_chance3','media_victories_home','media_victories_away']]
    df_conj.dropna(inplace=True)
    
    # Dividir em conjunto de treino e teste para o Q-Learning
    train_q, test_q = train_test_split(df_conj, test_size=0.05, random_state=42)
    
    # Inicializar e treinar o agente Q-Learning
    agent = QLearningDoubleChance()
    agent.train(train_q)  # Ajuste o número de episódios conforme necessário
    
    # Avaliar o modelo Q-Learning
    q_evaluation = agent.evaluate(test_q)
    print(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
    print(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
    print(f"unidades: {q_evaluation['uni']}")
    
    print("\nAcurácia por tipo de Double Chance (Q-Learning):")
    for action, accuracy in q_evaluation['accuracy_by_action'].items():
        dc_type = {0: "DC1 (Casa ou Empate)", 1: "DC2 (Fora ou Empate)", 2: "DC3 (Casa ou Fora)"}
        correct = q_evaluation['results_by_action'][action]['correct']
        total = q_evaluation['results_by_action'][action]['total']
        print(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    with open('q_learning_dc.txt','w+') as f:
        f.write(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
        f.write(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
        f.write(f"unidades: {q_evaluation['uni']}")
        
        f.write("\nAcurácia por tipo de Double Chance (Q-Learning):")
        for action, accuracy in q_evaluation['accuracy_by_action'].items():
            dc_type = {0: "DC1 (Casa ou Empate)", 1: "DC2 (Fora ou Empate)", 2: "DC3 (Casa ou Fora)"}
            correct = q_evaluation['results_by_action'][action]['correct']
            total = q_evaluation['results_by_action'][action]['total']
            f.write(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    # Salvar o modelo Q-Learning
    agent.save_model('q_learning_dc_model_final.pkl')
    
        

def ql_dnb(df=df_temp):
    '''cria q learning para draw_no_bet'''
    # Pré-processamento
    '''
    df['prob_odds_dnb1'] = 1 / df['odds_dnb1']
    df['prob_odds_dnb2'] = 1 / df['odds_dnb2']
    tot = df['prob_odds_dnb1'] + df['prob_odds_dnb2']
    df['prob_odds_dnb1'] = df['prob_odds_dnb1'] / tot
    df['prob_odds_dnb2'] = df['prob_odds_dnb2'] / tot
    '''
    df_conj = df.copy()
    df_conj['goal_diff'] = df_conj['media_goals_home'] - df_conj['media_goals_away']
      
    df_conj['h2h_diff'] = df_conj['home_h2h_mean'] - df_conj['away_h2h_mean']
    
    df_conj['goals_ratio_home'] = df_conj['media_goals_home'] - df_conj['media_goals_sofridos_home']
    df_conj['goals_ratio_away'] = df_conj['media_goals_away'] - df_conj['media_goals_sofridos_away']
    df_conj['vic_ratio'] = df_conj['home_h2h_win_rate'] - df_conj['away_h2h_win_rate']
    df_conj['victory_diff'] = df_conj['media_victories_home'] - df_conj['media_victories_away']
    df_conj = df_conj[['league',  
         'odds_dnb1', 
        'odds_dnb2',
        'goal_diff', 'victory_diff', 'h2h_diff','goals_ratio_home','draw_no_bet_team1','draw_no_bet_team2','goals_ratio_away','vic_ratio','home_h2h_win_rate','away_h2h_win_rate','media_victories_home','media_victories_away','h2h_total_games','dnb1_ganha','dnb2_ganha']]
    df_conj.dropna(inplace=True)

    def transformar_res(row):
        if row['dnb1_ganha'] == 1:
            return 0
        elif row['dnb2_ganha'] == 1:
            return 1
        else:
            return None
        

    df_conj['resultado'] = df_conj.apply(transformar_res, axis=1)
    df_conj = df_conj[df_conj['resultado'].notna()].copy()
    df_conj.drop(columns=['dnb1_ganha','dnb2_ganha'],inplace=True)
    # **** PASSO 1: TREINAR O MODELO Q-LEARNING ****
    print("\n=== Treinando modelo Q-Learning para DNB ===")
    
    # Dividir em conjunto de treino e teste para o Q-Learning
    train_q, test_q = train_test_split(df_conj, test_size=0.05, random_state=42)
    
    # Inicializar e treinar o agente Q-Learning
    agent = QLearningDrawNoBet()
    agent.train(train_q)  # Ajuste o número de episódios conforme necessário
    
    # Avaliar o modelo Q-Learning
    q_evaluation = agent.evaluate(test_q)
    print(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
    print(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
    print(f"unidades: {q_evaluation['uni']}")
    
    print("\nAcurácia por tipo de DNB (Q-Learning):")
    for action, accuracy in q_evaluation['accuracy_by_action'].items():
        dc_type = {0: "vitoria time home", 1: "vitoria time away"}
        correct = q_evaluation['results_by_action'][action]['correct']
        total = q_evaluation['results_by_action'][action]['total']
        print(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    with open('q_learning_dnb.txt','w+') as f:
        f.write(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
        f.write(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
        f.write(f"unidades: {q_evaluation['uni']}")
        
        f.write("\nAcurácia por tipo de DNB (Q-Learning):")
        for action, accuracy in q_evaluation['accuracy_by_action'].items():
            dc_type = {0: "vitoria time home", 1: "vitoria time away"}
            correct = q_evaluation['results_by_action'][action]['correct']
            total = q_evaluation['results_by_action'][action]['total']
            f.write(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    # Salvar o modelo Q-Learning
    agent.save_model('q_learning_dnb_model_final.pkl')



def ql_h(df=df_temp):
    '''cria q learning para handicap'''
    # Pré-processamento
    df_temporario = df.copy()

    #CONJ

    df_conj = df_temporario.copy()
    df_conj['favorite_by_odds'] = df_conj['odds_ah1'] < df_conj['odds_ah2']
    df_conj['odds_ratio'] = df_conj['odds_ah1'] / df_conj['odds_ah2']
    df_conj['goals_diff'] = df_conj['media_goals_home'] - df_conj['media_goals_away']
    df_conj['h2h_diff'] = df_conj['home_h2h_mean'] - df_conj['away_h2h_mean']
    
    df_conj['goals_ratio_home'] = df_conj['media_goals_home'] - df_conj['media_goals_sofridos_home']
    df_conj['goals_ratio_away'] = df_conj['media_goals_away'] - df_conj['media_goals_sofridos_away']
    df_conj['vic_ratio'] = df_conj['home_h2h_win_rate'] - df_conj['away_h2h_win_rate']
    df_conj['victory_diff'] = df_conj['media_victories_home'] - df_conj['media_victories_away']
    
    def transformar_resultado(row):
        if row['ah1_positivo'] == 1:
            return 0
        elif row['ah2_positivo'] == 1:
            return 1
        else:
            return None
        

    df_conj['resultado'] = df_conj.apply(transformar_resultado, axis=1)
    df_conj = df_conj[['asian_handicap1_1', 'asian_handicap1_2','asian_handicap2_1', 'asian_handicap2_2','team_ah1','odds_ah1', 'team_ah2','odds_ah2','h2h_diff','goals_diff', 'league','goals_ratio_home','goals_ratio_away','vic_ratio','victory_diff','home_h2h_win_rate','away_h2h_win_rate','h2h_total_games','media_victories_away','media_victories_home','media_goals_sofridos_away','media_goals_sofridos_home','media_goals_home','media_goals_away','resultado','home_h2h_mean','away_h2h_mean']].copy()
    df_conj = df_conj[df_conj['resultado'].notna()].copy()
    df_conj.dropna(inplace=True)
    # **** PASSO 1: TREINAR O MODELO Q-LEARNING ****
    print("\n=== Treinando modelo Q-Learning para ah ===")
    
    # Dividir em conjunto de treino e teste para o Q-Learning
    train_q, test_q = train_test_split(df_conj, test_size=0.05, random_state=42)
    
    # Inicializar e treinar o agente Q-Learning
    agent = QLearningHandicap()
    agent.train(train_q)  # Ajuste o número de episódios conforme necessário
    
    # Avaliar o modelo Q-Learning
    q_evaluation = agent.evaluate(test_q)
    print(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
    print(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
    print(f"unidades: {q_evaluation['uni']}")
    
    print("\nAcurácia por tipo de AH (Q-Learning):")
    for action, accuracy in q_evaluation['accuracy_by_action'].items():
        dc_type = {0: "vitoria time home", 1: "vitoria time away"}
        correct = q_evaluation['results_by_action'][action]['correct']
        total = q_evaluation['results_by_action'][action]['total']
        print(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    with open('q_learning_h.txt','w+') as f:
        f.write(f"\nModelo Q-Learning - Acurácia: {q_evaluation['accuracy']:.4f}")
        f.write(f"Previsões corretas: {q_evaluation['correct_predictions']}/{q_evaluation['total_predictions']}")
        f.write(f"unidades: {q_evaluation['uni']}")
        
        f.write("\nAcurácia por tipo de DNB (Q-Learning):")
        for action, accuracy in q_evaluation['accuracy_by_action'].items():
            dc_type = {0: "vitoria time home", 1: "vitoria time away"}
            correct = q_evaluation['results_by_action'][action]['correct']
            total = q_evaluation['results_by_action'][action]['total']
            f.write(f"{dc_type[action]}: {accuracy:.4f} ({correct}/{total})")
    
    # Salvar o modelo Q-Learning
    agent.save_model('q_learning_h_model_final.pkl')


def atua():
    '''atualiza o df_temp_preprocessado_teste caso seja necessario para criação dos q-learnings'''
    df = df_temp.copy()
    df = preProcessGeneral(df)
    df.to_csv('df_temp_preprocessado_teste.csv', index=False)