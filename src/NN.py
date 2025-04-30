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




logger = logging.getLogger(__name__)

#'10048705', 'Esoccer GT Leagues - 12 mins play' ;'10047781', 'Esoccer Battle - 8 mins play'
#testar pegar evento 171732570 mais tarde   171790606  172006772 9723272 172006783
load_dotenv()
api = os.getenv("API_KEY")
apiclient = BetsAPIClient(api_key=api)

def dia_anterior():
        """Retorna o dia anterior ao atual no formato YYYYMMDD."""
        ontem = datetime.now() - timedelta(days=1)
        return ontem.strftime("%Y%m%d")


#df_temp = pd.read_csv(r"C:\Users\Leoso\Downloads\projBotAposta\src\resultados_novo.csv")
df_temp = pd.read_csv('resultados_60.csv')

def preProcessGeneral(df=df_temp):
    df = preProcessEstatisticasGerais(df)
    df = preProcessOverUnder(df)
    df = preProcessHandicap(df)
    df = preProcessGoalLine(df)
    df = preProcessDoubleChance(df)
    df = preProcessDrawNoBet(df)
    return df


def preProcessGeneral_x(df):
    
    df = preProcessEstatisticasGerais_X(df)
    df = preProcessHandicap_X(df)
    df = preProcessGoalLine_X(df)
    return df

def criaNNs():
    df = df_temp.copy()
    df = preProcessGeneral(df)
    z_over_under_positivo, z_over_under_negativo = NN_over_under(df)
    z_handicap = NN_handicap(df)
    z_goal_line = NN_goal_line(df)
    z_double_chance = NN_double_chance(df)
    z_draw_no_bet = NN_draw_no_bet(df)
    lista = [z_over_under_positivo, z_over_under_negativo, z_handicap, z_goal_line, z_double_chance , z_draw_no_bet]
    return lista
def estatisticas_ultimos_10(home_team, away_team):
    df = df_temp.copy()
    # Filtrar os últimos 10 jogos do home_team apenas como mandante
    df_home = df[df['home'] == home_team].head(10)
    media_gols_home = df_home['home_goals'].mean() if not df_home.empty else None
    vitorias_home = (df_home['home_goals'] > df_home['away_goals']).mean() if not df_home.empty else None
    # Filtrar os últimos 10 jogos do away_team apenas como visitante
    df_away = df[df['away'] == away_team].head(10)
    media_gols_away = df_away['away_goals'].mean() if not df_away.empty else None
    vitorias_away = (df_away['away_goals'] > df_away['home_goals']).mean() if not df_away.empty else None
    return media_gols_home, vitorias_home, media_gols_away, vitorias_away, df_away

def preProcessEstatisticasGerais(df):
    # Atualiza a chamada para passar também a data do jogo
    df[['media_goals_home', 'media_victories_home', 'media_goals_away', 'media_victories_away']] = df.apply(
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
    
    return df





def preProcessEstatisticasGerais_X(df):
    logger.info("🧮 Iniciando preProcessEstatisticasGerais_X")
    try:

        # Calcular as estatísticas das últimas 5 partidas
        df[['media_goals_home', 'media_victories_home', 'media_goals_away', 'media_victories_away']] = df.apply(
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

        logger.debug("📊 Estatísticas H2H calculadas")
        return df

    except Exception as e:
        logger.exception("❌ Erro em preProcessEstatisticasGerais_X")
        return df



def preProcessOverUnder(df=df_temp):
    df['res_goals_over_under'] = df['tot_goals'] > df['goals_over_under'].astype(float)
    return df




def preProcessHandicap(df=df_temp):
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
    df = calcular_resultado_double_chance(df)
    return df

def preProcessDrawNoBet(df=df_temp):
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










#standarization
def normalizacao_and_split(X, y):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_standardized,y, random_state=42, test_size=0.2)
    return x_train, x_test, y_train, y_test

#standarization
def normalizacao(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized
    
def split(X_standardized, y):
    x_train, x_test, y_train, y_test = train_test_split(X_standardized,y, random_state=42, test_size=0.2)
    return x_train, x_test, y_train, y_test

#ESTATISTICAS GERAIS DE CONFRONTO




from datetime import datetime

def estatisticas_ultimos_5(home_team, away_team):
    try:
    

        # Filtra os jogos anteriores à data atual e do time como mandante
        df_home = df_temp[(df_temp['home'] == home_team)].head(5)
        media_gols_home = df_home['home_goals'].mean() if not df_home.empty else np.nan
        vitorias_home = (df_home['home_goals'] > df_home['away_goals']).mean() if not df_home.empty else np.nan

        # Filtra os jogos anteriores à data atual e do time como visitante
        df_away = df_temp[(df_temp['away'] == away_team)].head(5)
        media_gols_away = df_away['away_goals'].mean() if not df_away.empty else np.nan
        vitorias_away = (df_away['away_goals'] > df_away['home_goals']).mean() if not df_away.empty else np.nan

        return pd.Series({
            'media_goals_home': media_gols_home,
            'media_victories_home': vitorias_home,
            'media_goals_away': media_gols_away,
            'media_victories_away': vitorias_away
        })

    except Exception as e:
        print(f"❌ Erro em estatisticas_ultimos_5 para {home_team} x {away_team}: {e}")
        return pd.Series({
            'media_goals_home': np.nan,
            'media_victories_home': np.nan,
            'media_goals_away': np.nan,
            'media_victories_away': np.nan
        })



def calcular_medias_h2h(home_id, away_id, index):
    """
    Calcula três estatísticas de confronto direto com base na data do jogo:
    - h2h_mean: média de gols totais nos confrontos anteriores
    - home_h2h_mean: média de gols marcados pelo time atual como mandante
    - away_h2h_mean: média de gols marcados pelo time atual como visitante
    """
    df = df_temp.copy()

    # Filtrar confrontos entre os dois times (em qualquer ordem)
    confrontos = df[((df['home'] == home_id) & (df['away'] == away_id)) |
                    ((df['home'] == away_id) & (df['away'] == home_id))]

    # Filtrar apenas confrontos ocorridos antes da data do jogo atual
    confrontos_passados = confrontos.loc[index + 1:].head(10)

    if confrontos_passados.empty:
        return {
            'h2h_mean': None,
            'home_h2h_mean': None,
            'away_h2h_mean': None
        }
    

    # Média de gols totais por confronto
    h2h_mean = confrontos_passados['tot_goals'].mean()

    # Ajustar médias por lado (quem está sendo analisado como home/away no jogo atual)
    home_goals = []
    away_goals = []
    for _, row in confrontos_passados.iterrows():
        if row['home'] == home_id:
            home_goals.append(row['home_goals'])
            away_goals.append(row['away_goals'])
        else:
            home_goals.append(row['away_goals'])
            away_goals.append(row['home_goals'])

    return {
        'h2h_mean': h2h_mean,
        'home_h2h_mean': np.mean(home_goals),
        'away_h2h_mean': np.mean(away_goals)
    }


def calcular_medias_h2h_X(home_id, away_id):
    """
    Calcula três estatísticas de confronto direto, considerando apenas os últimos 5 confrontos
    anteriores à data do jogo:
    - h2h_mean: média de gols totais (home_goals + away_goals) nos confrontos anteriores
    - home_h2h_mean: média de gols marcados pelo time da casa (home_id) nos confrontos
    - away_h2h_mean: média de gols marcados pelo time visitante (away_id) nos confrontos
    """
    df = df_temp.copy()

    # Filtrar todos os confrontos entre os times
    confrontos = df[((df['home'] == home_id) & (df['away'] == away_id)) |
                    ((df['home'] == away_id) & (df['away'] == home_id))]

    if confrontos.empty:
        return {'h2h_mean': None, 'home_h2h_mean': None, 'away_h2h_mean': None}

    # Filtrar apenas os confrontos anteriores à data do jogo
    confrontos_passados = confrontos.head(10)

    if confrontos_passados.empty:
        return {'h2h_mean': None, 'home_h2h_mean': None, 'away_h2h_mean': None}

    # Pegar os últimos 5 confrontos mais recentes antes da data do jogo

    # Calcular média geral de gols totais
    h2h_mean = confrontos_passados['tot_goals'].mean()

    # Calcular média de gols específicos por time
    home_goals = []
    away_goals = []
    for _, row in confrontos_passados.iterrows():
        if row['home'] == home_id:
            home_goals.append(row['home_goals'])
            away_goals.append(row['away_goals'])
        else:
            home_goals.append(row['away_goals'])
            away_goals.append(row['home_goals'])

    home_h2h_mean = np.mean(home_goals) if home_goals else None
    away_h2h_mean = np.mean(away_goals) if away_goals else None

    return {
        'h2h_mean': h2h_mean,
        'home_h2h_mean': home_h2h_mean,
        'away_h2h_mean': away_h2h_mean
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

def classify_asian_handicap_i1(team, ah1, ah2, diff_goals):
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
            return 'meia'

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
    required_columns = ['home','away','times','odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean','league']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNOver_under_X': {', '.join(missing_cols)}")
        return None, None
    
    
    df_temporario = df[required_columns].copy()
    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNoverunder*")
        return None, None
    z = df_temporario[['times','odd_goals_over1', 'odd_goals_under1','league']].copy()

    X = df_temporario[['odd_goals_over1', 'odd_goals_under1','media_goals_home','media_goals_away' ,'h2h_mean','league']]

    try:
        with open('scaler_over_under.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_standardized = scaler.transform(X)
        X = pd.DataFrame(X_standardized, columns=['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean','league']).reset_index(drop=True)
    except Exception as e:
        print('Erro over_under')
        return None, None
    return X, z

def prepNNHandicap_X(df=df_temp):
    required_columns = ['home','away','times','media_goals_home', 'media_goals_away','home_h2h_mean', 'away_h2h_mean',
                        'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 
                        'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2','league']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNHandicap_X': {', '.join(missing_cols)}")
        return None, None

    df_temporario = df[required_columns].copy()
    df_temporario = preparar_df_handicaps_X(df_temporario)

    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNhandicap*")
        return None, None
    z = df_temporario[['times','team_ah','asian_handicap_1', 'asian_handicap_2', 'odds','league']]
    z = z.sort_values('team_ah').reset_index(drop=True)
    df_temporario = df_temporario.sort_values('team_ah').reset_index(drop=True)
    
    df_temporario = pd.get_dummies(df_temporario, columns=['team_ah'], prefix='team_ah')
    X = df_temporario[['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds','league']]
    try:
        with open('scaler_handicap.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_standardized = scaler.transform(X)
        X = pd.DataFrame(X_standardized, columns=['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds','league']).reset_index(drop=True)
        type_df = df_temporario[['team_ah_1','team_ah_2']].reset_index(drop=True)
    except Exception as e:
        print('Erro handicap')
        return None, None
    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z

def prepNNGoal_line_X(df=df_temp):
    required_columns = ['home','away','times','h2h_mean' ,'media_goals_home' ,'media_goals_away',
                        'goal_line1_1','goal_line1_2','type_gl1','odds_gl1', 'odds_gl2',
                        'goal_line2_1','goal_line2_2','type_gl2','league']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNGoal_line_X': {', '.join(missing_cols)}")
        return None, None

    df_temporario = df[required_columns].copy()
    df_temporario = preparar_df_goallines_X(df_temporario)
    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNgl*")
        return None, None
    z = df_temporario[['times','goal_line_1', 'goal_line_2','type_gl', 'odds_gl','league']].copy()
    
    df_temporario = pd.get_dummies(df_temporario, columns=['type_gl'], prefix='type_gl')
    X = df_temporario[['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2','league']].copy()
    try:
        with open('scaler_goal_line.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_standardized = scaler.transform(X)
        X = pd.DataFrame(X_standardized, columns=['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2','league']).reset_index(drop=True)
        type_df = df_temporario[['type_gl_1', 'type_gl_2']].reset_index(drop=True)
    except Exception as e:
        print('Erro goal_line')
        return None, None
    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z

def prepNNDouble_chance_X(df=df_temp):
    required_columns = ['home','away','times','media_goals_home','media_goals_away','media_victories_home', 'media_victories_away', 
                        'home_h2h_mean', 'away_h2h_mean', 'double_chance1','odds_dc1', 
                        'double_chance2', 'odds_dc2', 'double_chance3', 'odds_dc3','league']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNDouble_chance_X': {', '.join(missing_cols)}")
        return None, None

    df_temporario = df[required_columns].copy()
    df_temporario = preparar_df_double_chance_X(df_temporario)
    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNdc*")
        return None, None
    z = df_temporario[['times','double_chance', 'odds','league']].copy()
    
    
    df_temporario = pd.get_dummies(df_temporario, columns=['double_chance'], prefix='double_chance_type')
    X = df_temporario[['media_goals_home', 'media_goals_away', 'media_victories_home',
                      'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']].copy()
    try:
        with open('scaler_double_chance.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_standardized = scaler.transform(X)
        X = pd.DataFrame(X_standardized, columns=['media_goals_home', 'media_goals_away', 'media_victories_home',
                                     'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']).reset_index(drop=True)
        type_df = df_temporario[['double_chance_type_1', 'double_chance_type_2','double_chance_type_3']].reset_index(drop=True)
    except Exception as e:
        print('Erro double_chance')
        return None, None
    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z

def prepNNDraw_no_bet_X(df=df_temp):
    required_columns = ['home','away','times', 'media_goals_home', 'media_goals_away', 'media_victories_home','media_victories_away',
                        'home_h2h_mean','away_h2h_mean', 'draw_no_bet_team1', 'odds_dnb1', 'draw_no_bet_team2', 'odds_dnb2','league']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"❌ Colunas ausentes em 'prepNNDraw_no_bet_X': {', '.join(missing_cols)}")
        return None, None

    df_temporario = df[required_columns].copy()
    df_temporario = preparar_df_draw_no_bet_X(df_temporario)
    df_temporario.dropna(inplace=True)
    if df_temporario.empty:
        print("❌ DataFrame vazio após dropna em prepNNdnb*")
        return None, None
    z = df_temporario[['times','draw_no_bet_team', 'odds','league']].copy()
  
    df_temporario = pd.get_dummies(df_temporario, columns=['draw_no_bet_team'], prefix='draw_no_bet_type')
    X = df_temporario[['media_goals_home', 'media_goals_away', 'media_victories_home','media_victories_away', 
                      'home_h2h_mean', 'away_h2h_mean', 'odds','league']].copy()
    try:
        with open('scaler_draw_no_bet.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_standardized = scaler.transform(X)
        X = pd.DataFrame(X_standardized, columns=['media_goals_home', 'media_goals_away', 'media_victories_home',
                                     'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']).reset_index(drop=True)
        type_df = df_temporario[['draw_no_bet_type_1', 'draw_no_bet_type_2']].reset_index(drop=True)
    except Exception as e:
        print('Erro draw_no_bet')
        return None, None
    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z



#NN over_under
def prepNNOver_under(df=df_temp):
    df_temporario = df[['home','away','odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean','res_goals_over_under','league']].copy()
    df_temporario.dropna(inplace=True)
    z = df_temporario[['home_name','away_name','odd_goals_over1', 'odd_goals_under1','league']].copy()
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    X = df[['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean','league']]
    
    X = normalizacao(X)
    X = pd.DataFrame(X, columns=['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean','league'])
    return X, z

#NN over_under
'''
def prepNNOver_under_X(df=df_temp):
    df_temporario = df[['home','away','odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean']].copy()
    null_cols = df_temporario.columns[df_temporario.isnull().any()]

    # Printar essas colunas
    print("Colunas com valores nulos:")
    print(null_cols)
    df_temporario.dropna(inplace=True)
    z = df_temporario[['home','away','odd_goals_over1', 'odd_goals_under1']].copy()
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    X = df[['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean']]
    try:
        X = normalizacao(X)
        X = pd.DataFrame(X, columns=['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean'])
    except:
        print('over_under/faltou dados')
    
    return X, z
'''
def NN_over_under(df):

    df_temporario = df[['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean','res_goals_over_under','league']].copy()
    df_temporario.dropna(inplace=True)
    X = df_temporario[['odd_goals_over1', 'odd_goals_under1', 'media_goals_home','media_goals_away' ,'h2h_mean','league']]
    y = df_temporario['res_goals_over_under']
    scaler_over_under = StandardScaler()
    X_standardized = scaler_over_under.fit_transform(X)
    with open('scaler_over_under.pkl', 'wb') as f:
        pickle.dump(scaler_over_under, f)
    x_train, x_test, y_train, y_test = split(X_standardized,y)

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
    hist_bin = model_over_under.fit(x_train, y_train, epochs=30)
    y_pred_probs = model_over_under.predict(x_test)
    
    melhor_z_positivo = encontrar_melhor_z_binario_positivo(y_test, y_pred_probs)
    melhor_z_negativo = encontrar_melhor_z_binario_negativo(y_test, y_pred_probs)
 
    model_over_under.save("model_binario_over_under.keras")  # Salva em formato nativo do Keras
    return melhor_z_positivo, melhor_z_negativo


#junta handicaps
def preparar_df_handicaps(df):
    

    # Seleciona e renomeia o df_temporario1
    df1 = df[['home','away','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
              'asian_handicap1_1', 'asian_handicap1_2','team_ah1', 'odds_ah1',
              'ah1_indefinido', 'ah1_negativo', 'ah1_positivo', 'ah1_reembolso','league']].copy()
    df1.columns = ['home','away','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
                   'asian_handicap_1', 'asian_handicap_2','team_ah', 'odds',
                   'indefinido', 'negativo', 'positivo', 'reembolso','league']

    # Seleciona e renomeia o df_temporario2
    df2 = df[['home','away','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
              'asian_handicap2_1', 'asian_handicap2_2','team_ah2', 'odds_ah2',
              'ah2_indefinido', 'ah2_negativo', 'ah2_positivo', 'ah2_reembolso','league']].copy()
    df2.columns = ['home','away','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
                   'asian_handicap_1', 'asian_handicap_2','team_ah', 'odds',
                   'indefinido', 'negativo', 'positivo', 'reembolso','league']

    # Concatena os dois dataframes
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final

def prepNNHandicap(df=df_temp):
    df_temporario = df[['home','away','media_goals_home', 'media_goals_away','home_h2h_mean', 'away_h2h_mean',
                       'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 
                       'ah1_indefinido','ah1_negativo', 'ah1_positivo','ah1_reembolso', 
                       'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2', 
                       'ah2_indefinido','ah2_negativo', 'ah2_positivo','ah2_reembolso','league']].copy()
    df_temporario = preparar_df_handicaps(df_temporario)
    
    df_temporario = df_temporario[df_temporario['indefinido'] == False]
    df_temporario.dropna(inplace=True)
    
    z = df_temporario[['home','away','team_ah','asian_handicap_1', 'asian_handicap_2', 'odds','league']]
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    df_temporario = pd.get_dummies(df_temporario, columns=['team_ah'], prefix='team_ah')
    X = df_temporario[['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds','league']]
    X = normalizacao(X)
    X = pd.DataFrame(X, columns=['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds','league']).reset_index(drop=True)
    type_df = df_temporario[['team_ah_1.0',	'team_ah_2.0']]
    type_df = type_df.reset_index(drop=True)
    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z

def preparar_df_handicaps_X(df):
    

    # Seleciona e renomeia o df_temporario1
    df1 = df[['home','away','times','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
              'asian_handicap1_1', 'asian_handicap1_2','team_ah1', 'odds_ah1','league']].copy()
    df1.columns = ['home','away','times','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
                   'asian_handicap_1', 'asian_handicap_2','team_ah', 'odds','league']

    # Seleciona e renomeia o df_temporario2
    df2 = df[['home','away','times','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
              'asian_handicap2_1', 'asian_handicap2_2','team_ah2', 'odds_ah2','league']].copy()
    df2.columns = ['home','away','times','media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean',
                   'asian_handicap_1', 'asian_handicap_2','team_ah', 'odds','league']

    # Concatena os dois dataframes
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final
'''
def prepNNHandicap_X(df=df_temp):
    df_temporario = df[['home','away','media_goals_home', 'media_goals_away','home_h2h_mean', 'away_h2h_mean',
                       'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 
                       'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2']].copy()
 
    df_temporario = preparar_df_handicaps_X(df_temporario)
 
    null_cols = df_temporario.columns[df_temporario.isnull().any()]

    # Printar essas colunas
    print("Colunas com valores nulos:")
    print(null_cols)
    df_temporario.dropna(inplace=True)
    
    z = df_temporario[['home','away','team_ah','asian_handicap_1', 'asian_handicap_2', 'odds']]
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    df_temporario = pd.get_dummies(df_temporario, columns=['team_ah'], prefix='team_ah')

    X = df_temporario[['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds']]

    try:
        X = normalizacao(X)
        X = pd.DataFrame(X, columns=['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds']).reset_index(drop=True)
    except:
        print('faltou dados handicap')
    type_df = df_temporario[['team_ah_1','team_ah_2']]
    type_df = type_df.reset_index(drop=True)
    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z
'''
#NN handicap
def NN_handicap(df=df_temp):
    
    
    
    # Agora pode prosseguir com a seleção das colunas
    df_temporario = df[['home','away','media_goals_home', 'media_goals_away','home_h2h_mean', 'away_h2h_mean',
                       'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 
                       'ah1_indefinido','ah1_negativo', 'ah1_positivo','ah1_reembolso', 
                       'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2', 
                       'ah2_indefinido','ah2_negativo', 'ah2_positivo','ah2_reembolso','league']].copy()
    df_temporario = preparar_df_handicaps(df_temporario)

    df_temporario = pd.get_dummies(df_temporario, columns=['team_ah'], prefix='team_ah')
    
    
    df_temporario = df_temporario[df_temporario['indefinido'] == False]
    
    
    df_temporario.dropna(inplace=True)
    X = df_temporario[['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds','league']]
    scaler_handicap = StandardScaler()
    X_standardized = scaler_handicap.fit_transform(X)
    with open('scaler_handicap.pkl', 'wb') as f:
        pickle.dump(scaler_handicap, f)
    
    X = pd.DataFrame(X_standardized, columns=['media_goals_home', 'media_goals_away', 'home_h2h_mean', 'away_h2h_mean','asian_handicap_1', 'asian_handicap_2', 'odds','league']).reset_index(drop=True)
    type_df = df_temporario[['team_ah_1.0',	'team_ah_2.0']]
    type_df = type_df.reset_index(drop=True)
    X_final = pd.concat([X, type_df], axis=1)
    y = df_temporario[['negativo', 'positivo', 'reembolso']].copy()
    

    y_binario = df_temporario['positivo'].astype(int)

    # 2. Reutilizando o X_final já preparado e normalizado
    x_train_bin, x_test_bin, y_train_bin, y_test_bin = split(X_final, y_binario)



        # 3. Criando o modelo binário
    modelo_binario = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train_bin.shape[1],)),
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

    modelo_binario.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Treinamento
    hist_bin = modelo_binario.fit(x_train_bin, y_train_bin, epochs=30)
    y_pred_probs = modelo_binario.predict(x_test_bin)
    melhor_z_positivo = encontrar_melhor_z_binario_positivo(y_test_bin, y_pred_probs )

    modelo_binario.save("model_handicap_binario.keras")  # Salva em formato nativo do Keras

    return melhor_z_positivo

    # 5. Previsões
    
    
    '''
    x_train, x_test, y_train, y_test = split(X_final, y)
    model_handicap = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model_handicap.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_handicap.fit(x_train, y_train, epochs=30)
    y_pred_probs = model_handicap.predict(x_test)
    melhor_z_positivo = encontrar_melhor_z_softmax_positivo(y_test, y_pred_probs)

    model_handicap.save("model_handicap.keras")  # Salva em formato nativo do Keras

    return melhor_z_positivo
    '''


#junta goal_lines
def preparar_df_goallines(df):
    # Seleciona e renomeia as colunas relacionadas à goal line 1
    df1 = df[['home','away','h2h_mean', 'media_goals_home', 'media_goals_away',
              'goal_line1_1', 'goal_line1_2', 'type_gl1','odds_gl1',
              'gl1_indefinido', 'gl1_negativo', 'gl1_positivo', 'gl1_reembolso','league']].copy()
    df1.columns = ['home','away','h2h_mean', 'media_goals_home', 'media_goals_away',
                   'goal_line_1', 'goal_line_2', 'type_gl','odds_gl',
                   'indefinido', 'negativo', 'positivo', 'reembolso','league']

    # Seleciona e renomeia as colunas relacionadas à goal line 2
    df2 = df[['home','away','h2h_mean', 'media_goals_home', 'media_goals_away',
              'goal_line2_1', 'goal_line2_2', 'type_gl2','odds_gl2',
              'gl2_indefinido', 'gl2_negativo', 'gl2_positivo', 'gl2_reembolso','league']].copy()
    df2.columns = ['home','away','h2h_mean', 'media_goals_home', 'media_goals_away',
                   'goal_line_1', 'goal_line_2', 'type_gl','odds_gl',
                   'indefinido', 'negativo', 'positivo', 'reembolso','league']

    # Concatena os dois dataframes
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final

def prepNNGoal_line(df=df_temp):
    df_temporario = df[['home','away','h2h_mean' ,'media_goals_home' ,'media_goals_away','goal_line1_1','goal_line1_2','type_gl1','odds_gl1', 'odds_gl2', 'goal_line2_1','goal_line2_2','type_gl2', 'gl1_indefinido','gl1_negativo', 'gl1_positivo', 'gl1_reembolso', 'gl2_indefinido', 'gl2_negativo', 'gl2_positivo', 'gl2_reembolso','league']].copy()

    df_temporario = preparar_df_goallines(df_temporario)
    
    
    
    df_temporario = df_temporario[df_temporario['indefinido'] == False]
    df_temporario.dropna(inplace=True)
    z = df_temporario[['home','away','goal_line_1', 'goal_line_2','type_gl', 'odds_gl','league']].copy()
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    df_temporario = pd.get_dummies(df_temporario, columns=['type_gl'], prefix='type_gl')
    X = df_temporario[['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2','league']].copy()
    X = normalizacao(X)
    X = pd.DataFrame(X, columns=['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2','league']).reset_index(drop=True)
    type_df = df_temporario[['type_gl_1.0', 'type_gl_2.0']]
    type_df = type_df.reset_index(drop=True)
    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z

#junta goal_lines
def preparar_df_goallines_X(df):
    # Seleciona e renomeia as colunas relacionadas à goal line 1
    df1 = df[['home','away','times','h2h_mean', 'media_goals_home', 'media_goals_away',
              'goal_line1_1', 'goal_line1_2', 'type_gl1','odds_gl1','league']].copy()
    df1.columns = ['home','away','times','h2h_mean', 'media_goals_home', 'media_goals_away',
                   'goal_line_1', 'goal_line_2', 'type_gl','odds_gl','league']

    # Seleciona e renomeia as colunas relacionadas à goal line 2
    df2 = df[['home','away','times','h2h_mean', 'media_goals_home', 'media_goals_away',
              'goal_line2_1', 'goal_line2_2', 'type_gl2','odds_gl2','league']].copy()
    df2.columns = ['home','away','times','h2h_mean', 'media_goals_home', 'media_goals_away',
                   'goal_line_1', 'goal_line_2', 'type_gl','odds_gl','league']

    # Concatena os dois dataframes
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final
'''
def prepNNGoal_line_X(df=df_temp):
    df_temporario = df[['home','away','h2h_mean' ,'media_goals_home' ,'media_goals_away','goal_line1_1','goal_line1_2','type_gl1','odds_gl1', 'odds_gl2', 'goal_line2_1','goal_line2_2','type_gl2']].copy()

    df_temporario = preparar_df_goallines_X(df_temporario)
    print(df_temporario.columns[df_temporario.isnull().any()])

    df_temporario.to_csv('prep_goalline.csv')

    df_temporario.dropna(inplace=True)
    z = df_temporario[['home','away','goal_line_1', 'goal_line_2','type_gl', 'odds_gl']].copy()
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    df_temporario = pd.get_dummies(df_temporario, columns=['type_gl'], prefix='type_gl')
    X = df_temporario[['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2']].copy()

    try:

        X = normalizacao(X)
        X = pd.DataFrame(X, columns=['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2']).reset_index(drop=True)
    except:
        print('faltou dados goal_line')
    type_df = df_temporario[['type_gl_1', 'type_gl_2']]
    type_df = type_df.reset_index(drop=True)
    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z
'''
#NN goal_line
def NN_goal_line(df=df_temp):
    
    df_temporario = df[['home','away','h2h_mean' ,'media_goals_home' ,'media_goals_away','goal_line1_1','goal_line1_2','type_gl1', 'odds_gl1','odds_gl2','goal_line2_1','goal_line2_2','type_gl2', 'gl1_indefinido','gl1_negativo', 'gl1_positivo', 'gl1_reembolso', 'gl2_indefinido', 'gl2_negativo', 'gl2_positivo', 'gl2_reembolso','league']].copy()
    df_temporario = preparar_df_goallines(df_temporario)
    
    df_temporario = pd.get_dummies(df_temporario, columns=['type_gl'], prefix='type_gl')

    df_temporario = df_temporario[df_temporario['indefinido'] == False]
    df_temporario.dropna(inplace=True)

    X = df_temporario[['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2','league']].copy()
    scaler_goal_line = StandardScaler()
    X_standardized = scaler_goal_line.fit_transform(X)
    with open('scaler_goal_line.pkl', 'wb') as f:
        pickle.dump(scaler_goal_line, f)
    X = pd.DataFrame(X_standardized, columns=['h2h_mean', 'media_goals_home', 'media_goals_away','odds_gl', 'goal_line_1', 'goal_line_2','league']).reset_index(drop=True)
    type_df = df_temporario[['type_gl_1.0', 'type_gl_2.0']]
    type_df = type_df.reset_index(drop=True)
    X_final = pd.concat([X, type_df], axis=1)
    
    
    
    # 1. Criando o y binário: positivo (1) ou não (0)
    y_binario = df_temporario['positivo'].astype(int)

    # 2. Reutilizando o X_final já preparado e normalizado
    x_train_bin, x_test_bin, y_train_bin, y_test_bin = split(X_final, y_binario)

        # 3. Criando o modelo binário
    modelo_binario_goal_line = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train_bin.shape[1],)),
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

    modelo_binario_goal_line.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Treinamento
    hist_bin = modelo_binario_goal_line.fit(x_train_bin, y_train_bin, epochs=30)
    y_pred_probs = modelo_binario_goal_line.predict(x_test_bin)
    melhor_z_positivo = encontrar_melhor_z_binario_positivo(y_test_bin, y_pred_probs )

    modelo_binario_goal_line.save("model_binario_goal_line.keras")  # Salva em formato nativo do Keras

    return melhor_z_positivo


'''

    y = df_temporario[['negativo', 'positivo', 'reembolso']].copy()

    x_train, x_test, y_train, y_test = split(X_final, y)

    model_goal_line = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model_goal_line.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_goal_line.fit(x_train, y_train, epochs=30)

    y_pred_probs = model_goal_line.predict(x_test)
    melhor_z_positivo = encontrar_melhor_z_softmax_positivo(y_test, y_pred_probs)

    model_goal_line.save("model_goal_line.keras")  # Salva em formato nativo do Keras

    return melhor_z_positivo
'''

#juntar double_chances
def preparar_df_double_chance(df):
    colunas_comuns = ['home','away','media_goals_home', 'media_goals_away', 'media_victories_home',
                      'media_victories_away', 'home_h2h_mean', 'away_h2h_mean','league']
    
    # Cria df para cada linha de double chance
    df1 = df[colunas_comuns + ['odds_dc1', 'res_double_chance1']].copy()
    df1['double_chance'] = 1
    df1.rename(columns={'odds_dc1': 'odds', 'res_double_chance1': 'resultado'}, inplace=True)

    df2 = df[colunas_comuns + ['odds_dc2', 'res_double_chance2']].copy()
    df2['double_chance'] = 2
    df2.rename(columns={'odds_dc2': 'odds', 'res_double_chance2': 'resultado'}, inplace=True)

    df3 = df[colunas_comuns + ['odds_dc3', 'res_double_chance3']].copy()
    df3['double_chance'] = 3
    df3.rename(columns={'odds_dc3': 'odds', 'res_double_chance3': 'resultado'}, inplace=True)

    # Concatena os três em um só
    df_final = pd.concat([df1, df2, df3], ignore_index=True)

    return df_final

def prepNNDouble_chance(df=df_temp):
    df_temporario =df[['home','away','media_goals_home',
        'media_goals_away','media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'double_chance1',
       'odds_dc1', 'double_chance2', 'odds_dc2', 'double_chance3', 'odds_dc3',
       'res_double_chance1', 'res_double_chance2', 'res_double_chance3','league']]
    df_temporario = preparar_df_double_chance(df_temporario)
    

    df_temporario.dropna(inplace=True)
    z = df_temporario[['home','away','double_chance', 'odds','league']].copy()
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    df_temporario = pd.get_dummies(df_temporario, columns=['double_chance'], prefix='double_chance_type')

    X = df_temporario[['media_goals_home', 'media_goals_away', 'media_victories_home',
                      'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']].copy()
    X = normalizacao(X)
    X = pd.DataFrame(X, columns=['media_goals_home', 'media_goals_away', 'media_victories_home',
                             'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']).reset_index(drop=True)
    type_df = df_temporario[['double_chance_type_1', 'double_chance_type_2','double_chance_type_3' ]]
    type_df = type_df.reset_index(drop=True)
    X_final = pd.concat([X, type_df], axis=1)

    return X_final, z

def preparar_df_double_chance_X(df):
    colunas_comuns = ['home','away','times','media_goals_home', 'media_goals_away', 'media_victories_home',
                      'media_victories_away', 'home_h2h_mean', 'away_h2h_mean','league']
    
    # Cria df para cada linha de double chance
    df1 = df[colunas_comuns + ['odds_dc1']].copy()
    df1['double_chance'] = 1
    df1.rename(columns={'odds_dc1': 'odds'}, inplace=True)

    df2 = df[colunas_comuns + ['odds_dc2']].copy()
    df2['double_chance'] = 2
    df2.rename(columns={'odds_dc2': 'odds'}, inplace=True)

    df3 = df[colunas_comuns + ['odds_dc3']].copy()
    df3['double_chance'] = 3
    df3.rename(columns={'odds_dc3': 'odds'}, inplace=True)

    # Concatena os três em um só
    df_final = pd.concat([df1, df2, df3], ignore_index=True)

    return df_final
'''
def prepNNDouble_chance_X(df=df_temp):
    df_temporario =df[['home','away','media_goals_home',
        'media_goals_away','media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'double_chance1',
       'odds_dc1', 'double_chance2', 'odds_dc2', 'double_chance3', 'odds_dc3']]
    df_temporario = preparar_df_double_chance_X(df_temporario)
    

    df_temporario.dropna(inplace=True)
    z = df_temporario[['home','away','double_chance', 'odds']].copy()
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    df_temporario = pd.get_dummies(df_temporario, columns=['double_chance'], prefix='double_chance_type')

    X = df_temporario[['media_goals_home', 'media_goals_away', 'media_victories_home',
                      'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds']].copy()
   
    try:
        X = normalizacao(X)
        X = pd.DataFrame(X, columns=['media_goals_home', 'media_goals_away', 'media_victories_home',
                                'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds']).reset_index(drop=True)
    except:
        print('faltou dados double_chance', )
    type_df = df_temporario[['double_chance_type_1', 'double_chance_type_2','double_chance_type_3' ]]
    type_df = type_df.reset_index(drop=True)
    
    
    X_final = pd.concat([X, type_df], axis=1)

    return X_final, z
'''
#NN double_chance
def NN_double_chance(df=df_temp):
    df_temporario =df[['home','away','media_goals_home',
        'media_goals_away','media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'double_chance1',
       'odds_dc1', 'double_chance2', 'odds_dc2', 'double_chance3', 'odds_dc3',
       'res_double_chance1', 'res_double_chance2', 'res_double_chance3','league']]
    df_temporario = preparar_df_double_chance(df_temporario)
    df_temporario = pd.get_dummies(df_temporario, columns=['double_chance'], prefix='double_chance_type')

    df_temporario.dropna(inplace=True)

    X = df_temporario[['media_goals_home', 'media_goals_away', 'media_victories_home',
                      'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']].copy()
    scaler_double_chance = StandardScaler()
    X_standardized = scaler_double_chance.fit_transform(X)
    with open('scaler_double_chance.pkl', 'wb') as f:
        pickle.dump(scaler_double_chance, f)
    X = pd.DataFrame(X_standardized, columns=['media_goals_home', 'media_goals_away', 'media_victories_home',
                             'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']).reset_index(drop=True)
    type_df = df_temporario[['double_chance_type_1', 'double_chance_type_2','double_chance_type_3' ]]
    type_df = type_df.reset_index(drop=True)
    
    
    X_final = pd.concat([X, type_df], axis=1)

    y = df_temporario['resultado'].copy()

    x_train, x_test, y_train, y_test = split(X_final, y)
    
    model_double_chance = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_double_chance.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    hist_bin = model_double_chance.fit(x_train, y_train, epochs=30)

    y_pred_probs = model_double_chance.predict(x_test)
    melhor_z_positivo = encontrar_melhor_z_binario_positivo(y_test, y_pred_probs)

    model_double_chance.save("model_binario_double_chance.keras")  # Salva em formato nativo do Keras

    return melhor_z_positivo

#junta draw_no_bet
def preparar_df_draw_no_bet(df):
    # Seleciona e renomeia o lado 1
    df1 = df[['home','away','home_goals', 'away_goals', 'media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team1', 'odds_dnb1', 'dnb1_indefinido', 'dnb1_perde', 'dnb1_ganha', 'dnb1_reembolso','league']].copy()

    df1.columns = ['home','away','home_goals', 'away_goals', 'media_goals_home', 'media_goals_away',
                   'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
                   'draw_no_bet_team', 'odds', 'indefinido', 'perde', 'ganha', 'reembolso','league']

    # Seleciona e renomeia o lado 2
    df2 = df[['home','away','home_goals', 'away_goals', 'media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team2', 'odds_dnb2', 'dnb2_indefinido', 'dnb2_perde', 'dnb2_ganha', 'dnb2_reembolso','league']].copy()

    df2.columns = ['home','away','home_goals', 'away_goals', 'media_goals_home', 'media_goals_away',
                   'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
                   'draw_no_bet_team', 'odds', 'indefinido', 'perde', 'ganha', 'reembolso','league']

    # Concatena os dois lados
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final

def prepNNDraw_no_bet(df=df_temp):
    df_temporario = df[['home','away','home_goals', 'away_goals','media_goals_home', 
       'media_goals_away', 'media_victories_home','media_victories_away', 'home_h2h_mean','away_h2h_mean', 'draw_no_bet_team1', 'odds_dnb1', 'draw_no_bet_team2', 'odds_dnb2', 'dnb1_indefinido' , 'dnb1_perde','dnb1_ganha', 'dnb1_reembolso',
       'dnb2_indefinido', 'dnb2_perde', 'dnb2_ganha', 'dnb2_reembolso','league']]
    df_temporario = preparar_df_draw_no_bet(df_temporario)


    df_temporario = df_temporario[df_temporario['indefinido'] == False]

    df_temporario.dropna(inplace=True)
    z = df_temporario[['home','away','draw_no_bet_team','odds','league']].copy()
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    df_temporario = pd.get_dummies(df_temporario, columns=['draw_no_bet_team'], prefix='draw_no_bet_team')
    

    X = df_temporario[['home_goals', 'away_goals', 'media_goals_home', 'media_goals_away','media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']].copy()
    
    X = normalizacao(X)
    X = pd.DataFrame(X, columns=['home_goals', 'away_goals', 'media_goals_home', 'media_goals_away',
                             'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']).reset_index(drop=True)

    type_df = df_temporario[['draw_no_bet_team_1.0', 'draw_no_bet_team_2.0']]
    type_df = type_df.reset_index(drop=True)

    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z


def preparar_df_draw_no_bet_X(df):
    # Seleciona e renomeia o lado 1
    df1 = df[['home','away','times', 'media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team1', 'odds_dnb1','league']].copy()

    df1.columns = ['home','away','times', 'media_goals_home', 'media_goals_away',
                   'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
                   'draw_no_bet_team', 'odds','league']

    # Seleciona e renomeia o lado 2
    df2 = df[['home','away', 'times','media_goals_home', 'media_goals_away',
              'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
              'draw_no_bet_team2', 'odds_dnb2','league']].copy()

    df2.columns = ['home','away','times', 'media_goals_home', 'media_goals_away',
                   'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean',
                   'draw_no_bet_team', 'odds','league']

    # Concatena os dois lados
    df_final = pd.concat([df1, df2], ignore_index=True)

    return df_final
'''
def prepNNDraw_no_bet_X(df=df_temp):
    df_temporario = df[['home','away', 'media_goals_home', 
       'media_goals_away', 'media_victories_home','media_victories_away', 'home_h2h_mean','away_h2h_mean', 'draw_no_bet_team1', 'odds_dnb1', 'draw_no_bet_team2', 'odds_dnb2', 'dnb1_indefinido' , 'dnb1_perde','dnb1_ganha', 'dnb1_reembolso',
       'dnb2_indefinido', 'dnb2_perde', 'dnb2_ganha', 'dnb2_reembolso']]
    df_temporario = preparar_df_draw_no_bet_X(df_temporario)



    df_temporario.dropna(inplace=True)
    z = df_temporario[['home','away','draw_no_bet_team','odds']].copy()
    if len(z) == 1:
        z = z.iloc[[0]].copy()
    df_temporario = pd.get_dummies(df_temporario, columns=['draw_no_bet_team'], prefix='draw_no_bet_team')
    

    X = df_temporario[['media_goals_home', 'media_goals_away','media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds']].copy()

    try:
        X = normalizacao(X)
        X = pd.DataFrame(X, columns=['media_goals_home', 'media_goals_away',
                                'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds']).reset_index(drop=True)
    except:
        print('faltou dados dnb')
    type_df = df_temporario[['draw_no_bet_team_1.0', 'draw_no_bet_team_2.0']]
    type_df = type_df.reset_index(drop=True)

    X_final = pd.concat([X, type_df], axis=1)
    return X_final, z
'''
#NN draw_no_bet
def NN_draw_no_bet(df=df_temp):

    df_temporario = df[['home','away','home_goals', 'away_goals','media_goals_home', 
       'media_goals_away', 'media_victories_home','media_victories_away', 'home_h2h_mean','away_h2h_mean', 'draw_no_bet_team1', 'odds_dnb1', 'draw_no_bet_team2', 'odds_dnb2', 'dnb1_indefinido' , 'dnb1_perde','dnb1_ganha', 'dnb1_reembolso',
       'dnb2_indefinido', 'dnb2_perde', 'dnb2_ganha', 'dnb2_reembolso','league']]
    df_temporario = preparar_df_draw_no_bet(df_temporario)

    df_temporario = pd.get_dummies(df_temporario, columns=['draw_no_bet_team'], prefix='draw_no_bet_team')
    df_temporario = df_temporario[df_temporario['indefinido'] == False]

    df_temporario.dropna(inplace=True)
    

    X = df_temporario[['media_goals_home', 'media_goals_away','media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']].copy()
    
    scaler_draw_no_bet = StandardScaler()
    X_standardized = scaler_draw_no_bet.fit_transform(X)
    with open('scaler_draw_no_bet.pkl', 'wb') as f:
        pickle.dump(scaler_draw_no_bet, f)
    X = pd.DataFrame(X_standardized, columns=['media_goals_home', 'media_goals_away',
                             'media_victories_home', 'media_victories_away', 'home_h2h_mean', 'away_h2h_mean', 'odds','league']).reset_index(drop=True)

    type_df = df_temporario[['draw_no_bet_team_1.0', 'draw_no_bet_team_2.0']]
    type_df = type_df.reset_index(drop=True)

    X_final = pd.concat([X, type_df], axis=1)
    #   1. Criando o y binário: positivo (1) ou não (0)
    y_binario = df_temporario['ganha'].astype(int)

    # 2. Reutilizando o X_final já preparado e normalizado
    x_train_bin, x_test_bin, y_train_bin, y_test_bin = split(X_final, y_binario)
            # 3. Criando o modelo binário

    modelo_binario_draw_no_bet = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train_bin.shape[1],)),
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

    modelo_binario_draw_no_bet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Treinamento
    hist_bin = modelo_binario_draw_no_bet.fit(x_train_bin, y_train_bin, epochs=30)
    y_pred_probs = modelo_binario_draw_no_bet.predict(x_test_bin)
    melhor_z_positivo = encontrar_melhor_z_binario_positivo(y_test_bin, y_pred_probs)

    modelo_binario_draw_no_bet.save("model_binario_draw_no_bet.keras")  # Salva em formato nativo do Keras

    return melhor_z_positivo
    '''
    y = df_temporario[['perde', 'ganha', 'reembolso']].copy()

    x_train, x_test, y_train, y_test = split(X_final, y)
    
    model_draw_no_bet = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model_draw_no_bet.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model_draw_no_bet.fit(x_train,y_train, epochs=30)

    y_pred_probs = model_draw_no_bet.predict(x_test)
    melhor_z_positivo = encontrar_melhor_z_softmax_positivo(y_test, y_pred_probs)

    model_draw_no_bet.save("model_draw_no_bet.keras")  # Salva em formato nativo do Keras

    return melhor_z_positivo
    '''
