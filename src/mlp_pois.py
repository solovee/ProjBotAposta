import numpy as np
import tensorflow as tf
import pandas as pd
from main import logger
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import NN
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def contar_registros_por_liga(caminho_csv, ligas_interesse):
    """
    Lê um arquivo CSV e conta a quantidade de registros para ligas específicas.

    Args:
        caminho_csv (str): O caminho para o arquivo CSV.
        ligas_interesse (list): Uma lista de números de ligas para as quais contar os registros.

    Returns:
        dict: Um dicionário onde as chaves são os números das ligas e os valores são as contagens de registros.
              Retorna None se o arquivo não for encontrado ou se a coluna 'league' não existir.
    """
    try:
        # 1. Carregar o CSV para um DataFrame
        df = pd.read_csv(caminho_csv, encoding='utf-8')
        print(f"Arquivo '{caminho_csv}' carregado com sucesso. Total de {len(df)} registros.")

        # 2. Verificar se a coluna 'league' existe
        if 'league' not in df.columns:
            print("Erro: Coluna 'league' não encontrada no DataFrame.")
            return None

        contagem_por_liga = {}
        for liga in ligas_interesse:
            # 3. Filtrar o DataFrame para a liga específica e contar os registros
            quantidade_registros = len(df[df['league'] == liga])
            
            contagem_por_liga[liga] = quantidade_registros
            print(f"Quantidade de registros para a liga {liga}: {quantidade_registros}")

        return contagem_por_liga

    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_csv}' não foi encontrado.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Erro: O arquivo '{caminho_csv}' está vazio.")
        return None
    except pd.errors.ParserError as e:
        print(f"Erro de parsing no CSV: {e}. Verifique o formato do CSV e o encoding.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None



import pandas as pd

def calcular_media_gols_por_liga(caminho_csv, ligas_interesse):
    """
    Lê um arquivo CSV, calcula a média da coluna 'tot_goals' para ligas específicas.

    Args:
        caminho_csv (str): O caminho para o arquivo CSV.
        ligas_interesse (list): Uma lista de números de ligas para as quais calcular a média.

    Returns:
        dict: Um dicionário onde as chaves são os números das ligas e os valores são as médias de 'tot_goals'.
              Retorna None se o arquivo não for encontrado ou se 'tot_goals' ou 'league' não existirem.
    """
    try:
        # 1. Carregar o CSV para um DataFrame
        df = pd.read_csv(caminho_csv, encoding='utf-8')
        print(f"Arquivo '{caminho_csv}' carregado com sucesso. Total de {len(df)} registros.")

        # 2. Verificar se as colunas necessárias existem
        if 'tot_goals' not in df.columns:
            print("Erro: Coluna 'tot_goals' não encontrada no DataFrame.")
            return None
        if 'league' not in df.columns:
            print("Erro: Coluna 'league' não encontrada no DataFrame.")
            return None

        medias_por_liga = {}
        for liga in ligas_interesse:
            # 3. Filtrar o DataFrame para a liga específica
            df_liga = df[df['league'] == liga]

            if not df_liga.empty:
                # 4. Calcular a média de 'tot_goals' para a liga filtrada
                media_gols = df_liga['tot_goals'].mean()
                medias_por_liga[liga] = media_gols / 2
                print(f"Média de 'tot_goals' para a liga {liga}: {media_gols:.2f}")
            else:
                print(f"Não foram encontrados registros para a liga {liga}.")
                medias_por_liga[liga] = None # Ou pode definir como 0, dependendo da sua necessidade

        return medias_por_liga

    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_csv}' não foi encontrado.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Erro: O arquivo '{caminho_csv}' está vazio.")
        return None
    except pd.errors.ParserError as e:
        print(f"Erro de parsing no CSV: {e}. Verifique o formato do CSV e o encoding.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None



import pandas as pd
import numpy as np


def preprocessor(df):
    """Prepara o DataFrame para o modelo, removendo colunas desnecessárias e convertendo tipos."""
    df_filtered = df.dropna()
    with open('src/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    # Aplicando transformação
    X_scaled = pd.DataFrame(loaded_scaler.transform(df_filtered), columns=df_filtered.columns)
    X_knn = adicionar_knn_pred_via_modelo(X_scaled)
    with open('src/scaler_2.pkl', 'rb') as f:
        loaded_scaler_2 = pickle.load(f)
    X_scaled_knn = loaded_scaler_2.transform(X_knn)
    X_scaled_knn = pd.DataFrame(X_scaled_knn, columns=X_knn.columns)

    return X_scaled_knn

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

def adicionar_knn_pred(X: pd.DataFrame, y, n_splits=5, n_neighbors=5, random_state=42):
    """
    Adiciona as previsões out-of-fold de um KNN como uma nova coluna ao DataFrame X.
    
    Parâmetros:
        X (pd.DataFrame): DataFrame com os dados de entrada. Deve conter a coluna 'id'.
        y (pd.Series ou array): Valores alvo (target).
        n_splits (int): Número de splits do KFold.
        n_neighbors (int): Número de vizinhos no KNN.
        random_state (int): Semente para garantir reproducibilidade.

    Retorna:
        pd.DataFrame: DataFrame X com a nova coluna 'knn_pred'.
    """
    # Garantir que X e y sejam DataFrames/Séries com índices alinhados
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    # Remover coluna 'id' para o treino
    X_treino_sem_id = X.drop('id', axis=1)

    # Vetor para armazenar previsões
    knn_preds = np.zeros(len(X))

    # KFold para gerar previsões fora do fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(X):
        X_train_fold = X_treino_sem_id.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]

        X_val_fold = X_treino_sem_id.iloc[val_idx]

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_fold, y_train_fold)

        knn_preds[val_idx] = knn.predict(X_val_fold)

    # Adiciona a coluna de predições
    X['knn_pred'] = knn_preds

    return X

import pickle
import pandas as pd

def adicionar_knn_pred_via_modelo(df: pd.DataFrame, caminho_modelo='src/modelo_knn_final.pkl'):
    """
    Carrega um modelo KNN salvo e adiciona as previsões como uma nova coluna 'knn_pred' no DataFrame.

    Parâmetros:
        df (pd.DataFrame): DataFrame original com a coluna 'id'.
        caminho_modelo (str): Caminho para o arquivo .pkl do modelo KNN.

    Retorna:
        pd.DataFrame: O DataFrame original com a coluna 'knn_pred' adicionada.
    """
    # Carrega o modelo
    with open(caminho_modelo, 'rb') as f:
        knn_model = pickle.load(f)

    # Cópia para não modificar o original diretamente
    df_copy = df.copy()

    # Remove colunas que não são usadas no modelo

    colunas_usaveis = [col for col in df_copy.columns]
    
    # Extrai os dados para predição
    X_para_prever = df_copy[colunas_usaveis]

    # Faz a predição
    predicoes = knn_model.predict(X_para_prever)

    # Adiciona a nova coluna
    df_copy['knn_pred'] = predicoes
    

    return df_copy




def individualiza_jogo_duplo_com_resultado(df, media_8, media_12): # Adicionado resultados_medias como argumento
    '''regressao'''

    colunas_comuns_base = [
        'id','league',
        'media_goals_home', 'media_goals_away',
        'media_victories_home', 'media_victories_away',
        'home_h2h_mean', 'away_h2h_mean',
        'media_goals_sofridos_home', 'media_goals_sofridos_away',
        'home_h2h_win_rate', 'away_h2h_win_rate', 'h2h_total_games','home_goals','away_goals'
    ]

    # Garante que o DataFrame de entrada tenha as colunas necessárias
    for col in colunas_comuns_base:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame de entrada.")
            
    


    novas_linhas = []

    for index, row in df.iterrows():
        # Define a média da liga com base no ID da liga
        if int(row['league']) == 8:
            media_league = media_8
        else: # Assumindo que 12 é a outra liga principal, ajuste se houver mais
            media_league = media_12

        # --- Linha para o Time da Casa (HOME) ---
        linha_home = {}
        
        
        # Resultados do jogo para a perspectiva do time da casa
      
        linha_home['resultado_vitoria'] = row['home_goals']# >= row['away_goals'] # Vitória do HOME

        linha_home['home'] = 1
      

        # Features calculadas para o time da casa
        linha_home['fa'] = row['media_goals_home'] / media_league # Força de ataque do time
        linha_home['fd'] = row['media_goals_sofridos_home'] / media_league # Força de defesa do time
        
        # Features do oponente na perspectiva do time da casa
        linha_home['opponent_fa'] = row['media_goals_away'] / media_league
        linha_home['opponent_fd'] = row['media_goals_sofridos_away'] / media_league

        # Diferenças de forças
        linha_home['fa_diff'] = linha_home['fa'] - linha_home['opponent_fa']
        linha_home['fd_diff'] = linha_home['fd'] - linha_home['opponent_fd']
        linha_home['fa/fd_diff'] = linha_home['fa'] - linha_home['opponent_fd']

        # Adiciona as colunas gerais ajustadas para a perspectiva 'home'
        linha_home['league'] = row['league']
        linha_home['media_goals'] = row['media_goals_home']
        linha_home['media_goals_sofridos'] = row['media_goals_sofridos_home']
        linha_home['media_victories'] = row['media_victories_home']
        linha_home['h2h_mean'] = row['home_h2h_mean']
        linha_home['h2h_win_rate'] = row['home_h2h_win_rate']
        linha_home['h2h_total_games'] = row['h2h_total_games']
        
        # Gols esperados
        linha_home['expected_goals'] = linha_home['fa'] * linha_home['opponent_fd'] * media_league
        linha_home['opponent_expected_goals'] = linha_home['opponent_fa'] * linha_home['fd'] * media_league
        linha_home['expected_goals_diff'] = linha_home['expected_goals'] - linha_home['opponent_expected_goals']
      
        
        # Métricas do adversário (away) para a linha 'home'
        linha_home['opponent_media_goals'] = row['media_goals_away']
        linha_home['opponent_media_goals_sofridos'] = row['media_goals_sofridos_away']
        linha_home['opponent_media_victories'] = row['media_victories_away']
        linha_home['opponent_h2h_mean'] = row['away_h2h_mean']
        linha_home['opponent_h2h_win_rate'] = row['away_h2h_win_rate']

        linha_home['media_goals_diff'] = linha_home['media_goals'] - linha_home['opponent_media_goals']
        linha_home['media_goals_sofridos_diff'] = linha_home['media_goals_sofridos'] - linha_home['opponent_media_goals_sofridos']
        linha_home['media_victories_diff'] = linha_home['media_victories'] - linha_home['opponent_media_victories']
        linha_home['h2h_mean_diff'] = linha_home['h2h_mean'] - linha_home['opponent_h2h_mean']
        linha_home['h2h_win_rate_diff'] = linha_home['h2h_win_rate'] - linha_home['opponent_h2h_win_rate']


        novas_linhas.append(linha_home)

        # --- Linha para o Time Visitante (AWAY) ---
        linha_away = {}
        
    
        
        # Resultados do jogo para a perspectiva do time visitante
        # O resultado é invertido em relação ao time da casa
        linha_away['resultado_vitoria'] = row['away_goals']# >= row['home_goals'] # Vitória do AWAY
        linha_away['home'] = 0
        
    

        # Features calculadas para o time visitante
        linha_away['fa'] = row['media_goals_away'] / media_league
        linha_away['fd'] = row['media_goals_sofridos_away'] / media_league
        
        # Features do oponente na perspectiva do time visitante
        linha_away['opponent_fa'] = row['media_goals_home'] / media_league
        linha_away['opponent_fd'] = row['media_goals_sofridos_home'] / media_league
        
        # Diferenças de forças
        linha_away['fa_diff'] = linha_away['fa'] - linha_away['opponent_fa']

        linha_away['fd_diff'] = linha_away['fd'] - linha_away['opponent_fd']
        linha_away['fa/fd_diff'] = linha_away['fa'] - linha_away['opponent_fd']

        # Adiciona as colunas gerais ajustadas para a perspectiva 'away'
        linha_away['league'] = row['league']
        linha_away['media_goals'] = row['media_goals_away']
        linha_away['media_goals_sofridos'] = row['media_goals_sofridos_away']
        linha_away['media_victories'] = row['media_victories_away']
        linha_away['h2h_mean'] = row['away_h2h_mean']
        linha_away['h2h_win_rate'] = row['away_h2h_win_rate']
        linha_away['h2h_total_games'] = row['h2h_total_games']
        
        # Gols esperados
        linha_away['expected_goals'] = linha_away['fa'] * linha_away['opponent_fd'] * media_league
        linha_away['opponent_expected_goals'] = linha_away['opponent_fa'] * linha_away['fd'] * media_league
        linha_away['expected_goals_diff'] = linha_away['expected_goals'] - linha_away['opponent_expected_goals']
      

        # Métricas do adversário (home) para a linha 'away'
        linha_away['opponent_media_goals'] = row['media_goals_home']
        linha_away['opponent_media_goals_sofridos'] = row['media_goals_sofridos_home']
        linha_away['opponent_media_victories'] = row['media_victories_home']
        linha_away['opponent_h2h_mean'] = row['home_h2h_mean']
        linha_away['opponent_h2h_win_rate'] = row['home_h2h_win_rate']

        linha_away['media_goals_diff'] = linha_away['media_goals'] - linha_away['opponent_media_goals']
        linha_away['media_goals_sofridos_diff'] = linha_away['media_goals_sofridos'] - linha_away['opponent_media_goals_sofridos']
        linha_away['media_victories_diff'] = linha_away['media_victories'] - linha_away['opponent_media_victories']
        linha_away['h2h_mean_diff'] = linha_away['h2h_mean'] - linha_away['opponent_h2h_mean']
        linha_away['h2h_win_rate_diff'] = linha_away['h2h_win_rate'] - linha_away['opponent_h2h_win_rate']
        
        novas_linhas.append(linha_away)

    df_individualizado = pd.DataFrame(novas_linhas)

    return df_individualizado

def individualiza_jogo_duplo(df, media_8, media_12): # Adicionado resultados_medias como argumento
    '''regressao'''

    colunas_comuns_base = [
        'id','league',
        'media_goals_home', 'media_goals_away',
        'media_victories_home', 'media_victories_away',
        'home_h2h_mean', 'away_h2h_mean',
        'media_goals_sofridos_home', 'media_goals_sofridos_away',
        'home_h2h_win_rate', 'away_h2h_win_rate', 'h2h_total_games'
    ]

    # Garante que o DataFrame de entrada tenha as colunas necessárias
    for col in colunas_comuns_base:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame de entrada.")
    for col in colunas_comuns_base:
        if df[col].isnull().any():
            logger.warning(f"⚠️ Coluna '{col}' contém valores NA/None no DataFrame de entrada")


    novas_linhas = []

    for index, row in df.iterrows():
        # Define a média da liga com base no ID da liga
        if int(row['league']) == 8:
            media_league = media_8
        else: # Assumindo que 12 é a outra liga principal, ajuste se houver mais
            media_league = media_12

        # --- Linha para o Time da Casa (HOME) ---
        linha_home = {}
        
        
        # Resultados do jogo para a perspectiva do time da casa
      

        linha_home['home'] = 1
      

        # Features calculadas para o time da casa
        linha_home['fa'] = row['media_goals_home'] / media_league # Força de ataque do time
        linha_home['fd'] = row['media_goals_sofridos_home'] / media_league # Força de defesa do time
        
        # Features do oponente na perspectiva do time da casa
        linha_home['opponent_fa'] = row['media_goals_away'] / media_league
        linha_home['opponent_fd'] = row['media_goals_sofridos_away'] / media_league

        # Diferenças de forças
        linha_home['fa_diff'] = linha_home['fa'] - linha_home['opponent_fa']
        linha_home['fd_diff'] = linha_home['fd'] - linha_home['opponent_fd']
        linha_home['fa/fd_diff'] = linha_home['fa'] - linha_home['opponent_fd']

        # Adiciona as colunas gerais ajustadas para a perspectiva 'home'
        linha_home['league'] = row['league']
        linha_home['media_goals'] = row['media_goals_home']
        linha_home['media_goals_sofridos'] = row['media_goals_sofridos_home']
        linha_home['media_victories'] = row['media_victories_home']
        linha_home['h2h_mean'] = row['home_h2h_mean']
        linha_home['h2h_win_rate'] = row['home_h2h_win_rate']
        linha_home['h2h_total_games'] = row['h2h_total_games']
        
        # Gols esperados
        linha_home['expected_goals'] = linha_home['fa'] * linha_home['opponent_fd'] * media_league
        linha_home['opponent_expected_goals'] = linha_home['opponent_fa'] * linha_home['fd'] * media_league
        linha_home['expected_goals_diff'] = linha_home['expected_goals'] - linha_home['opponent_expected_goals']
      
        
        # Métricas do adversário (away) para a linha 'home'
        linha_home['opponent_media_goals'] = row['media_goals_away']
        linha_home['opponent_media_goals_sofridos'] = row['media_goals_sofridos_away']
        linha_home['opponent_media_victories'] = row['media_victories_away']
        linha_home['opponent_h2h_mean'] = row['away_h2h_mean']
        linha_home['opponent_h2h_win_rate'] = row['away_h2h_win_rate']

        linha_home['media_goals_diff'] = linha_home['media_goals'] - linha_home['opponent_media_goals']
        linha_home['media_goals_sofridos_diff'] = linha_home['media_goals_sofridos'] - linha_home['opponent_media_goals_sofridos']
        linha_home['media_victories_diff'] = linha_home['media_victories'] - linha_home['opponent_media_victories']
        linha_home['h2h_mean_diff'] = linha_home['h2h_mean'] - linha_home['opponent_h2h_mean']
        linha_home['h2h_win_rate_diff'] = linha_home['h2h_win_rate'] - linha_home['opponent_h2h_win_rate']


        novas_linhas.append(linha_home)

        # --- Linha para o Time Visitante (AWAY) ---
        linha_away = {}
        
    
        
        # Resultados do jogo para a perspectiva do time visitante
        # O resultado é invertido em relação ao time da casa
       

        linha_away['home'] = 0
        
    

        # Features calculadas para o time visitante
        linha_away['fa'] = row['media_goals_away'] / media_league
        linha_away['fd'] = row['media_goals_sofridos_away'] / media_league
        
        # Features do oponente na perspectiva do time visitante
        linha_away['opponent_fa'] = row['media_goals_home'] / media_league
        linha_away['opponent_fd'] = row['media_goals_sofridos_home'] / media_league
        
        # Diferenças de forças
        linha_away['fa_diff'] = linha_away['fa'] - linha_away['opponent_fa']

        linha_away['fd_diff'] = linha_away['fd'] - linha_away['opponent_fd']
        linha_away['fa/fd_diff'] = linha_away['fa'] - linha_away['opponent_fd']

        # Adiciona as colunas gerais ajustadas para a perspectiva 'away'
        linha_away['league'] = row['league']
        linha_away['media_goals'] = row['media_goals_away']
        linha_away['media_goals_sofridos'] = row['media_goals_sofridos_away']
        linha_away['media_victories'] = row['media_victories_away']
        linha_away['h2h_mean'] = row['away_h2h_mean']
        linha_away['h2h_win_rate'] = row['away_h2h_win_rate']
        linha_away['h2h_total_games'] = row['h2h_total_games']
        
        # Gols esperados
        linha_away['expected_goals'] = linha_away['fa'] * linha_away['opponent_fd'] * media_league
        linha_away['opponent_expected_goals'] = linha_away['opponent_fa'] * linha_away['fd'] * media_league
        linha_away['expected_goals_diff'] = linha_away['expected_goals'] - linha_away['opponent_expected_goals']
      

        # Métricas do adversário (home) para a linha 'away'
        linha_away['opponent_media_goals'] = row['media_goals_home']
        linha_away['opponent_media_goals_sofridos'] = row['media_goals_sofridos_home']
        linha_away['opponent_media_victories'] = row['media_victories_home']
        linha_away['opponent_h2h_mean'] = row['home_h2h_mean']
        linha_away['opponent_h2h_win_rate'] = row['home_h2h_win_rate']

        linha_away['media_goals_diff'] = linha_away['media_goals'] - linha_away['opponent_media_goals']
        linha_away['media_goals_sofridos_diff'] = linha_away['media_goals_sofridos'] - linha_away['opponent_media_goals_sofridos']
        linha_away['media_victories_diff'] = linha_away['media_victories'] - linha_away['opponent_media_victories']
        linha_away['h2h_mean_diff'] = linha_away['h2h_mean'] - linha_away['opponent_h2h_mean']
        linha_away['h2h_win_rate_diff'] = linha_away['h2h_win_rate'] - linha_away['opponent_h2h_win_rate']
        
        novas_linhas.append(linha_away)

    df_individualizado = pd.DataFrame(novas_linhas)

    return df_individualizado




import pandas as pd
from sklearn.preprocessing import MinMaxScaler




import numpy as np
from scipy.stats import poisson

def prever_placares_futebol_poisson(lambda_casa: float, lambda_visitante: float, max_gols: int = 7):
    """
    Calcula as probabilidades de placares exatos e resultados finais de uma partida de futebol
    usando o Modelo de Poisson.

    Args:
        lambda_casa (float): Gols esperados (lambda) para o time da casa.
        lambda_visitante (float): Gols esperados (lambda) para o time visitante.
        max_gols (int): O número máximo de gols para calcular as probabilidades
                        da matriz de placar (ex: 5 para placares até 5x5).

    Returns:
        tuple: Uma tupla contendo:
               - np.ndarray: Matriz de probabilidades de placar (linhas=gols casa, colunas=gols visitante).
               - dict: Dicionário com as probabilidades de vitória da casa, empate e vitória do visitante.
    """

    # Inicializa uma matriz de zeros para armazenar as probabilidades de placar
    # As dimensões são (max_gols + 1) x (max_gols + 1) para incluir o 0-0 até max_gols x max_gols
    prob_placar = np.zeros((max_gols + 1, max_gols + 1))

    # Preenche a matriz calculando a probabilidade de cada placar
    for gols_casa in range(max_gols + 1):
        for gols_visitante in range(max_gols + 1):
            # Calcula a probabilidade do Time da Casa marcar 'gols_casa' gols
            p_gols_casa = poisson.pmf(gols_casa, lambda_casa)
            # Calcula a probabilidade do Time Visitante marcar 'gols_visitante' gols
            p_gols_visitante = poisson.pmf(gols_visitante, lambda_visitante)

            # A probabilidade do placar específico é o produto das probabilidades individuais
            prob_placar[gols_casa, gols_visitante] = p_gols_casa * p_gols_visitante

    # Calcula as probabilidades de Vitória, Empate, Derrota a partir da matriz de placares
    prob_vitoria_casa = 0
    prob_empate = 0
    prob_vitoria_visitante = 0

    for gols_casa in range(max_gols + 1):
        for gols_visitante in range(max_gols + 1):
            prob = prob_placar[gols_casa, gols_visitante]
            
            if gols_casa > gols_visitante:
                prob_vitoria_casa += prob
            elif gols_casa == gols_visitante:
                prob_empate += prob
            else: # gols_casa < gols_visitante
                prob_vitoria_visitante += prob

    # Armazena as probabilidades de resultado final em um dicionário
    prob_resultados = {
        "vitoria_casa": prob_vitoria_casa,
        "empate": prob_empate,
        "vitoria_visitante": prob_vitoria_visitante
    }

    return prob_placar, prob_resultados



import numpy as np



import numpy as np
from scipy.stats import poisson



def calcular_over_under(matriz_probabilidades_placar: np.ndarray, goal_line1: float, goal_line2: float):
   
    # --- Cálculo para goal_line1 ---
    prob_over1 = 0
    prob_under1 = 0
    prob_push1 = 0
    is_integer_line1 = (goal_line1 == int(goal_line1))

    for gols_casa in range(matriz_probabilidades_placar.shape[0]):
        for gols_visitante in range(matriz_probabilidades_placar.shape[1]):
            total_gols = gols_casa + gols_visitante
            prob_placar_atual = matriz_probabilidades_placar[gols_casa, gols_visitante]

            if total_gols > goal_line1:
                prob_over1 += prob_placar_atual
            elif is_integer_line1 and total_gols == goal_line1:
                prob_push1 += prob_placar_atual
            else: # total_gols < goal_line1
                prob_under1 += prob_placar_atual

    # --- Cálculo para goal_line2 ---
    prob_over2 = 0
    prob_under2 = 0
    prob_push2 = 0
    is_integer_line2 = (goal_line2 == int(goal_line2))

    for gols_casa in range(matriz_probabilidades_placar.shape[0]):
        for gols_visitante in range(matriz_probabilidades_placar.shape[1]):
            total_gols = gols_casa + gols_visitante
            prob_placar_atual = matriz_probabilidades_placar[gols_casa, gols_visitante]

            if total_gols > goal_line2:
                prob_over2 += prob_placar_atual
            elif is_integer_line2 and total_gols == goal_line2:
                prob_push2 += prob_placar_atual
            else: # total_gols < goal_line2
                prob_under2 += prob_placar_atual

    # --- Cálculo da Média ---
    avg_prob_over = (prob_over1 + prob_over2) / 2
    avg_prob_under = (prob_under1 + prob_under2) / 2
    avg_prob_push = (prob_push1 + prob_push2) / 2

    return {
        "over": round(avg_prob_over, 4),
        "under": round(avg_prob_under, 4),
        "push": round(avg_prob_push, 4)
    }

def calcular_handicap(matriz_probabilidades_placar: np.ndarray,
                       handicap_casa1: float, handicap_casa2: float,
                       handicap_fora1: float, handicap_fora2: float):
 

    def _calcular_single_handicap_outcome(matriz, handicap_val, team_type):
        """
        Calcula as probabilidades de vitória para um único handicap.
        team_type: 'home' ou 'away'
        """
        prob_casa_vence = 0
        prob_fora_vence = 0
        prob_push = 0
        is_integer_line = (handicap_val == int(handicap_val))

        for gols_casa in range(matriz.shape[0]):
            for gols_visitante in range(matriz.shape[1]):
                prob_placar_atual = matriz[gols_casa, gols_visitante]

                if team_type == 'home':
                    gols_casa_ajustado = gols_casa + handicap_val
                    gols_fora_ajustado = gols_visitante
                else: # team_type == 'away'
                    gols_casa_ajustado = gols_casa
                    gols_fora_ajustado = gols_visitante + handicap_val

                if gols_casa_ajustado > gols_fora_ajustado:
                    prob_casa_vence += prob_placar_atual
                elif is_integer_line and gols_casa_ajustado == gols_fora_ajustado:
                    prob_push += prob_placar_atual
                else: # gols_casa_ajustado < gols_fora_ajustado
                    prob_fora_vence += prob_placar_atual
        return prob_casa_vence, prob_fora_vence, prob_push

    # --- Cálculos para cada uma das 4 linhas de handicap ---
    prob_casa_vence_h1, prob_fora_vence_h1, prob_push_h1 = _calcular_single_handicap_outcome(
        matriz_probabilidades_placar, handicap_casa1, 'home'
    )
    prob_casa_vence_h2, prob_fora_vence_h2, prob_push_h2 = _calcular_single_handicap_outcome(
        matriz_probabilidades_placar, handicap_casa2, 'home'
    )
    prob_casa_vence_a1, prob_fora_vence_a1, prob_push_a1 = _calcular_single_handicap_outcome(
        matriz_probabilidades_placar, handicap_fora1, 'away'
    )
    prob_casa_vence_a2, prob_fora_vence_a2, prob_push_a2 = _calcular_single_handicap_outcome(
        matriz_probabilidades_placar, handicap_fora2, 'away'
    )

    # --- Cálculo da Média de todas as 4 probabilidades ---
    avg_prob_casa_vence = (prob_casa_vence_h1 + prob_casa_vence_h2 + prob_casa_vence_a1 + prob_casa_vence_a2) / 4
    avg_prob_fora_vence = (prob_fora_vence_h1 + prob_fora_vence_h2 + prob_fora_vence_a1 + prob_fora_vence_a2) / 4
    avg_prob_push = (prob_push_h1 + prob_push_h2 + prob_push_a1 + prob_push_a2) / 4

    return {
        "vitoria_casa": round(avg_prob_casa_vence, 4),
        "vitoria_visitante": round(avg_prob_fora_vence, 4),
        "push": round(avg_prob_push, 4)
    }




def decide_aposta(probs_ou, probs_gl, probs_vic, probs_h, odds_ou, odds_gl, odds_dc, odds_dnb, odds_h):
    """
    Decide se deve apostar com base nas probabilidades e odds fornecidas.
    
    Args:
        probs_ou (dict): Probabilidades de Over/Under.
        probs_gl (dict): Probabilidades de gols exatos.
        probs_vic (dict): Probabilidades de vitória.
        probs_h (dict): Probabilidades de handicap.
        odds_ou (float): Odds para Over/Under.
        odds_gl (float): Odds para gols exatos.
        odds_vic (float): Odds para vitória.
        odds_h (float): Odds para handicap.

    Returns:
        str: Decisão sobre a aposta.
    """
    print(probs_ou, probs_gl, probs_vic, probs_h, odds_ou, odds_gl, odds_dc, odds_dnb, odds_h)
    decisions = {'ou': None, 'gl': None, 'dc': None,'dnb': None, 'h': None}
    if probs_ou is not None and odds_ou is not None:
        if float(probs_ou['over']) > float(probs_ou['under']) and float(odds_ou['over']) > 1.5 and float(probs_ou['over']) > 0.6:
            decisions['ou'] = 0
        elif float(probs_ou['under']) > float(probs_ou['over']) and float(odds_ou['under']) > 1.5 and float(probs_ou['under']) > 0.6:
            decisions['ou'] = 1
    if probs_gl is not None and odds_gl is not None:
        if float(probs_gl['over']) > float(probs_gl['under']) and float(odds_gl['over']) > 1.5 and (float(probs_gl['over']) + float(probs_gl['push'])) > 0.6:
            decisions['gl'] = 0
        elif float(probs_gl['under']) > float(probs_gl['over']) and float(odds_gl['under']) > 1.5 and (float(probs_gl['under']) + float(probs_gl['push'])) > 0.6:
            decisions['gl'] = 1
    if probs_vic is not None and odds_dc is not None:
        if float(odds_dc['ambos']) > 1.5 and float(probs_vic['empate']) < 0.6:
            decisions['dc'] = 2
        elif float(probs_vic['vitoria_casa']) > float(probs_vic['vitoria_visitante']) and float(odds_dc['vitoria_casa']) > 1.5 and (float(probs_vic['vitoria_casa']) + float(probs_vic['empate'])) > 0.6:
            decisions['dc'] = 0
        elif float(probs_vic['vitoria_visitante']) > float(probs_vic['vitoria_casa']) and float(odds_dc['vitoria_visitante']) > 1.5 and (float(probs_vic['vitoria_visitante']) + float(probs_vic['empate'])) > 0.6:
            decisions['dc'] = 1
    if probs_vic is not None and odds_dnb is not None:
        if float(probs_vic['vitoria_casa']) > float(probs_vic['vitoria_visitante']) and float(odds_dnb['vitoria_casa']) > 1.5 and (float(probs_vic['vitoria_casa']) + float(probs_vic['empate'])) > 0.6:
            decisions['dnb'] = 0
        elif float(probs_vic['vitoria_visitante']) > float(probs_vic['vitoria_casa']) and float(odds_dnb['vitoria_visitante']) > 1.5 and (float(probs_vic['vitoria_visitante']) + float(probs_vic['empate'])) > 0.6:
            decisions['dnb'] = 1
    if probs_h is not None and odds_h is not None:
        if float(probs_h['vitoria_casa']) > float(probs_h['vitoria_visitante']) and float(odds_h['vitoria_casa']) > 1.5 and (float(probs_h['vitoria_casa']) + float(probs_h['push'])) > 0.6:
            decisions['h'] = 0
        elif float(probs_h['vitoria_visitante']) > float(probs_h['vitoria_casa']) and float(odds_h['vitoria_visitante']) > 1.5 and (float(probs_h['vitoria_visitante']) + float(probs_h['push'])) > 0.6:
            decisions['h'] = 1
    print(decisions)
    decisions['h'] = None
    decisions['ou'] = None
    return decisions
    
def associa_odds_e_stats(df_stats, df_odds):
    """
    Associa as odds de apostas com as estatísticas de partidas.

    Args:
        df_stats (pd.DataFrame): DataFrame contendo as estatísticas das partidas.
        df_odds (pd.DataFrame): DataFrame contendo as odds das apostas.

    Returns:
        pd.DataFrame: DataFrame resultante com as odds associadas às estatísticas.
    """
    # Verifica se os DataFrames têm a coluna 'id' para associação
    if 'id' not in df_stats.columns or 'id' not in df_odds.columns:
        raise ValueError("Ambos os DataFrames devem conter a coluna 'id' para associação.")



    # Realiza o merge dos DataFrames com base na coluna 'id'
    df_merged = pd.merge(df_stats, df_odds, on='id', how='left')


    return df_stats.copy(), df_merged



def prepara_e_realiza_teste(df_stats, df_odds, model_regression_2):
    """
    Prepara os dados de teste e realiza previsões usando o modelo de regressão.

    Args:
        df_stats (pd.DataFrame): DataFrame contendo as estatísticas das partidas.
        df_odds (pd.DataFrame): DataFrame contendo as odds das apostas.
        model_regression_2 (tf.keras.Model): Modelo de regressão treinado.

    Returns:
        pd.DataFrame: DataFrame com as previsões e odds associadas.
    """
    # Associa as odds com as estatísticas
    df_stats, df_merged = associa_odds_e_stats(df_stats, df_odds[['id','odds_ah1','odds_ah2', 'odd_goals_over1', 'odd_goals_under1','odds_gl1','odds_gl2','odds_dc1','odds_dc2','odds_dc3','odds_dnb1','odds_dnb2', 'tot_goals', 'home_goals', 'away_goals', 'goal_line1_1','goal_line1_2', 'asian_handicap1_1', 'asian_handicap1_2', 'asian_handicap2_1', 'asian_handicap2_2']])
   

    
    ou_certo = 0
    tot_ou = 0
    uni_ou = 0
    gl_certo = 0
    tot_gl = 0
    uni_gl = 0
    dc_certo = 0
    tot_dc = 0
    uni_dc = 0
    dnb_certo = 0
    tot_dnb = 0
    uni_dnb = 0
    h_certo = 0
    tot_h = 0
    uni_h = 0
    df_stats.drop(columns=['id'], inplace=True)
    for i in range(0, df_merged.shape[0], 2):
        pred1 = model_regression_2.predict(df_stats.iloc[i: i + 1], verbose=0)
        pred2 = model_regression_2.predict(df_stats.iloc[i + 1: i + 2], verbose=0)
        

        matriz, win_probs = prever_placares_futebol_poisson(pred1, pred2, max_gols=7)
        try:
            probabilidades_ou = calcular_over_under(matriz, 2.5, 2.5)
        except:
            probabilidades_ou = None
        try:
            probabilidades_gl = calcular_over_under(matriz, df_merged.iloc[i].loc['goal_line1_1'], df_merged.iloc[i].loc['goal_line1_2'])
        except:
            probabilidades_gl = None

        probabilidades_vic = win_probs
        
        try:
            probabilidades_h = calcular_handicap(matriz, df_merged.iloc[i].loc['asian_handicap1_1'], df_merged.iloc[i].loc['asian_handicap1_2'],df_merged.iloc[i].loc['asian_handicap2_1'], df_merged.iloc[i].loc['asian_handicap2_2'])
        except:
            probabilidades_h = None
        odds_ou = {
            'over': df_merged.iloc[i].loc['odd_goals_over1'],
            'under': df_merged.iloc[i].loc['odd_goals_under1']
        }
        odds_gl = {
            'over': df_merged.iloc[i].loc['odds_gl1'],
            'under': df_merged.iloc[i].loc['odds_gl2']
        }
        odds_h = {
            'vitoria_casa': df_merged.iloc[i].loc['odds_ah1'],
            'vitoria_visitante': df_merged.iloc[i].loc['odds_ah2']
        }
        odds_dnb = {
            'vitoria_casa': df_merged.iloc[i].loc['odds_dnb1'],
            'vitoria_visitante': df_merged.iloc[i].loc['odds_dnb2']
        }
        odds_dc = {
            'vitoria_casa': df_merged.iloc[i].loc['odds_dc1'],
            'vitoria_visitante': df_merged.iloc[i].loc['odds_dc2'],
            'ambos': df_merged.iloc[i].loc['odds_dc3']
        }
        previsoes = decide_aposta(probabilidades_ou, probabilidades_gl, probabilidades_vic, probabilidades_h, odds_ou=odds_ou, odds_gl=odds_gl, odds_dc=odds_dc, odds_dnb=odds_dnb, odds_h=odds_h)
        if previsoes['ou'] is not None:
            if previsoes['ou'] == 0:
                if df_merged.iloc[i].loc['tot_goals'] > 2.5:
                    ou_certo += 1
                    uni_ou += float(odds_ou['over']) - 1
                else:
                    uni_ou -= 1
            elif previsoes['ou'] == 1:
                if df_merged.iloc[i].loc['tot_goals'] < 2.5:
                    ou_certo += 1
                    uni_ou += float(odds_ou['under']) - 1
                else:
                    uni_ou -= 1
            tot_ou += 1
        if previsoes['gl'] is not None:
            gl1 = df_merged.iloc[i].loc['goal_line1_1']
            gl2 = df_merged.iloc[i].loc['goal_line1_2']
            total = df_merged.iloc[i].loc['tot_goals']
            
            if previsoes['gl'] == 0:  # Over (acima)
                res1 = total - gl1
                res2 = total - gl2

                if res1 > 0 and res2 > 0:  # 🟢 ganho total
                    gl_certo += 1
                    uni_gl += float(odds_gl['over']) - 1
                elif (res1 > 0 and res2 == 0) or (res1 == 0 and res2 > 0):  # 🟡 meia vitória
                    gl_certo += 0.5
                    uni_gl += (float(odds_gl['over']) - 1) /2
                elif res1 == 0 and res2 == 0:  # ⚪ empate total (push)
                    pass  # dinheiro devolvido
                elif (res1 < 0 and res2 == 0) or (res1 == 0 and res2 < 0):  # 🔴 meia perda
                    uni_gl -= 0.5
                elif res1 < 0 and res2 < 0:  # ❌ perda total
                    uni_gl -= 1

            elif previsoes['gl'] == 1:  # Under (abaixo)
                res1 = gl1 - total
                res2 = gl2 - total

                if res1 > 0 and res2 > 0:  # 🟢 ganho total
                    gl_certo += 1
                    uni_gl += float(odds_gl['under']) - 1
                elif (res1 > 0 and res2 == 0) or (res1 == 0 and res2 > 0):  # 🟡 meia vitória
                    gl_certo += 0.5
                    uni_gl += (float(odds_gl['under']) - 1) * 0.5
                elif res1 == 0 and res2 == 0:  # ⚪ empate total
                    pass  # dinheiro devolvido
                elif (res1 < 0 and res2 == 0) or (res1 == 0 and res2 < 0):  # 🔴 meia perda
                    uni_gl -= 0.5
                elif res1 < 0 and res2 < 0:  # ❌ perda total
                    uni_gl -= 1

            tot_gl += 1

        if previsoes['dc'] is not None:
            res = df_merged.iloc[i].loc['home_goals'] - df_merged.iloc[i].loc['away_goals']
            if previsoes['dc'] == 0:
                if res >= 0:
                    
                    dc_certo += 1
                    uni_dc += float(odds_dc['vitoria_casa']) - 1

                else:
                    uni_dc -= 1
            elif previsoes['dc'] == 1:
                if res <= 0:
                    
                    dc_certo += 1
                    uni_dc += float(odds_dc['vitoria_visitante']) - 1
                else:
                    uni_dc -= 1
            elif previsoes['dc'] == 2:
                if res < 0 or res > 0:
                    dc_certo += 1
                    uni_dc += float(odds_dc['ambos']) - 1
                else:
                    uni_dc -= 1
            tot_dc += 1
        if previsoes['dnb'] is not None:
            res = df_merged.iloc[i].loc['home_goals'] - df_merged.iloc[i].loc['away_goals']
            if previsoes['dnb'] == 0:
                if res > 0:
                    
                    dnb_certo += 1
                    uni_dnb += float(odds_dnb['vitoria_casa']) - 1
                elif res < 0:
                    uni_dnb -= 1
            elif previsoes['dnb'] == 1:
                if res < 0:
                    
                    dnb_certo += 1
                    uni_dnb += float(odds_dnb['vitoria_visitante']) - 1
                elif res > 0:
                    uni_dnb -= 1
            tot_dnb += 1
        if previsoes['h'] is not None:
            res = df_merged.iloc[i].loc['home_goals'] - df_merged.iloc[i].loc['away_goals']
            
            if previsoes['h'] == 0:  # Vitória casa
                h1 = df_merged.iloc[i].loc['asian_handicap1_1']
                h2 = df_merged.iloc[i].loc['asian_handicap1_2']
                
                res1 = res - h1
                res2 = res - h2
                
                # Avaliação
                if res1 > 0 and res2 > 0:  # 🟢 ganho total
                    h_certo += 1
                    uni_h += float(odds_h['vitoria_casa']) - 1
                elif (res1 == 0 and res2 > 0) or (res1 > 0 and res2 == 0):  # 🟡 meia vitória
                    h_certo += 0.5
                    uni_h += (float(odds_h['vitoria_casa']) - 1) /2
                elif res1 == 0 and res2 == 0:  # ⚪ empate total
                    pass  # dinheiro devolvido
                elif (res1 < 0 and res2 == 0) or (res1 == 0 and res2 < 0):  # 🔴 meia perda
                    uni_h -= 0.5
                elif res1 < 0 and res2 < 0:  # ❌ perda total
                    uni_h -= 1

            elif previsoes['h'] == 1:  # Vitória visitante
                h1 = -df_merged.iloc[i].loc['asian_handicap2_1']
                h2 = -df_merged.iloc[i].loc['asian_handicap2_2']
                
                res1 = -res - h1  # porque res = home - away
                res2 = -res - h2

                # Avaliação
                if res1 > 0 and res2 > 0:  # 🟢 ganho total
                    h_certo += 1
                    uni_h += float(odds_h['vitoria_visitante']) - 1
                elif (res1 == 0 and res2 > 0) or (res1 > 0 and res2 == 0):  # 🟡 meia vitória
                    h_certo += 0.5
                    uni_h += (float(odds_h['vitoria_visitante']) - 1) * 0.5
                elif res1 == 0 and res2 == 0:  # ⚪ empate total
                    pass
                elif (res1 < 0 and res2 == 0) or (res1 == 0 and res2 < 0):  # 🔴 meia perda
                    h_certo += 0
                    uni_h -= 0.5
                elif res1 < 0 and res2 < 0:  # ❌ perda total
                    uni_h -= 1

            tot_h += 1

        
        results = {
        'ou_certo': f'{ou_certo} / {tot_ou}',
        'uni_ou': uni_ou,
        'gl_certo': f'{gl_certo} / {tot_gl}',
        'uni_gl': uni_gl,
        'dc_certo': f'{dc_certo} / {tot_dc}',
        'uni_dc': uni_dc,
        'dnb_certo': f'{dnb_certo} / {tot_dnb}',
        'uni_dnb': uni_dnb,
        'h_certo': f'{h_certo} / {tot_h}',
        'uni_h': uni_h}
        print(results)
    

    return results

def prepara_e_preve(df_stats,df_odds, model_regression_path='src/model_regression_2.keras'):
    """
    Prepara os dados de teste e realiza previsões usando o modelo de regressão.

    Args:
        df_stats (pd.DataFrame): DataFrame contendo as estatísticas das partidas.
        df_odds (pd.DataFrame): DataFrame contendo as odds das apostas.
        model_regression_path (str): Caminho para o modelo de regressão treinado.

    Returns:
        pd.DataFrame: DataFrame com as previsões e odds associadas.
    """
    # Associa as odds com as estatísticas
   
    #[['id','odds_ah1','odds_ah2', 'odd_goals_over1', 'odd_goals_under1','odds_gl1','odds_gl2','odds_dc1','odds_dc2','odds_dc3','odds_dnb1','odds_dnb2', 'tot_goals', 'home_goals', 'away_goals', 'goal_line1_1','goal_line1_2', 'asian_handicap1_1', 'asian_handicap1_2', 'asian_handicap2_1', 'asian_handicap2_2']]

    try:
        model = tf.keras.models.load_model(model_regression_path)
  
        pred1 = model.predict(df_stats.iloc[[0]], verbose=1)
        pred2 = model.predict(df_stats.iloc[[1]], verbose=1)
    except:
        raise ValueError("Erro ao carregar o modelo de regressão. Verifique o caminho do modelo.")
    linha = df_odds.iloc[0]
    matriz, win_probs = prever_placares_futebol_poisson(pred1, pred2, max_gols=7)
    try:
        probabilidades_ou = calcular_over_under(matriz, 2.5, 2.5)
    except:
        probabilidades_ou = None
    try:
        probabilidades_gl = calcular_over_under(matriz, 
        linha.get('goal_line1_1'), 
        linha.get('goal_line1_2'))
    except:
        probabilidades_gl = None

    probabilidades_vic = win_probs
    
    try:
        probabilidades_h = calcular_handicap(matriz, 
        linha.get('asian_handicap1_1'), 
        linha.get('asian_handicap1_2'),
        linha.get('asian_handicap2_1'), 
        linha.get('asian_handicap2_2'))
    except:
        probabilidades_h = None
    

    odds_ou = {
        'over': linha.get('odd_goals_over1'),
        'under': linha.get('odd_goals_under1')
    }
    odds_gl = {
        'over': linha.get('odds_gl1'),
        'under': linha.get('odds_gl2')
    }
    odds_h = {
        'vitoria_casa': linha.get('odds_ah1'),
        'vitoria_visitante': linha.get('odds_ah2')
    }
    odds_dnb = {
        'vitoria_casa': linha.get('odds_dnb1'),
        'vitoria_visitante': linha.get('odds_dnb2')
    }
    odds_dc = {
        'vitoria_casa': linha.get('odds_dc1'),
        'vitoria_visitante': linha.get('odds_dc2'),
        'ambos': linha.get('odds_dc3')
    }

    previsoes = decide_aposta(probabilidades_ou, probabilidades_gl, probabilidades_vic, probabilidades_h, odds_ou=odds_ou, odds_gl=odds_gl, odds_dc=odds_dc, odds_dnb=odds_dnb, odds_h=odds_h)
    return previsoes


def treino():

    df_temp = pd.read_csv('resultados_60.csv')
    df_temp = NN.preProcessGeneral(df_temp)
    df_temp.to_csv('df_temp_preprocessado.csv', index=False)
    df = pd.read_csv('df_temp_preprocessado.csv')

    df_new = individualiza_jogo_duplo_com_resultado(df)

    df_filtered = df_new.dropna()

    X = df_filtered.drop(columns=['resultado_vitoria'], axis=1)

    y = df_filtered['resultado_vitoria']

    
    columns_to_scale = [
        'home', 'fa', 'fd', 'opponent_fa', 'opponent_fd', 'fa_diff',
        'fd_diff', 'fa/fd_diff', 'league', 'media_goals',
        'media_goals_sofridos', 'media_victories', 'h2h_mean', 'h2h_win_rate',
        'h2h_total_games', 'expected_goals', 'opponent_expected_goals',
        'expected_goals_diff', 'opponent_media_goals',
        'opponent_media_goals_sofridos', 'opponent_media_victories',
        'opponent_h2h_mean', 'opponent_h2h_win_rate', 'media_goals_diff',
        'media_goals_sofridos_diff', 'media_victories_diff', 'h2h_mean_diff',
        'h2h_win_rate_diff'
    ]
    #'goal_line1_1', 'goal_line1_2', 'odds_gl1', 'odds_gl2',
    # 3. Aplicar o MinMaxScaler
    X, scaler = apply_min_max_scaler_to_columns(X, columns_to_scale)



    # Configurações
    n_splits = 5
    n_neighbors = 5

    # Garantir que X e y sejam DataFrames/Séries com índices alinhados
    X_treino_sem_id = pd.DataFrame(X).drop('id', axis=1).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    # Arrays para armazenar previsões do KNN
    knn_preds = np.zeros(len(X))

    # KFold para gerar previsões "fora do fold"
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]

        X_val_fold = X.iloc[val_idx]

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_fold, y_train_fold)

        knn_preds[val_idx] = knn.predict(X_val_fold)


    X['knn_pred'] = knn_preds

    columns_to_scale = [
        'home', 'fa', 'fd', 'opponent_fa', 'opponent_fd', 'fa_diff',
        'fd_diff', 'fa/fd_diff', 'league', 'media_goals',
        'media_goals_sofridos', 'media_victories', 'h2h_mean', 'h2h_win_rate',
        'h2h_total_games', 'expected_goals', 'opponent_expected_goals',
        'expected_goals_diff', 'opponent_media_goals',
        'opponent_media_goals_sofridos', 'opponent_media_victories',
        'opponent_h2h_mean', 'opponent_h2h_win_rate', 'media_goals_diff',
        'media_goals_sofridos_diff', 'media_victories_diff', 'h2h_mean_diff',
        'h2h_win_rate_diff', 'knn_pred'
    ]
    df_scaled, scaler = apply_min_max_scaler_to_columns(X, columns_to_scale)
    

    model_regression_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model_regression_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss= tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    model_regression_2.fit(df_scaled, y, epochs=100)
    model_regression_2.save('src/model_regression_2.keras')



def apply_min_max_scaler_to_columns(df, columns_to_scale):
    """
    Aplica MinMaxScaler a uma lista específica de colunas em um DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame original.
        columns_to_scale (list): Uma lista de strings com os nomes das colunas
                                 às quais o MinMaxScaler deve ser aplicado.

    Returns:
        pd.DataFrame: O DataFrame com as colunas especificadas escaladas.
        MinMaxScaler: O objeto scaler ajustado, para ser usado em novos dados.
    """
    # Cria uma cópia do DataFrame para evitar modificar o original diretamente
    df_scaled = df.copy()

    # Inicializa o MinMaxScaler
    scaler = MinMaxScaler()

    # Aplica o scaler apenas às colunas especificadas
    # O .fit_transform() ajusta o scaler aos dados e os transforma
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

    return df_scaled, scaler