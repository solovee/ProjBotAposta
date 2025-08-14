import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import random


def discretizar_goals_balance_diff_num(goals_balance_diff):
    """
    Discretiza a diferença de saldo de gols em valores numéricos
    Limites ampliados para capturar diferenças mais significativas
    """
    if goals_balance_diff >= 2.5:
        return 4  # casa_muito_superior
    elif goals_balance_diff >= 0.5:
        return 3  # casa_superior
    elif goals_balance_diff >= -0.5:
        return 2  # equilibrado
    elif goals_balance_diff >= -2.5:
        return 1  # fora_superior
    else:
        return 0  # fora_muito_superior

# Discretization functions
def discretizar_numero_indice(numero):
    """
    Discretiza um número inteiro de 0 a 10 em intervalos, retornando índices.
    
    Args:
        numero (int): Número inteiro entre 0 e 10
        
    Returns:
        int: Índice do intervalo (0 para <=3, 1 para <=7, 2 para <=10)
    """
    if not isinstance(numero, int) or numero < 0 or numero > 10:
        raise ValueError("O número deve ser um inteiro entre 0 e 10")
    
    if numero <= 3:
        return 0
    elif numero <= 7:
        return 1
    else:
        return 2
import math

def discretizar_vitorias(valor):
    """
    Discretiza a média de vitórias (que pode ser negativa) em intervalos de 0.25, retornando um inteiro.
    
    Exemplo de faixas:
    - [-∞, -1.0)    → -5
    - [-1.0, -0.75) → -4
    - [-0.75, -0.5) → -3
    - [-0.5, -0.25) → -2
    - [-0.25, 0.0)  → -1
    - [ 0.0, 0.25)  → 0
    - [ 0.25, 0.5)  → 1
    - [ 0.5, 0.75)  → 2
    - [ 0.75, 1.0)  → 3
    - valor == 1.0 → 4
    """
    if valor == 1.0:
        return 4

    return math.floor(valor * 4)



import math

def discretizar_goals(valor):
    """
    Discretiza a média de gols (positiva ou negativa) em intervalos de 0.5.
    
    Exemplo:
    - [-0.5, 0.0)  → -1
    - [ 0.0, 0.5)  → 0
    - [ 0.5, 1.0)  → 1
    - [ 1.0, 1.5)  → 2
    - ...
    """
    return math.floor(valor * 2)


def discretizar_odds(valor):
    """
    Discretiza as odds (valores ≥ 1.0) em intervalos de 0.25:
    - 1.0-1.25: 0
    - 1.25-1.5: 1
    - ...
    - 2.75-3.0 e acima: 7
    Odds acima de 3.0 são agrupadas com o intervalo 2.75-3.0.
    """
    # Limita o valor máximo a 3.0
    valor = min(valor, 3.0)

    # Calcula o índice com base em intervalos de 0.25 a partir de 1.0
    indice_intervalo = int((valor - 1.0) / 0.25)

    return indice_intervalo

'''
def discretizar_goal_diff(valor):
    """
    Discretiza a diferença de gols em intervalos de 0.5, com centro entre -0.5 e 0.5:
    - valor <= -3.0: -6
    - [-3.0, -2.5): -5
    - [-2.5, -2.0): -4
    - [-2.0, -1.5): -3
    - [-1.5, -1.0): -2
    - [-1.0, -0.5): -1
    - [-0.5, 0.5):   0
    - [0.5, 1.0):    1
    - [1.0, 1.5):    2
    - [1.5, 2.0):    3
    - [2.0, 2.5):    4
    - [2.5, 3.0):    5
    - valor >= 3.0:  6
    """
    if valor <= -3.0:
        return -6
    elif valor >= 3.0:
        return 6

    # Com intervalo central de [-0.5, 0.5), basta dividir por 0.5 e arredondar
    indice_intervalo = int(valor / 0.5)

    return indice_intervalo
'''
def discretizar_goal_diff(valor):
    """
    Discretiza a diferença de gols em intervalos de 1.0, com centro entre -0.5 e 0.5:
    - valor < -2.5: -3
    - [-2.5, -1.5): -2
    - [-1.5, -0.5): -1
    - [-0.5, 0.5):   0
    - [0.5, 1.5):    1
    - [1.5, 2.5):    2
    - valor >= 2.5:  3
    """
    if valor < -2.5:
        return -3
    elif valor >= 2.5:
        return 3
    elif valor >= -2.5 and valor < -1.5:
        return -2
    elif valor >= -1.5 and valor < -0.5:
        return -1
    elif valor >= -0.5 and valor < 0.5:
        return 0
    elif valor >= 0.5 and valor < 1.5:
        return 1
    elif valor >= 1.5 and valor < 2.5:
        return 2

def discretizar_league(league_id):
    """
    Discretiza o ID da liga para um valor inteiro
    """
    # Se league_id já for um inteiro, apenas retorna
    # Caso contrário, mapeia para valores inteiros específicos
    # Aqui, estamos assumindo que league_id já é um valor discreto adequado
    return int(league_id)

class QLearningDoubleChance:
    def __init__(self, alpha=0.1, gamma=0.0, epsilon=0.2):
        """
        Inicializa o agente de Q-Learning para apostas Double Chance
        
        Parâmetros:
        - alpha: taxa de aprendizado (0-1)
        - gamma: fator de desconto para recompensas futuras (0-1)
        - epsilon: parâmetro de exploração para a estratégia epsilon-greedy (0-1)
        """
        self.q_table = {}  # Tabela Q vazia (será um dicionário, pois temos muitos estados)
        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon  # Parâmetro de exploração
        self.actions = [0, 1, 2]  # 0: Double Chance 1, 1: Double Chance 2, 2: Double Chance 3
    
    def get_q_value(self, state, action):
        """Retorna o valor Q para um par estado-ação"""
        if state not in self.q_table:
            # Se o estado não existir, inicializa com zeros para todas as ações
            self.q_table[state] = [0, 0, 0]
        return self.q_table[state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        """Atualiza o valor Q para um par estado-ação"""
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        
        # Se o próximo estado existir, considera o máximo Q-valor das próximas ações
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state])
        else:
            max_next_q = 0
        
        # Atualiza o valor Q usando a fórmula de Q-Learning
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def choose_action(self, state, epsilon=None):
        """
        Escolhe uma ação usando a estratégia epsilon-greedy
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Com probabilidade epsilon, escolhe uma ação aleatória (exploração)
        if random.random() < epsilon:
            return random.choice(self.actions)
        
        # Caso contrário, escolhe a melhor ação (explotação)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        
        # Em caso de empate, escolhe aleatoriamente entre os melhores
        max_q = max(self.q_table[state])
        best_actions = [a for a, q in enumerate(self.q_table[state]) if q == max_q]
        return random.choice(best_actions)
    
    def train(self, df, num_episodes=150):
        """
        Treina o modelo de Q-Learning usando o DataFrame fornecido
        
        Parâmetros:
        - df: DataFrame com os dados de jogos
        - num_episodes: número de episódios de treinamento
        """
        # Garantir que temos as colunas necessárias
        required_cols = ['league',  
         'odds_dc1', 
        'odds_dc2', 
         'odds_dc3', 'h2h_total_games','home_h2h_win_rate','away_h2h_win_rate',
        'res_double_chance1', 'res_double_chance2', 'res_double_chance3','media_victories_home','media_victories_away']
        
            

        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no DataFrame")
        
        # Preparar o DataFrame para o treinamento
        df_copy = df[['league',  
         'odds_dc1', 
        'odds_dc2', 
         'odds_dc3', 'h2h_total_games','home_h2h_win_rate','away_h2h_win_rate',
        'res_double_chance1', 'res_double_chance2', 'res_double_chance3','media_victories_home','media_victories_away']].copy()
        df_copy.dropna(inplace=True)
        # Adicionar uma coluna para o índice do jogo
        df_copy['game_idx'] = df_copy.index
        
        # Loop para os episódios de treinamento
        for episode in tqdm(range(num_episodes), desc="Treinamento Q-Learning"):
            # Embaralhar o DataFrame para treinamento
            df_train = df_copy.sample(frac=1)
            
            for idx, (_, game) in enumerate(df_train.iterrows()):
                # Criar o estado atual baseado nas características do jogo
                current_state = q_learning_dc(game)
                
                # Escolher uma ação (Double Chance 1, 2 ou 3)
                action = self.choose_action(current_state)
                
                # Determinar a recompensa baseada no resultado do jogo
                if action == 0 and game['res_double_chance1'] == 1:
                    reward = game['odds_dc1'] - 1  # Se escolheu DC1 e acertou
                elif action == 1 and game['res_double_chance2'] == 1:
                    reward = game['odds_dc2'] - 1  # Se escolheu DC2 e acertou
                elif action == 2 and game['res_double_chance3'] == 1:
                    reward = game['odds_dc3'] - 1  # Se escolheu DC3 e acertou
                else:
                    reward = -1  # Se errou a previsão
                
                # Se não for o último jogo, usar o próximo jogo como próximo estado
                if idx < len(df_train) - 1:
                    next_game = df_train.iloc[idx + 1]
                    next_state = q_learning_dc(next_game)
                else:
                    next_state = None
                
                # Atualizar o valor Q
                if next_state:
                    self.update_q_value(current_state, action, reward, next_state)
                else:
                    # Se for o último estado, não há próximo estado para considerar
                    if current_state not in self.q_table:
                        self.q_table[current_state] = [0, 0, 0]
                    self.q_table[current_state][action] = (1 - self.alpha) * self.get_q_value(current_state, action) + self.alpha * reward
    
    def save_model(self, filename='q_learning_dc_model2.pkl'):
        """Salva o modelo em um arquivo"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }, f)
        print(f"Modelo salvo em {filename}")
    
    def load_model(self, filename='q_learning_dc_model.pkl'):
        """Carrega o modelo de um arquivo"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            self.q_table = model_data['q_table']
            self.alpha = model_data['alpha']
            self.gamma = model_data['gamma']
            self.epsilon = model_data['epsilon']
        print(f"Modelo carregado de {filename}")
    
    def evaluate(self, df_test):
        """
        Avalia o modelo em um conjunto de teste
        
        Retorna um dicionário com métricas de avaliação
        """
        correct_predictions = 0
        total_predictions = 0
        uni = 0
        
        results_by_action = {0: {'correct': 0, 'total': 0}, 
                            1: {'correct': 0, 'total': 0}, 
                            2: {'correct': 0, 'total': 0}}
        
        for _, game in df_test.iterrows():
            
            state = q_learning_dc(game)
            action = self.choose_action(state, epsilon=0)  # Sem exploração na avaliação
            
            # Verificar se a previsão está correta
            is_correct = False
            if action == 0 and game['res_double_chance1'] == 1:
                odd = game['odds_dc1']
                is_correct = True
            elif action == 1 and game['res_double_chance2'] == 1:
                odd = game['odds_dc2']
                is_correct = True
            elif action == 2 and game['res_double_chance3'] == 1:
                odd = game['odds_dc3']
                is_correct = True
            
            if is_correct:
                correct_predictions += 1
                results_by_action[action]['correct'] += 1
                uni += odd - 1
            
            total_predictions += 1
            results_by_action[action]['total'] += 1
        
        # Calcular métricas
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        # Calcular accuracy por ação
        accuracy_by_action = {}
        for action, results in results_by_action.items():
            action_accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
            accuracy_by_action[action] = action_accuracy
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'accuracy_by_action': accuracy_by_action,
            'results_by_action': results_by_action,
            'uni': uni
        }



def q_learning_dc(row):
    """
    Converte uma linha do DataFrame em um estado discreto para a Q-table.
    'league',  
         'odds_dc1', 
        'odds_dc2', 
         'odds_dc3',
        'goal_diff', 'victory_diff', 'h2h_diff','goals_ratio_home','goals_ratio_away','vic_ratio'
        'res_double_chance1', 'res_double_chance2', 'res_double_chance3'
    """
   
    victory_diff = row['media_victories_home'] - row['media_victories_away']
    estado = (
        discretizar_league(row['league']),
        discretizar_odds(row['odds_dc1']),
        discretizar_odds(row['odds_dc2']),
        discretizar_odds(row['odds_dc3']),

        discretizar_vitorias(victory_diff),

        discretizar_numero_indice(int(row['h2h_total_games'])),

        discretizar_vitorias(row['home_h2h_win_rate'] - row['away_h2h_win_rate'])
       
    )
    
    return estado

class QLearningGoalLine:
    def __init__(self, alpha=0.1, gamma=0.0, epsilon=0.2):
        """
        Inicializa o agente de Q-Learning para apostas Goal Line
        
        Parâmetros:
        - alpha: taxa de aprendizado (0-1)
        - gamma: fator de desconto para recompensas futuras (0-1)
        - epsilon: parâmetro de exploração para a estratégia epsilon-greedy (0-1)
        """
        self.q_table = {}  # Tabela Q vazia (será um dicionário, pois temos muitos estados)
        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon  # Parâmetro de exploração
        self.actions = [0, 1]  # 0: over 1, 1: under 2
    
    def get_q_value(self, state, action):
        """Retorna o valor Q para um par estado-ação"""
        if state not in self.q_table:
            # Se o estado não existir, inicializa com zeros para todas as ações
            self.q_table[state] = [0, 0]
        return self.q_table[state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        """Atualiza o valor Q para um par estado-ação"""
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        
        # Se o próximo estado existir, considera o máximo Q-valor das próximas ações
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state])
        else:
            max_next_q = 0
        
        # Atualiza o valor Q usando a fórmula de Q-Learning
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def choose_action(self, state, epsilon=None):
        """
        Escolhe uma ação usando a estratégia epsilon-greedy
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Com probabilidade epsilon, escolhe uma ação aleatória (exploração)
        if random.random() < epsilon:
            return random.choice(self.actions)
        
        # Caso contrário, escolhe a melhor ação (explotação)
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        
        # Em caso de empate, escolhe aleatoriamente entre os melhores
        max_q = max(self.q_table[state])
        best_actions = [a for a, q in enumerate(self.q_table[state]) if q == max_q]
        return random.choice(best_actions)
    
    def train(self, df, num_episodes=150):
        """
        Treina o modelo de Q-Learning usando o DataFrame fornecido
        
        Parâmetros:
        - df: DataFrame com os dados de jogos
        - num_episodes: número de episódios de treinamento
        """
        # Garantir que temos as colunas necessárias
        df_of = df.copy()
        if 'resultado' not in df_of.columns:
            def transformar_target(row):
                if (row['gl1_positivo'] == 1):
                    return 0
                elif (row['gl2_positivo'] == 1):
                    return 1
                else:
                    return None

            df_of['resultado'] = df_of.apply(transformar_target, axis=1)

        required_cols = ['goal_line1_1', 'goal_line1_2','league','odds_gl1','odds_gl2','media_goals_home','media_goals_sofridos_home','media_goals_away','media_goals_sofridos_away','h2h_mean','h2h_total_games','resultado']

        
        for col in required_cols:
            if col not in df_of.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no DataFrame")
        
        # Preparar o DataFrame para o treinamento
        df_copy = df_of.copy()
        
        
        # Adicionar uma coluna para o índice do jogo
        df_copy['game_idx'] = df_copy.index
        


        # Loop para os episódios de treinamento
        for episode in tqdm(range(num_episodes), desc="Treinamento Q-Learning"):
            # Embaralhar o DataFrame para treinamento
            df_train = df_copy.sample(frac=1)
            
            for idx, (_, game) in enumerate(df_train.iterrows()):
                # Criar o estado atual baseado nas características do jogo
                current_state = q_learning_gl(game)
                
                # Escolher uma ação (Double Chance 1, 2 ou 3)
                action = self.choose_action(current_state)
                
                # Determinar a recompensa baseada no resultado do jogo
                if action == 0 and game['resultado'] == 0:
                    reward = game['odds_gl1'] - 1  # Se escolheu DC1 e acertou
                elif action == 1 and game['resultado'] == 1:
                    reward =  game['odds_gl2'] - 1  # Se escolheu DC2 e acertou
                else:
                    reward = -1  # Se errou a previsão
                
                # Se não for o último jogo, usar o próximo jogo como próximo estado
                if idx < len(df_train) - 1:
                    next_game = df_train.iloc[idx + 1]
                    next_state = q_learning_gl(next_game)
                else:
                    next_state = None
                
                # Atualizar o valor Q
                if next_state:
                    self.update_q_value(current_state, action, reward, next_state)
                else:
                    # Se for o último estado, não há próximo estado para considerar
                    if current_state not in self.q_table:
                        self.q_table[current_state] = [0, 0, 0]
                    self.q_table[current_state][action] = (1 - self.alpha) * self.get_q_value(current_state, action) + self.alpha * reward
    
    def save_model(self, filename='q_learning_gl_model.pkl'):
        """Salva o modelo em um arquivo"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }, f)
        print(f"Modelo salvo em {filename}")
    
    def load_model(self, filename='q_learning_gl_model.pkl'):
        """Carrega o modelo de um arquivo"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            self.q_table = model_data['q_table']
            self.alpha = model_data['alpha']
            self.gamma = model_data['gamma']
            self.epsilon = model_data['epsilon']
        print(f"Modelo carregado de {filename}")
    
    def evaluate(self, df_test):
        """
        Avalia o modelo em um conjunto de teste
        
        Retorna um dicionário com métricas de avaliação
        """
        correct_predictions = 0
        total_predictions = 0
        uni = 0
        
        results_by_action = {0: {'correct': 0, 'total': 0}, 
                            1: {'correct': 0, 'total': 0}}
        
        for _, game in df_test.iterrows():
            
            state = q_learning_gl(game)
            action = self.choose_action(state, epsilon=0)  # Sem exploração na avaliação
            
            # Verificar se a previsão está correta
            is_correct = False
            if action == 0 and game['resultado'] == 0:
                odd = game['odds_gl1']
                is_correct = True
            elif action == 1 and game['resultado'] == 1:
                odd = game['odds_gl2']
                is_correct = True
            
            if is_correct:
                correct_predictions += 1
                results_by_action[action]['correct'] += 1
                uni += odd - 1
            
            total_predictions += 1
            results_by_action[action]['total'] += 1
        
        # Calcular métricas
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        # Calcular accuracy por ação
        accuracy_by_action = {}
        for action, results in results_by_action.items():
            action_accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
            accuracy_by_action[action] = action_accuracy
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'accuracy_by_action': accuracy_by_action,
            'results_by_action': results_by_action,
            'uni': uni
        }


def q_learning_gl(row):
    """
    Converte uma linha do DataFrame em um estado discreto para a Q-table.
    'goal_line1_1', 'goal_line1_2','league','odds_gl1','odds_gl2','media_goals_home','media_goals_sofridos_home','media_goals_away','media_goals_sofridos_away','h2h_mean','h2h_total_games','resultado'
    """
    
    estado = (
        float(row['goal_line1_1']),
        float(row['goal_line1_2']),
        discretizar_league(row['league']),
        discretizar_odds(row['odds_gl1']),
        discretizar_odds(row['odds_gl2']),
        discretizar_goals(((row['media_goals_home'] + row['media_goals_sofridos_home']) + (row['media_goals_away'] + row['media_goals_sofridos_away']))/2),
        discretizar_goals(row['h2h_mean']),
        discretizar_numero_indice(int(row['h2h_total_games']))
    )
    
    return estado


class QLearningDrawNoBet:
    def __init__(self, alpha=0.1, gamma=0.0, epsilon=0.2):
        """
        Inicializa o agente de Q-Learning para apostas Goal Line
        
        Parâmetros:
        - alpha: taxa de aprendizado (0-1)
        - gamma: fator de desconto para recompensas futuras (0-1)
        - epsilon: parâmetro de exploração para a estratégia epsilon-greedy (0-1)
        """
        self.q_table = {}  # Tabela Q vazia (será um dicionário, pois temos muitos estados)
        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon  # Parâmetro de exploração
        self.actions = [0, 1]  # 0: over 1, 1: under 2
    
    def get_q_value(self, state, action):
        """Retorna o valor Q para um par estado-ação"""
        if state not in self.q_table:
            # Se o estado não existir, inicializa com zeros para todas as ações
            self.q_table[state] = [0, 0]
        return self.q_table[state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        """Atualiza o valor Q para um par estado-ação"""
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        
        # Se o próximo estado existir, considera o máximo Q-valor das próximas ações
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state])
        else:
            max_next_q = 0
        
        # Atualiza o valor Q usando a fórmula de Q-Learning
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def choose_action(self, state, epsilon=None):
        """
        Escolhe uma ação usando a estratégia epsilon-greedy
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Com probabilidade epsilon, escolhe uma ação aleatória (exploração)
        if random.random() < epsilon:
            return random.choice(self.actions)
        
        # Caso contrário, escolhe a melhor ação (explotação)
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        
        # Em caso de empate, escolhe aleatoriamente entre os melhores
        max_q = max(self.q_table[state])
        best_actions = [a for a, q in enumerate(self.q_table[state]) if q == max_q]
        return random.choice(best_actions)
    
    def train(self, df, num_episodes=150):
        """
        Treina o modelo de Q-Learning usando o DataFrame fornecido
        
        Parâmetros:
        - df: DataFrame com os dados de jogos
        - num_episodes: número de episódios de treinamento
        """
        # Garantir que temos as colunas necessárias
        df_of = df.copy()
        if 'resultado' not in df_of.columns:
            def transformar_res(row):
                if row['dnb1_ganha'] == 1:
                    return 0
                elif row['dnb2_ganha'] == 1:
                    return 1
                else:
                    return None
                

            df_of['resultado'] = df_of.apply(transformar_res, axis=1)
        required_cols = ['league',  
         'odds_dnb1', 
        'odds_dnb2',
        'h2h_total_games','home_h2h_win_rate','away_h2h_win_rate','media_victories_home','media_victories_away','resultado']
        
        for col in required_cols:
            if col not in df_of.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no DataFrame")
        
        # Preparar o DataFrame para o treinamento
        df_copy = df_of.copy()
        
        # Adicionar uma coluna para o índice do jogo
        df_copy['game_idx'] = df_copy.index
        
        # Loop para os episódios de treinamento
        for episode in tqdm(range(num_episodes), desc="Treinamento Q-Learning"):
            # Embaralhar o DataFrame para treinamento
            df_train = df_copy.sample(frac=1)
            
            for idx, (_, game) in enumerate(df_train.iterrows()):
                # Criar o estado atual baseado nas características do jogo
                current_state = q_learning_dnb(game)
                
                # Escolher uma ação (Double Chance 1, 2 ou 3)
                action = self.choose_action(current_state)
                
                # Determinar a recompensa baseada no resultado do jogo
                if action == 0 and game['resultado'] == 0:
                    reward =  game['odds_dnb1'] - 1  # Se escolheu DC1 e acertou
                elif action == 1 and game['resultado'] == 1:
                    reward =  game['odds_dnb2'] - 1  # Se escolheu DC2 e acertou
                else:
                    reward = -1  # Se errou a previsão
                
                # Se não for o último jogo, usar o próximo jogo como próximo estado
                if idx < len(df_train) - 1:
                    next_game = df_train.iloc[idx + 1]
                    next_state = q_learning_dnb(next_game)
                else:
                    next_state = None
                
                # Atualizar o valor Q
                if next_state:
                    self.update_q_value(current_state, action, reward, next_state)
                else:
                    # Se for o último estado, não há próximo estado para considerar
                    if current_state not in self.q_table:
                        self.q_table[current_state] = [0, 0, 0]
                    self.q_table[current_state][action] = (1 - self.alpha) * self.get_q_value(current_state, action) + self.alpha * reward
    
    def save_model(self, filename='q_learning_dnb_model.pkl'):
        """Salva o modelo em um arquivo"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }, f)
        print(f"Modelo salvo em {filename}")
    
    def load_model(self, filename='q_learning_dnb_model.pkl'):
        """Carrega o modelo de um arquivo"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            self.q_table = model_data['q_table']
            self.alpha = model_data['alpha']
            self.gamma = model_data['gamma']
            self.epsilon = model_data['epsilon']
        print(f"Modelo carregado de {filename}")
    
    def evaluate(self, df_test):
        """
        Avalia o modelo em um conjunto de teste
        
        Retorna um dicionário com métricas de avaliação
        """
        correct_predictions = 0
        total_predictions = 0
        uni = 0
        
        results_by_action = {0: {'correct': 0, 'total': 0}, 
                            1: {'correct': 0, 'total': 0}}
        
        for _, game in df_test.iterrows():
            
            state = q_learning_dnb(game)
            action = self.choose_action(state, epsilon=0)  # Sem exploração na avaliação
            
            # Verificar se a previsão está correta
            is_correct = False
            if action == 0 and game['resultado'] == 0:
                odd = game['odds_dnb1']
                is_correct = True
            elif action == 1 and game['resultado'] == 1:
                odd = game['odds_dnb2']
                is_correct = True
            
            if is_correct:
                correct_predictions += 1
                results_by_action[action]['correct'] += 1
                uni += odd - 1
            
            total_predictions += 1
            results_by_action[action]['total'] += 1
        
        # Calcular métricas
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        # Calcular accuracy por ação
        accuracy_by_action = {}
        for action, results in results_by_action.items():
            action_accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
            accuracy_by_action[action] = action_accuracy
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'accuracy_by_action': accuracy_by_action,
            'results_by_action': results_by_action,
            'uni': uni
        }



def q_learning_dnb(row):
    """
    Converte uma linha do DataFrame em um estado discreto para a Q-table.
    'league',  
         'odds_dnb1', 
        'odds_dnb2',
        'goal_diff', 'victory_diff', 'h2h_diff','goals_ratio_home','goals_ratio_away','vic_ratio','h2h_total_games','draw_no_bet_team1','draw_no_bet_team2','resultado'
    """

    
    
    estado = (
        
        discretizar_league(row['league']),
        discretizar_odds(row['odds_dnb1']),
        discretizar_odds(row['odds_dnb2']),
        discretizar_vitorias(row['media_victories_home'] - row['media_victories_away']),
        discretizar_numero_indice(int(row['h2h_total_games'])),
 

        discretizar_vitorias(row['home_h2h_win_rate'] - row['away_h2h_win_rate'])

    )
    
    return estado



class QLearningHandicap:
    def __init__(self, alpha=0.1, gamma=0.0, epsilon=0.2):
        """
        Inicializa o agente de Q-Learning para apostas Goal Line
        
        Parâmetros:
        - alpha: taxa de aprendizado (0-1)
        - gamma: fator de desconto para recompensas futuras (0-1)
        - epsilon: parâmetro de exploração para a estratégia epsilon-greedy (0-1)
        """
        self.q_table = {}  # Tabela Q vazia (será um dicionário, pois temos muitos estados)
        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon  # Parâmetro de exploração
        self.actions = [0, 1]  # 0: over 1, 1: under 2
    
    def get_q_value(self, state, action):
        """Retorna o valor Q para um par estado-ação"""
        if state not in self.q_table:
            # Se o estado não existir, inicializa com zeros para todas as ações
            self.q_table[state] = [0, 0]
        return self.q_table[state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        """Atualiza o valor Q para um par estado-ação"""
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        
        # Se o próximo estado existir, considera o máximo Q-valor das próximas ações
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state])
        else:
            max_next_q = 0
        
        # Atualiza o valor Q usando a fórmula de Q-Learning
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def choose_action(self, state, epsilon=None):
        """
        Escolhe uma ação usando a estratégia epsilon-greedy
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Com probabilidade epsilon, escolhe uma ação aleatória (exploração)
        if random.random() < epsilon:
            return random.choice(self.actions)
        
        # Caso contrário, escolhe a melhor ação (explotação)
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        
        # Em caso de empate, escolhe aleatoriamente entre os melhores
        max_q = max(self.q_table[state])
        best_actions = [a for a, q in enumerate(self.q_table[state]) if q == max_q]
        return random.choice(best_actions)
    
    def train(self, df, num_episodes=150):
        """
        Treina o modelo de Q-Learning usando o DataFrame fornecido
        
        Parâmetros:
        - df: DataFrame com os dados de jogos
        - num_episodes: número de episódios de treinamento
        """
        # Garantir que temos as colunas necessárias
        required_cols = ['media_goals_home', 'media_goals_away',
                        'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 
                        'asian_handicap2_1', 'asian_handicap2_2','team_ah2','odds_ah2','league','home_h2h_win_rate','away_h2h_win_rate','media_goals_home','media_goals_away','h2h_total_games','media_goals_sofridos_home','media_goals_sofridos_away','media_victories_away','media_victories_home','home_h2h_mean','away_h2h_mean']

        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no DataFrame")
        
        # Preparar o DataFrame para o treinamento
        df_copy = df.copy()
        df_copy.dropna(inplace=True)
        
        # Adicionar uma coluna para o índice do jogo
        df_copy['game_idx'] = df_copy.index
        
        # Loop para os episódios de treinamento
        for episode in tqdm(range(num_episodes), desc="Treinamento Q-Learning"):
            # Embaralhar o DataFrame para treinamento
            df_train = df_copy.sample(frac=1)
            
            for idx, (_, game) in enumerate(df_train.iterrows()):
                # Criar o estado atual baseado nas características do jogo
                current_state = q_learning_h(game)
                
                # Escolher uma ação (Double Chance 1, 2 ou 3)
                action = self.choose_action(current_state)
                
                # Determinar a recompensa baseada no resultado do jogo
                if action == 0 and game['resultado'] == 0:
                    reward = game['odds_ah1'] - 1  # Se escolheu DC1 e acertou
                elif action == 1 and game['resultado'] == 1:
                    reward =  game['odds_ah2'] - 1  # Se escolheu DC2 e acertou
                else:
                    reward = -1  # Se errou a previsão
                
                # Se não for o último jogo, usar o próximo jogo como próximo estado
                if idx < len(df_train) - 1:
                    next_game = df_train.iloc[idx + 1]
                    next_state = q_learning_h(next_game)
                else:
                    next_state = None
                
                # Atualizar o valor Q
                if next_state:
                    self.update_q_value(current_state, action, reward, next_state)
                else:
                    # Se for o último estado, não há próximo estado para considerar
                    if current_state not in self.q_table:
                        self.q_table[current_state] = [0, 0, 0]
                    self.q_table[current_state][action] = (1 - self.alpha) * self.get_q_value(current_state, action) + self.alpha * reward
    
    def save_model(self, filename='q_learning_h_model.pkl'):
        """Salva o modelo em um arquivo"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }, f)
        print(f"Modelo salvo em {filename}")
    
    def load_model(self, filename='q_learning_h_model.pkl'):
        """Carrega o modelo de um arquivo"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            self.q_table = model_data['q_table']
            self.alpha = model_data['alpha']
            self.gamma = model_data['gamma']
            self.epsilon = model_data['epsilon']
        print(f"Modelo carregado de {filename}")
    
    def evaluate(self, df_test):
        """
        Avalia o modelo em um conjunto de teste
        
        Retorna um dicionário com métricas de avaliação
        """
        correct_predictions = 0
        total_predictions = 0
        uni = 0
        
        results_by_action = {0: {'correct': 0, 'total': 0}, 
                            1: {'correct': 0, 'total': 0}}
        
        for _, game in df_test.iterrows():
            
            state = q_learning_h(game)
            action = self.choose_action(state, epsilon=0)  # Sem exploração na avaliação
            
            # Verificar se a previsão está correta
            is_correct = False
            if action == 0 and game['resultado'] == 0:
                odd = game['odds_ah1']
                is_correct = True
            elif action == 1 and game['resultado'] == 1:
                odd = game['odds_ah2']
                is_correct = True
            
            if is_correct:
                correct_predictions += 1
                results_by_action[action]['correct'] += 1
                uni += odd - 1
            
            total_predictions += 1
            results_by_action[action]['total'] += 1
        
        # Calcular métricas
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        # Calcular accuracy por ação
        accuracy_by_action = {}
        for action, results in results_by_action.items():
            action_accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
            accuracy_by_action[action] = action_accuracy
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'accuracy_by_action': accuracy_by_action,
            'results_by_action': results_by_action,
            'uni': uni
        }



def q_learning_h(row):
    """
    Converte uma linha do DataFrame em um estado discreto para a Q-table.
    'asian_handicap1_1', 'asian_handicap1_2','team_ah1','odds_ah1', 'team_ah2','odds_ah2','h2h_diff','goals_diff', 'league','goals_ratio_home','goals_ratio_away','vic_ratio','h2h_total_games','resultado'
    """
    home_ratio = row['media_goals_home'] - row['media_goals_sofridos_home']
    away_ratio = row['media_goals_away'] - row['media_goals_sofridos_away']
    vic_ratio = row['home_h2h_win_rate'] - row['away_h2h_win_rate']
    victory_diff = row['media_victories_home'] - row['media_victories_away']
    goals_diff = row['media_goals_home'] - row['media_goals_away']
    h2h_diff = row['home_h2h_mean'] - row['away_h2h_mean']
    estado = (
        
        discretizar_goal_diff((h2h_diff + goals_diff)/2),
        discretizar_goals_balance_diff_num(home_ratio - away_ratio),
        float(row['asian_handicap1_1']),
        float(row['asian_handicap1_2']),
        float(row['asian_handicap2_1']),
        float(row['asian_handicap2_2']),
        discretizar_odds(row['odds_ah1']),
        discretizar_odds(row['odds_ah2']),
        discretizar_league(row['league']),
        discretizar_numero_indice(int(row['h2h_total_games'])),
        discretizar_vitorias(vic_ratio),
        discretizar_vitorias(victory_diff)
        
    )
    
    return estado


