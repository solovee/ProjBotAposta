import time as time_module  # para ainda usar time.sleep
from datetime import datetime, timedelta, time
import qlearning

import tensorflow as tf
from api import BetsAPIClient, dia_anterior
import pandas as pd
from dotenv import load_dotenv
import os
import threading
import NN
import telegramBot as tb
import logging
import json
import threading
import os
import signal
import sys
import pickle

#arrumar multiclass handicap saida

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#NORMALIZAR OS DADOS

load_dotenv()

api = os.getenv("API_KEY")
chat_id = int(os.getenv("CHAT_ID"))
#novo -4954876315
# -1002610837223
chats = [chat_id, -4954876315]



apiclient = BetsAPIClient(api_key=api)

#talvez adicionar liga como parametro da NN (one hot ou label encoder?), talvez nao normalizar a linha do handicap?, definir um th bem menor que o esperado? melhorar os retornos?

#df = pd.read_csv('src\resultados_novo.csv')
#CSV_FILE = r"C:\Users\Leoso\Downloads\projBotAposta\src\resultados_novo.csv"
CSV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resultados_60.csv')
#lista dos thresholds das nns
lista_th = [0.575,0.4,0.625,0.6,0.6,0.6]
list_checa = [{"id": "175474951", "mercado": "double_chance", "time": "AC MILAN (KLAUS)", "odd": 1.5, "jogo": "AC MILAN (KLAUS) X INTER (VENDETTA)"},{"id": "175474956", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.000", "jogo": "NAPOLI (DENNIS) X JUVENTUS (DEMPSEY)"},{"id": "175474960", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.727", "jogo": "AC MILAN (KLAUS) X JUVENTUS (DEMPSEY)"},{"id": "175474964", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.250", "jogo": "NAPOLI (DENNIS) X LAZIO (JACK)"},{"id": "175474967", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.750", "jogo": "LAZIO (JACK) X JUVENTUS (DEMPSEY)"},{"id": "175474985", "mercado": "double_chance", "time": "AC MILAN (KLAUS)", "odd": 1.5, "jogo": "NAPOLI (DENNIS) X AC MILAN (KLAUS)"},{"id": "175474997", "mercado": "double_chance", "time": "INTER (VENDETTA)", "odd": 1.5, "jogo": "INTER (VENDETTA) X AC MILAN (KLAUS)"},{"id": "175475023", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.700", "jogo": "JUVENTUS (DEMPSEY) X AC MILAN (KLAUS)"},{"id": "175475029", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.150", "jogo": "LAZIO (JACK) X NAPOLI (DENNIS)"},{"id": "175475035", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.950", "jogo": "ARSENAL (DEMPSEY) X INTER (JACK)"},{"id": "175475044", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.100", "jogo": "ARSENAL (DEMPSEY) X REAL MADRID (VENDETTA)"},{"id": "175475041", "mercado": "double_chance", "time": "BAYERN (KLAUS)", "odd": 1.55, "jogo": "BAYERN (KLAUS) X INTER (JACK)"},{"id": "175475066", "mercado": "handicap", "time": "REAL MADRID (VENDETTA)", "linha": "0.0 , -0.5", "odd": 1.875, "jogo": "INTER (JACK) X REAL MADRID (VENDETTA)"},{"id": "175475077", "mercado": "goal_line", "tipo": "under", "linha": "2.5", "odd": 1.925, "jogo": "PSG (DENNIS) X ARSENAL (DEMPSEY)"},{"id": "175475072", "mercado": "handicap", "time": "REAL MADRID (VENDETTA)", "linha": "0.0", "odd": 1.875, "jogo": "BAYERN (KLAUS) X REAL MADRID (VENDETTA)"},{"id": "175475097", "mercado": "double_chance", "time": "PSG (DENNIS)", "odd": 1.533, "jogo": "PSG (DENNIS) X INTER (JACK)"},{"id": "175475084", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.900", "jogo": "BAYERN (KLAUS) X ARSENAL (DEMPSEY)"},{"id": "175475105", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.750", "jogo": "INTER (JACK) X ARSENAL (DEMPSEY)"},{"id": "175482966", "mercado": "goal_line", "tipo": "over", "linha": "4.5 , 5.0", "odd": 1.8, "jogo": "NAPOLI (NIKKITTA) X FIORENTINA (CL1VLIND)"},{"id": "175482986", "mercado": "double_chance", "time": "NAPOLI (NIKKITTA)", "odd": 1.533, "jogo": "BOLOGNA (SENIOR) X NAPOLI (NIKKITTA)"},{"id": "175475136", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.250", "jogo": "ARSENAL (DEMPSEY) X PSG (DENNIS)"},{"id": "175475145", "mercado": "double_chance", "time": "PSG (DENNIS)", "odd": 1.727, "jogo": "INTER (JACK) X PSG (DENNIS)"},{"id": "175475141", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.700", "jogo": "ARSENAL (DEMPSEY) X BAYERN (KLAUS)"},{"id": "175475160", "mercado": "handicap", "time": "PAOK (DENNIS)", "linha": "0.0", "odd": 1.85, "jogo": "TOTTENHAM (VENDETTA) X PAOK (DENNIS)"},{"id": "175475150", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.750", "jogo": "LAZIO (DEMPSEY) X A.BILBAO (JACK)"},{"id": "175475171", "mercado": "handicap", "time": "TOTTENHAM (VENDETTA)", "linha": "0.0", "odd": 1.875, "jogo": "LAZIO (DEMPSEY) X TOTTENHAM (VENDETTA)"},{"id": "175483038", "mercado": "goal_line", "tipo": "over", "linha": "4.0", "odd": 1.825, "jogo": "NAPOLI (NIKKITTA) X BOLOGNA (SENIOR)"},{"id": "175475174", "mercado": "goal_line", "tipo": "under", "linha": "3.0", "odd": 1.9, "jogo": "LYON (KLAUS) X PAOK (DENNIS)"},{"id": "175475176", "mercado": "handicap", "time": "TOTTENHAM (VENDETTA)", "linha": "0.0 , -0.5", "odd": 1.95, "jogo": "A.BILBAO (JACK) X TOTTENHAM (VENDETTA)"},{"id": "175483068", "mercado": "goal_line", "tipo": "under", "linha": "5.0", "odd": 1.925, "jogo": "BOLOGNA (SENIOR) X JUVENTUS (ARCOS)"},{"id": "175475184", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.050", "jogo": "PAOK (DENNIS) X LAZIO (DEMPSEY)"},{"id": "175475191", "mercado": "double_chance", "time": "LAZIO (DEMPSEY)", "odd": 1.615, "jogo": "LYON (KLAUS) X LAZIO (DEMPSEY)"},{"id": "175475203", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.727", "jogo": "A.BILBAO (JACK) X LAZIO (DEMPSEY)"},{"id": "175475208", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.300", "jogo": "PAOK (DENNIS) X TOTTENHAM (VENDETTA)"},{"id": "175475218", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.100", "jogo": "TOTTENHAM (VENDETTA) X LAZIO (DEMPSEY)"},{"id": "175475241", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.250", "jogo": "TOTTENHAM (VENDETTA) X A.BILBAO (JACK)"},{"id": "175475230", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "2.250", "jogo": "PAOK (DENNIS) X LYON (KLAUS)"},{"id": "175475247", "mercado": "handicap", "time": "LYON (KLAUS)", "linha": "0.0 , 0.5", "odd": 1.85, "jogo": "TOTTENHAM (VENDETTA) X LYON (KLAUS)"},{"id": "175475252", "mercado": "handicap", "time": "PAOK (DENNIS)", "linha": "0.0", "odd": 1.85, "jogo": "LAZIO (DEMPSEY) X PAOK (DENNIS)"},{"id": "175475256", "mercado": "double_chance", "time": "LAZIO (DEMPSEY)", "odd": 1.571, "jogo": "LAZIO (DEMPSEY) X LYON (KLAUS)"},{"id": "175494080", "mercado": "goal_line", "tipo": "over", "linha": "3.0 , 3.5", "odd": 1.875, "jogo": "ROMA (SHELBY) X OLYMPIAKOS (ARTHUR)"},{"id": "175494086", "mercado": "goal_line", "tipo": "under", "linha": "4.5", "odd": 1.9, "jogo": "AJAX (THOR) X REAL SOCIEDAD (PRINCE)"},{"id": "175494098", "mercado": "over_under", "tipo": "under", "linha": "2.5", "odd": "1.800", "jogo": "OLYMPIAKOS (ARTHUR) X MAN UTD (PROFESSOR)"},{"id": "175494101", "mercado": "goal_line", "tipo": "over", "linha": "4.0 , 4.5", "odd": 1.925, "jogo": "AJAX (THOR) X MAN UTD (PROFESSOR)"},{"id": "175494134", "mercado": "handicap", "time": "MAN UTD (PROFESSOR)", "linha": "0.0", "odd": 1.925, "jogo": "ROMA (SHELBY) X MAN UTD (PROFESSOR)"},{"id": "175494139", "mercado": "handicap", "time": "REAL SOCIEDAD (PRINCE)", "linha": "0.0", "odd": 1.9, "jogo": "ROMA (SHELBY) X REAL SOCIEDAD (PRINCE)"},{"id": "175494137", "mercado": "double_chance", "time": "OLYMPIAKOS (ARTHUR)", "odd": 1.55, "jogo": "OLYMPIAKOS (ARTHUR) X AJAX (THOR)"},{"id": "175494143", "mercado": "goal_line", "tipo": "under", "linha": "2.5", "odd": 1.925, "jogo": "MAN UTD (PROFESSOR) X OLYMPIAKOS (ARTHUR)"},{"id": "175494141", "mercado": "goal_line", "tipo": "over", "linha": "4.0 , 4.5", "odd": 1.85, "jogo": "ROMA (SHELBY) X AJAX (THOR)"},{"id": "175494145", "mercado": "goal_line", "tipo": "over", "linha": "4.0 , 4.5", "odd": 1.925, "jogo": "MAN UTD (PROFESSOR) X AJAX (THOR)"},{"id": "175506216", "mercado": "handicap", "time": "FENERBAHCE (WBOY)", "linha": "0.0 , -0.5", "odd": 1.9, "jogo": "TOTTENHAM (BOMB1TO) X FENERBAHCE (WBOY)"},{"id": "175494150", "mercado": "handicap", "time": "INTER (PRINCE)", "linha": "0.0", "odd": 1.925, "jogo": "ARSENAL (PROFESSOR) X INTER (PRINCE)"},{"id": "175494154", "mercado": "goal_line", "tipo": "over", "linha": "3.0 , 3.5", "odd": 1.875, "jogo": "REAL MADRID (SHELBY) X PSG (ARTHUR)"},{"id": "175494157", "mercado": "handicap", "time": "INTER (PRINCE)", "linha": "-0.5 , -1.0", "odd": 1.925, "jogo": "BAYERN (THOR) X INTER (PRINCE)"},{"id": "175494177", "mercado": "goal_line", "tipo": "under", "linha": "5.0", "odd": 1.85, "jogo": "INTER (PRINCE) X REAL MADRID (SHELBY)"},{"id": "175524962", "mercado": "goal_line", "tipo": "under", "linha": "3.5", "odd": 1.825, "jogo": "LAZIO (MADISSON) X TOTTENHAM (SENSEI)"},{"id": "175494182", "mercado": "goal_line", "tipo": "over", "linha": "2.5 , 3.0", "odd": 1.925, "jogo": "PSG (ARTHUR) X ARSENAL (PROFESSOR)"},{"id": "175524710", "mercado": "handicap", "time": "PSG (GLUMAC)", "linha": "-0.5", "odd": 1.9, "jogo": "REAL MADRID (GIOX) X PSG (GLUMAC)"},{"id": "175524974", "mercado": "goal_line", "tipo": "over", "linha": "4.0 , 4.5", "odd": 1.9, "jogo": "A.BILBAO (ZANGIEF) X TOTTENHAM (SENSEI)"},{"id": "175524720", "mercado": "goal_line", "tipo": "under", "linha": "5.5 , 6.0", "odd": 1.8, "jogo": "REAL MADRID (GIOX) X MAN CITY (MASLJA)"},{"id": "175524724", "mercado": "goal_line", "tipo": "under", "linha": "5.5 , 6.0", "odd": 1.8, "jogo": "RIVER PLATE (PECONI) X PSG (GLUMAC)"},{"id": "175494192", "mercado": "goal_line", "tipo": "over", "linha": "3.0 , 3.5", "odd": 1.875, "jogo": "PSG (ARTHUR) X REAL MADRID (SHELBY)"},{"id": "175525035", "mercado": "goal_line", "tipo": "over", "linha": "3.0 , 3.5", "odd": 1.9, "jogo": "A.BILBAO (ZANGIEF) X LAZIO (MADISSON)"},{"id": "175524734", "mercado": "goal_line", "tipo": "over", "linha": "5.0", "odd": 1.9, "jogo": "RIVER PLATE (PECONI) X MAN CITY (MASLJA)"}]




#!pega uma vez ao virar o dia e depois de 20 em 20 min pra testar
data_hoje = datetime.now().date().strftime('%Y%m%d')
#!reseta todo dia as 00:00, guarda jogos programados do dia, deve zerar ao mudar o dia
programado = []

def checa_virada_do_dia():
    global data_hoje, programado
    while True:
        novo_dia = datetime.now().date().strftime('%Y%m%d')
        if novo_dia != data_hoje:
            data_hoje = novo_dia
            programado = []
            logger.info("🔄 Novo dia detectado, resetando variáveis...")
        time_module.sleep(60)


def agendar_processar_dia_anterior():
    agora = datetime.now()
    alvo = datetime.combine(agora.date(), datetime.min.time()) + timedelta(hours=0, minutes=5)

    if agora >= alvo:
        alvo += timedelta(days=1)

    delay = (alvo - agora).total_seconds()
    logger.info(f"⏰ Agendando processamento do dia anterior para {alvo}")
    threading.Timer(delay, processar_dia_anterior).start()

def incremental_learning():
    NN.atua()
    df = pd.read_csv('df_temp_preprocessado_teste.csv')
    hoje = datetime.today()
    dia_atual = hoje.strftime('%Y%m%d')
    dia_anterior = (hoje - timedelta(days=1)).strftime('%Y%m%d')

    # Filtra o DataFrame para incluir apenas essas duas datas
    dois_dias_recentes = df[df['event_day'].isin([dia_atual, dia_anterior])]
    
    
    dc = qlearning.QLearningDoubleChance()
    dc.load_model('q_learning_dc_model_final.pkl')
    dc.alpha = 0.01
    dc.gamma = 0.85
    dc.epsilon = 0.01
    dc.train(dois_dias_recentes,num_episodes=50)
    dc.save_model('q_learning_dc_model_final.pkl')

    gl = qlearning.QLearningGoalLine()
    gl.load_model('q_learning_gl_model_final.pkl')
    gl.alpha = 0.01
    gl.gamma = 0.85
    gl.epsilon = 0.01
    gl.train(dois_dias_recentes,num_episodes=50)
    gl.save_model('q_learning_gl_model_final.pkl')

    h = qlearning.QLearningHandicap()
    h.load_model('q_learning_h_model_final.pkl')
    h.alpha = 0.01
    h.gamma = 0.85
    h.epsilon = 0.01
    h.train(dois_dias_recentes,num_episodes=50)
    h.save_model('q_learning_h_model_final.pkl')

    dnb = qlearning.QLearningDrawNoBet()
    dnb.load_model('q_learning_dnb_model_final.pkl')
    dnb.alpha = 0.05
    dnb.gamma = 0.85
    dnb.epsilon = 0.01
    dnb.train(dois_dias_recentes,num_episodes=50)
    dnb.save_model('q_learning_dnb_model_final.pkl')
    
def agendar_treino_incremental():
    agora = datetime.now()
    alvo = datetime.combine(agora.date(), datetime.min.time()) + timedelta(hours=0, minutes=10)

    if agora >= alvo:
        alvo += timedelta(days=1)

    delay = (alvo - agora).total_seconds()

    def tarefa():
        logger.info("🧠 Iniciando treino incremental de ql...")
        incremental_learning()
        logger.info("✅ Modelos de ql treinados com sucesso")

    threading.Timer(delay, tarefa).start()

def agendar_criacao_nns():
    agora = datetime.now()
    alvo = datetime.combine(agora.date(), datetime.min.time()) + timedelta(hours=0, minutes=15)

    if agora >= alvo:
        alvo += timedelta(days=1)

    delay = (alvo - agora).total_seconds()
    logger.info(f"⏰ Agendando criação de NNs para {alvo}")

    def tarefa():
        logger.info("🧠 Iniciando criação de modelos de rede neural...")
        criaTodasNNs()
        logger.info("✅ Modelos de rede neural criados com sucesso")
        print(lista_th)

    threading.Timer(delay, tarefa).start()


def agendar_verificacao_diaria():
    agora = datetime.now()
    
    alvo = datetime.combine(agora.date(), time(0, 30))
    
    if agora >= alvo:
        alvo += timedelta(days=1)
    
    delay = (alvo - agora).total_seconds()
    
    logger.info(f"⏰ Agendando verificação diária para {alvo.strftime('%d/%m/%Y %H:%M')}")

    def tarefa():
        logger.info("🔍 Iniciando verificação diária de apostas...")
        try:
            global list_checa
            checa()  # Executa a verificação
            list_checa = []
        except Exception as e:
            logger.error(f"❌ Erro na verificação diária: {e}")
        # Reagenda para o próximo dia
        agendar_verificacao_diaria()

    threading.Timer(delay, tarefa).start()

def verificar_aposta(aposta, df_resultados):
    try:
        # Extrair dados da aposta
        id = str(aposta['id'])  # Convert to string to ensure consistent type
        jogo = aposta['jogo']
        linha = aposta.get('linha')
        tipo = aposta.get('tipo')
        time = aposta.get('time')
        mercado = aposta['mercado']
        
        # Log para debug
        print(f"Verificando: ID={id}, mercado={mercado}, time={time}, linha={linha}, tipo={tipo}")
        
        # Verificar se ID existe no DataFrame
        df_resultados['id'] = df_resultados['id'].astype(str)  # Ensure ID column is string
        resultado = df_resultados[df_resultados['id'] == id]
        
        if resultado.empty:
            print(f"ID não encontrado: {id}")
            return None
            
        if mercado == 'goal_line' and not tipo:
            print(f"Aposta ID={id}: tipo não fornecido para goal_line")
            return None

        if (mercado == 'handicap' or mercado == 'draw_no_bet') and not time:
            print(f"Aposta ID={id}: time não fornecido para {mercado}")
            return None
        
        home_time = jogo.split(' X ')[0].strip()
        away_time = jogo.split(' X ')[1].strip()

        row = resultado.iloc[0]

        if mercado == 'over_under':
            if tipo == 'over':
                tipo = 1.0
            else:
                tipo = 2.0
                
            tot_goals = float(row['tot_goals'])
            if pd.isna(tot_goals):
                print(f"Total de gols não disponível para ID={id}")
                return None
                
            if (tipo == 1.0) and (tot_goals > 2.5):
                return 1
            elif (tipo == 1.0) and (tot_goals < 2.5):
                return -1
            elif (tipo == 2.0) and (tot_goals > 2.5):
                return -1
            elif (tipo == 2.0) and (tot_goals < 2.5):
                return 1
            else:
                return 0  # Empate exato em 2.5

        elif mercado == 'goal_line':
            if tipo == 'over':
                tipo = 1.0
            else:
                tipo = 2.0

            # Convert to float and handle NaN values
            type_gl1 = float(row['type_gl1']) if not pd.isna(row['type_gl1']) else None
            type_gl2 = float(row['type_gl2']) if not pd.isna(row['type_gl2']) else None

            if type_gl1 == tipo:
                if row['gl1_positivo']:
                    return 1
                elif row['gl1_negativo']:
                    return -1
                elif row['gl1_reembolso']:
                    return 0
                elif row['gl1_meio_ganho']:
                    return 0.5
                elif row['gl1_meia_perda']:
                    return -0.5
                
            elif type_gl2 == tipo:
                if row['gl2_positivo']:
                    return 1
                elif row['gl2_negativo']:
                    return -1
                elif row['gl2_reembolso']:
                    return 0
                elif row['gl2_meio_ganho']:
                    return 0.5
                elif row['gl2_meia_perda']:
                    return -0.5
                
            print(f"Retornando None para ID={id} porque não encontrou goal_line correspondente")
            return None

        elif mercado == 'handicap':
            if time == home_time:
                time = 1.0
            else:
                time = 2.0

            # Convert to float and handle NaN values
            team_ah1 = float(row['team_ah1']) if not pd.isna(row['team_ah1']) else None
            team_ah2 = float(row['team_ah2']) if not pd.isna(row['team_ah2']) else None

            if team_ah1 == time:
                if row['ah1_positivo']:
                    return 1
                elif row['ah1_negativo']:
                    return -1
                elif row['ah1_reembolso']:
                    return 0
                elif row['ah1_meio_ganho']:
                    return 0.5
                elif row['ah1_meia_perda']:
                    return -0.5
                
            elif team_ah2 == time:
                if row['ah2_positivo']:
                    return 1
                elif row['ah2_negativo']:
                    return -1
                elif row['ah2_reembolso']:
                    return 0
                elif row['ah2_meio_ganho']:
                    return 0.5
                elif row['ah2_meia_perda']:
                    return -0.5
                
            print(f"Retornando None para ID={id} porque não encontrou handicap correspondente")
            return None

        elif mercado == 'draw_no_bet':
            if time == home_time:
                time = 1.0
            else:
                time = 2.0

            # Convert to float and handle NaN values
            draw_no_bet_team1 = float(row['draw_no_bet_team1']) if not pd.isna(row['draw_no_bet_team1']) else None
            draw_no_bet_team2 = float(row['draw_no_bet_team2']) if not pd.isna(row['draw_no_bet_team2']) else None

            if draw_no_bet_team1 == time:
                if row['dnb1_ganha']:
                    return 1
                elif row['dnb1_perde']:
                    return -1
                else:
                    return 0
                
            elif draw_no_bet_team2 == time:
                if row['dnb2_ganha']:
                    return 1
                elif row['dnb2_perde']:
                    return -1
                else:
                    return 0
            
            print(f"Retornando None para ID={id} porque não encontrou draw_no_bet correspondente")
            return None

        elif mercado == 'double_chance':
            if time == home_time:
                time = 1.0
            elif time == away_time:
                time = 2.0
            else:
                time = 3.0

            if time == 1.0:
                if row['res_double_chance1']:
                    return 1
                else:
                    return -1
            elif time == 2.0:
                if row['res_double_chance2']:
                    return 1
                else:
                    return -1
            elif time == 3.0:
                if row['res_double_chance3']:
                    return 1
                else:
                    return -1
            else:
                print(f"Retornando None para ID={id} porque não encontrou double_chance correspondente")
                return None

        print(f"Retornando None para ID={id} porque mercado não reconhecido: {mercado}")
        return None
    except Exception as e:
        print(f"Erro ao verificar aposta: {e}")
        return None

def jogos_do_dia():
    # Obter os dados para o dia anterior e para o dia atual
    ids_anterior, dicio_anterior = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia_anterior())
    ids_atual, dicio_atual = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=data_hoje)  # Adicionando o dia atual
    
    # Filtrar odds para os jogos do dia anterior e do dia atual
    odds_anterior = apiclient.filtraOddsNovo(ids_anterior)
    odds_atual = apiclient.filtraOddsNovo(ids_atual)
    
    # Transformar os dados de odds
    df_odds_anterior = apiclient.transform_betting_data(odds_anterior)
    df_odds_atual = apiclient.transform_betting_data(odds_atual)
    
    novos_dados = []  # ✅ declarar a lista aqui
    
    # Juntar dados do evento para o dia anterior
    for dados_evento in dicio_anterior:
        event_id = dados_evento.get('id')
        odds_transformadas = df_odds_anterior[df_odds_anterior['id'] == event_id].to_dict('records')
        
        if odds_transformadas:
            merged = {**dados_evento, **odds_transformadas[0], "event_day": dia_anterior()}  # Usando dia anterior
        else:
            merged = {**dados_evento, "event_day": dia_anterior()}  # Usando dia anterior
        
        novos_dados.append(merged)
    
    # Juntar dados do evento para o dia atual
    for dados_evento in dicio_atual:
        event_id = dados_evento.get('id')
        odds_transformadas = df_odds_atual[df_odds_atual['id'] == event_id].to_dict('records')
        
        if odds_transformadas:
            merged = {**dados_evento, **odds_transformadas[0], "event_day": data_hoje}  # Usando dia atual
        else:
            merged = {**dados_evento, "event_day": data_hoje}  # Usando dia atual
        
        novos_dados.append(merged)
    
    # Criando o DataFrame com todos os dados
    df_dados = pd.DataFrame(novos_dados)
    df = df_dados.copy()
    
    # Pré-processamento dos dados
    df = NN.preProcessEstatisticasGerais(df.copy())
    df = NN.preProcessOverUnder(df.copy())
    df = NN.preProcessHandicap_i(df.copy())
    df = NN.preProcessGoalLine_i(df.copy())
    df = NN.preProcessDoubleChance(df.copy())
    df = NN.preProcessDrawNoBet_i(df.copy())
    
    return df
def jogos_do_dia1():
    # Obter os dados para o dia anterior e para o dia atual
    ids_anterior, dicio_anterior = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day="20250424")
    ids_atual, dicio_atual = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day="20250425")  # Adicionando o dia atual
    
    # Filtrar odds para os jogos do dia anterior e do dia atual
    odds_anterior = apiclient.filtraOddsNovo(ids_anterior)
    odds_atual = apiclient.filtraOddsNovo(ids_atual)
    
    # Transformar os dados de odds
    df_odds_anterior = apiclient.transform_betting_data(odds_anterior)
    df_odds_atual = apiclient.transform_betting_data(odds_atual)
    
    novos_dados = []  # ✅ declarar a lista aqui
    
    # Juntar dados do evento para o dia anterior
    for dados_evento in dicio_anterior:
        event_id = dados_evento.get('id')
        odds_transformadas = df_odds_anterior[df_odds_anterior['id'] == event_id].to_dict('records')
        
        if odds_transformadas:
            merged = {**dados_evento, **odds_transformadas[0], "event_day": dia_anterior()}  # Usando dia anterior
        else:
            merged = {**dados_evento, "event_day": dia_anterior()}  # Usando dia anterior
        
        novos_dados.append(merged)
    
    # Juntar dados do evento para o dia atual
    for dados_evento in dicio_atual:
        event_id = dados_evento.get('id')
        odds_transformadas = df_odds_atual[df_odds_atual['id'] == event_id].to_dict('records')
        
        if odds_transformadas:
            merged = {**dados_evento, **odds_transformadas[0], "event_day": data_hoje}  # Usando dia atual
        else:
            merged = {**dados_evento, "event_day": data_hoje}  # Usando dia atual
        
        novos_dados.append(merged)
    
    # Criando o DataFrame com todos os dados
    df_dados = pd.DataFrame(novos_dados)
    df = df_dados.copy()
    
    # Pré-processamento dos dados
    df = NN.preProcessEstatisticasGerais(df.copy())
    df = NN.preProcessOverUnder(df.copy())
    df = NN.preProcessHandicap_i(df.copy())
    df = NN.preProcessGoalLine_i(df.copy())
    df = NN.preProcessDoubleChance(df.copy())
    df = NN.preProcessDrawNoBet_i(df.copy())
    
    return df





def checa():
    df_odds = jogos_do_dia()

    resultados_verificados = []
    contador_none = 0
    contador_validos = 0
    # Suponha que você tenha a seguinte lista de dicionários
    
    # Extraindo os valores da chave 'id' em uma nova lista
    # Converte ambos para o mesmo tipo (por exemplo, string)
    df_odds['id'] = df_odds['id'].astype(str)  # Se o id no CSV for string
    ids = [str(dicionario["id"]) for dicionario in list_checa]  # Se a lista de ids for inteira


    df_filtrado = df_odds[df_odds['id'].isin(ids)]


    for aposta in list_checa:
        resultado = verificar_aposta(aposta, df_odds)
        
        # Verifica se o resultado é None/nulo
        if resultado is None:
            contador_none += 1
        else:
            contador_validos += 1
            
        resultados_verificados.append({
            **aposta,
            'resultado': resultado
        })
    
    df_verificacao = pd.DataFrame(resultados_verificados)

    # Conversão para garantir que o 'resultado' seja numérico, com None sendo preservado
    df_verificacao['resultado'] = pd.to_numeric(df_verificacao['resultado'], errors='coerce')

    # Suponha que cada aposta tenha uma coluna 'odd'
    df_verificacao['odd'] = df_verificacao.get('odd')

    # Cálculo das unidades
    df_verificacao['unidade'] = df_verificacao['resultado'].apply(
        lambda x: 1 if x == 1 else
                -1 if x == -1 else
                0.5 if x == 0.5 else
                -0.5 if x == -0.5 else
                0 if x == 0 else
                None
    )


    
    # Cálculo do lucro (None resulta em 0)
    df_verificacao['lucro'] = df_verificacao.apply(
        lambda row: (float(row['odd']) - 1) * row['unidade'] if row['unidade'] == 1 else
                    -1 if row['unidade'] == -1 else
                    (float(row['odd']) - 1) * row['unidade'] if row['unidade'] == 0.5 else
                    -0.5 if row['unidade'] == -0.5 else
                    0, 
        axis=1
    )


    # Estatísticas
    total_unidades = df_verificacao['lucro'].sum()
    total_apostas = len(df_verificacao)
    total_apostas_validas = contador_validos
    roi = (total_unidades / total_apostas_validas) * 100 if total_apostas_validas > 0 else 0
    percentual_none = (contador_none / total_apostas) * 100 if total_apostas > 0 else 0
    # Pega o dia anterior
    data_anterior = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    nome_arquivo = f"verificacao_diaria_{data_anterior}.txt"

    # Gera string de resumo para envio ao bot
    resumo_str = (
        f"📊 Estatísticas Detalhadas – {data_anterior}\n"
        f"✅ Total de Apostas: {total_apostas}\n"
        f"✅ Apostas Válidas: {total_apostas_validas}\n"
        f"❓ Apostas None/Nulas: {contador_none} ({percentual_none:.1f}%)\n"
        f"💰 Total de Unidades: {total_unidades:.2f}\n"
        f"📈 ROI (apenas válidas): {roi:.2f}%\n"
    )
    for chat in chats:
        tb.sendMessages(chat, resumo_str)
   
    with open(nome_arquivo, "a", encoding="utf-8") as f:
        f.write(f"\n📅 Verificação referente ao dia {data_anterior}\n")
        f.write(f"✅ Total de Apostas: {total_apostas}\n")
        f.write(f"✅ Apostas Válidas: {total_apostas_validas}\n")
        f.write(f"❓ Apostas None/Nulas: {contador_none} ({percentual_none:.1f}%)\n")
        f.write(f"💰 Total de Unidades: {total_unidades:.2f}\n")
        f.write(f"📈 ROI (apenas válidas): {roi:.2f}%\n")
        f.write("-" * 40 + "\n")

    return {
        'dataframe': df_verificacao,
        'total_unidades': total_unidades,
        'roi': roi,
        'apostas_total': total_apostas,
        'apostas_validas': total_apostas_validas,
        'apostas_none': contador_none,
        'percentual_none': percentual_none
    }




def loop_pega_jogos():
    while True:
        logger.info("🔎 Buscando jogos programados para hoje...")
        df_jogos = pegaJogosDoDia()
        if not df_jogos.empty:
            logger.info(f"📅 Encontrados {len(df_jogos)} jogos para hoje")
            pegaOddsEvento(df_jogos)
        else:
            logger.info("ℹ️ Nenhum jogo encontrado por agora")
        time_module.sleep(10 * 60)  # 10 minutos, ajustado para o intervalo correto

def atualizar_csv_dia_atual():
    COLUNAS_PADRAO = [
        'id', 'event_day', 'home', 'away','league', 'time','home_goals', 'away_goals', 'tot_goals',
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

    dia = datetime.now().strftime("%Y%m%d")
    logger.info(f"🔄 Atualizando jogos do dia {dia}")

    try:
        # Carregar dados existentes primeiro
        ids_existentes = set()
        if os.path.exists(CSV_FILE):
            df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
            ids_existentes = set(df_existente['id'].astype(str))
            logger.info(f"📊 Total de registros existentes: {len(ids_existentes)}")

        logger.info("🔎 Buscando IDs e dicionário de eventos...")
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
        logger.info(f"✔️ {len(ids)} eventos encontrados.")

        logger.info("📊 Filtrando e transformando odds...")
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        df_odds = apiclient.transform_betting_data(odds_data)

        novos_dados = []
        for dados_evento in dicio:
            event_id = str(dados_evento.get('id'))
            
            # Verificar se o ID já existe
            if event_id in ids_existentes:
                logger.info(f"⚠️ ID {event_id} já existe no CSV, pulando...")
                continue
                
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')

            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
            else:
                merged = {**dados_evento, "event_day": dia}

            novos_dados.append(merged)

        if not novos_dados:
            logger.info("⚠️ Nenhum dado novo para adicionar.")
            return

        df_novo = pd.DataFrame(novos_dados)
        logger.info(f"📝 {len(df_novo)} novos registros para adicionar")

        # Garantir que id é string
        df_novo['id'] = df_novo['id'].astype(str)

        colunas_adicionadas = []
        for coluna in COLUNAS_PADRAO:
            if coluna not in df_novo.columns:
                df_novo[coluna] = None
                colunas_adicionadas.append(coluna)

        if colunas_adicionadas:
            logger.info(f"➕ Colunas adicionadas automaticamente: {', '.join(colunas_adicionadas)}")

        df_novo = df_novo[COLUNAS_PADRAO]

        if os.path.exists(CSV_FILE):
            logger.info("📂 CSV existente encontrado, mesclando dados...")
            df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
            df_existente['id'] = df_existente['id'].astype(str)
            
            # Verificar duplicatas antes da concatenação
            duplicatas = set(df_novo['id']).intersection(set(df_existente['id']))
            if duplicatas:
                logger.warning(f"⚠️ Encontrados {len(duplicatas)} IDs que já existem no CSV")
                for dup in duplicatas:
                    logger.warning(f"ID duplicado: {dup}")
            
            # Filtrar apenas registros novos que não existem no CSV
            df_novo = df_novo[~df_novo['id'].isin(duplicatas)]
            logger.info(f"📝 Após remover duplicatas, {len(df_novo)} registros novos para adicionar")
            
            if len(df_novo) > 0:
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                df_final['id'] = df_final['id'].astype(str)
                df_final = df_final.drop_duplicates(subset=['id'], keep='last')
                
                # Ordenar por data do evento
                df_final["time"] = df_final["time"].astype(int)
                df_final = df_final.sort_values(by="time", ascending=False).reset_index(drop=True)
                
                # Verificar duplicatas finais
                duplicatas_finais = df_final[df_final.duplicated(subset=['id'], keep=False)]
                if not duplicatas_finais.empty:
                    logger.warning(f"⚠️ Ainda existem {len(duplicatas_finais)} duplicatas após a concatenação")
                    for id_dup in duplicatas_finais['id'].unique():
                        logger.warning(f"ID duplicado final: {id_dup}")
                
                df_final.to_csv(CSV_FILE, index=False)
                logger.info(f"✅ CSV atualizado com {len(df_novo)} eventos adicionados")
            else:
                logger.info("ℹ️ Nenhum registro novo para adicionar após remoção de duplicatas")
        else:
            logger.info("📄 Nenhum CSV encontrado, criando novo arquivo...")
            df_novo.to_csv(CSV_FILE, index=False)
            logger.info(f"✅ Novo CSV criado com {len(df_novo)} eventos")
        
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar dados do dia {dia}: {type(e).__name__}: {e}")
        raise  # Re-lança a exceção para ser tratada pelo chamador

def remover_duplicatas():
    global CSV_FILE
    try:
        logger.info("🔍 Iniciando remoção de duplicatas...")
        df_final = pd.read_csv(CSV_FILE)
        total_antes = len(df_final)
        
        # Garantir que id é string
        df_final['id'] = df_final['id'].astype(str)
        
        # Log dos IDs duplicados antes da remoção
        duplicados = df_final[df_final.duplicated(subset=['id'], keep=False)]
        if not duplicados.empty:
            logger.info(f"⚠️ Encontrados {len(duplicados)} registros duplicados:")
            for id_dup in duplicados['id'].unique():
                logger.info(f"ID duplicado: {id_dup}")
        
        # Remover duplicatas mantendo o mais recente
        df_final = df_final.drop_duplicates(subset=['id'], keep='last')
        
        total_depois = len(df_final)
        removidos = total_antes - total_depois
        
        if removidos > 0:
            logger.info(f"✅ Removidas {removidos} duplicatas")
        
        df_final.to_csv(CSV_FILE, index=False)
        logger.info("✅ CSV atualizado após remoção de duplicatas")
    except Exception as e:
        logger.error(f"❌ Erro ao remover duplicatas: {str(e)}")

def agendar_atualizacao_csv():
    logger.info("🔄 Agendando atualização do CSV...")
    atualizar_csv_dia_atual()
    # Reagenda para daqui 30 minutos
    threading.Timer(1800, agendar_atualizacao_csv).start()

def main():
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting background worker...")
    
    # Start the day check thread
    threading.Thread(target=checa_virada_do_dia, daemon=True).start()
    
    # Start the continuous loop for fetching games
    threading.Thread(target=loop_pega_jogos, daemon=True).start() 
    
    # Schedule daily tasks
    agendar_processar_dia_anterior()
 
    agendar_treino_incremental()
    agendar_verificacao_diaria()
    agendar_atualizacao_csv()

    # Start the main loop to keep the program alive and running
    while True:
        time_module.sleep(60)  # Sleep to keep the program running and allow threads to work

def pegaJogosDoDia():
    try:
        dias_para_buscar = [str(data_hoje)]
        if datetime.now().hour >= 20:
            dia_seg = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
            dias_para_buscar.append(dia_seg)
        logger.info(f"📅 Dias para buscar: {dias_para_buscar}")

        ids, tempo, nome_time, times_id, league_durations = [], [], [], [], []
        for dia in dias_para_buscar:
            logger.info(f"🔍 Buscando jogos para o dia {dia}")
            r_ids, r_tempo, r_nome_time, r_times_id, r_league_durations = apiclient.getUpcoming(leagues=apiclient.leagues_ids, day=dia)
            logger.info(f"📊 Jogos encontrados: {len(r_ids)}")
            ids.extend(r_ids)
            tempo.extend(r_tempo)
            nome_time.extend(r_nome_time)
            times_id.extend(r_times_id)
            league_durations.extend(r_league_durations)

        if not ids:
            logger.warning("⚠️ Nenhum ID de jogo retornado pela API")
            return pd.DataFrame()

        dados = [{
            "id_jogo": i,
            "horario": h,
            "times": k,
            "home": z,
            "away": t,
            "league_duration": d
        } for i, h, k, (z, t), d in zip(ids, tempo, nome_time, times_id, league_durations)]

        dados_dataframe = pd.DataFrame(dados)
        logger.info(f"📋 Total de jogos antes da filtragem: {len(dados_dataframe)}")
        print(dados_dataframe)
        
        dados_dataframe = dados_dataframe[~dados_dataframe['id_jogo'].isin(programado)]
        logger.info(f"📋 Jogos após remover programados: {len(dados_dataframe)}")

        if dados_dataframe.empty:
            logger.info("ℹ️ Todos os jogos já estão programados")
            return dados_dataframe

        agora = int(time_module.time())
        dados_dataframe['horario'] = dados_dataframe['horario'].astype(int)
        dados_dataframe['send_time'] = dados_dataframe['horario'] - 320
        logger.info(f"⏰ Tempo atual: {agora}")
        print(dados_dataframe)
        logger.info(f"⏰ Primeiro horário de jogo: {dados_dataframe['horario'].min()}")
        logger.info(f"⏰ Primeiro send_time: {dados_dataframe['send_time'].min()}")
        
        dados_dataframe = dados_dataframe[dados_dataframe['send_time'] > (agora - (7 * 60))]
        logger.info(f"📋 Jogos após filtragem por horário: {len(dados_dataframe)}")
        
        dados_dataframe = dados_dataframe.sort_values(by="horario").reset_index(drop=True)
        print(dados_dataframe)

        programados = dados_dataframe['id_jogo'].tolist()
        programado.extend(programados)
        logger.info(f"📌 Adicionados {len(programados)} novos jogos à lista de programados")
        return dados_dataframe

    except Exception as e:
        logger.error(f"❌ Erro ao obter jogos do dia: {str(e)}")
        return pd.DataFrame()




#!roda apos pegajogosDoDia, mas cada acao do jogo sera executada em seu tempo send_timer
def pegaOddsEvento(df):
    agora = time_module.time()  # timestamp atual em segundos
    logger.info(f"⏳ Agendando {len(df)} eventos...")

    for _, row in df.iterrows():
        delay = row['send_time'] - agora  # tempo até a ação acontecer
        delay = max(0, delay)  # evita delays negativos

        threading.Timer(delay, acao_do_jogo, args=(row,)).start()
        threading.Timer(delay + 1300, checa_jogos_do_dia, args=(row['id_jogo'],)).start()
        print(f"Agendado jogo {row['id_jogo']} para {datetime.fromtimestamp(row['send_time'])}")


def checa_jogos_do_dia(id,tentativa=0):
    global list_checa
    df = pd.read_csv(CSV_FILE)
    
    # Garante que 'event_day' é string
    df["event_day"] = df["event_day"].astype(str)

    # Identifica os dois dias mais recentes
    ultimos_dois_dias = sorted(df["event_day"].unique())[-2:]

    # Filtra o DataFrame
    df_apenas_dois_dias = df[df["event_day"].isin(ultimos_dois_dias)]

    # Pré-processamento dos dados
    df_apenas_dois_dias = NN.preProcessEstatisticasGerais(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessOverUnder(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessHandicap_i(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessGoalLine_i(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessDoubleChance(df_apenas_dois_dias.copy())
    df_apenas_dois_dias = NN.preProcessDrawNoBet_i(df_apenas_dois_dias.copy())
    

    for a in list_checa:
        if a['id'] == id:
            res = verificar_aposta(a,df_apenas_dois_dias)
            if res is not None:
                if res == 1:
                    a['resultado'] = 'ganhou'
                    
                elif res == 0.5:
                    a['resultado'] = 'meio ganho'
                    
                elif res == -0.5:
                    a['resultado'] = 'meio ganho'
                elif res == -1:
                    a['resultado'] = 'perdeu'
                else:
                    a['resultado'] = 'empate'
                a = pd.DataFrame([a])
                a.drop(columns=['id'], inplace=True)
                
                mens = df_para_string(a)
                for chat in chats:
                    tb.sendMessages(chat, mens)
            else:
                if tentativa  < 2:
                    time_module.sleep(2000)
                    checa_jogos_do_dia(id,tentativa+1)
                else:
                    logger.info(f"❌ Jogo {id} não retornou resultado após 2 tentativas")
                    return 0

# Função que será executada para cada jogo
def acao_do_jogo(row):
    try:
        global list_checa
        logger.info(f"⚽ Processando jogo {row['id_jogo']}")
        odds = apiclient.filtraOddsNovo([row['id_jogo']])
        if not odds:
            logger.warning(f"⚠️ Nenhuma odd encontrada para o jogo {row['id_jogo']}")
            return 0
        df_odds = apiclient.transform_betting_data(odds)
        
        df_odds['home'] = int(row['home'])
        df_odds['away'] = int(row['away'])
        df_odds['times'] = str(row['times'])
        df_odds['league'] = int(row['league_duration'])
        df_odds['horario'] = row['horario']  # Add game time to df_odds
        
        id = row['id_jogo']
        df_odds = NN.preProcessGeneral_x(df_odds)
        lista_bets_a_enviar, listas_para_checar = preve(df_odds, id)
        with open("checar_bets.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- NOVO LOTE ({datetime.now().isoformat()}) ---\n")
            for aposta in listas_para_checar:
                list_checa.append(aposta)
                print(list_checa)
                f.write(json.dumps(aposta, ensure_ascii=False) + "\n")


        if lista_bets_a_enviar:
            logger.info(f"📩 Enviando {len(lista_bets_a_enviar)} previsões para o Telegram")
            for bet in lista_bets_a_enviar:
                for id in chats:
                    tb.sendMessages(id, bet)
        else:
            logger.info("ℹ️ Nenhuma aposta recomendada para este jogo")

    except Exception as e:
        logger.error(f"❌ Erro ao processar jogo {row['id_jogo']}: {str(e)}")
        return 0

#! roda todo dia as 00:15
def criaTodasNNs():
    global lista_th 
    logger.info("🔧 Criando todos os modelos de rede neural...")
    lista_th = NN.criaNNs()
    logger.info(f"📊 Thresholds definidos: {lista_th}")

def preve(df_linha, id):
    logger.info("🔮 Fazendo previsões para o jogo atual...")
    try:
        if not lista_th:
            logger.warning("⚠️ Thresholds de modelos ainda não definidos")
            return []
        prepOverUnder, dados_OU = NN.prepNNOver_under_X(df_linha)

        if prepOverUnder is None:
            tipo_over_under, res_under_over = None, None
        else:
            tipo_over_under, res_under_over = predicta_over_under(prepOverUnder, dados_OU)
        
        prepHandicap, dados_ah = NN.prepNNHandicap_X(df_linha)
        prepHandicap_conj = NN.prepNNHandicap_X_conj(df_linha)

        if prepHandicap is None or prepHandicap_conj is None:
            time_handicap, res_handicap = None, None
        else:
            time_handicap, res_handicap = predicta_handicap(prepHandicap,prepHandicap_conj, dados_ah)
        
        prepGoal_line, dados_gl = NN.prepNNGoal_line_X(df_linha)
        prepGoal_line_conj = NN.prepNNGoal_line_X_conj(df_linha)

        if prepGoal_line is None:
            linha_gl, res_goal_line = None, None
        else:
            linha_gl, res_goal_line = predicta_goal_line(prepGoal_line, prepGoal_line_conj, dados_gl)
        
        prepDouble_chance, dados_dc = NN.prepNNDouble_chance_X(df_linha)
        prepDouble_chance_conj = NN.prepNNDouble_chance_X_conj(df_linha)

        if prepDouble_chance is None:
            type_dc, res_double_chance = None, None
        else:
            type_dc, res_double_chance = predicta_double_chance(prepDouble_chance,prepDouble_chance_conj, dados_dc)
        
        prepDraw_no_bet, dados_dnb = NN.prepNNDraw_no_bet_X(df_linha)
        prepDraw_no_bet_conj = NN.prepNNDraw_no_bet_X_conj(df_linha)

        if prepDraw_no_bet is None:
            time_draw_no_bet, res_draw_no_bet = None, None
        else:
            time_draw_no_bet, res_draw_no_bet = predicta_draw_no_bet(prepDraw_no_bet,prepDraw_no_bet_conj, dados_dnb)

        lista_preds_true = [tipo_over_under, res_under_over, time_handicap, res_handicap, linha_gl, res_goal_line, type_dc, res_double_chance,time_draw_no_bet, res_draw_no_bet]
        
        logger.info(f"🧠 Predições retornadas: {lista_preds_true}")

        list_res = []
        
        if res_under_over:
            list_res.append(res_under_over)
        if res_handicap:
            list_res.append(res_handicap)
        if res_goal_line:
            list_res.append(res_goal_line)
        if res_double_chance:
            list_res.append(res_double_chance)
        if res_draw_no_bet:
            list_res.append(res_draw_no_bet)
        print(list_res)
        melhor = None

        try:
            melhor = max(list_res)
            print(melhor)
        except:
            logger.info("❌ Nenhuma previsão foi considerada válida.")

        #[(0) - over, (1) - under, (2) - handicap home, (3) - handicap away, (4) - over goal_line, (5) - under goal_line, (6) dc1, dc2, dc3, dnb1, dnb2]
        list_true = []
        list_final = []
        list_check = []
        if melhor:
            if (tipo_over_under is not None and (res_under_over == melhor)):
                dados_OU['🔔 Jogo'] = times_para_jogo(str(dados_OU['times'].iloc[0]))
                dados_OU.drop('times', axis=1, inplace=True)
                dados_OU['📊 Tipo'] = tipo_over_under
                if tipo_over_under == 'over':
                    dados_OU['⭐ Odd'] = dados_OU['odd_goals_over1']
                else:
                    dados_OU['⭐ Odd'] = dados_OU['odd_goals_under1']
                dados_OU.drop(columns=['odd_goals_over1','odd_goals_under1'], inplace=True)
                dados_OU['⚽ Linha'] = '2.5'
                dados_OU['⏰ Horário'] = datetime.fromtimestamp(df_linha['horario'].iloc[0]).strftime('%H:%M')
                dados_OU = dados_OU[['🔔 Jogo','⏰ Horário', '📊 Tipo', '⭐ Odd', '⚽ Linha']]
                list_true.append(dados_OU)
                list_check.append({
                    'id': id,
                    'mercado': 'over_under',
                    'tipo': tipo_over_under,
                    'linha': '2.5',
                    'odd': dados_OU['⭐ Odd'].iloc[0],
                    'jogo': dados_OU['🔔 Jogo'].iloc[0]
                })
                

            if (time_handicap is not None and (res_handicap == melhor)):
                if (time_handicap== 1):
                    dados_temp = dados_ah.iloc[[0]].copy()
                elif (time_handicap == 2):
                    dados_temp = dados_ah.iloc[[1]].copy()
                dados_temp['🔔 Jogo'] = times_para_jogo(str(dados_temp['times'].iloc[0]))

                ah1 = str(dados_temp['asian_handicap_1'].iloc[0])
                ah2 = str(dados_temp['asian_handicap_2'].iloc[0])
                if ah1 == ah2:
                    dados_temp['⚽ handicap'] = ah1
                else:
                    dados_temp['⚽ handicap'] = f"{ah1} , {ah2}"

                home, away = home_e_away(str(dados_temp['times'].iloc[0]))
                dados_temp.drop(columns=['asian_handicap_1', 'asian_handicap_2', 'times'], inplace=True)
                dados_temp = dados_temp.rename(columns={'team_ah': '🚀 time'})
                team_val = dados_temp['🚀 time'].iloc[0]
                dados_temp.at[dados_temp.index[0], '🚀 time'] = home if team_val == 1 else away
                dados_temp = dados_temp.rename(columns={'odds': '⭐ Odd'})
                dados_temp['⏰ Horário'] = datetime.fromtimestamp(df_linha['horario'].iloc[0]).strftime('%H:%M')
                dados_temp = dados_temp[['🔔 Jogo','⏰ Horário','🚀 time', '⭐ Odd', '⚽ handicap']]
                list_true.append(dados_temp)
                list_check.append({
                    'id': id,
                    'mercado': 'handicap',
                    'time': dados_temp['🚀 time'].iloc[0],
                    'linha': dados_temp['⚽ handicap'].iloc[0],
                    'odd': dados_temp['⭐ Odd'].iloc[0],
                    'jogo': dados_temp['🔔 Jogo'].iloc[0]
                })



            if (linha_gl is not None and (res_goal_line == melhor)):
                if (linha_gl == 1):
                    dados_temp = dados_gl.iloc[[0]].copy()
                elif (linha_gl == 2):
                    dados_temp = dados_gl.iloc[[1]].copy()

                dados_temp['🔔 Jogo'] = times_para_jogo(str(dados_temp['times'].iloc[0]))

                gl1 = str(dados_temp['goal_line_1'].iloc[0])
                gl2 = str(dados_temp['goal_line_2'].iloc[0])
                if gl1 == gl2:
                    dados_temp['⚽ Linha'] = f'{gl1}'
                else:
                    dados_temp['⚽ Linha'] = f"{gl1} , {gl2}"

                dados_temp = dados_temp.rename(columns={'type_gl': '📊 Tipo'})
                dados_temp = dados_temp.rename(columns={'odds_gl': '⭐ Odd'})
                tipo_valor = dados_temp['📊 Tipo'].iloc[0]
                dados_temp.at[dados_temp.index[0], '📊 Tipo'] = 'over' if tipo_valor == 1 else 'under'
                dados_temp.drop(columns=['goal_line_1', 'goal_line_2', 'times'], inplace=True)
                dados_temp['⏰ Horário'] = datetime.fromtimestamp(df_linha['horario'].iloc[0]).strftime('%H:%M')
                dados_temp = dados_temp[['🔔 Jogo','⏰ Horário','📊 Tipo','⭐ Odd','⚽ Linha']]
                list_true.append(dados_temp)
                list_check.append({
                    'id': id,
                    'mercado': 'goal_line',
                    'tipo': dados_temp['📊 Tipo'].iloc[0],
                    'linha': dados_temp['⚽ Linha'].iloc[0],
                    'odd': dados_temp['⭐ Odd'].iloc[0],
                    'jogo': dados_temp['🔔 Jogo'].iloc[0]
                })

            if (type_dc is not None and (res_double_chance == melhor)):
                if (type_dc == 1):
                    dados_temp = dados_dc.iloc[[0]].copy()
                elif (type_dc == 2):
                    dados_temp = dados_dc.iloc[[1]].copy()
                elif (type_dc == 3):
                    dados_temp = dados_dc.iloc[[2]].copy()

                dados_temp['🔔 Jogo'] = times_para_jogo(str(dados_temp['times'].iloc[0]))
                home, away = home_e_away(str(dados_temp['times'].iloc[0]))
                val_dc = dados_temp['double_chance'].iloc[0]
                if val_dc == 1:
                    dc = home
                elif val_dc == 2:
                    dc = away
                else:
                    dc = f'{home} e {away}'
                dados_temp.at[dados_temp.index[0], 'double_chance'] = dc
                dados_temp = dados_temp.rename(columns={'double_chance': '📊 Double Chance'})
                dados_temp = dados_temp.rename(columns={'odds': '⭐ Odd'})
                dados_temp.drop('times', axis=1, inplace=True)
                dados_temp['⏰ Horário'] = datetime.fromtimestamp(df_linha['horario'].iloc[0]).strftime('%H:%M')
                dados_temp = dados_temp[['🔔 Jogo','⏰ Horário', '📊 Double Chance', '⭐ Odd']]
                list_true.append(dados_temp)
                list_check.append({
                    'id': id,
                    'mercado': 'double_chance',
                    'time': dados_temp['📊 Double Chance'].iloc[0],
                    'odd': dados_temp['⭐ Odd'].iloc[0],
                    'jogo': dados_temp['🔔 Jogo'].iloc[0]
                })

            if (time_draw_no_bet is not None and (res_draw_no_bet == melhor)):
                if (time_draw_no_bet == 1):
                    dados_temp = dados_dnb.iloc[[0]].copy()
                elif (time_draw_no_bet == 2):
                    dados_temp = dados_dnb.iloc[[1]].copy()

                dados_temp['🔔 Jogo'] = times_para_jogo(str(dados_temp['times'].iloc[0]))
                home, away = home_e_away(str(dados_temp['times'].iloc[0]))
                team_val = dados_temp['draw_no_bet_team'].iloc[0]
                dados_temp.at[dados_temp.index[0], 'draw_no_bet_team'] = home if team_val == 1 else away
                dados_temp = dados_temp.rename(columns={'draw_no_bet_team': '📊 Draw No Bet'})
                dados_temp = dados_temp.rename(columns={'odds': '⭐ Odd'})
                dados_temp.drop('times', axis=1, inplace=True)
                dados_temp['⏰ Horário'] = datetime.fromtimestamp(df_linha['horario'].iloc[0]).strftime('%H:%M')
                dados_temp = dados_temp[['🔔 Jogo','⏰ Horário','📊 Draw No Bet', '⭐ Odd']]
                list_true.append(dados_temp)
                list_check.append({
                    'id': id,
                    'mercado': 'draw_no_bet',
                    'time': dados_temp['📊 Draw No Bet'].iloc[0],
                    'odd': dados_temp['⭐ Odd'].iloc[0],
                    'jogo': dados_temp['🔔 Jogo'].iloc[0]
                })
 
            for df in list_true:
                men = df_para_string(df)
                list_final.append(men)

        if list_final:
            logger.info("✅ Previsões recomendadas:")
            for recomendacao in list_final:
                logger.info(f"👉 {recomendacao}")
        else:
            logger.info("❌ Nenhuma previsão foi considerada válida.")

        return list_final, list_check
    except Exception as e:
        logger.error(f"❌ Erro durante a previsão: {str(e)}")
        return []
2

def times_para_jogo(times):
    #('arsenal','mai')
    c = times.find(',')
    time_a = times[2:c-1]
    time_b = times[c+3:-2]
    final = f'{time_a.upper()} X {time_b.upper()}'
    return final

def home_e_away(times):
    #('arsenal','mai')
    c = times.find(',')
    time_a = times[2:c-1].upper()
    time_b = times[c+3:-2].upper()
    
    return time_a, time_b

def df_para_string(df):
    
    mensagens = []

    for _, row in df.iterrows():
        msg = ""
        #msg = "🔎 *LINHA IDENTIFICADA:*\n\n"
        cont=0
        for col in df.columns:
            if cont == 0:
                msg += f"{col}: {row[col]}\n\n"
            else:
                msg += f"{col}: {row[col]}\n"
            cont+=1
        mensagens.append(msg.strip())

    return mensagens


def predicta_over_under(prepOverUnder_df, dados):
    model_over_under = tf.keras.models.load_model('model_binario_over_under.keras')
    preds = model_over_under.predict(prepOverUnder_df)

    pred_over = float(preds[0])
    preds = [pred_over]

    th_ve = 1.025  # Valor Esperado mínimo
    recomendacoes = []

    # Odds de over e under
    odd_over = float(dados['odd_goals_over1'])
    odd_under = float(dados['odd_goals_under1'])

    # Cálculo do Valor Esperado para over e under
    ve_over = pred_over * odd_over
    ve_under = (1 - pred_over) * odd_under

    logger.info(f"📊 Over/Under - Predição: {pred_over}, Odd Over: {odd_over}, Odd Under: {odd_under}")
    th_odd = 1.5
    # Verificar as condições para recomendação
    if (ve_over >= th_ve) and (pred_over >= lista_th[0]) and (odd_over >= th_odd):
        recomendacoes.append(('over', ve_over, pred_over, odd_over))
    if (ve_under >= th_ve) and (pred_over <= lista_th[1]) and (odd_under >= th_odd):
        recomendacoes.append(('under', ve_under, 1 - pred_over, odd_under))

    if recomendacoes:
        # Escolher a melhor opção com maior valor esperado
        melhor_opcao = max(recomendacoes, key=lambda x: x[2])
        logger.info(f"✅ {melhor_opcao[0]} recomendado (VE: {melhor_opcao[1]:.3f}, Prob: {melhor_opcao[2]:.3f}, Odd: {melhor_opcao[3]:.2f})")
        return (melhor_opcao[0], melhor_opcao[2])
    else:
        logger.info("❌ Nenhuma recomendação em Over/Under")
        return (None, None)



from autogluon.tabular import TabularPredictor

def predicta_handicap(prepHandicap_df,prepHandicap_df_conj, dados):
    predictor = TabularPredictor.load("autogluon_handicap_model")
    predictor_conj = TabularPredictor.load("autogluon_handicap_model_conj")

    # Garante que usaremos o melhor modelo
    best_model = predictor.model_best
    print(f"🔍 Modelo selecionado: {best_model}")
    best_model_conj = predictor_conj.model_best
    
    # Usa explicitamente o melhor modelo para previsão
    preds_proba = predictor.predict_proba(prepHandicap_df, model=best_model)
    pred_conj = predictor_conj.predict(prepHandicap_df_conj, model=best_model_conj).iloc[0]
    try:
        ql_h = qlearning.QLearningHandicap()
        ql_h.load_model('q_learning_h_model_final.pkl')
        first_row = prepHandicap_df_conj.iloc[0]
        estado = qlearning.q_learning_h(first_row)
        if estado in ql_h.q_table:
            pred_ql = ql_h.choose_action(estado, epsilon=0)
        else:
            print('Estado H nao encontrado')
            pred_ql = pred_conj

    except:
        print(f'PROBLEMAS COM QL GL')
        pred_ql = pred_conj

    pred_handicap_1 = float(preds_proba[0][1]) 
    pred_handicap_2 = float(preds_proba[1][1]) 

    preds = [pred_handicap_1, pred_handicap_2]

    th_ve = 1.0
    recomendacoes = []
    th_odd = 1.5
    for i in range(2):
        prob = preds[i]
        odd = float(dados['odds'].iloc[i])
        ve = prob * odd
        if (ve >= th_ve) and (prob >= lista_th[2]) and (odd >= th_odd):
            if prob > preds[1 - i]:
                if pred_conj == i and pred_conj == pred_ql:
                    recomendacoes.append((i + 1, ve, prob, odd))

    logger.info(f"📊 Handicap - Predições: {preds}, Odds: {dados['odds']}")
    logger.info(f"📊 Handicap - Pred: {pred_conj}")

    if recomendacoes:
        melhor_opcao = max(recomendacoes, key=lambda x: x[2])
        logger.info(f"✅ Handicap opção {melhor_opcao[0]} recomendada (VE: {melhor_opcao[1]:.3f}, Prob: {melhor_opcao[2]:.3f}, Odd: {melhor_opcao[3]:.2f})")
        return (melhor_opcao[0], melhor_opcao[2])
    else:
        logger.info("❌ Nenhuma recomendação em Handicap")
        return (None, None)





def predicta_goal_line(prepGoal_line_df,prepGoal_line_df_conj, dados):
    predictor = TabularPredictor.load("autogluon_goal_line_model")
    predictor_conj = TabularPredictor.load("autogluon_goal_line_model_conj")

    
    # Garante que usaremos o melhor modelo
    best_model = predictor.model_best
    print(f"🔍 Modelo selecionado: {best_model}")
    best_model_conj = predictor_conj.model_best
    
    # Usa explicitamente o melhor modelo para previsão de probabilidades
    preds_proba = predictor.predict_proba(prepGoal_line_df, model=best_model)
    pred = predictor_conj.predict(prepGoal_line_df_conj, model=best_model_conj).iloc[0]
    try:
        ql_gl = qlearning.QLearningGoalLine()
        ql_gl.load_model('q_learning_gl_model_final.pkl')
        first_row = prepGoal_line_df_conj.iloc[0]
        estado = qlearning.q_learning_gl(first_row)
        if estado in ql_gl.q_table:
            pred_ql = ql_gl.choose_action(estado, epsilon=0)
        else:
            print('Estado GL nao encontrado')
            pred_ql = pred

    except:
        print('PROBLEMAS COM QL GL')
        pred_ql = pred
    if pred == 'over':
        pred = 0
    elif pred == 'under':
        pred = 1
    pred_goal_line_1 = float(preds_proba[0][1])
    pred_goal_line_2 = float(preds_proba[1][1])
    preds = [pred_goal_line_1, pred_goal_line_2]

    th_ve = 1.0  # Valor Esperado mínimo
    recomendacoes = []
    th_odd = 1.5
    for i in range(2):
        prob = preds[i]
        odd = float(dados['odds_gl'].iloc[i])
        ve = prob * odd
        if (ve >= th_ve) and (prob >= lista_th[3]) and (odd >= th_odd):
            if prob > preds[1 - i]:  # Comparação com a outra opção
                if pred == i and pred == pred_ql:
                    recomendacoes.append((i + 1, ve, prob, odd))

    logger.info(f"📊 Goal Line - Predições: {preds}, Odds GL: {dados['odds_gl']}")
    logger.info(f"📊 gl pred - Pred: {pred}")
    logger.info(f"📊 ql gl pred - Pred: {pred_ql}")

    if recomendacoes:
        melhor_opcao = max(recomendacoes, key=lambda x: x[2])
        logger.info(f"✅ Goal Line opção {melhor_opcao[0]} recomendada (VE: {melhor_opcao[1]:.3f}, Prob: {melhor_opcao[2]:.3f}, Odd: {melhor_opcao[3]:.2f})")
        return (melhor_opcao[0], melhor_opcao[2])
    else:
        logger.info("❌ Nenhuma recomendação em Goal Line")
        return (None, None)





def predicta_double_chance(prepDoubleChance_df, prepDoubleChance_df_conj, dados):
    try:
        predictor = TabularPredictor.load("autogluon_double_chance_model")
        predictor_conj = TabularPredictor.load("autogluon_double_chance_model_conj")
        
        # Garante que usaremos o melhor modelo
        best_model = predictor.model_best
        logger.info(f"🔍 Modelo selecionado (Double Chance): {best_model}")
        best_model_conj = predictor_conj.model_best

        # Usa explicitamente o melhor modelo para 
        preds_proba = predictor.predict_proba(prepDoubleChance_df, model=best_model)
        pred = predictor_conj.predict(prepDoubleChance_df_conj, model=best_model_conj).iloc[0]

        # Correção do problema com Q-Learning
        try:
            ql_dc = qlearning.QLearningDoubleChance()
            ql_dc.load_model('q_learning_dc_model_final.pkl')
            
            # CORREÇÃO: Tratar o DataFrame antes de passar para q_learning_dc
            try:
                # Método 1: Tentar normalmente
                estado = qlearning.q_learning_dc(prepDoubleChance_df_conj)
                
            except ValueError as ve:
                if "truth value of a Series is ambiguous" in str(ve):
                    logger.warning("⚠️ Erro de Series ambígua detectado, tentando correções...")
                    
        
                    try:
                        first_row = prepDoubleChance_df_conj.iloc[0]
                        logger.info(f"📊 Tentando com primeira linha: {first_row.shape}")
                        estado = qlearning.q_learning_dc(first_row)
                    except Exception:
                        logger.error("❌ Todas as tentativas falharam")
                        raise Exception("Não foi possível gerar estado para Q-Learning")
                else:
                    raise ve
            
            logger.info(f"✅ Estado gerado: {estado} (tipo: {type(estado)})")
            
            if estado in ql_dc.q_table:
                pred_dc = ql_dc.choose_action(estado, epsilon=0)
                logger.info(f"✅ Ação escolhida pelo Q-Learning: {pred_dc}")
            else:
                logger.warning(f"⚠️ Estado '{estado}' não encontrado na Q-table")
                pred_dc = pred
                
        except Exception as e:
            logger.error(f"❌ PROBLEMA COM QL - {type(e).__name__}: {str(e)}")
            pred_dc = pred
        
        # Resto do código permanece igual...
        # Log das dimensões das previsões
        logger.info(f"📊 Formato das previsões: {preds_proba.shape}")
        logger.info(f"📊 Conteúdo das previsões: {preds_proba}")
        
        if preds_proba.shape[0] < 3:
            logger.error(f"❌ Número insuficiente de previsões: {preds_proba.shape[0]}")
            return (None, None)

        try:
            pred_dc_1 = float(preds_proba.iloc[0, 1])
            logger.info(f"📊 Previsão 1: {pred_dc_1}")
        except Exception as e:
            logger.error(f"❌ Erro ao processar previsão 1: {str(e)}")
            return (None, None)

        try:
            pred_dc_2 = float(preds_proba.iloc[1, 1])
            logger.info(f"📊 Previsão 2: {pred_dc_2}")
        except Exception as e:
            logger.error(f"❌ Erro ao processar previsão 2: {str(e)}")
            return (None, None)

        try:
            pred_dc_3 = float(preds_proba.iloc[2, 1])
            logger.info(f"📊 Previsão 3: {pred_dc_3}")
        except Exception as e:
            logger.error(f"❌ Erro ao processar previsão 3: {str(e)}")
            return (None, None)

        preds = [pred_dc_1, pred_dc_2, pred_dc_3]
        logger.info(f"📊 Lista de previsões: {preds}")
        logger.info(f"📊 pred dc: {pred_dc}") 
        logger.info(f"📊 pred conj  dc: {pred}") # Corrigido para usar pred_dc

        th_ve = 1.0
        th_odd = 1.5
        recomendacoes = []

        for i in range(3):
            try:
                prob = preds[i]
                odd = float(dados['odds'].iloc[i])
                ve = prob * odd
                logger.info(f"📊 Iteração {i+1} - Prob: {prob}, Odd: {odd}, VE: {ve}")
                
                if (ve >= th_ve) and (prob >= lista_th[4]) and (odd >= th_odd):
                    if i in [0, 1]:
                        if prob > preds[1 - i]: 
                            if pred == i and pred == pred_dc:
                                recomendacoes.append((i + 1, ve, prob, odd))
                    else:
                        recomendacoes.append((i + 1, ve, prob, odd))
            except Exception as e:
                logger.error(f"❌ Erro ao processar iteração {i+1}: {str(e)}")
                continue

        logger.info(f"📊 Double Chance - Predições: {preds}, Odds: {dados['odds']}")
        logger.info(f"📊 Recomendações encontradas: {recomendacoes}")

        if recomendacoes:
            melhor_opcao = max(recomendacoes, key=lambda x: x[2])
            logger.info(f"✅ Double Chance opção {melhor_opcao[0]} recomendada (VE: {melhor_opcao[1]:.3f}, Prob: {melhor_opcao[2]:.3f}, Odd: {melhor_opcao[3]:.2f})")
            return (melhor_opcao[0], melhor_opcao[2])
        else:
            logger.info("❌ Nenhuma recomendação em Double Chance")
            return (None, None)
            
    except Exception as e:
        logger.error(f"❌ Erro durante a previsão do Double Chance: {str(e)}")
        return (None, None)



def predicta_draw_no_bet(pred_draw_no_bet_df,pred_draw_no_bet_df_conj, dados):
    # Carregar o modelo treinado para "Draw No Bet"
    predictor = TabularPredictor.load("autogluon_draw_no_bet_model")
    predictor_conj = TabularPredictor.load("autogluon_draw_no_bet_model_conj")
    
    # Garantir que usamos o melhor modelo
    best_model = predictor.model_best
    logger.info(f"🔍 Modelo selecionado (Draw No Bet): {best_model}")
    best_model_conj = predictor_conj.model_best

    # Obter as probabilidades de predição
    preds_proba = predictor.predict_proba(pred_draw_no_bet_df, model=best_model)
    pred = predictor_conj.predict(pred_draw_no_bet_df_conj, model=best_model_conj).iloc[0]

    try:
        ql_dnb = qlearning.QLearningDrawNoBet()
        ql_dnb.load_model('q_learning_dnb_model_final.pkl')
        first_row = pred_draw_no_bet_df_conj.iloc[0]
        estado = qlearning.q_learning_gl(first_row)
        if estado in ql_dnb.q_table:
            pred_ql = ql_dnb.choose_action(estado, epsilon=0)
        else:
            print('Estado DNB nao encontrado')
            pred_ql = pred

    except:
        print('PROBLEMAS COM QL dnb')
        pred_ql = pred

    # Convertendo as predições para valores flutuantes
    pred_dnb_1 = float(preds_proba[0][1])
    pred_dnb_2 = float(preds_proba[1][1])
    preds = [pred_dnb_1, pred_dnb_2]

    th_ve = 1.0  # Valor esperado mínimo
    th_odd = 1.5  # Odd mínima
    recomendacoes = []

    for i in range(2):
        prob = preds[i]
        odd = float(dados['odds'].iloc[i])
        ve = prob * odd

        # Verifica se o Valor Esperado é maior que o limite e a probabilidade e odd estão boas
        if (ve >= th_ve) and (prob >= lista_th[5]) and (odd >= th_odd):
            if prob > preds[1 - i]:  # Compara entre as duas opções possíveis
                if pred == i and pred == pred_ql:
                    recomendacoes.append((i + 1, ve, prob, odd))

    logger.info(f"📊 Draw No Bet - Predições: {preds}, Odds: {dados['odds']}")
    logger.info(f"📊 pred dnb: {pred}")

    if recomendacoes:
        # Seleciona a melhor recomendação com base na maior probabilidade
        melhor_opcao = max(recomendacoes, key=lambda x: x[2])
        logger.info(f"✅ Draw No Bet opção {melhor_opcao[0]} recomendada (VE: {melhor_opcao[1]:.3f}, Prob: {melhor_opcao[2]:.3f}, Odd: {melhor_opcao[3]:.2f})")
        return (melhor_opcao[0], melhor_opcao[2])
    else:
        logger.info("❌ Nenhuma recomendação em Draw No Bet")
        return (None, None)


def processar_dia_anterior():
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
    dia = dia_anterior()
    print(f"🔄 Processando jogos do dia {dia}")

    try:
        print("🔎 Buscando IDs e dicionário de eventos...")
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
        print(f"✔️ {len(ids)} eventos encontrados.")

        print("📊 Filtrando e transformando odds...")
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        df_odds = apiclient.transform_betting_data(odds_data)  # Corrigido para usar a função correta

        novos_dados = []
        for dados_evento in dicio:
            event_id = dados_evento.get('id')
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')

            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0], "event_day": dia}
            else:
                merged = {**dados_evento, "event_day": dia}

            novos_dados.append(merged)

        if novos_dados:
            print(f"🧩 {len(novos_dados)} eventos com odds processados.")
            df_novo = pd.DataFrame(novos_dados)

            colunas_adicionadas = []
            for coluna in COLUNAS_PADRAO:
                if coluna not in df_novo.columns:
                    df_novo[coluna] = None
                    colunas_adicionadas.append(coluna)

            if colunas_adicionadas:
                print(f"➕ Colunas adicionadas automaticamente: {', '.join(colunas_adicionadas)}")

            df_novo = df_novo[COLUNAS_PADRAO]

            if os.path.exists(CSV_FILE):
                print("📂 CSV existente encontrado, mesclando dados...")
                df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                df_final["time"] = df_final["time"].astype(int)
                df_final = df_final.sort_values(by="time", ascending=False).reset_index(drop=True)

                primeiro_dia = df_final["event_day"].min()
                df_final = df_final[df_final["event_day"] != primeiro_dia]
            else:
                print("📄 Nenhum CSV encontrado, criando novo arquivo...")
                df_final = df_novo
                primeiro_dia = "N/A"

            df_final.to_csv(CSV_FILE, index=False)
            remover_duplicatas()
            print(f"✅ Dados atualizados com sucesso! Dia {primeiro_dia} removido, dia {dia} adicionado.")
        else:
            print(f"⚠️ Nenhum dado encontrado para o dia {dia}")

    except Exception as e:
        print(f"❌ Erro ao processar dia {dia}: {type(e).__name__}: {e}")




# Variável global para controlar o estado do servidor
server_running = True

def signal_handler(signum, frame):
    logger.info("Received shutdown signal. Cleaning up...")
    sys.exit(0)

if __name__ == "__main__":
    main()

import pandas as pd
import os
from datetime import datetime

def atualizar_csv_dia_atual():
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

    # Definir dias para buscar (igual à função pegaJogosDoDia)
    dias_para_buscar = [datetime.now().strftime("%Y%m%d")]
    if datetime.now().hour >= 20:
        dia_seguinte = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
        dias_para_buscar.append(dia_seguinte)
    
    logger.info(f"📅 Dias para buscar: {dias_para_buscar}")

    try:
        # Carregar dados existentes primeiro
        ids_existentes = set()
        if os.path.exists(CSV_FILE):
            df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
            ids_existentes = set(df_existente['id'].astype(str))
            logger.info(f"📊 Total de registros existentes: {len(ids_existentes)}")

        # Buscar dados para todos os dias
        todos_ids = []
        todos_dicionarios = []
        
        for dia in dias_para_buscar:
            logger.info(f"🔎 Buscando IDs e dicionário de eventos para o dia {dia}...")
            ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
            logger.info(f"✔️ {len(ids)} eventos encontrados para o dia {dia}.")
            
            # Adicionar o dia do evento a cada dicionário
            for evento in dicio:
                evento['event_day'] = dia
            
            todos_ids.extend(ids)
            todos_dicionarios.extend(dicio)

        logger.info(f"✔️ Total de {len(todos_ids)} eventos encontrados em todos os dias.")

        if not todos_ids:
            logger.info("⚠️ Nenhum evento encontrado para os dias especificados.")
            return

        logger.info("📊 Filtrando e transformando odds...")
        odds_data = apiclient.filtraOddsNovo(ids=todos_ids)
        df_odds = apiclient.transform_betting_data(odds_data)

        novos_dados = []
        for dados_evento in todos_dicionarios:
            event_id = str(dados_evento.get('id'))
            
            # Verificar se o ID já existe
            if event_id in ids_existentes:
                logger.info(f"⚠️ ID {event_id} já existe no CSV, pulando...")
                continue
                
            odds_transformadas = df_odds[df_odds['id'] == event_id].to_dict('records')

            if odds_transformadas:
                merged = {**dados_evento, **odds_transformadas[0]}
            else:
                merged = dados_evento.copy()

            novos_dados.append(merged)

        if not novos_dados:
            logger.info("⚠️ Nenhum dado novo para adicionar.")
            return

        df_novo = pd.DataFrame(novos_dados)
        logger.info(f"📝 {len(df_novo)} novos registros para adicionar")

        # Garantir que id é string
        df_novo['id'] = df_novo['id'].astype(str)

        colunas_adicionadas = []
        for coluna in COLUNAS_PADRAO:
            if coluna not in df_novo.columns:
                df_novo[coluna] = None
                colunas_adicionadas.append(coluna)

        if colunas_adicionadas:
            logger.info(f"➕ Colunas adicionadas automaticamente: {', '.join(colunas_adicionadas)}")

        df_novo = df_novo[COLUNAS_PADRAO]

        if os.path.exists(CSV_FILE):
            logger.info("📂 CSV existente encontrado, mesclando dados...")
            df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})
            df_existente['id'] = df_existente['id'].astype(str)
            
            # Verificar duplicatas antes da concatenação
            duplicatas = set(df_novo['id']).intersection(set(df_existente['id']))
            if duplicatas:
                logger.warning(f"⚠️ Encontrados {len(duplicatas)} IDs que já existem no CSV")
                for dup in duplicatas:
                    logger.warning(f"ID duplicado: {dup}")
            
            # Filtrar apenas registros novos que não existem no CSV
            df_novo = df_novo[~df_novo['id'].isin(duplicatas)]
            logger.info(f"📝 Após remover duplicatas, {len(df_novo)} registros novos para adicionar")
            
            if len(df_novo) > 0:
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                df_final['id'] = df_final['id'].astype(str)
                df_final = df_final.drop_duplicates(subset=['id'], keep='last')
                
                # Ordenar por data do evento
                df_final["time"] = df_final["time"].astype(int)
                df_final = df_final.sort_values(by="time", ascending=False).reset_index(drop=True)
                
                # Verificar duplicatas finais
                duplicatas_finais = df_final[df_final.duplicated(subset=['id'], keep=False)]
                if not duplicatas_finais.empty:
                    logger.warning(f"⚠️ Ainda existem {len(duplicatas_finais)} duplicatas após a concatenação")
                    for id_dup in duplicatas_finais['id'].unique():
                        logger.warning(f"ID duplicado final: {id_dup}")
                
                df_final.to_csv(CSV_FILE, index=False)
                logger.info(f"✅ CSV atualizado com {len(df_novo)} eventos adicionados")
            else:
                logger.info("ℹ️ Nenhum registro novo para adicionar após remoção de duplicatas")
        else:
            logger.info("📄 Nenhum CSV encontrado, criando novo arquivo...")
            df_novo.to_csv(CSV_FILE, index=False)
            logger.info(f"✅ Novo CSV criado com {len(df_novo)} eventos")
        
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar dados dos dias {dias_para_buscar}: {type(e).__name__}: {e}")
        raise  # Re-lança a exceção para ser tratada pelo chamador