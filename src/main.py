import time
from datetime import datetime, timedelta
import tensorflow as tf
from api import BetsAPIClient, dia_anterior
import pandas as pd
from dotenv import load_dotenv
import os
import threading
import NN
import telegramBot as tb
import logging

#tentar arrumar as estatisticas
#ver os team_ah e type_gl

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
chat_id = os.getenv("CHAT_ID")



apiclient = BetsAPIClient(api_key=api)

#df = pd.read_csv('src\resultados_novo.csv')
CSV_FILE = r"C:\Users\Leoso\Downloads\projBotAposta\src\resultados_novo.csv"
#lista dos thresholds das nns
lista_th = [0.7,0.2,0.7,0.8,0.8,0.9]



#!pega uma vez ao virar o dia e depois de 20 em 20 min pra testar
data_hoje = datetime.now().date()
#!reseta todo dia as 00:00, guarda jogos programados do dia, deve zerar ao mudar o dia
programado = []

def checa_virada_do_dia():
    global data_hoje, programado
    while True:
        if datetime.now().date() != data_hoje:
            data_hoje = datetime.now().date()
            programado = []
            logger.info("🔄 Novo dia detectado, resetando variáveis...")
        time.sleep(60)

def agendar_processar_dia_anterior():
    agora = datetime.now()
    alvo = datetime.combine(agora.date(), datetime.min.time()) + timedelta(hours=0, minutes=5)

    if agora >= alvo:
        alvo += timedelta(days=1)

    delay = (alvo - agora).total_seconds()
    logger.info(f"⏰ Agendando processamento do dia anterior para {alvo}")
    threading.Timer(delay, processar_dia_anterior).start()


def agendar_criacao_nns():
    agora = datetime.now()
    alvo = datetime.combine(agora.date(), datetime.min.time()) + timedelta(hours=0, minutes=15)

    # Se o horário atual já passou de 00:15, agendar para o próximo dia
    if agora >= alvo:
        alvo += timedelta(days=1)

    delay = (alvo - agora).total_seconds()
    logger.info(f"⏰ Agendando criação de NNs para {alvo}")

    def tarefa():
        logger.info("🧠 Iniciando criação de modelos de rede neural...")
        df = pd.read_csv(CSV_FILE)
        criaTodasNNs(df)
        logger.info("✅ Modelos de rede neural criados com sucesso")

    threading.Timer(delay, tarefa).start()


def loop_pega_jogos():
    while True:
        logger.info("🔎 Buscando jogos programados para hoje...")
        df_jogos = pegaJogosDoDia()
        if not df_jogos.empty:
            logger.info(f"📅 Encontrados {len(df_jogos)} jogos para hoje")
            pegaOddsEvento(df_jogos)
        else:
            logger.info("ℹ️ Nenhum jogo encontrado por agora")
        time.sleep(20 * 60)  # 20 minutos

def main():
    logger.info("🚀 Iniciando aplicação de apostas esportivas")
    # Thread para verificar virada do dia e resetar programado
    threading.Thread(target=checa_virada_do_dia, daemon=True).start()

    # Agendar processamento do dia anterior
    agendar_processar_dia_anterior()

    # Agendar criação das NNs
    agendar_criacao_nns()

    # Iniciar loop que pega os jogos de tempos em tempos
    threading.Thread(target=loop_pega_jogos, daemon=True).start()
    tb.start_bot()
    logger.info("✅ Todos os serviços foram iniciados")


def pegaJogosDoDia():
    try:
        ids , tempo, time, times_id = apiclient.getUpcoming(leagues=apiclient.leagues_ids)
               
        if not ids:
            logger.warning("⚠️ Nenhum ID de jogo retornado pela API")
            return pd.DataFrame()
        # adicionar tratamento pra no caso de vazio
        dados = [{"id_jogo": i, "horario": h, "times": k, "home": z, "away": t} for i, h, k, z, t in zip(ids, tempo, time, times_id[0], times_id[1])]
        
        dados_dataframe = pd.DataFrame(dados)
        dados_dataframe = dados_dataframe[~dados_dataframe['id_jogo'].isin(programado)]
        if dados_dataframe.empty:
            logger.info("ℹ️ Todos os jogos já estão programados")
            return dados_dataframe
        
        dados_dataframe['horario'] = dados_dataframe['horario'].astype(int)
        dados_dataframe['send_time'] = dados_dataframe['horario'] - 350
        dados_dataframe = dados_dataframe.sort_values(by="horario").reset_index(drop=True)
        programados = dados_dataframe['id_jogo'].tolist()
        programado.extend(programados)
        logger.info(f"📌 Adicionados {len(programados)} novos jogos à lista de programados")
        return dados_dataframe
    
    except Exception as e:
        logger.error(f"❌ Erro ao obter jogos do dia: {str(e)}")
        return pd.DataFrame()


#!roda apos pegajogosDoDia, mas cada acao do jogo sera executada em seu tempo send_timer
def pegaOddsEvento(df):
    agora = time.time()  # timestamp atual em segundos
    logger.info(f"⏳ Agendando {len(df)} eventos...")

    for _, row in df.iterrows():
        delay = row['send_time'] - agora  # tempo até a ação acontecer
        delay = max(0, delay)  # evita delays negativos

        threading.Timer(delay, acao_do_jogo, args=(row,)).start()
        print(f"Agendado jogo {row['id_jogo']} para {datetime.fromtimestamp(row['send_time'])}")




# Função que será executada para cada jogo
def acao_do_jogo(row):
    try:
        logger.info(f"⚽ Processando jogo {row['id_jogo']}")
        odds = apiclient.filtraOddsNovo([row['id_jogo']])
        if not odds:
            logger.warning(f"⚠️ Nenhuma odd encontrada para o jogo {row['id_jogo']}")
            return 0
        df_odds = apiclient.transform_betting_data(odds)
        df_odds['home'] = row['home']
        df_odds['away'] = row['away']
        df_odds = NN.preProcessGeneral_x(df_odds)
        
        #implementar as previsões e requisitos (threshold e odd)
        lista_bets_a_enviar = preve(df_odds)
        if lista_bets_a_enviar:
            logger.info(f"📩 Enviando {len(lista_bets_a_enviar)} previsões para o Telegram")
            for bet in lista_bets_a_enviar:
                tb.sendMessage(chat_id, bet)
        else:
            logger.info("ℹ️ Nenhuma aposta recomendada para este jogo")

    except Exception as e:
        logger.error(f"❌ Erro ao processar jogo {row['id_jogo']}: {str(e)}")
        return 0

#! roda todo dia as 00:15
def criaTodasNNs(df):
    global lista_th 
    logger.info("🔧 Criando todos os modelos de rede neural...")
    lista_th = NN.criaNNs(df)
    logger.info(f"📊 Thresholds definidos: {lista_th}")

def preve(df_linha):

    logger.debug("🔮 Fazendo previsões para o jogo atual...")
    try:
        if not lista_th:
            logger.warning("⚠️ Thresholds de modelos ainda não definidos")
            return []
        prepOverUnder, dados_OU = NN.prepNNOver_under_X(df_linha)
        prepHandicap, dados_ah = NN.prepNNHandicap_X(df_linha)
        prepGoal_line, dados_gl = NN.prepNNGoal_line_X(df_linha)
        prepDouble_chance, dados_dc = NN.prepNNDouble_chance_X(df_linha)
        prepDraw_no_bet, dados_dnb = NN.prepNNDraw_no_bet_X(df_linha)

        tipo_over_under, res_under_over = predicta_over_under(prepOverUnder, dados_OU)
        time_handicap, res_handicap = predicta_handicap(prepHandicap, dados_ah)
        linha_gl, res_goal_line = predicta_goal_line(prepGoal_line, dados_gl)
        type_dc, res_double_chance = predicta_double_chance(prepDouble_chance, dados_dc)
        res_draw_no_bet = predicta_draw_no_bet(prepDraw_no_bet, dados_dnb)
        lista_preds_true = [tipo_over_under, res_under_over,time_handicap, res_handicap, linha_gl, res_goal_line, type_dc, res_double_chance, res_draw_no_bet]
        list_true = []
        if lista_preds_true[0] and lista_preds_true[1]:
            dados_OU['tipo'] = lista_preds_true[0]
            dados_OU['linha'] = '2.5'
            dados_OU = dados_OU.rename(columns={'odds_goals_over1': 'odds_over'})
            dados_OU = dados_OU.rename(columns={'odd_goals_under1': 'odds_under'})
            list_true.append(dados_OU)
        if lista_preds_true[2] and lista_preds_true[3]:
            if (lista_preds_true[2] == 1):
                dados_temp = dados_ah.iloc[[0]].copy()     
                dados_temp['linha'] = dados_temp['asian_handicap_1'] + ' , ' + dados_temp['asian_handicap_2']
                dados_temp = dados_temp.drop(columns=['asian_handicap_1', 'asian_handicap_2'])
                dados_temp = dados_temp.rename(columns={'team_ah': 'time'})
                list_true.append(dados_temp)
            elif (lista_preds_true[2] == 2):
                dados_temp = dados_ah.iloc[[1]].copy()
                dados_temp['linha'] = dados_temp['asian_handicap_1'] + ' , ' + dados_temp['asian_handicap_2']
                dados_temp = dados_temp.drop(columns=['asian_handicap_1', 'asian_handicap_2'])
                dados_temp = dados_temp.rename(columns={'team_ah': 'time'})
                list_true.append(dados_temp) 
        if lista_preds_true[4] and lista_preds_true[5]:
            if (lista_preds_true[4] == 1):
                dados_temp = dados_gl.iloc[[0]].copy()   
                dados_temp['linha'] = dados_temp['goal_line_1'] + ' , ' + dados_temp['goal_line_2']
                dados_temp = dados_temp.drop(columns=['goal_line_1', 'goal_line_2'])
                dados_temp = dados_temp.rename(columns={'type_gl': 'tipo over(0)/under(1)'})  
                dados_temp = dados_temp.rename(columns={'odds_gl': 'odds'})  
                list_true.append(dados_temp)
            elif (lista_preds_true[4] == 2):
                dados_temp = dados_gl.iloc[[1]].copy()   
                dados_temp['linha'] = dados_temp['goal_line_1'] + ' , ' + dados_temp['goal_line_2']
                dados_temp = dados_temp.drop(columns=['goal_line_1', 'goal_line_2'])
                dados_temp = dados_temp.rename(columns={'type_gl': 'tipo over(0)/under(1)'})  
                dados_temp = dados_temp.rename(columns={'odds_gl': 'odds'})     
                list_true.append(dados_temp)
            
        if lista_preds_true[6] and lista_preds_true[7]:
            if (lista_preds_true[6] == 1):
                dados_temp = dados_dc.iloc[[0]].copy()
                dados_temp = dados_temp.rename(columns={'double_chance': 'double_chance(home=1,away=2,both=3)'})  
                list_true.append(dados_temp)
            elif (lista_preds_true[6] == 2):
                dados_temp = dados_dc.iloc[[1]].copy()
                dados_temp = dados_temp.rename(columns={'double_chance': 'double_chance(home=1,away=2,both=3)'})  
                list_true.append(dados_temp)
            elif (lista_preds_true[6] == 3):
                dados_temp = dados_dc.iloc[[2]].copy()
                dados_temp = dados_temp.rename(columns={'double_chance': 'double_chance(home=1,away=2,both=3)'})  
                list_true.append(dados_temp)
        if lista_preds_true[8] and lista_preds_true[9]:
            if (lista_preds_true[8] == 1):
                dados_temp = dados_dnb.iloc[[0]].copy()
                list_true.append(dados_temp)
            elif (lista_preds_true[8] == 2):
                dados_temp = dados_dnb.iloc[[1]].copy()
                dados_temp = dados_temp.rename(columns={'draw_no_bet_team': 'draw_no_bet_team(home=1,away=2)'})  
                list_true.append(dados_temp)
        list_final = []
        for df in list_true:
            men = df_para_string(df)
            list_final.append(men)
        return list_final
    
    except Exception as e:
        logger.error(f"❌ Erro durante a previsão: {str(e)}")
        return []


def df_para_string(df):
    mensagens = []

    for _, row in df.iterrows():
        msg = ""
        for col in df.columns:
            msg += f"{col}: {row[col]}\n"
        mensagens.append(msg.strip())  # remove o \n final

    return mensagens


def predicta_over_under(prepOverUnder_df, dados):
    model_over_under = tf.keras.models.load_model('model_over_under.keras')
    preds = model_over_under.predict(prepOverUnder_df)

    if (preds >= lista_th[0]) and (dados['odd_goals_over1'] > 1.6):
        return ('over', True)
    elif (preds <= lista_th[1]) and (dados['odd_goals_under1'] > 1.6):
        return ('under', True)
    else:
        return (None, False)
def predicta_handicap(prepHandicap_df, dados):
    model_handicap = tf.keras.models.load_model('model_handicap_binario.keras')
    
    preds = model_handicap.predict(prepHandicap_df)

    pred_handicap_1 = preds[0]  
    pred_handicap_2 = preds[1] 
    if (pred_handicap_1 >= lista_th[2]) and (dados['odds'] > 1.6):
        return (1, True)
    elif (pred_handicap_2 >= lista_th[2]) and (dados['odds'] > 1.6):
        return (2, True)
    else:
        return (None, False)
def predicta_goal_line(prepGoal_line_df, dados):
    model_goal_line = tf.keras.models.load_model('mode_binario_goal_line.keras')

    preds = model_goal_line.predict(prepGoal_line_df)

    pred_goal_line_1 = preds[0]  
    pred_goal_line_2 = preds[1] 

    if (pred_goal_line_1 >= lista_th[3]) and (dados['odds_gl'] > 1.6):
        return (1, True)
    if (pred_goal_line_2 >= lista_th[3]) and (dados['odds_gl'] > 1.6):
        return (2, True)
    else:
        return (None, False)
def predicta_double_chance(pred_double_chance_df, dados):
    model_double_chance = tf.keras.models.load_model('mode_double_chance.keras')
    preds = model_double_chance.predict(pred_double_chance_df)

    pred_double_chance_1 = preds[0]  
    pred_double_chance_2 = preds[1] 
    pred_double_chance_3 = preds[2] 

    if (pred_double_chance_1 >= lista_th[4]) and (dados['odds'] > 1.6):
        return (1, True)
    elif (pred_double_chance_2 >= lista_th[4]) and (dados['odds'] > 1.6):
        return (2, True)
    elif (pred_double_chance_3 >= lista_th[4]) and (dados['odds'] > 1.6):
        return (3, True)
    else:
        return (None, False)
    
def predicta_draw_no_bet(pred_draw_no_bet_df, dados):
    model_draw_no_bet = tf.keras.models.load_model('model_binario_draw_no_bet.keras')
    
    preds = model_draw_no_bet.predict(pred_draw_no_bet_df)

    pred_draw_no_bet_1 = preds[0]  
    pred_draw_no_bet_2 = preds[1] 
 
    if (pred_draw_no_bet_1 >= lista_th[5]) and (dados['odds'] > 1.6):
        return (1, True)
    if (pred_draw_no_bet_2 >= lista_th[5]) and (dados['odds'] > 1.6):
        return (2, True)
    else:
        return (None, False)


#! roda as 00:05
def processar_dia_anterior():
    dia = dia_anterior()
    print(f"🔄 Processando jogos do dia {dia}")

    try:
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=dia)
        odds_data = apiclient.filtraOddsNovo(ids=ids)
        df_odds = apiclient.transform_betting_data(odds_data)

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
            df_novo = pd.DataFrame(novos_dados)
            colunas_ordenadas = ['id', 'event_day'] + [col for col in df_novo.columns if col not in ['id', 'event_day']]
            df_novo = df_novo[colunas_ordenadas]

            if os.path.exists(CSV_FILE):
                df_existente = pd.read_csv(CSV_FILE, dtype={"event_day": str})

                # Adiciona os dados novos
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)

                # Ordena cronologicamente
                df_final = df_final.sort_values(by="event_day").reset_index(drop=True)

                # Remove o primeiro (mais antigo) dia
                primeiro_dia = df_final["event_day"].min()
                df_final = df_final[df_final["event_day"] != primeiro_dia]
            else:
                df_final = df_novo

            df_final.to_csv(CSV_FILE, index=False)
            print(f"✅ Dados atualizados com sucesso! Dia {primeiro_dia} removido, dia {dia} adicionado.")
        else:
            print(f"⚠️ Nenhum dado encontrado para o dia {dia}")

    except Exception as e:
        print(f"❌ Erro ao processar dia {dia}: {e}")

if __name__ == "__main__":
    main()

