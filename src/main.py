import time
from datetime import datetime, timedelta
import tensorflow as tf
from api import BetsAPIClient
import pandas as pd
from dotenv import load_dotenv
import os
import threading
import NN


load_dotenv()

api = os.getenv("API_KEY")



apiclient = BetsAPIClient(api_key=api)

#df = pd.read_csv('src\resultados_novo.csv')


#pega uma vez ao dia, provavelmente umas 00:30
def pegaJogosDoDia():
    ids , tempo, time = apiclient.getUpcoming(leagues=apiclient.leagues_ids)
    dados = [{"id_jogo": i, "horario": h, "times": k} for i, h, k in zip(ids, tempo, time)]
    
    dados_dataframe = pd.DataFrame(dados)
    print(dados_dataframe.columns)
    dados_dataframe['horario'] = dados_dataframe['horario'].astype(int)
    dados_dataframe['send_time'] = dados_dataframe['horario'] - 350
    dados_dataframe = dados_dataframe.sort_values(by="horario").reset_index(drop=True)
    return dados_dataframe


#roda apos pegar os jogos do dia, mas cada acao do jogo sera executada em seu tempo send_timer
def pegaOddsEvento(df):
    agora = time.time()  # timestamp atual em segundos

    for _, row in df.iterrows():
        delay = row['send_time'] - agora  # tempo até a ação acontecer
        delay = max(0, delay)  # evita delays negativos

        threading.Timer(delay, acao_do_jogo, args=(row['id'],)).start()
        print(f"Agendado jogo {row['id']} para {datetime.fromtimestamp(row['send_time'])}")




# Função que será executada para cada jogo
def acao_do_jogo(jogo_id):
    odds = apiclient.filtraOddsNovo([jogo_id])
    df_odds = apiclient.transform_betting_data(odds)
    df_odds = NN.preProcessGeneral(df_odds)
    #implementar as previsões e requisitos (threshold e odd)
    preve()


def criaTodasNNs(df):
    lista = NN.criaNNs(df)
    return lista


def preve(lista, df_linha):
    model_over_under = tf.keras.models.load_model('model_over_under.keras')
    model_handicap = tf.keras.models.load_model('model_handicap_binario.keras')
    model_goal_line = tf.keras.models.load_model('mode_binario_goal_line.keras')
    model_double_chance = tf.keras.models.load_model('mode_double_chance.keras')
    model_draw_no_bet = tf.keras.models.load_model('model_binario_draw_no_bet.keras')

    pred_over_under = model_over_under.predict(NN.preProcessOverUnder(df_linha))
    pred_handicap = model_handicap.predict(NN.preProcessHandicap(df_linha))
    pred_goal_line = model_goal_line.predict(NN.preProcessGoalLine(df_linha))
    pred_double_chance = model_double_chance.predict(NN.preProcessDoubleChance(df_linha))
    pred_draw_no_bet = model_draw_no_bet.predict(NN.preProcessDrawNoBet(df_linha))

day_ids = pegaJogosDoDia()
print(day_ids)
#lista = criaTodasNNs(df)
