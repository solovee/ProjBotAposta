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
CSV_FILE = "resultados.csv"
#lista dos thresholds das nns
lista_th = []
#guarda jogos programados do dia, deve zerar ao mudar o dia
programado = []
#pega uma vez ao virar o dia e depois de 20 em 20 min pra testar
def pegaJogosDoDia(programado):
    ids , tempo, time = apiclient.getUpcoming(leagues=apiclient.leagues_ids)
    # adicionar tratamento pra no caso de vazio
    dados = [{"id_jogo": i, "horario": h, "times": k} for i, h, k in zip(ids, tempo, time)]
    
    dados_dataframe = pd.DataFrame(dados)
    dados_dataframe = dados_dataframe[~dados_dataframe['id_jogo'].isin(programado)]
    print(dados_dataframe.columns)
    dados_dataframe['horario'] = dados_dataframe['horario'].astype(int)
    dados_dataframe['send_time'] = dados_dataframe['horario'] - 350
    dados_dataframe = dados_dataframe.sort_values(by="horario").reset_index(drop=True)
    programados = [dados['id_jogo'] for dado in dados]
    programado.extend(programados)
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
    preve(df_linha=df_odds)


def criaTodasNNs(df):
    lista_th = NN.criaNNs(df)
    


def preve(df_linha):
    
    model_over_under = tf.keras.models.load_model('model_over_under.keras')
    model_handicap = tf.keras.models.load_model('model_handicap_binario.keras')
    model_goal_line = tf.keras.models.load_model('mode_binario_goal_line.keras')
    model_double_chance = tf.keras.models.load_model('mode_double_chance.keras')
    model_draw_no_bet = tf.keras.models.load_model('model_binario_draw_no_bet.keras')
    
    prepOverUnder, dados_OU = NN.prepNNOver_under(df_linha)
    prepHandicap, dados_ah = NN.prepNNHandicap(df_linha)
    prepGoal_line, dados_gl = NN.prepNNGoal_line(df_linha)
    prepDouble_chance, dados_dc = NN.prepNNDouble_chance(df_linha)
    prepDraw_no_bet, dados_dnb = NN.prepNNDraw_no_bet(df_linha)
   
    
    pred_over_under_pos, pred_over_under_neg = model_over_under.predict(prepOverUnder)
    pred_handicap = model_handicap.predict(prepHandicap)
    pred_goal_line = model_goal_line.predict(prepGoal_line)
    pred_double_chance = model_double_chance.predict(prepDouble_chance)
    pred_draw_no_bet = model_draw_no_bet.predict(prepDraw_no_bet)
    lista_resultados = [prepOverUnder, prepHandicap, prepGoal_line, prepDouble_chance, prepDraw_no_bet]
    return lista_resultados

def predicta_over_under():
    pass

def processar_dia_anterior():
    dia = apiclient.dia_anterior()
    print(f"🔄 Processando jogos do dia {dia}")

    try:
        ids, dicio = apiclient.getAllOlds(leagues=apiclient.leagues_ids, day=int(dia))
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

df = pegaJogosDoDia()
print(df)