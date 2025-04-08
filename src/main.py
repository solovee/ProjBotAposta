import time
from datetime import datetime, timedelta
from api import BetsAPIClient
import pandas as pd
from dotenv import load_dotenv
import os
import threading
import NN


load_dotenv()

api = os.getenv("API_KEY")



apiclient = BetsAPIClient(api_key=api)


def pegaJogosDoDia():
    ids , time = apiclient.getUpcoming(leagues=apiclient.leagues_ids)
    dados = [{"id_jogo": i, "horario": h} for i, h in zip(ids, time)]
    
    dados_dataframe = pd.DataFrame(dados)

    dados_dataframe['send_time'] = dados_dataframe['horario'] - 300
    dados_dataframe = dados_dataframe.sort_values(by="time").reset_index(drop=True)
    return dados_dataframe



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




# Chamar função
agendar_acoes(df)
