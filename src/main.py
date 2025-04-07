import time
from datetime import datetime, timedelta
from api import BetsAPIClient
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

api = os.getenv("API_KEY")



apiclient = BetsAPIClient(api_key=api)
apiclient = BetsAPIClient()
def pegaJogosDoDia():
    ids , time = apiclient.getUpcoming(leagues=apiclient.leagues_ids)
    dados = [{"id_jogo": i, "horario": h} for i, h in zip(ids, time)]
    dados_dataframe = pd.DataFrame(dados).sort_values(by="time").reset_index(drop=True)
    
    

