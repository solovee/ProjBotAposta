



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
import random
import ast
import main

load_dotenv()

api = os.getenv("API_KEY")
chat_id = int(os.getenv("CHAT_ID"))



apiclient = BetsAPIClient(api_key=api)
CSV_FILE = r"C:\Users\Leoso\Downloads\projBotAposta\resultados_60_ofc.csv"

main.criaTodasNNs()