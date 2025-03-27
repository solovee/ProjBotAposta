import requests
from api import BetsAPIClient
from dotenv import load_dotenv
import os
from datetime import datetime






#'10048705', 'Esoccer GT Leagues - 12 mins play' ;'10047781', 'Esoccer Battle - 8 mins play'

#testar pegar evento 171732570 mais tarde   171790606  172006772 9723272 172006783
load_dotenv()

api = os.getenv("API_KEY")



apiclient = BetsAPIClient(api_key=api)


ids = apiclient.getUpcoming(leagues=apiclient.leagues_ids)

res = apiclient.filtraOdds(ids=ids)





