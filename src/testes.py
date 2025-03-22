import requests
from api import BetsAPIClient
#'10048705', 'Esoccer GT Leagues - 12 mins play' ;'10047781', 'Esoccer Battle - 8 mins play'

#testar pegar evento 171732570 mais tarde


api = "217012-1ic4weHvqmtEZ0"


apiclient = BetsAPIClient(api_key=api)


ids = []
res = apiclient.get_fifa_matches()

for r in res['results']:
    ids.append(r['id'])

for id in ids:
    odds = apiclient.get_odds(FI=id)['results'][0]['goals']['sp']['goals_over_under']
    print(odds)



