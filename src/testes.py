import requests


url = 'https://api.b365api.com/v1/league'
api = "217012-1ic4weHvqmtEZ0"

params = {
    'sport_id': 1,
    'token': api
}
res = requests.get("https://api.b365api.com/v1/bet365/inplay_filter?sport_id=1&token=217012-1ic4weHvqmtEZ0").json()
result = res['results']
final = []
for resu in result:
    a = resu['league']['name']
    if 'Esoccer' in a:
        final.append(a)

print(final)

with open('testes.txt','w+') as f:
    f.write(str(final))


