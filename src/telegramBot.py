from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, CallbackContext
#from api_client import BetsAPIClient
import os
from dotenv import load_dotenv
from desregulation_calcs import calculaMediaDeGols, eventosParaOdds


load_dotenv()


# Carregar tokens de ambiente
#API_KEY = os.getenv('BETSAPI_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Inicializar os clientes
#bets_api_client = BetsAPIClient(API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)

# Comando para iniciar o bot

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Olá! Eu sou o bot de análise de apostas fifa. Aguarde enquanto eu faço as análises.')
'''
def get_matches(update: Update, context: CallbackContext) -> None:
    try:
        matches = bets_api_client.get_fifa_matches()

        if matches['success'] == 1:
            response = '📊 Jogos disponíveis:
'
            for match in matches['results'][:5]:  # Mostra os 5 primeiros jogos
                response += f"- {match['home']['name']} x {match['away']['name']}
"
            update.message.reply_text(response)
        else:
            update.message.reply_text('Nenhum jogo encontrado no momento.')
    except Exception as e:
        update.message.reply_text(f'Erro ao buscar os jogos: {str(e)}')
'''

def main() -> None:
    updater = Updater(TELEGRAM_TOKEN)

    # Registrar comandos
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('matches', get_matches))

    # Iniciar o bot
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
