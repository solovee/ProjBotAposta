from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import os
from dotenv import load_dotenv
from telegram import Bot
import logging
import requests


logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
bot = Bot(token=TELEGRAM_TOKEN)
#chatid 7815720131

# Comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    print(f"📩 Chat ID: {chat_id}")
    await update.message.reply_text("👋 Olá! Eu sou o bot de análise de apostas FIFA.")

# Comando /matches (exemplo)
async def matches(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensagens = [
        "⚽ Jogo 1: Time A x Time B - Over 2.5 recomendado",
        "⚽ Jogo 2: Time C x Time D - Under 2.5 recomendado",
        "⚽ Jogo 3: Time E x Time F - Handicap Asiático -1.5"
    ]
    for msg in mensagens:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

# Função principal
def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("matches", matches))

    print("🤖 Bot rodando...")
    app.run_polling()



def sendMessages(chat_id, text):
    TOKEN = '7857822617:AAH_pNvbi7M1254hwLDeJA4KKyKiZYdHTzM'
    chat_id = '7815720131'
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
    params = {
        'chat_id': chat_id,
        'text': text
    }

    response = requests.get(url, params=params)
    print(response.json())
    '''
    try:
        bot.send_message(chat_id=chat_id, text=text)  # ou sem parse_mode para testar
        logger.info(f"✅ Mensagem enviada com sucesso: {text}")
    except Exception as e:
        logger.error(f"❌ Falha ao enviar mensagem: {e}")
import requests





url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
params = {
    'chat_id': chat_id,
    'text': text
}

response = requests.get(url, params=params)
print(response.json())
'''
