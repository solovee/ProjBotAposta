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


# Função principal
def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    

    print("🤖 Bot rodando...")
    app.run_polling()



def sendMessages(chat_id, text):
    TOKEN = '7857822617:AAH_pNvbi7M1254hwLDeJA4KKyKiZYdHTzM'
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
    params = {
        'chat_id': chat_id,
        'text': text
    }

    response = requests.get(url, params=params)
    print(response.json())
    
