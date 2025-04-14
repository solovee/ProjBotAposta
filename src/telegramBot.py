from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import os
from dotenv import load_dotenv
from telegram import Bot



load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
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



def sendMessage(chat_id, text):
    bot.send_message(chat_id=chat_id, text=text)
