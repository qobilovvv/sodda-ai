import asyncio
import logging
import os
import re
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.chat_action import ChatActionSender
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from brain import ask_ai
from sync import sync_vector_db
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Assalomu alaykum! Sodda.uz onlayn do'konining botiga xush kelibsiz.\n"
        "Sizga mahsulotlarni topishda va savollaringizga javob berishda yordam bera olaman.\n\n"
        "Yordam uchun /help buyrug'ini yuboring."
    )

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "Botdan foydalanish bo'yicha yordam:\n"
        "1. Mahsulot qidirish uchun uning nomini yoki turini yozing (masalan: 'iPhone 15' yoki 'muzlatgich').\n"
        "2. Mahsulot haqida batafsil ma'lumot olish uchun uni tanlang yoki batafsil so'rang.\n"
        "3. /clear - suhbat tarixini tozalash.\n"
        "4. Savollaringiz bo'lsa, @sodda_admin bilan bog'laning."
    )

@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    from brain import get_session_history
    history = get_session_history(str(message.chat.id))
    history.clear()
    await message.answer("Suhbat tarixi tozalandi!")

@dp.message()
async def handle_message(message: types.Message):
    if not message.text:
        return

    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        try:
            answer = ask_ai(message.text, str(message.chat.id))
            
            # Robust image extraction using regex
            image_match = re.search(r"IMAGE:\s*(https?://[^\s\n]+)", answer)
            
            if image_match:
                image_url = image_match.group(1).strip()
                # Remove the IMAGE: line from the text
                caption = re.sub(r"IMAGE:\s*https?://[^\s\n]+", "", answer).strip()
                
                if image_url and image_url.lower() != "none":
                    try:
                        await message.answer_photo(
                            photo=image_url, 
                            caption=caption, 
                            parse_mode="Markdown"
                        )
                        return
                    except Exception as img_error:
                        logging.error(f"Image send failed: {img_error}")
                        # Fallback if image fails
                        await message.answer(answer, parse_mode="Markdown")
                        return

            await message.answer(answer, parse_mode="Markdown")

        except Exception as e:
            logging.error(f"Error handling message: {e}")
            await message.answer("Kechirasiz, texnik nosozlik yuz berdi. Birozdan so'ng urinib ko'ring.")

async def main():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(sync_vector_db, 'cron', hour=0, minute=0)
    scheduler.start()

    logging.info("🚀 Bot is starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")