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

from aiogram.utils.media_group import MediaGroupBuilder

...

@dp.message()
async def handle_message(message: types.Message):
    if not message.text:
        return

    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        try:
            answer = ask_ai(message.text, str(message.chat.id))
            
            # 1. Look for IMAGES: tag (can be multiple comma-separated URLs)
            images_match = re.search(r"IMAGES?:\s*([^\n]+)", answer)
            
            if images_match:
                image_links_str = images_match.group(1).strip()
                # Remove the IMAGES line from the text for the caption/answer
                caption = re.sub(r"IMAGES?:\s*[^\n]+", "", answer).strip()
                
                # Split and filter out placeholders
                image_urls = [u.strip() for u in image_links_str.split(",") if "http" in u and "IMAGE_LINKS" not in u]
                
                if image_urls:
                    if len(image_urls) == 1:
                        # Single image
                        try:
                            await message.answer_photo(
                                photo=image_urls[0], 
                                caption=caption, 
                                parse_mode="Markdown"
                            )
                            return
                        except Exception as img_error:
                            logging.error(f"Single image send failed: {img_error}")
                    else:
                        # Multiple images (Album)
                        try:
                            media_group = MediaGroupBuilder(caption=caption)
                            # Telegram media group limit is 10
                            for url in image_urls[:10]:
                                media_group.add_photo(media=url)
                            
                            await message.answer_media_group(media=media_group.build())
                            return
                        except Exception as album_error:
                            logging.error(f"Album send failed: {album_error}")
                            # Fallback to text if album fails
                            await message.answer(caption, parse_mode="Markdown")
                            return

            # 2. If no valid image URL was found, clean the text and send as text
            clean_answer = re.sub(r"IMAGES?:.*", "", answer).strip()
            await message.answer(clean_answer, parse_mode="Markdown")

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