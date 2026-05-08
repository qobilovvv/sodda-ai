import asyncio
import logging
import os
import re
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.chat_action import ChatActionSender
from aiogram.utils.media_group import MediaGroupBuilder
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

async def get_valid_image_url(url: str) -> str | None:
    """
    Checks the /thumbs/ path first, then falls back to the main /products/ path.
    Returns None if both return 404 or fail.
    """
    # Extract the base filename (e.g., 'A8HuMrc2npMJZDAPwP2a6eyIYpgpKpFCgooCDnkm.webp')
    filename = url.split("/")[-1]
    
    thumb_url = f"https://sodda.uz/storage/products/thumbs/{filename}"
    main_url = f"https://sodda.uz/storage/products/{filename}"

    # Use aiohttp to send a lightweight HEAD request
    async with aiohttp.ClientSession() as session:
        try:
            # 1. Try thumb first
            async with session.head(thumb_url, timeout=2) as resp:
                if resp.status == 200:
                    return thumb_url
        except Exception as e:
            logging.debug(f"Thumb URL failed: {e}")
            pass
        
        try:
            # 2. Try main image if thumb fails
            async with session.head(main_url, timeout=2) as resp:
                if resp.status == 200:
                    return main_url
        except Exception as e:
            logging.debug(f"Main URL failed: {e}")
            pass
            
    # 3. Both failed (e.g., 404 Not Found)
    return None


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
            # 🚀 1. Get AI response in a background thread to prevent freezing
            answer = await asyncio.to_thread(ask_ai, message.text, str(message.chat.id))

            # 2. Split products
            parts = answer.split("---PRODUCT---")

            # 3. Send intro text
            intro_text = parts[0].strip()
            if intro_text:
                await message.answer(intro_text, parse_mode="Markdown")
                await asyncio.sleep(0.5)

            # 4. Process products
            for part in parts[1:]:
                part = part.strip()

                if not part:
                    continue

                # Extract IMAGES tag
                images_match = re.search(r"IMAGES?:\s*([^\n]+)", part)

                # Remove image line from caption
                caption = re.sub(r"IMAGES?:\s*[^\n]+", "", part).strip()

                # Telegram caption limit
                if len(caption) > 1024:
                    caption = caption[:1020] + "..."

                image_urls = []

                if images_match:
                    image_links_str = images_match.group(1).strip()

                    raw_urls = [
                        u.strip()
                        for u in image_links_str.split(",")
                        if "http" in u
                        and "IMAGE_LINKS" not in u
                        and "None" not in u
                    ]

                    # Validate image URLs concurrently
                    if raw_urls:
                        tasks = [
                            get_valid_image_url(u)
                            for u in raw_urls[:10]
                        ]

                        results = await asyncio.gather(*tasks)

                        # Remove failed URLs (the Nones)
                        image_urls = [
                            url for url in results
                            if url is not None
                        ]

                # Send logic
                if image_urls:

                    # Single image
                    if len(image_urls) == 1:
                        try:
                            await message.answer_photo(
                                photo=image_urls[0],
                                caption=caption,
                                parse_mode="Markdown"
                            )

                        except Exception as img_error:
                            logging.error(f"Single image failed: {img_error}")
                            await message.answer(
                                caption,
                                parse_mode="Markdown"
                            )

                    # Multiple images
                    else:
                        try:
                            media_group = MediaGroupBuilder(caption=caption)

                            for url in image_urls:
                                media_group.add_photo(media=url)

                            await message.answer_media_group(
                                media=media_group.build()
                            )

                        except Exception as album_error:
                            logging.error(f"Album failed: {album_error}")

                            await message.answer(
                                caption,
                                parse_mode="Markdown"
                            )

                else:
                    # No valid images
                    await message.answer(
                        caption,
                        parse_mode="Markdown"
                    )

                # Small delay between products
                await asyncio.sleep(0.5)

        except Exception as e:
            logging.error(f"Error handling message: {e}")

            await message.answer(
                "Kechirasiz, texnik nosozlik yuz berdi. "
                "Birozdan so'ng urinib ko'ring."
            )

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