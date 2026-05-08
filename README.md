# Sodda AI - Telegram Bot

Sodda.uz uchun sun'iy intellektga asoslangan Telegram bot. Bot mijozlarga mahsulotlarni topishda va ular haqida ma'lumot olishda yordam beradi.

## Imkoniyatlar

- **RAG (Retrieval-Augmented Generation):** Mahsulotlar bazasidan qidirish va GPT-4o-mini yordamida javob qaytarish.
- **Persistent Chat History:** Suhbatlar tarixi SQLite bazasida saqlanadi.
- **Bilingual Support:** O'zbek va Rus tillarida javob bera oladi.
- **Auto-Sync:** Mahsulotlar bazasi har kuni avtomatik ravishda yangilanadi.
- **Robust Image Handling:** Mahsulot rasmlarini Telegram orqali yuborish.

## O'rnatish

1.  Repozitoriyani klonlang.
2.  Virtual muhit yarating va faollashtiring:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    ```
3.  Zarur kutubxonalarni o'rnating:
    ```bash
    pip install -r requirements.txt
    ```
4.  `.env` faylini yarating va quyidagi o'zgaruvchilarni to'ldiring:
    ```env
    TELEGRAM_TOKEN=your_token
    OPENAI_API_KEY=your_key
    DB_HOST=127.0.0.1
    DB_USERNAME=root
    DB_PASSWORD=secret
    DB_DATABASE=sodda_db
    ```
5.  Vektor bazasini birinchi marta sinxronizatsiya qiling:
    ```bash
    python sync.py
    ```
6.  Botni ishga tushiring:
    ```bash
    python bot.py
    ```

## Fayllar tuzilmasi

- `bot.py`: Telegram bot interfeysi (aiogram).
- `brain.py`: AI mantiqi va RAG (LangChain).
- `sync.py`: MySQL bazasidan ma'lumotlarni ChromaDB ga o'tkazish.
- `chat_history.db`: Suhbatlar tarixi saqlanadigan SQLite fayli.
- `chroma_db/`: Vektorli ma'lumotlar bazasi.

## Kelajakdagi rejalar

- [ ] Mahsulotlarni buyurtma qilish imkoniyatini qo'shish.
- [ ] Foydalanuvchi statistikasini yig'ish.
- [ ] Redis orqali kesh tizimini joriy qilish.
