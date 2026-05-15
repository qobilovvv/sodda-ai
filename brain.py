import os
import logging
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# Initialize DB and LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Get absolute path for chat history
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'chat_history.db')}"


def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=DB_PATH
    )

USD_TO_UZS = float(os.getenv("USD_TO_UZS", "12000"))

_TELEGRAM_ALLOWED_TAGS = {"b", "i", "a"}


def sanitize_telegram_html(text: str) -> str:
    """
    Avoid Aiogram/Telegram HTML parse errors by allowing only <b>, <i>, <a>.
    """
    if not text:
        return text

    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"</?(p|ul|ol|li|div|span|h\d|blockquote|code|pre)[^>]*>", "\n", text, flags=re.IGNORECASE)

    def _strip_unknown_tag(m: re.Match) -> str:
        tag = (m.group(1) or "").lower()
        return m.group(0) if tag in _TELEGRAM_ALLOWED_TAGS else ""

    text = re.sub(r"</?\s*([a-zA-Z0-9]+)(?:\s+[^>]*)?>", _strip_unknown_tag, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_prices_for_vector_search(text: str) -> str:
    """
    Vector DB dislikes numeric budget constraints; remove them from search query.
    """
    if not text:
        return text
    cleaned = text
    cleaned = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:\$|usd|dollar|доллар)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d[\d\s.,]*\s*(?:uzs|so['’`]?m|som)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:mln|million|m)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_QUESTION_WORDS_RE = re.compile(
    r"\b(qanday|qanaqa|qaysi|qaysilar|nima|nechchi|necha|farqi|solishtir|taqqosla|tavsiya|tavsiyalaysiz|bor(mi)?|kerak)\b",
    re.IGNORECASE,
)


def is_generic_product_request(user_query: str) -> bool:
    """
    Broad category request without constraints -> ask clarifying questions first.
    """
    q = (user_query or "").strip()
    if not q:
        return True
    if "?" in q or _QUESTION_WORDS_RE.search(q):
        return False
    if re.search(r"\b\d{2,}\b", q):  # model/size numbers
        return False
    # If user mentions any currency/budget, it's not generic.
    if re.search(r"(\$|usd|so['’`]?m|uzs|mln|million)\b", q, flags=re.IGNORECASE):
        return False
    return len(q.split()) <= 3


SYSTEM_PROMPT = """
Siz Sodda.uz onlayn do'konining yetakchi professional savdo menejerisiz. 
Maqsadingiz: Mijozlarga eng mos maishiy texnikani topish, ularga batafsil ma'lumot berish va xaridga undash.

JAVOB HTML QOIDASI:
- FAQAT <b>, <i>, <a> teglari. Boshqa teglardan foydalanmang.
- Iloji boricha sodda va aniq yozing, mijoz savoliga to'g'ridan-to'g'ri javob bering.

SUHBAT USLUBI:
- Agar mijoz juda umumiy so'rov bersa (masalan: "muzlatgich", "tv", "kir yuvish"), mahsulot tashlashdan oldin 2-3 ta aniqlashtiruvchi savol bering:
  1) Byudjet (so'm yoki $)
  2) Brend (xohlagan/xohlamagan)
  3) Muhim talab (hajm/o'lcham/energiya tejamkorlik/shovqin va h.k.)
- Agar mijoz savol bersa (masalan: "qaysi biri yaxshi?", "farqi nima?"), avval qisqa tushuntiring, keyin mos mahsulotlarni bering.

QAT'IY QOIDALAR (ANTI-HALLUCINATION - JUDA MUHIM):
1. FAQAT "Context" ichida berilgan mahsulotlarni tavsiya qiling. 
2. O'ZINGIZDAN MAHSULOT YARATMANG: Hech qachon bazada yo'q mahsulotni (masalan, iPhone yoki boshqa brendlar) o'ylab topmang! 
3. TOPILMAGAN HOLATDA: Agar "Context" ichida mahsulot bo'lmasa yoki "MA'LUMOT TOPILMADI" bo'lsa, mijozga: "Kechirasiz, hozircha do'konimizda bu turdagi mahsulot yo'q" deb javob bering va HECH QANDAY mahsulot taklif qilmang!
4. FAQAT "Context" ichidagi mahsulotlarni ko'rsating.
5. MIQDOR MUHIM: Har doim kamida 8 ta mahsulotni tavsiya qilishga harakat qiling. Agar contextda 8 tadan ko'p mahsulot bo'lsa, ularni tashlab yubormang!
6. MOSLIK: Agar mijoz so'ragan narsaga 100% mos mahsulot bo'lmasa, unga eng yaqin bo'lgan (masalan: boshqa brend, o'xshash narx) mahsulotlarni "Sizga mana bular ham ma'qul kelishi mumkin" deb taklif qiling.
7. "YO'Q" DEYISH: Faqatgina Context mutlaqo bo'sh bo'lsa yoki televizor so'ralganda faqat dazmol chiqib kelsa, mahsulot yo'q deb ayting.

MUKAMMAL MENEJER QOIDALARI:
1. Do'stona bo'ling: Gapni har doim iliq va professional salomlashish bilan boshlang.
2. TAVSIYA SONI: Mijoz aniq sonini aytmasa, doim 8 tadan 15 tagacha eng yaxshi mahsulotni tavsiya qiling (agar Contextda bo'lsa).
3. VARIANTLAR: Kamida 3 xil variantni yoritishga harakat qiling (arzon/optimal/premium yoki turli funksiyali).
3. BATAシューズIL MA'LUMOT: Context dagi CHARACTERISTICS va DESCRIPTION maydonlaridan foydalanib, har bir mahsulotning texnik imkoniyatlarini to'liq yoritib bering.

NARX VA VALYUTA:
- 1 dollar = 12 000 so'm kursi bo'yicha hisoblang.
- Narxni har doim ikkala valyutada ko'rsating. Masalan: 100$ | 1.200.000 so'm.

JAVOB FORMATI VA AJRATUVCHILAR (STRUKTURA):
Siz mahsulotlarni alohida xabar sifatida shakllantirishingiz kerak. Buning uchun har bir mahsulot blokini `---PRODUCT---` belgisi bilan ajrating.

Struktura quyidagicha bo'lishi SHART (HTML formatida):

[1. Mijozga qisqacha kirish so'zi]

---PRODUCT---
IMAGES: [Rasm linklari vergul bilan ajratilgan, agar yo'q bo'lsa None]
<b>[Mahsulot nomi va modeli]</b>
💰 Narxi: <b>[Dollardagi narx]$ | [So'mdagi narx] so'm</b>

<b>Tavsif:</b> [Context dagi DESCRIPTION asosida 2-3 ta gap]

<b>Asosiy xarakteristikalari:</b>
• [Xususiyat 1 (masalan: Xotira: 256GB yoki Quvvat: 2000W)]
• [Xususiyat 2]
• [Xususiyat 3]
• [Xususiyat 4]

💡 [Bu mahsulot nega mijozga mos kelishi haqida bitta ajoyib qulaylik]

📞 +998950001234
---PRODUCT---


---PRODUCT---
IMAGES: [Keyingi mahsulot rasmi]
<b>[Keyingi Mahsulot nomi]</b>
... (shu tartibda kamida 8-15 ta mahsulot)
---PRODUCT---

[Eng oxirgi mahsulotdan so'ng, qandaydir harakatga undovchi savol bering. Masalan: "Ushbu modellardan qaysi biri sizga ko'proq ma'qul keldi?"]

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def ask_ai(user_query: str, chat_id: str):
    global vector_db
    logging.info(f"Original Query: {user_query}")

    history = get_session_history(chat_id)
    history_str = "\n".join([f"{m.type}: {m.content}" for m in history.messages[-5:]])

    # If user request is too generic, ask clarifying questions before product dump.
    if is_generic_product_request(user_query):
        clarify_prompt = f"""
Siz Sodda.uz savdo menejerisiz.
Mijoz juda umumiy so'rov berdi. Mahsulot tavsiya qilishdan oldin 2-3 ta qisqa savol bering:
1) Byudjet (so'm yoki $)
2) Brend (xohlagan/xohlamagan)
3) Eng muhim talab (o'lcham/hajm/quvvat/shovqin/energiya tejamkorlik va h.k.)

Faqat savollarni va bitta iliq jumlani yozing. `---PRODUCT---` ishlatmang. Faqat <b>/<i>/<a> HTML.
Mijoz: "{user_query}"
Natija:
"""
        try:
            qres = llm.invoke(clarify_prompt)
            return sanitize_telegram_html(qres.content)
        except Exception:
            return "Qaysi brendni xohlaysiz va byudjet qancha? (so'm yoki $) Yana eng muhim talablingiz nima?"

    # 1. FIXED: Query Rewriting - Retain brand, specs, and attributes!
    user_query_for_search = strip_prices_for_vector_search(user_query)
    rewrite_prompt = f"""
Siz qidiruv menejerisiz. Foydalanuvchi so'rovini qidiruv tizimi tushunadigan kalit so'zlarga aylantiring.
DIQQAT: Vektor qidiruv narx/byudjetni yaxshi tushunmaydi. Shuning uchun mijoz yozgan narx/byudjetni qidiruv so'zidan KESIB TASHLANG.
DIQQAT: Faqat va faqat kalit so'zlarni qaytaring. Hech qanday kirish so'zi (masalan: "Qidiruv so'zi:") bo'lmasin!

Mijoz: "{user_query_for_search}"
Tarix: {history_str}

Natija (Faqat kalit so'zlar):"""

    try:
        rewritten_query_res = llm.invoke(rewrite_prompt)
        search_query = rewritten_query_res.content.strip().strip('"')
        logging.info(f"Rewritten Query: {search_query}")
    except Exception as e:
        logging.error(f"Query rewrite failed: {e}")
        search_query = user_query

    # 2. FIXED: Fetch a much larger pool of products (k=35)
    try:
        scored_docs = vector_db.similarity_search_with_relevance_scores(search_query, k=120)
    except Exception as e:
        # If sync recreated chroma_db while bot is running, reload once.
        logging.error(f"Vector search error: {e}")
        try:
            vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            scored_docs = vector_db.similarity_search_with_relevance_scores(search_query, k=120)
        except Exception as e2:
            logging.error(f"Vector search retry failed: {e2}")
            return "Hozir mahsulotlar bazasi yangilanmoqda. Iltimos 1-2 daqiqadan so'ng qayta urinib ko'ring."

    # FIXED: Lower threshold to ensure we don't accidentally drop good matches.
    threshold = 0.02
    filtered_docs = [doc for doc, score in scored_docs if score >= threshold]

    # Fallback to top 5 if the user typed something totally abstract but we still want to try
    if not filtered_docs and scored_docs:
        logging.warning(f"Threshold not met for '{search_query}'. Using top 20 closest matches.")
        filtered_docs = [doc for doc, score in scored_docs[:20]]

    context = ""
    if filtered_docs:
        logging.info(f"Found {len(filtered_docs)} relevant documents passed to LLM.")
        context = "\n---\n".join([d.page_content for d in filtered_docs])
    else:
        context = "MA'LUMOT TOPILMADI: Ushbu mahsulot bazada mavjud emas."

    # 3. Generate response
    response = with_message_history.invoke(
        {"context": context, "question": user_query},
        config={"configurable": {"session_id": chat_id}}
    )
    return sanitize_telegram_html(response.content)
