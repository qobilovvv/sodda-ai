import os
import warnings

from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from products import Product, format_price_ui, truncate


warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'chat_history.db')}"


def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id, connection=DB_PATH)


SYSTEM_PROMPT = """
Siz Sodda.uz onlayn do'konining quvnoq, samimiy va yordamchi savdo menejerisiz.
Ohang: iliq, do'stona, sotuvga yo'naltirilgan (agressiv emas).

MUHIM CHEGARALAR:
- Do'kon faqat quyidagi yo'nalishlarda sotadi: Katta maishiy texnika, Kichik maishiy texnika, Iqlim texnikasi, Uy uchun elektronika.
- Boshqa mavzular (ovqat, kiyim, xizmatlar, umumiy suhbat, siyosat, tibbiyot va h.k.) bo'yicha javob bermang; muloyim tarzda do'kondagi yo'nalishlarni ayting.

Sizga faqat bitta mahsulot haqida ma'lumot beriladi. Javoblaringiz faqat shu mahsulotga tegishli bo'lsin.
Agar berilgan ma'lumotda javob yo'q bo'lsa, buni aniq ayting va 1 ta aniqlashtiruvchi savol bering.
JAVOB BERISHDA FAQAT HTML TEGLARIDAN (<b>, <i>) FOYDALANING.
"""


llm = ChatOpenAI(model=os.getenv("SODDA_CHAT_MODEL", "gpt-4o-mini"), temperature=0, max_tokens=1500)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def _product_context(p: Product) -> str:
    price_txt = format_price_ui(p.price_uzs)
    return "\n".join(
        [
            f"PRODUCT_ID: {p.id}",
            f"TITLE: {p.title}",
            f"MODEL: {p.model}",
            f"BRAND: {p.brand}",
            f"CATEGORY: {p.category}",
            f"PRICE: {price_txt}",
            f"DESCRIPTION: {truncate(p.description, 900)}",
            f"CHARACTERISTICS: {truncate(p.specs, 1100)}",
        ]
    )


def ask_ai_about_product(user_query: str, chat_id: str, product: Product) -> str:
    question = f"Mahsulot ma'lumoti:\n{_product_context(product)}\n\nMijoz savoli: {user_query}"
    res = with_message_history.invoke({"question": question}, config={"configurable": {"session_id": chat_id}})
    return res.content


def summarize_product(chat_id: str, product: Product) -> str:
    question = (
        "Quyidagi mahsulot bo‘yicha mijozga ko‘rsatish uchun qisqa sotuv kartasi tayyorlang.\n"
        "Talablar:\n"
        "- Narxni faqat USD ko‘rsating.\n"
        "- <b>Tavsif:</b> DESCRIPTION + CHARACTERISTICS asosida 2-3 gap yozing.\n"
        "- <b>Asosiy xarakteristikalari:</b> 4-6 ta punkt.\n"
        "- Yakunda: “Savolingizni yozing” deb chaqiring.\n\n"
        f"Mahsulot ma'lumoti:\n{_product_context(product)}"
    )
    res = with_message_history.invoke({"question": question}, config={"configurable": {"session_id": chat_id}})
    return res.content


def summarize_product_50w(product: Product) -> str:
    """
    Returns a short (<= ~50 words) Uzbek description for list view.
    Not chat-history dependent; should be cheap and reusable/cached.
    """
    question = (
        "Quyidagi mahsulot uchun mijozga ko‘rsatish uchun juda qisqa tavsif yozing.\n"
        "Qoidalar:\n"
        "- Uzbek tilida.\n"
        "- Maksimum 50 ta so‘z.\n"
        "- Faqat 1 paragraf.\n"
        "- <b> va <i> dan boshqa HTML teglardan foydalanmang (ul/li/br ishlatma).\n"
        "- DESCRIPTION va CHARACTERISTICS asosida yozing.\n\n"
        f"{_product_context(product)}"
    )
    res = llm.invoke([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}])
    return res.content


GENERAL_SYSTEM_PROMPT = """
Siz Sodda.uz onlayn do'konining quvnoq va samimiy savdo maslahatchisisiz.
Foydalanuvchi istalgan mavzuda yozishi mumkin: siz muloyim javob bering, lekin suhbatni do'kon yordamiga bog'lab boring.

Do'kon yo'nalishlari: Katta maishiy texnika, Kichik maishiy texnika, Iqlim texnikasi, Uy uchun elektronika.
Sizda yetkazib berish/to'lov siyosati haqida aniq ma'lumot bo'lmasa, uydirmang — aniqlashtiruvchi savol bering.

JAVOB BERISHDA FAQAT HTML TEGLARIDAN (<b>, <i>) FOYDALANING.
"""


def general_chat(chat_id: str, user_query: str) -> str:
    """
    Free-form chat mode. Used when a message isn't a clear product query or DB returns no matches.
    Keeps a friendly seller tone and gently steers toward the shop catalog.
    """
    local_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERAL_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    local_chain = local_prompt | llm
    local_with_history = RunnableWithMessageHistory(
        local_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    res = local_with_history.invoke({"question": user_query}, config={"configurable": {"session_id": chat_id}})
    return res.content
