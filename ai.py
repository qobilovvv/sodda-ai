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

MUHIM:
- Agar suhbat allaqachon boshlangan bo'lsa (history mavjud bo'lsa), qayta "Salom/Assalomu alaykum" deb boshlamang.
"""


llm = ChatOpenAI(model=os.getenv("SODDA_CHAT_MODEL", "gpt-4o-mini"), temperature=0, max_tokens=1500)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

def _last_messages(chat_id: str, limit: int = 10):
    """
    Returns last `limit` messages (human+ai) from persisted SQLite history.
    """
    hist = get_session_history(chat_id)
    try:
        return list(hist.messages[-limit:])
    except Exception:
        return list(hist.messages)


def _invoke_with_last_history(system_prompt: str, chat_id: str, user_text: str, limit: int = 10) -> str:
    """
    Manual history windowing: send only the latest N messages to the model.
    Persists the new exchange back into SQLite.
    """
    hist = get_session_history(chat_id)
    last = _last_messages(chat_id, limit=limit)
    sys = system_prompt
    if last:
        sys = (sys or "") + "\n\nConversation already started. Do NOT greet again."
    msgs = [{"role": "system", "content": sys}]
    for m in last:
        role = "assistant" if getattr(m, "type", "") == "ai" else "user"
        content = getattr(m, "content", "") or ""
        if content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_text})

    res = llm.invoke(msgs)
    out = res.content or ""
    try:
        hist.add_user_message(user_text)
        hist.add_ai_message(out)
    except Exception:
        pass
    return out


def _product_context(p: Product, *, names_only: bool = False) -> str:
    price_txt = format_price_ui(p.price_uzs)
    base = [
        f"PRODUCT_ID: {p.id}",
        f"TITLE: {p.title}",
        f"MODEL: {p.model}",
        f"BRAND: {p.brand}",
        f"CATEGORY: {p.category}",
        f"PRICE: {price_txt}",
    ]
    if names_only:
        return "\n".join(base)
    return "\n".join(
        base
        + [
            f"DESCRIPTION: {truncate(p.description, 900)}",
            f"CHARACTERISTICS: {truncate(p.specs, 1100)}",
        ]
    )


def ask_ai_about_product(user_query: str, chat_id: str, product: Product) -> str:
    question = f"Mahsulot ma'lumoti:\n{_product_context(product)}\n\nMijoz savoli: {user_query}"
    return _invoke_with_last_history(SYSTEM_PROMPT, chat_id, question, limit=10)


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
    return _invoke_with_last_history(SYSTEM_PROMPT, chat_id, question, limit=10)


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
        f"{_product_context(product, names_only=True)}"
    )
    res = llm.invoke([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}])
    return res.content


GENERAL_SYSTEM_PROMPT = """
Siz Sodda.uz onlayn do'konining quvnoq va samimiy savdo maslahatchisisiz.
Foydalanuvchi istalgan mavzuda yozishi mumkin: siz muloyim javob bering, lekin suhbatni do'kon yordamiga bog'lab boring.

Do'kon yo'nalishlari: Katta maishiy texnika, Kichik maishiy texnika, Iqlim texnikasi, Uy uchun elektronika.
Sizda yetkazib berish/to'lov siyosati haqida aniq ma'lumot bo'lmasa, uydirmang — aniqlashtiruvchi savol bering.

JAVOB BERISHDA FAQAT HTML TEGLARIDAN (<b>, <i>) FOYDALANING.

MUHIM:
- Agar suhbat allaqachon boshlangan bo'lsa (history mavjud bo'lsa), qayta "Salom/Assalomu alaykum" deb boshlamang.
"""


def general_chat(chat_id: str, user_query: str) -> str:
    """
    Free-form chat mode. Used when a message isn't a clear product query or DB returns no matches.
    Keeps a friendly seller tone and gently steers toward the shop catalog.
    """
    return _invoke_with_last_history(GENERAL_SYSTEM_PROMPT, chat_id, user_query, limit=10)
