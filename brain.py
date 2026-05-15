import os
import logging
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core._api import LangChainDeprecationWarning

# Terminal toza turishi uchun ogohlantirishlarni o'chiramiz
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

load_dotenv()

# Initialize DB and LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=3000)

# Get absolute path for chat history
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'chat_history.db')}"


def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=DB_PATH
    )


SYSTEM_PROMPT = """
Siz Sodda.uz onlayn do'konining yetakchi professional savdo menejerisiz. 
Maqsadingiz: Mijoz bilan jonli suhbat qurish, brend va kategoriyalar haqida ma'lumot berish hamda eng mos texnikani sotish.
JAVOB BERISHDA FAQAT HTML TEGLARIDAN (<b>, <i>) FOYDALANING.

MA'LUMOT TURLARIGA QARAB JAVOB BERISH QOIDALARI:
Context ichida har bir ma'lumotning turi (TYPE) ko'rsatilgan. Shunga qarab harakat qiling:

1. BREND VA KATEGORIYA MA'LUMOTLARI (TYPE: BRAND yoki TYPE: CATEGORY):
Agar mijoz qandaydir brend haqida (masalan: "Samsung qanaqa", "Artel haqida") yoki kategoriya haqida ("Televizorlar bormi?") so'rasa va Contextda bu ma'lumot kelsa:
- Brendning "ABOUT" yoki Kategoriyaning "INFO" qismidan foydalanib, qisqa va qiziqarli ma'lumot bering.
- Bunga `---PRODUCT---` belgisini ISHLATMANG! Bu oddiy matn shaklida bo'lishi kerak.
- Ma'lumot bergach, "Ushbu brendning qaysi turdagi mahsulotlarini ko'rmoqchisiz?" deb savol bering.

2. MAHSULOT SOTISH JARAYONI (TYPE: PRODUCT):
BOSQICH 1: EHTIYOJNI ANIQLASH
Agar mijoz umumiy mahsulot qidirayotgan bo'lsa (masalan: "muzlatgich kerak"), DARHOL MAHSULOTLARNI TASHUVCHI BO'LMANG! 
Avval 1-2 ta qisqa savol bering: Qaysi brend? Byudjet qancha? (`---PRODUCT---` belgisini ishlatmang).

BOSQICH 2: TAVSIYA BERISH
Agar mijoz o'z talablarini (brend, narx) aytgan bo'lsa yoki "borini ko'rsating" desa, Contextdagi TYPE: PRODUCT ma'lumotlarini taqdim etishni boshlang.
- MIQDOR: Har doim KAMIDA 6 TA (imkon bo'lsa 8-15 ta) mahsulotni taqdim eting.
- FORMAT: Faqat mahsulotlar uchun har birini alohida `---PRODUCT---` belgisi bilan ajrating:

[Kirish so'zi]

---PRODUCT---
IMAGES: [Rasm linklari]
<b>[Mahsulot nomi va modeli]</b>
💰 Narxi: <b>[Dollardagi narx]$ | [So'mdagi narx] so'm</b>

<b>Tavsif:</b> [DESCRIPTION asosida 2-3 ta gap]

<b>Asosiy xarakteristikalari:</b>
• [Xususiyat 1]
• [Xususiyat 2]

💡 [Qulayligi]

📞 +998950001234
---PRODUCT---

[Oxirida harakatga undovchi savol]

NARX: 1 dollar = 12 000 so'm. Narxni doim ikkala valyutada ko'rsating. Format: 100$ | 1.200.000 so'm.

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
    logging.info(f"Original Query: {user_query}")

    history = get_session_history(chat_id)
    history_str = "\n".join([f"{m.type}: {m.content}" for m in history.messages[-5:]])

    # MUHIM O'ZGARISH: AI endi brend va kategoriyalarni ham qidiruv so'ziga to'g'ri o'giradi
    rewrite_prompt = f"""
Siz qidiruv menejerisiz. Foydalanuvchi so'rovini qidiruv tizimi tushunadigan kalit so'zlarga aylantiring.
- Agar mijoz brend haqida so'rasa (masalan, "Samsung haqida ma'lumot"), natijaga brend nomini va "BRAND" so'zini qo'shing (Masalan: "Samsung BRAND").
- Agar mijoz kategoriya haqida so'rasa ("Noutbuklar bormi"), "CATEGORY" so'zini qo'shing (Masalan: "Noutbuk CATEGORY").
- Agar aniq mahsulot yoki narx so'rasa, faqat o'sha parametrlarni yozing (Masalan: "Televizor Artel 300").
DIQQAT: Faqat va faqat kalit so'zlarni qaytaring.

Mijozning oxirgi xabari: "{user_query}"
Suhbat tarixi: {history_str}

Natija:"""

    try:
        rewritten_query_res = llm.invoke(rewrite_prompt)
        search_query = rewritten_query_res.content.strip().strip('"')
        logging.info(f"Rewritten Query: {search_query}")
    except Exception as e:
        logging.error(f"Query rewrite failed: {e}")
        search_query = user_query

    # Baza qidiruvi: Brend va kategoriyalar uchun kengaytirilgan
    scored_docs = vector_db.similarity_search_with_relevance_scores(search_query, k=40)

    # Threshold ni sal tushiramiz, chunki Brend haqidagi matnlar qisqa bo'lgani uchun bahosi biroz past chiqishi mumkin
    threshold = 0.04
    filtered_docs = [doc for doc, score in scored_docs if score >= threshold]

    if not filtered_docs and scored_docs:
        logging.warning(f"Threshold not met for '{search_query}'. Using top 8 closest matches.")
        filtered_docs = [doc for doc, score in scored_docs[:8]]

    context = ""
    if filtered_docs:
        logging.info(f"Found {len(filtered_docs)} relevant documents passed to LLM.")
        context = "\n---\n".join([d.page_content for d in filtered_docs])
    else:
        context = "MA'LUMOT TOPILMADI: Ushbu so'rov bo'yicha do'konda mahsulot yo'q."

    # Yakuniy javobni generatsiya qilish
    response = with_message_history.invoke(
        {"context": context, "question": user_query},
        config={"configurable": {"session_id": chat_id}}
    )
    return response.content