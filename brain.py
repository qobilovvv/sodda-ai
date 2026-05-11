import os
import logging
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


SYSTEM_PROMPT = """
Siz Sodda.uz onlayn do'konining yetakchi professional savdo menejerisiz. 
Maqsadingiz: Mijozlarga eng mos maishiy texnikani topish, ularga batafsil ma'lumot berish va xaridga undash.

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
    logging.info(f"Original Query: {user_query}")
    
    history = get_session_history(chat_id)
    history_str = "\n".join([f"{m.type}: {m.content}" for m in history.messages[-5:]])
    
    # 1. FIXED: Query Rewriting - Retain brand, specs, and attributes!
    rewrite_prompt = f"""
Siz qidiruv menejerisiz. Foydalanuvchi so'rovini qidiruv tizimi tushunadigan kalit so'zlarga aylantiring.
DIQQAT: Faqat va faqat kalit so'zlarni qaytaring. Hech qanday kirish so'zi (masalan: "Qidiruv so'zi:") bo'lmasin!

Mijoz: "{user_query}"
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
    scored_docs = vector_db.similarity_search_with_relevance_scores(search_query, k=40)
    
    # FIXED: Lower threshold to ensure we don't accidentally drop good matches.
    threshold = 0.05 
    filtered_docs = [doc for doc, score in scored_docs if score >= threshold]
    
    # Fallback to top 5 if the user typed something totally abstract but we still want to try
    if not filtered_docs and scored_docs:
        logging.warning(f"Threshold not met for '{search_query}'. Using top 5 closest matches.")
        filtered_docs = [doc for doc, score in scored_docs[:5]]
    
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
    return response.content