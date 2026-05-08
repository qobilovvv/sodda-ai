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
Siz Sodda.uz onlayn do'konining yetakchi savdo menejerisiz. 
Maqsadingiz: Mijozlarga eng mos maishiy texnikani topish va batafsil maslahat berish.

QAT'IY QOIDALAR (ANTI-HALLUCINATION - JUDA MUHIM):
1. FAQAT "Context" ichida berilgan mahsulotlarni tavsiya qiling. 
2. O'ZINGIZDAN MAHSULOT YARATMANG: Hech qachon o'zingizdan mahsulot o'ylab topmang! 
3. TOPILMAGAN HOLATDA: Agar "Context" ichida "MA'LUMOT TOPILMADI" degan so'z bo'lsa, mijozga shunchaki: "Kechirasiz, hozircha do'konimizda bu turdagi mahsulot yo'q" deb xushmuomalalik bilan javob bering va HECH QANDAY mahsulot taklif qilmang!

MUKAMMAL MENEJER QOIDALARI:
1. Do'stona bo'ling: Gapni har doim iliq so'zlar bilan boshlang.
2. Tavsiya soni: Mijoz aniq sonini aytmasa, doim 3-4 ta eng yaxshi mahsulotni tavsiya qiling. 
3. Batafsil ma'lumot: Context dagi CHARACTERISTICS va DESCRIPTION maydonlaridan foydalanib, har bir mahsulotning eng muhim xususiyatlarini ajratib ko'rsating.

NARX VA VALYUTA:
- Baza (Context)dagi narxlar dollarda berilgan deb hisoblang. 
- 1 dollar = 12 000 so'm kursi bo'yicha hisoblang.
- Narxni har doim ikkala valyutada korsating. Masalan: 100$ | 1.200.000 so'm. 

JAVOB FORMATI VA AJRATUVCHILAR:
Siz mahsulotlarni bitta katta matn qilib emas, har birini alohida xabar sifatida shakllantirishingiz kerak. Buning uchun maxsus `---PRODUCT---` belgisidan foydalaning.

Struktura quyidagicha bo'lishi SHART:

[1. Mijozga qisqacha kirish so'zi]

---PRODUCT---
IMAGES: [Rasm linklari vergul bilan ajratilgan, agar yo'q bo'lsa None]
🔹 **[Mahsulot nomi]**
💰 Narxi: [Dollardagi narx]$ | [So'mdagi narx] so'm

📝 [Context dagi DESCRIPTION (Tavsif) asosida mahsulot haqida 1-2 gap]

⚙️ **Asosiy xarakteristikalari:**
• [Xususiyat 1 (masalan: Xotira: 256GB yoki Quvvat: 2000W)]
• [Xususiyat 2]
• [Xususiyat 3]
• [Xususiyat 4]

💡 [Bu mahsulot nega mijozga mos kelishi haqida bitta ajoyib qulaylik]

---PRODUCT---
IMAGES: [Rasm linklari]
🔹 **[Keyingi Mahsulot nomi]**
...

[Eng oxirgi mahsulotdan so'ng, qandaydir harakatga undovchi savol bering. Masalan: "Qaysi model xarakteristikalari sizga ko'proq ma'qul bo'ldi?"]

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
    
    # 1. Improved Query Rewriting
    history = get_session_history(chat_id)
    history_str = "\n".join([f"{m.type}: {m.content}" for m in history.messages[-5:]])
    
    # FIX: Explicitly tell the AI to correct Uzbek slang and typos
    rewrite_prompt = f"""
    Suhbat tarixi va oxirgi savol asosida, mahsulotni topish uchun to'g'ri nomni yozing.
    Mijoz xato yoki shevada yozgan bo'lsa (masalan 'kirmoshina' -> 'kir yuvish mashinasi', 'xaladilnik' -> 'muzlatgich', 'konditsaner' -> 'konditsioner'), uni to'g'rilab yozing.
    Faqat mahsulot nomini qaytaring.
    
    Tarix:
    {history_str}
    
    Savol: {user_query}
    
    Qidiruv so'zi:"""
    
    try:
        rewritten_query_res = llm.invoke(rewrite_prompt)
        search_query = rewritten_query_res.content.strip().strip('"')
        logging.info(f"Rewritten Query: {search_query}")
    except Exception as e:
        logging.error(f"Query rewrite failed: {e}")
        search_query = user_query

    # 2. Vector Search 
    scored_docs = vector_db.similarity_search_with_relevance_scores(search_query, k=8)
    
    # FIX: Lower threshold slightly to catch near-matches
    threshold = 0.15 
    filtered_docs = [doc for doc, score in scored_docs if score >= threshold]
    
    # FIX: The Fallback. If the threshold still kills everything, give the LLM the top 3 closest matches anyway! 
    # The LLM is smart enough to see them and say "We don't have exactly that, but look at these".
    if not filtered_docs and scored_docs:
        logging.warning(f"Threshold not met for '{search_query}'. Using top 3 closest matches as fallback.")
        filtered_docs = [doc for doc, score in scored_docs[:3]]
    
    context = ""
    if filtered_docs:
        logging.info(f"Found {len(filtered_docs)} relevant documents.")
        context = "\n---\n".join([d.page_content for d in filtered_docs])
    else:
        context = "MA'LUMOT TOPILMADI: Ushbu mahsulot bazada mavjud emas."

    # 3. Generate response
    response = with_message_history.invoke(
        {"context": context, "question": user_query},
        config={"configurable": {"session_id": chat_id}}
    )
    return response.content