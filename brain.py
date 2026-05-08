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
Siz Sodda.uz onlayn do'konining professional va xushmuomala menejerisiz. 

QAT'IY QOIDALAR (ANTI-HALLUCINATION):
1. FAQAT KONTEKSTGA TAYANING: Agar so'ralayotgan mahsulot quyidagi "Context" bo'limida bo'lmasa, u bizda yo'q deb javob bering. Hech qachon mahsulot o'ylab topmang (masalan, iPhone 17 bazada yo'q bo'lsa, uni bor deb aytish qat'iyan man etiladi).
2. BILMAGAN NARSANGIZNI AYTMANG: Agar mahsulot Context ichida topilmasa, xushmuomalalik bilan "Kechirasiz, bazamizda bunday mahsulot topilmadi" deb javob bering.
3. KONTEKST USTUNLIGI: Sizning ushbu mahsulot haqidagi umumiy bilimingizdan ko'ra, Context ichidagi ma'lumot muhimroq. Agar Context bo'sh bo'lsa, demak mahsulot topilmadi.

VAZIFANGIZ:
1. Kontekstdagi (Context) mahsulotlar asosida mijozga yordam berish.
2. Mijoz bilan do'stona gaplashing. Faqatgina quruq ro'yxat bermasdan, gapni "Ha, albatta, bizda quyidagi modellar bor:" kabi jumlalar bilan boshlang.
3. Agar mijoz avvalgi gaplarda ma'lum bir mahsulot haqida so'ragan bo'lsa, suhbat tarixidan (History) foydalanib o'sha mahsulot haqida suhbatni davom ettiring.

NARX VA VALYUTA:
- Mijoz narxni so'mda so'rasa: 1 dollar = 12,800 so'm kursi bo'yicha hisoblab bering va bu taxminiy ekanligini ayting.

QOIDALAR:
1. TIL: Mijoz qaysi tilda yozsa, o'sha tilda javob bering.
2. FORMAT:
   - RO'YXAT (Bir nechta mahsulot):
     **[Mahsulot nomi]**
     💰 Narxi: [Price] $
     📌 [Tavsifdan 1 ta qisqa gap]
     📝 *Yanada koproq malumot olishni istasangiz ushbu model nomini yozing.*
     ---
   - BATAFSIL (Bitta mahsulot):
     IMAGES: [Kontekstdagi IMAGE_LINKS qatoridagi barcha linklarni vergul bilan ajratilgan holda shu yerga qo'ying]
     **[Mahsulot nomi]**
     💰 Narxi: [Price] $
     🛠 Xarakteristikalar:
     • [Xarakteristika 1]
     [To'liq tavsif]

MUHIM: Haqiqiy IMAGE_LINKS ishlatilganiga ishonch hosil qiling. Agar rasm bo'lmasa "IMAGES: None" deb yozing. 

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
    
    rewrite_prompt = f"""
    Suhbat tarixi va oxirgi savol asosida, mahsulotni topish uchun eng mos mahsulot nomini qaytaring.
    Faqat mahsulot nomini yoki modelini qaytaring, ortiqcha so'zlarsiz.
    Agar savol mahsulotga tegishli bo'lmasa, savolning o'zini qaytaring.
    
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

    # 2. Vector Search with higher threshold to prevent hallucinations
    scored_docs = vector_db.similarity_search_with_relevance_scores(search_query, k=8)
    
    # Higher threshold (0.25) to avoid matching unrelated garbage
    threshold = 0.25
    filtered_docs = [doc for doc, score in scored_docs if score >= threshold]
    
    context = ""
    if filtered_docs:
        logging.info(f"Found {len(filtered_docs)} relevant documents.")
        context = "\n---\n".join([d.page_content for d in filtered_docs])
    else:
        logging.warning(f"No documents met the relevance threshold ({threshold}) for query: {search_query}")
        # If it's a specific product keyword, explicitly mark it as not found
        product_keywords = ["iphone", "samsung", "plita", "mashina", "tv", "lg", "artel"]
        if any(kw in search_query.lower() for kw in product_keywords):
            context = "MA'LUMOT TOPILMADI: Ushbu mahsulot bazada mavjud emas."

    # 3. Generate response
    response = with_message_history.invoke(
        {"context": context, "question": user_query},
        config={"configurable": {"session_id": chat_id}}
    )
    return response.content
