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
Siz Sodda.uz onlayn do'konining professional menejerisiz. 

VAZIFANGIZ:
1. Kontekstdagi (Context) mahsulotlar asosida mijozga yordam berish.
2. Agar mijoz avvalgi gaplarda ma'lum bir mahsulot haqida so'ragan bo'lsa (masalan: "nima u?", "narxi necha?"), suhbat tarixidan (History) foydalanib o'sha mahsulot haqida ma'lumot bering.

QOIDALAR:
1. TIL: Mijoz qaysi tilda yozsa (O'zbek yoki Rus), o'sha tilda javob bering.
2. SALOMLASHISHSIZ: Ortiqcha gaplarsiz, darhol savolga javob bering.
3. NARX: Faqat dollarda ($). Masalan: 400 $.
4. FORMAT:
   - RO'YXAT (Agar bir nechta mahsulot bo'lsa):
     **[Mahsulot nomi]**
     💰 Narxi: [Price] $
     📌 [Tavsifdan 1 ta qisqa gap]
     ---
   - BATAFSIL (Bitta mahsulot haqida so'ralsa):
     IMAGE: [ImageURL]
     **[Mahsulot nomi]**
     💰 Narxi: [Price] $
     🛠 Xarakteristikalar:
     • [Xarakteristika 1]
     [To'liq tavsif]

Agar kontekstda ham, suhbat tarixida ham mos mahsulot bo'lmasa, "Kechirasiz, bunday mahsulot topilmadi." deb javob bering.

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
    logging.info(f"Query: {user_query}")
    
    # 1. First, search with a relevance score threshold to avoid hallucinations on irrelevant queries
    # Using similarity_search_with_relevance_scores
    # Note: 'text-embedding-3-small' with cosine similarity in Chroma often gives scores around 0.3-0.4 for good matches.
    scored_docs = vector_db.similarity_search_with_relevance_scores(user_query, k=5)
    
    # Filter docs by threshold
    threshold = 0.25
    filtered_docs = [doc for doc, score in scored_docs if score >= threshold]
    
    # If no good matches, we still provide history context because the user might be asking about a previously found item
    context = ""
    if filtered_docs:
        logging.info(f"Found {len(filtered_docs)} relevant documents.")
        context = "\n---\n".join([d.page_content for d in filtered_docs])
    else:
        logging.warning(f"No documents met the relevance threshold ({threshold}) for query: {user_query}")

    response = with_message_history.invoke(
        {"context": context, "question": user_query},
        config={"configurable": {"session_id": chat_id}}
    )
    return response.content
