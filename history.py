import os
from langchain_community.chat_message_histories import SQLChatMessageHistory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'chat_history.db')}"

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id, connection=DB_PATH)

def get_last_messages(chat_id: str, limit: int = 10):
    """
    Returns last `limit` messages (human+ai) from persisted SQLite history.
    """
    hist = get_session_history(chat_id)
    try:
        msgs = list(hist.messages)
        return msgs[-limit:] if limit > 0 else msgs
    except Exception:
        return []
