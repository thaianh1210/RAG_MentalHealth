import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from datetime import datetime
import toml
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.tools import FunctionTool
from src.global_settings import INDEX_STORAGE, CONVERSATION_FILE, SCORES_FILE
from src.prompts import CUSTORM_AGENT_SYSTEM_TEMPLATE
import logging


# Đọc API key từ file secrets.toml
secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "streamlit", "secrets.toml")
secrets = toml.load(secrets_path)
openai_api_key = secrets["openai"]["OPENAI_API_KEY"]

# Đặt API key cho OpenAI
os.environ["OPENAI_API_KEY"] = openai_api_key

def load_chat_store():
    logging.info(f"Loading chat store from {CONVERSATION_FILE}")
    if os.path.exists(CONVERSATION_FILE) and os.path.getsize(CONVERSATION_FILE) > 0:
        try:
            with open(CONVERSATION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chat_store = SimpleChatStore.from_dict(data)
                return chat_store
        except Exception as e:
            logging.error(f"Error loading chat store: {str(e)}")
            chat_store = SimpleChatStore()
    else:
        chat_store = SimpleChatStore()
        os.makedirs(os.path.dirname(CONVERSATION_FILE), exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        with open(CONVERSATION_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    return chat_store

def save_score(score, content, total_guess, username):
    """
    Lưu kết quả chẩn đoán sức khỏe tâm lý của người dùng
    
    Args:
        score (int): Mức độ nghiêm trọng (1-10)
        content (str): Kết quả chẩn đoán chi tiết
        total_guess (int): Số lượng triệu chứng được xác định
        username (str): Tên người dùng
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    diagnosis_entry = {
        "username": username,
        "timestamp": current_time,
        "severity_score": score,
        "diagnosis": content,
        "symptoms_count": total_guess
    }
    
    try:
        os.makedirs(os.path.dirname(SCORES_FILE), exist_ok=True)
        
        try:
            with open(SCORES_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        data.append(diagnosis_entry)
        with open(SCORES_FILE, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        return f"Đã lưu kết quả chẩn đoán của bạn"
    except Exception as e:
        error_msg = f"Error saving diagnosis: {str(e)}"
        return error_msg
    
def save_chat_store(chat_store):
    try:
        chat_dict = chat_store.to_dict()
        os.makedirs(os.path.dirname(CONVERSATION_FILE), exist_ok=True)
        with open(CONVERSATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error saving chat store: {str(e)}")

def initialize_chatbot(chat_store, container, username, user_info):
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000, 
        chat_store=chat_store, 
        chat_store_key=username
    )  
    storage_context = StorageContext.from_defaults(
        persist_dir=INDEX_STORAGE
    )
    index = load_index_from_storage(
        storage_context, 
        index_id="vector"
    )
    dsm5_engine = index.as_query_engine(
        similarity_top_k=3,
    )
    dsm5_tool = QueryEngineTool(
        query_engine=dsm5_engine, 
        metadata=ToolMetadata(
            name="dsm5",
            description=(
                f"Cung cấp các thông tin liên quan đến các bệnh "
                f"tâm thần theo tiêu chuẩn DSM5. Sử dụng câu hỏi văn bản thuần túy chi tiết làm đầu vào cho công cụ"
            ),
        )
    )   
    save_tool = FunctionTool.from_defaults(
        fn=save_score,
        name="save_score",
        description="Lưu điểm số đánh giá tình trạng sức khỏe của người dùng. Parameters: score (int): điểm số từ 1-10, content (str): nội dung đánh giá, total_guess (int): số lần đoán, username (str): tên người dùng"
    )
    agent = OpenAIAgent.from_tools(
        tools=[dsm5_tool, save_tool], 
        memory=memory,
        system_prompt=CUSTORM_AGENT_SYSTEM_TEMPLATE.format(user_info=user_info),
        api_key=openai_api_key
    )
    
    # Wrap agent's chat method để tự động lưu sau mỗi tin nhắn
    original_chat = agent.chat
    def chat_and_save(*args, **kwargs):
        response = original_chat(*args, **kwargs)
        save_chat_store(chat_store)
        return response
    agent.chat = chat_and_save
    
    return agent

