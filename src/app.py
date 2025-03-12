import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.conversation_engine import initialize_chatbot, load_chat_store

def main():
    print("🤖 Chào mừng bạn đến với Tư Vấn Tâm Lý AI!")
    print("Bot: Xin chào! Tôi là trợ lý AI có thể giúp bạn tư vấn về các vấn đề tâm lý.")
    print("Bot: Bạn có thể chia sẻ với tôi về những gì bạn đang gặp phải...")
    print("(Gõ 'quit' hoặc 'exit' để kết thúc cuộc trò chuyện)\n")
    
    # Khởi tạo chatbot
    chat_store = load_chat_store()
    username = "default_user"
    user_info = "No information"
    
    agent = initialize_chatbot(
        chat_store=chat_store,
        container=None,  # Không cần container vì không dùng streamlit
        username=username,
        user_info=user_info
    )

    # Chat loop
    while True:
        # Lấy input từ người dùng
        user_input = input("\nBạn: ")
        
        # Kiểm tra điều kiện thoát
        if user_input.lower() in ['quit', 'exit', 'bye', 'tạm biệt']:
            print("\nBot: Tạm biệt bạn! Hẹn gặp lại!")
            break
            
        # Lấy phản hồi từ agent
        try:
            response = agent.chat(user_input)
            print("\nBot:", response)
        except Exception as e:
            print("\nBot: Xin lỗi, đã có lỗi xảy ra:", str(e))
            print("Bot: Bạn có thể thử lại hoặc khởi động lại chương trình.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBot: Tạm biệt! Hẹn gặp lại bạn!")
