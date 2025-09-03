import threading
import time
import streamlit as st
import os
from datetime import datetime
from PIL import Image
from main import MedGuideAI
# import text_to_speech
from speech_module.ffmpeg_decoding import AudioPlayer
from speech_module.tts_mp3_stream import tts_stream
from speech_module.test_streamlit_stt import speech_to_text
import speed_to_text as sp

# Page configuration
st.set_page_config(
    page_title="MedGuide AI",
    page_icon="🏥",
    layout="centered"
)

# Define fragments
@st.fragment
def playback_controls(message_id: str, message_content: str):
    """Isolated fragment for managing playback controls for a single message."""
    # Get or create player reference in session state
    if 'audio_players' not in st.session_state:
        st.session_state.audio_players = {}
    
    # Initialize player for this message if needed
    if message_id not in st.session_state.audio_players:
        st.session_state.audio_players[message_id] = None
    
    player = st.session_state.audio_players[message_id]
    is_playing = player is not None and player.is_playing()
    
    # Display appropriate button based on playback state
    if is_playing:
        if st.button("⏹️", key=f"stop_{message_id}"):
            # Stop this specific playback
            if player:
                player.stop()
            # Clear player reference
            st.session_state.audio_players[message_id] = None
            st.rerun(scope="fragment")
            # The fragment will automatically rerun and show the Play button
            
        if isinstance(player.playback_thread, threading.Thread) and player.playback_thread.is_alive():
            # time.sleep(0.1)
            # st.rerun(scope="fragment")
            pass
        else:
            st.session_state.audio_players[message_id] = None
        # time.sleep(3)
        # st.rerun(scope="fragment")

    else:
        if st.button("🔊", key=f"play_{message_id}"):
            # First stop any active playback (enforce only one playback at a time)
            for msg_id, p in st.session_state.audio_players.items():
                if p is not None:
                    p.stop()
                    st.session_state.audio_players[msg_id] = None

            # Create new player for this message
            player = AudioPlayer()
            # Start playback in background thread
            def playback():
                player.play(tts_stream(message_content))
            
            t = threading.Thread(target=playback, daemon=True)
            st.session_state.audio_players[message_id] = player
            t.start()
            st.rerun(scope="fragment")

# Initialize AI
@st.cache_resource
def load_ai():
    return MedGuideAI()

if "audio_record_bytes" not in st.session_state:
    st.session_state.audio_record_bytes = None

def main():
    # Header
    st.title("🏥 MedGuide AI")
    st.markdown("### Tư vấn y tế thông minh với AI")

    # Initialize
    ai = load_ai()

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'patient_context' not in st.session_state:
        st.session_state.patient_context = {
            "medical_history": [],
            "medications": [],
            "allergies": [],
            "symptoms_timeline": []
        }
    if 'processing_image' not in st.session_state:
        st.session_state.processing_image = False
    if 'temp_image' not in st.session_state:
        st.session_state.temp_image = None

    # Welcome message - show full intro on first visit, short version afterwards
    if not st.session_state.messages:
        st.info("""
            👋 **Chào mừng đến với MedGuide AI!**
           
            🗣️ **Bạn có thể:**
            - Hỏi về triệu chứng, thuốc, xét nghiệm
            - Upload hình ảnh đơn thuốc hoặc kết quả xét nghiệm để phân tích
            - Trò chuyện liên tục với AI để được tư vấn chi tiết
           
            💡 **Cách sử dụng nhanh:**
            - Nhập câu hỏi và nhấn Enter để gửi
            - Dùng nút 📷 để gửi hình ảnh y tế
            - Sử dụng các câu hỏi mẫu bên dưới để bắt đầu
            """)
    else:
        st.info("### Tư vấn y tế thông minh với AI")

    # Display chat history with container for better scrolling
    if st.session_state.messages:
        st.markdown("### 💬 Cuộc trò chuyện")

        # Create container for chat messages
        chat_container = st.container()
        with chat_container:
            for message_index, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.write(message["content"])

                    # Show image if exists
                    if "image" in message:
                        st.image(message["image"], width=300)

                    # Show classification info
                    if message["role"] == "assistant" and "topic" in message:
                        topic_icons = {
                            'symptoms': '🩺',
                            'drug_groups': '💊',
                            'lab_results': '🧪',
                            'unknown': '❓'
                        }
                        # st.caption(f"{topic_icons.get(message['topic'], '❓')} {message['topic']}")

                    # Add audio player for assistant messages
                    if message["role"] == "assistant":
                        playback_controls(str(message_index), message["content"])

    # Show processing indicators right after chat history
    if st.session_state.get('processing', False):
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang xử lý..."):
                st.write("🤖 Đang xử lý câu hỏi của bạn...")

    if st.session_state.get('processing_image', False):
        with st.chat_message("assistant"):
            with st.spinner("🔍 Đang phân tích hình ảnh..."):
                st.write("🔍 Đang phân tích hình ảnh y tế...")

    # Input section at bottom
    st.markdown("---")

    # Combined input area with image upload
    col1, col2 = st.columns([6, 4])
    audio_bytes_ = st.audio_input("Speak something...", key="audio_recorder")
    if audio_bytes_:
        st.audio(audio_bytes_)
        print(audio_bytes_.size)
        st.session_state.audio_record_bytes = audio_bytes_
    if st.button("Send"):
        if st.session_state.audio_record_bytes is not None:
            
            audio_text =  sp.speech_to_text_raw_bytes(st.session_state.audio_record_bytes)
            
            audio_content = audio_text
            print(audio_content)
            st.session_state.audio_record_bytes = None

            st.session_state.messages.append({
                "role": "user",
                "content":  audio_content
            })

            st.session_state.processing = True
                    
    with col1:
        # Chat input
        user_text = st.chat_input(
            placeholder="Nhập câu hỏi y tế... (Enter để gửi)"
        )
        text_submit = bool(user_text)

    with col2:
        # Use dynamic key to clear file uploader after submit
        upload_key = f"file_upload_{st.session_state.get('upload_counter', 0)}"
        uploaded_file = st.file_uploader(
            "📷",
            type=['jpg', 'jpeg', 'png'],
            help="Gửi hình ảnh y tế (đơn thuốc, xét nghiệm...)",
            key=upload_key,
            label_visibility="collapsed"
        )
            

    # Show image preview when uploaded
    if uploaded_file and not st.session_state.get('processing_image', False):
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(uploaded_file, width=120, caption="Hình ảnh đã chọn")

        with col2:
            st.info("📷 Hình ảnh đã chọn! Nhập câu hỏi hoặc nhấn Enter để phân tích.")

    # Process text input (with or without image)
    if text_submit and (user_text.strip() or uploaded_file):
        # Determine content and processing type
        if uploaded_file:
            # Image with optional text
            user_content = user_text.strip() if user_text.strip() else "Phân tích hình ảnh này"
            st.session_state.messages.append({
                "role": "user",
                "content": user_content,
                "image": uploaded_file
            })
            st.session_state.processing_image = True
            st.session_state.temp_image = uploaded_file
        else:
            # Text only
            st.session_state.messages.append({
                "role": "user",
                "content": user_text
            })
            st.session_state.processing = True
        st.rerun()

    # Handle processing state (background processing)
    if st.session_state.get('processing', False):
        # Get the last user message
        last_user_msg = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break

        if last_user_msg:
            # Process with AI (no UI here, just processing)
            print("<duypv10 log> last_user_msg: ", last_user_msg)
            result = ai.process_user_query(last_user_msg)

            if "error" in result:
                response = f"❌ Lỗi: {result['error']}"
                topic = "error"
            else:
                response = result.get('ai_response', 'Không có phản hồi')
                topic = result.get('topic_classified', 'unknown')

                # Generate audio
                # audio_bytes = text_to_speech.run_audio(response)

            # Add AI response to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "topic": topic,
                # "audio": audio_bytes
            })

            # Clear processing state
            st.session_state.processing = False
            st.rerun()

    # Clear file uploader after processing
    if (text_submit and uploaded_file) or st.session_state.get('processing_image', False):
        if 'upload_counter' not in st.session_state:
            st.session_state.upload_counter = 0
        st.session_state.upload_counter += 1

    # Handle image processing state (background processing)
    if st.session_state.get('processing_image', False):
        # Process with AI (no UI here, just processing)
        temp_image = st.session_state.temp_image
        temp_image.seek(0)
        response = ai.analyze_medical_image(temp_image, "general")

        # Generate audio
        # audio_bytes = text_to_speech.run_audio(response)

        # Add AI response to messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "topic": "image_analysis",
            # "audio": audio_bytes
        })

        # Clear processing state and file uploader
        st.session_state.processing_image = False
        st.session_state.temp_image = None
        if 'upload_counter' not in st.session_state:
            st.session_state.upload_counter = 0
        st.session_state.upload_counter += 1

        st.rerun()

    # File upload section for Pinecone DB
    with st.sidebar:
        st.markdown("### 📁 Thêm tài liệu y tế")

        # Collection selection
        collection_choice = st.selectbox(
            "Chọn loại tài liệu:",
            ["Tự động phân loại", "Triệu chứng", "Thuốc", "Xét nghiệm"],
            help="AI sẽ tự động phân loại hoặc bạn có thể chọn trước"
        )

        # File uploader
        doc_file = st.file_uploader(
            "Upload file (.txt, .pdf, .docx):",
            type=['txt', 'pdf', 'docx'],
            help="Tài liệu y tế để bổ sung cơ sở dữ liệu"
        )

        if doc_file and st.button("📤 Thêm vào cơ sở dữ liệu", use_container_width=True):
            with st.spinner("Đang xử lý tài liệu..."):
                try:
                    # Read file content
                    if doc_file.type == "text/plain":
                        content = str(doc_file.read(), "utf-8")
                    else:
                        st.error("Hiện tại chỉ hỗ trợ file .txt")
                        content = None

                    if content:
                        # Process with Pinecone DB
                        if collection_choice == "Tự động phân loại":
                            additions = ai.pinecone_db.add_file_content_to_db(content, doc_file.name)
                        else:
                            # Manual classification
                            collection_map = {
                                "Triệu chứng": "symptoms",
                                "Thuốc": "drug_groups",
                                "Xét nghiệm": "lab_results"
                            }
                            target_collection = collection_map[collection_choice]
                            additions = ai.pinecone_db.add_to_specific_collection(content, doc_file.name, target_collection)

                        # Check for errors
                        if "error" in additions:
                            st.error(f"❌ Lỗi khi thêm dữ liệu: {additions['error']}")
                            if "No Pinecone connection" in additions['error']:
                                st.warning("⚠️ Vui lòng tạo file .env với PINECONE_API_KEY của bạn")
                        elif sum(additions.values()) == 0:
                            st.warning("⚠️ Không có dữ liệu nào được thêm vào. Kiểm tra nội dung file và kết nối Pinecone.")
                        else:
                            st.success(f"✅ Đã thêm: {additions}")

                        # Show collection stats
                        stats = ai.pinecone_db.get_collection_stats()
                        st.info(f"📊 Tổng: Triệu chứng({stats['symptoms']}), Thuốc({stats['drug_groups']}), XN({stats['lab_results']})")

                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")

        st.markdown("---")

    # # Quick actions - always show for easy access  
    # st.markdown("### 🚀 Câu hỏi mẫu:")
    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     if st.button("💊 Hỏi về thuốc"):
    #         st.session_state.messages.append({
    #             "role": "user",
    #             "content": "Paracetamol có tác dụng gì?"
    #         })
    #         st.rerun()

    # with col2:
    #     if st.button("🧪 Hỏi về xét nghiệm"):
    #         st.session_state.messages.append({
    #             "role": "user",
    #             "content": "Glucose 150 mg/dL có cao không?"
    #         })
    #         st.rerun()

    # with col3:
    #     if st.button("🩺 Hỏi về triệu chứng"):
    #         st.session_state.messages.append({
    #             "role": "user",
    #             "content": "Tôi bị đau đầu và chóng mặt"
    #         })
    #         st.rerun()

    # Clear chat
    if st.session_state.messages:
        if st.button("🗑️ Xóa cuộc trò chuyện", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # Footer
    st.markdown("---")
    st.caption("⚠️ Thông tin chỉ mang tính tham khảo, hãy tham khảo bác sĩ chuyên khoa")

if __name__ == "__main__":
    main()