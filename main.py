import streamlit as st
import json
import base64
import openai
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain.chains import LLMChain
from image_analysis.core import process_image_pipeline
from pinecone_integration import MedicalPineconeDB
from result_analysis.core import summarize_user_result, summarize_prescription
from sched_appointment import AppointmentProcessor

# # Load environment variables
load_dotenv()
# Setup session_state for audio caching
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
# Page configuration
st.set_page_config(
    page_title="MedGuide AI - Trợ lý Y tế Thông minh",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
few_shot_prompt_examples = """
    Example 1:
    Previous topic: sched_appointment
    Previous message: "Đặt lịch khám răng cho tôi ngày 20/10/2025 lúc 10h sáng"
    Current message: "Khám đau đầu"
    Output: sched_appointment

    Example 2:
    Previous topic: drug_groups
    Previous message: ""
    Current message: "Thuốc paracetamol dùng khi nào?"
    Output: drug_groups
"""

class MedGuideAI:
    def __init__(self):
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            base_url=st.secrets["OPENAI_ENDPOINT"],
            api_key=st.secrets["OPENAI_API_KEY"],
        )
        self.pinecone_db=MedicalPineconeDB()
       
        # Initialize session state for context management
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'patient_context' not in st.session_state:
            st.session_state.patient_context = {
                "medical_history": [],
                "medications": [],
                "allergies": [],
                "symptoms_timeline": []
            }
       
        # System prompt cho MedGuide AI
        self.system_prompt = """
        Bạn là MedGuideAI — trợ lý y tế thông minh và hữu ích, chuyên về thuốc, xét nghiệm và tư vấn y khoa tổng quát.

        NGUYÊN TẮC AN TOÀN:
        - Nếu câu hỏi thuộc lĩnh vực y tế:
          - Không tự ý chẩn đoán bệnh cụ thể.
          - Luôn kết thúc bằng: "Đây là thông tin tham khảo, bạn nên tham khảo bác sĩ để có hướng điều trị chính xác".
          - Khuyến khích thăm khám chuyên khoa khi cần.
        - Khi trả lời, ưu tiên chính xác, dễ hiểu, phù hợp cho người không có chuyên môn y khoa.
        """

    # Context Management Methods
    def add_to_context(self, category: str, data: Any):
        """Thêm thông tin vào context của bệnh nhân"""
        if category in st.session_state.patient_context:
            st.session_state.patient_context[category].append({
                "timestamp": datetime.now().isoformat(),
                "data": data
            })
   
    def add_conversation(self, role: str, content: str):
        """Thêm hội thoại vào lịch sử"""
        st.session_state.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
   
    def get_context_summary(self) -> str:
        """Tạo tóm tắt context để đưa vào prompt"""
        context_parts = []
       
        if st.session_state.patient_context["medical_history"]:
            history = [item["data"] for item in st.session_state.patient_context["medical_history"][-3:]]
            context_parts.append(f"Tiền sử bệnh: {'; '.join(history)}")
       
        if st.session_state.patient_context["medications"]:
            meds = [item["data"] for item in st.session_state.patient_context["medications"][-5:]]
            context_parts.append(f"Thuốc đang dùng: {'; '.join(meds)}")
       
        if st.session_state.patient_context["allergies"]:
            allergies = [item["data"] for item in st.session_state.patient_context["allergies"]]
            context_parts.append(f"Dị ứng: {'; '.join(allergies)}")
       
        return "\n".join(context_parts) if context_parts else "Chưa có thông tin bệnh nhân"
    
    def classify_user_query(self, user_input: str, previous_topic: str = None, previous_message: str = None) -> str:
        """Classify user query into topic categories"""
        try:
            classification_prompt = f"""
                You are a medical query classifier and conversational context analyzer.
                Your tasks:
                1. Classify the current query into one of these categories:
                - drug_groups: Questions about medications, drugs, or their uses/dosages/side effects.
                - get_prescription: Requests to view, retrieve, or check a prescription (e.g., doctor's prescription, medication order).
                - get_lab_results: Requests to view, retrieve, or check laboratory test results (e.g., blood test, urine test, imaging results).
                - sched_appointment: Requests to book or schedule a medical appointment (must include name, date/time, and reason if provided).
                - health_advice: Requests for health or symptom advice (e.g., describing symptoms, asking for medical advice or recommendations)
                - other: Any other type of query not covered above.

                2. If the query is vague, incomplete, or clearly a follow-up to the previous conversation and still relevant, return the previous topic.

                Previous topic category: {previous_topic}
                Previous user message: "{previous_message}"
                Current user message: "{user_input}"

                A follow-up means:
                - The current message refers to information in the previous message without repeating all the details
                - It continues the same intent or subject
                - It doesn't introduce a completely new topic

                Few-shot examples: {few_shot_prompt_examples}
                Return only the category name (drug_groups/get_prescription/get_lab_results/sched_appointment/other).
                """
            
            response = self.client.chat.completions.create(
                model="GPT-4o-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=50,
                temperature=0
            )
            
            category = response.choices[0].message.content.strip().lower()
            return category if category in ['symptoms', 'drug_groups', 'get_prescription', 'get_lab_results', 'compare_prescription', 'compare_lab_results', 'sched_appointment', 'other'] else 'other'
            
        except Exception as e:
            return 'other'  # Default fallback


    def process_user_query(self, user_input: str, get_latest_record, username, use_personal_data, get_records_in_range):
        """Main processing pipeline: classify -> query -> generate"""
        try:
            # Step 1: Text classification
            previous_topic = getattr(st.session_state, "current_topic", None)
            previous_message = getattr(st.session_state, "previous_user_message", "")

            topic = self.classify_user_query(user_input, previous_topic, previous_message)
            st.session_state.current_topic = topic
            st.session_state.previous_user_message = user_input

            print("current topic:", topic)

            search_results=""
            ai_response = "❌ Không tìm thấy kết quả phù hợp."
            user_test_result = "Hiện không có kết quả"
            user_prescription = "Hiện không có kết quả"
            previous_prescription = "Hiện không có kết quả"
            previous_test_result= "Hiện không có kết quả"

            should_personalize = False

            if topic in ("other", "drug_groups") and username and use_personal_data:
                should_personalize = True
                user_prescription = get_latest_record("patient_prescriptions", username)
                user_test_result = get_latest_record("patient_test_results", username)

            if topic in ("get_lab_results", "get_prescription") and username:
                should_personalize = True
                user_prescription = get_latest_record("patient_prescriptions", username)
                user_test_result = get_latest_record("patient_test_results", username)
                if use_personal_data:
                    previous_prescription = get_records_in_range("patient_prescriptions", username, 1)
                    previous_test_result = get_records_in_range("patient_test_results", username, 1)



            # Step 2: Query rel evant collection
            if topic == 'sched_appointment':
                print("handle function calling to schedule an appointment")
                processor = AppointmentProcessor(
                    base_url=st.secrets["OPENAI_ENDPOINT"],
                    api_key=st.secrets["OPENAI_API_KEY"],
                )
                result = processor.process_with_function_calling(user_input)
                ai_response = result["ai_response"]
            elif topic == "get_lab_results":
                ai_response = summarize_user_result(self.system_prompt, user_test_result, previous_test_result if should_personalize else "Hiện không có kết quả")
            elif topic == "get_prescription":
                ai_response = summarize_prescription(self.system_prompt, user_prescription, previous_prescription if should_personalize else "Hiện không có kết quả")
            elif topic == "drug_groups":
                search_results = self.pinecone_db.search_drug_groups(user_input, n_results=1)

                generation_prompt = f"""
                    Người dùng hỏi: "{user_input}"
                    
                    Kết quả tìm kiếm semantic gần nhất từ hệ thống RAG nội bộ (có thể đúng hoặc không liên quan):
                    {search_results if search_results else 'Không có kết quả'}
                    
                    Kết quả xét nghiệm gần nhất của người dùng: "{user_test_result if should_personalize else 'Hiện chưa có thông tin'}"
                    Đơn thuốc gần nhất của người dùng: "{user_prescription if should_personalize else 'Hiện chưa có thông tin'}"
                    
                    Hướng dẫn
                    1. Xác định:
                       - Nếu dữ liệu RAG liên quan → ưu tiên dùng và ghi nguồn: "Nguồn: MedGuideAI"
                       - Nếu dữ liệu RAG không liên quan hoặc không có → trả lời dựa trên kiến thức y khoa tổng quát từ nguồn uy tín (Bộ Y tế, WHO, PubMed…) và ghi nguồn được sử dụng. Ví dụ: "Nguồn: WHO"
                    2. Cấu trúc câu trả lời:
                       - Giới thiệu ngắn gọn về thuốc hoặc chủ đề được hỏi.
                       - Giới thiệu về các hoạt chất trong thuốc
                       - Tác dụng, cơ chế.
                       - Liều dùng khuyến nghị.
                       - Tác dụng phụ.
                       - **Cá nhân hóa**:
                            - Dựa vào kết quả xét nghiệm gần nhất của bệnh nhân(nếu có), đưa ra cụ thể những ảnh hưởng tốt và xấu của thuốc, những lưu ý, tác dụng phụ ảnh hưởng đến cơ thể người dùng khi dùng loại thuốc đang hỏi.
                            - Dựa vào đơn thuốc gần nhất của bệnh nhân(nếu có), đưa ra cụ thể những lưu ý, tương tác thuốc có thể có giữa đơn thuốc gần nhất với loại thuốc người dùng hỏi
                       - Nguồn:
                            - Nếu dữ liệu RAG liên quan → BẮT BUỘC ghi rõ nguồn như sau:"Nguồn: MedGuideAI"
                            - Nếu dữ liệu RAG không liên quan hoặc không có → ghi nguồn được sử dụng. Ví dụ: "Nguồn: WHO"
                    3. Nếu chưa có thông tin về kết quả xét nghiệm gần nhất của người dùng hoặc là đơn thuốc gần nhất của người dùng thì không đề cập trong mục cá nhân hóa
                    """
                response = self.client.chat.completions.create(
                    model="GPT-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": generation_prompt}
                    ],
                    temperature=0
                )

                ai_response = response.choices[0].message.content
            else:
                generation_prompt = f"""
                    Bạn là MedGuideAI — một chuyên gia trong lĩnh vực y tế.
                    
                    Người dùng đặt câu hỏi như sau: "{user_input}"
                    - Kết quả xét nghiệm gần nhất của người dùng: "{user_test_result or "Hiện không có thông tin kết quả xét nghiệm"}"
                    - Đơn thuốc gần nhất của người dùng: "{user_prescription or "Hiện chưa có thông tin đơn thuốc"}"
                    
                    Nhiệm vụ:
                    1. Đưa ra tư vấn cho người dùng kết hợp với tiền sử kết quả xét nghiệm gần nhất của người dùng (nếu có) và đơn thuốc gần nhất của người dùng (nếu có)
                    2. Nếu câu hỏi không liên quan đến y tế, thì chỉ cần trả lời: "MedguideAI không thể trả lời câu hỏi nằm ngoài lĩnh vực y tế" 
                    2. Dùng kiến thức y khoa từ nguồn uy tín (Bộ Y tế, WHO, PubMed...).
                    3. Khi trả lời giữ giọng văn chuyên nghiệp, dễ hiểu cho người không có chuyên môn y khoa
                    4. Trích dẫn những nguồn sử dụng
                    """
                response = self.client.chat.completions.create(
                    model="GPT-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": generation_prompt}
                    ],
                    temperature=0
                )
                ai_response = response.choices[0].message.content

            # print("ai_response: " + ai_response)
            # st.session_state.audio_bytes = text_to_speech.run_audio(ai_response)
            
            # Add to conversation history
            self.add_conversation("user", user_input)
            self.add_conversation("assistant", ai_response)
            
            return {
                "topic_classified": topic,
                "search_results": search_results,
                "ai_response": ai_response,
                "conversation_id": len(st.session_state.conversation_history)
            }
            
        except Exception as e:
            error_msg = f"Lỗi xử lý: {str(e)}"
            return {"error": error_msg}
    
    def encode_image(self, image_file):
        """Encode image to base64 for OpenAI Vision API"""
        try:
            return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            return None
 
    def analyze_medical_image(self, image_file, query_type="general", latest_prescription = None, latest_test_result = None):
        """Phân tích hình ảnh y tế với Vision API"""
        try:
            result = process_image_pipeline(image_file, latest_prescription, latest_test_result)
            ai_response = ""

            if result:
                if result["doc_type"] == "đơn thuốc" or result["doc_type"] == "kết quả xét nghiệm":
                    ai_response = result["data"]
                else:
                    ai_response = "❓ Loại tài liệu chưa được hỗ trợ."
            else:
                ai_response = "Không nhận diện được nội dung từ ảnh."

            return ai_response

        except Exception as e:
            return f"Lỗi khi phân tích hình ảnh: {str(e)}"