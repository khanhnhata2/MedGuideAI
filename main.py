import streamlit as st
import json
import os
import base64
import openai
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain.chains import LLMChain
from image_analysis.core import process_image_pipeline
from image_analysis.render import render_prescription, render_lab
from image_analysis.schemas import LabList
from pinecone_integration import MedicalPineconeDB
from result_analysis.core import handle_get_result, summarize_user_result, summarize_prescription
from result_analysis.render import render_latest_result, render_lab_comparison, render_latest_prescription
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
    Previous topic: None
    Previous message: ""
    Current message: "Tôi bị đau đầu và buồn nôn"
    Output: symptoms

    Example 2:
    Previous topic: sched_appointment
    Previous message: "Đặt lịch khám răng cho tôi ngày 20/10/2025 lúc 10h sáng"
    Current message: "Khám đau đầu"
    Output: sched_appointment

    Example 3:
    Previous topic: drug_groups
    Previous message: ""
    Current message: "Thuốc paracetamol dùng khi nào?"
    Output: drug_groups

    Example 4:
    Previous topic: symptoms
    Previous message: "Tôi bị ho và sốt"
    Current message: "Uống thuốc gì được?"
    Output: drug_groups

    Example 5:
    Previous topic: None
    Previous message: ""
    Current message: "So sánh kết quả xét nghiệm cholesterol tháng này và tháng trước"
    Output: compare_lab_results

    Example 6:
    Previous topic: None
    Previous message: ""
    Current message: "So sánh đơn thuốc bác sĩ A và bác sĩ B kê cho tôi"
    Output: compare_prescription
"""

class MedGuideAI:
    def __init__(self):
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            base_url=os.getenv("OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_API_KEY"),
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
        Bạn là MedGuide AI - Trợ lý y tế thông minh và hữu ích.
       
        NGUYÊN TẮC AN TOÀN:
        - Nếu câu hỏi nằm trong phạm trù y tế, luôn kết thúc với: "Đây là thông tin tham khảo, bạn nên tham khảo bác sĩ để có hướng điều trị chính xác"
        - Không tự ý chẩn đoán bệnh cụ thể
        - Khuyến khích thăm khám chuyên khoa khi cần thiết
        - Nếu câu hỏi KHÔNG nằm trong phạm trù y tế, trả lời: "Xin lỗi, MedGuide AI không thể trả lời câu hỏi nằm ngoài lĩnh vực y tế"
       
        Hãy trả lời một cách chi tiết, hữu ích và thực tế để người dùng hiểu rõ tình trạng sức khỏe của mình.
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
                    Classify the following medical query into one of these categories:
                    - drug_groups: Questions about medications, drugs, or prescriptions
                    - get_prescription: Question about getting prescriptions
                    - get_lab_results: Question about getting lab results
                    - compare_prescription: Question about comparing prescriptions
                    - compare_lab_results: Question about comparing lab results, must have the word "compare" or "so sánh" in the query
                    - sched_appointment: Requests to schedule an appointment, including name, date/time and reason for visit
                    - other: Other

                2. If the query is vague, incomplete, or clearly a follow-up to the previous conversation and still relevant, return the previous topic.

                Previous topic category: {previous_topic}
                Previous user message: "{previous_message}"
                Current user message: "{user_input}"

                A follow-up means:
                - The current message refers to information in the previous message without repeating all the details
                - It continues the same intent or subject
                - It doesn't introduce a completely new topic

                Few-shot examples: {few_shot_prompt_examples}
                Return only the category name (symptoms/drug_groups/get_prescription/get_lab_results/compare_prescription/compare_lab_results/sched_appointment/other).
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


    def process_user_query(self, user_input: str, user_test_result, user_prescription):
        """Main processing pipeline: classify -> query -> generate"""
        try:
            # Step 1: Text classification
            previous_topic = getattr(st.session_state, "current_topic", None)
            previous_message = getattr(st.session_state, "previous_user_message", "")

            topic = self.classify_user_query(user_input, previous_topic, previous_message)
            st.session_state.current_topic = topic
            st.session_state.previous_user_message = user_input

            print("<duypv10 log> current topic:", topic)
            print("<duypv10 log> previous topic:", previous_topic)
            print("<duypv10 log> previous user message:", previous_message)

            search_results=""
            ai_response = "❌ Không tìm thấy kết quả phù hợp."

            # Step 2: Query rel evant collection
            if topic == 'sched_appointment':
                print("<duypv10 log> handle function calling to schedule an appointment")
                processor = AppointmentProcessor(
                    base_url=os.getenv("OPENAI_ENDPOINT"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
                result = processor.process_with_function_calling(user_input)
                ai_response = result["ai_response"]
            elif topic == "get_lab_results":
                ai_response = summarize_user_result(self.system_prompt, user_test_result)
            # elif topic == "compare_lab_results":
            #     latest_lab_results = handle_get_result("lab", 2)
            #     prompt_result =  handle_compare_list_result(latest_lab_results)
            #     if prompt_result is not None:
            #         ai_response = render_lab_comparison(prompt_result)
            elif topic == "get_prescription":
                ai_response = summarize_prescription(self.system_prompt, user_prescription)
            # elif topic == "compare_prescription":
            #     latest_prescription_result = handle_get_result("prescription", 2)
            #     ai_response = handle_compare_list_medicines(latest_prescription_result)
            elif topic == "drug_groups":
                search_results = self.pinecone_db.search_drug_groups(user_input, n_results=1)

                generation_prompt = f"""
                    Bạn là MedGuideAI — trợ lý y tế chuyên về thuốc.
                    
                    Người dùng hỏi về loại thuốc như sau: "{user_input}"
                    
                    Kết quả tìm kiếm semantic gần nhất từ hệ thống RAG nội bộ (có thể đúng hoặc không liên quan):
                    {search_results if search_results else 'Không có kết quả'}
                    
                    Nhiệm vụ:
                    1. Xác định xem thông tin trên có phải là về **đúng loại thuốc** mà người dùng hỏi không.
                       - Nếu đúng → chỉ sử dụng dữ liệu này để trả lời.
                       - Nếu không liên quan hoặc không có dữ liệu → dùng kiến thức y khoa từ nguồn uy tín (Bộ Y tế, WHO, PubMed...).
                    
                    2. Khi trả lời:
                       - Giữ giọng văn chuyên nghiệp, dễ hiểu cho người không có chuyên môn y khoa.
                       - Cấu trúc:
                         - Mở đầu: Giới thiệu ngắn gọn về thuốc và mục đích sử dụng.
                         - Nội dung chính: Tác dụng, cơ chế (nếu có).
                         - Liều dùng: Liều khuyến nghị, tần suất sử dụng.
                         - Tác dụng phụ: Liệt kê tác dụng phụ thường gặp, kèm lời khuyên.
                         - Kết luận: Nhắc lại mục đích chính và khuyến cáo tham khảo ý kiến bác sĩ.
                    """
                response = self.client.chat.completions.create(
                    model="GPT-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": generation_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )

                ai_response = response.choices[0].message.content
            else:
                generation_prompt = f"""
                    Bạn là MedGuideAI — một chuyên gia trong lĩnh vực y tế.
                    
                    Người dùng đặt câu hỏi như sau: "{user_input}"
                    - Kết quả xét nghiệm gần nhất của người dùng: "{user_test_result or "Hiện không có thông tin kết quả xét nghiệm"}"
                    - Đơn thuốc gần nhất của người dùng: "{user_prescription or "Hiện chưa có thông tin đơn thuốc"}"
                    
                    Nhiệm vụ:
                    1. Đưa ra tư vấn cho người dùng kết hợp với tiền sử kết quả xét nghiệm gần nhất của người dùng 
                    và đơn thuốc gần nhất của người dùng (nếu có)
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
                    temperature=0.3
                )
                ai_response = response.choices[0].message.content

            # # thong's code start
            print("---go 222")
            print("ai_response: " + ai_response)
            # st.session_state.audio_bytes = text_to_speech.run_audio(ai_response)
            # # thong's code end
            
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
 
    def analyze_medical_image(self, image_file, query_type="general"):
        """Phân tích hình ảnh y tế với Vision API"""
        try:
            result = process_image_pipeline(image_file)
            ai_response = ""

            if result:
                if result["doc_type"] == "đơn thuốc":
                    meds = result["structured_data"]  # chắc chắn là list
                    ai_response = render_prescription(meds.medicines)
                elif result["doc_type"] == "kết quả xét nghiệm":
                    labs_structured: LabList = result["structured_data"]  # đây là LabList object
                    labs = labs_structured.lab  # lấy list LabItem bên trong
                    ai_response = render_lab(labs)
                else:
                    ai_response = "❓ Loại tài liệu chưa được hỗ trợ."
            else:
                ai_response = "Không nhận diện được nội dung từ ảnh."

            return ai_response

        except Exception as e:
            return f"Lỗi khi phân tích hình ảnh: {str(e)}"

    def analyze_lab_results(self, test_results: List[Dict], patient_age: int = None, patient_gender: str = None):
        """Phân tích chi tiết kết quả xét nghiệm và đưa ra tư vấn"""
        analysis = []
        abnormal_findings = []
        recommendations = []
       
        for test in test_results:
            test_name = test.get("test_name", "")
            value = test.get("value", 0)
            unit = test.get("unit", "")
           
            # Phân tích chi tiết từng chỉ số
            if "glucose" in test_name.lower() or "đường huyết" in test_name.lower():
                if value > 126:
                    status = "Cao hơn bình thường - có thể chỉ ra nguy cơ tiểu đường"
                    abnormal_findings.append(f"Đường huyết cao ({value} {unit})")
                    recommendations.extend([
                        "Giảm tiêu thụ đường và carbohydrate tinh chế",
                        "Tăng cường hoạt động thể chất (đi bộ 30 phút/ngày)",
                        "Chia nhỏ bữa ăn trong ngày",
                        "Theo dõi cân nặng"
                    ])
                elif value < 70:
                    status = "Thấp hơn bình thường - có thể do nhịn ăn hoặc vấn đề sức khỏe khác"
                    abnormal_findings.append(f"Đường huyết thấp ({value} {unit})")
                    recommendations.extend([
                        "Ăn đủ bữa, không bỏ bữa",
                        "Có sẵn kẹo hoặc nước ngọt khi cần",
                        "Theo dõi triệu chứng hạ đường huyết"
                    ])
                else:
                    status = "Trong giới hạn bình thường - tốt"
           
            elif "cholesterol" in test_name.lower() or "mỡ máu" in test_name.lower():
                if value > 240:
                    status = "Cao - tăng nguy cơ bệnh tim mạch"
                    abnormal_findings.append(f"Cholesterol cao ({value} {unit})")
                    recommendations.extend([
                        "Giảm thực phẩm nhiều chất béo bão hòa",
                        "Tăng omega-3 (cá, hạt óc chó)",
                        "Ăn nhiều rau xanh và trái cây",
                        "Tập thể dục đều đặn"
                    ])
                elif value > 200:
                    status = "Hơi cao - cần chú ý chế độ ăn"
                    recommendations.extend([
                        "Kiểm soát chế độ ăn",
                        "Tăng hoạt động thể chất"
                    ])
                else:
                    status = "Bình thường - tốt"
           
            elif "hemoglobin" in test_name.lower() or "hồng cầu" in test_name.lower():
                if value < 12 and patient_gender == "female":
                    status = "Thấp - có thể thiếu máu"
                    abnormal_findings.append(f"Hemoglobin thấp ({value} {unit})")
                    recommendations.extend([
                        "Ăn thực phẩm giàu sắt (thịt đỏ, gan, rau bina)",
                        "Kết hợp với vitamin C để tăng hấp thu sắt",
                        "Tránh uống trà/cà phê ngay sau bữa ăn"
                    ])
                elif value < 13 and patient_gender == "male":
                    status = "Thấp - có thể thiếu máu"
                    abnormal_findings.append(f"Hemoglobin thấp ({value} {unit})")
                else:
                    status = "Bình thường"
           
            else:
                # Phân tích chung cho các xét nghiệm khác
                status = "Cần tham khảo ý kiến bác sĩ để hiểu rõ ý nghĩa"
           
            analysis.append(f"• **{test_name}**: {value} {unit} - {status}")
       
        # Lưu vào context
        self.add_to_context("symptoms_timeline", f"Xét nghiệm: {', '.join([t['test_name'] for t in test_results])}")
       
        result = {
            "detailed_analysis": analysis,
            "abnormal_findings": abnormal_findings,
            "lifestyle_recommendations": list(set(recommendations)) if recommendations else ["Duy trì lối sống lành mạnh"],
            "follow_up_advice": "Theo dõi định kỳ và tham khảo bác sĩ để có kế hoạch điều chỉnh phù hợp"
        }
       
        return json.dumps(result, ensure_ascii=False, indent=2)
   
    def analyze_prescription(self, medications: List[Dict], current_medications: List[str] = None, allergies: List[str] = None):
        """Phân tích đơn thuốc chi tiết với thông tin hữu ích"""
        drug_analysis = []
        usage_tips = []
       
        # Lưu thuốc vào context
        for med in medications:
            self.add_to_context("medications", f"{med['drug_name']} {med['dosage']}")
       
        for med in medications:
            drug_name = med.get("drug_name", "")
            dosage = med.get("dosage", "")
            frequency = med.get("frequency", "")
            duration = med.get("duration", "")
           
            # Phân tích cơ bản theo tên thuốc (có thể mở rộng)
            usage_info = ""
            if any(keyword in drug_name.lower() for keyword in ["paracetamol", "acetaminophen"]):
                usage_info = " - Thuốc giảm đau, hạ sốt. Uống sau ăn, không quá 4g/ngày"
            elif any(keyword in drug_name.lower() for keyword in ["ibuprofen"]):
                usage_info = " - Thuốc chống viêm, giảm đau. Uống sau ăn để tránh đau dạ dày"
            elif any(keyword in drug_name.lower() for keyword in ["amoxicillin"]):
                usage_info = " - Kháng sinh. Uống đủ liều theo đơn, không tự ý ngừng"
            elif any(keyword in drug_name.lower() for keyword in ["omeprazole"]):
                usage_info = " - Thuốc dạ dày. Uống trước ăn sáng 30-60 phút"
           
            analysis = f"• **{drug_name}** ({dosage}, {frequency}){usage_info}"
            if duration:
                analysis += f" - Thời gian: {duration}"
               
            drug_analysis.append(analysis)
       
        # Lời khuyên chung
        general_tips = [
            "Uống thuốc đúng giờ theo chỉ định của bác sĩ",
            "Không tự ý tăng/giảm liều lượng",
            "Uống thuốc với nước lọc, tránh nước ngọt hoặc rượu bia",
            "Bảo quản thuốc nơi khô ráo, thoáng mát",
            "Thông báo với bác sĩ nếu có tác dụng phụ bất thường"
        ]
       
        result = {
            "medications_analysis": drug_analysis,
            "drug_interactions": [],
            "allergy_warnings": [],
            "usage_guidelines": general_tips,
            "important_notes": "Hoàn thành đủ liệu trình kháng sinh nếu có. Không chia sẻ thuốc với người khác."
        }
       
        return json.dumps(result, ensure_ascii=False, indent=2)
   
    def create_health_plan(self, health_goals: List[str], current_conditions: List[str] = None, lifestyle_factors: Dict = None):
        """Tạo kế hoạch chăm sóc sức khỏe cá nhân hóa"""
        plan = {
            "health_goals": health_goals,
            "nutrition_plan": ["Ăn nhiều rau xanh", "Giảm đường và muối"],
            "exercise_plan": ["Đi bộ 30 phút/ngày", "Yoga 2-3 lần/tuần"],
            "monitoring_schedule": ["Kiểm tra sức khỏe định kỳ"]
        }
       
        return json.dumps(plan, ensure_ascii=False, indent=2)