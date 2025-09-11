import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import easyocr
from PIL import Image
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import streamlit as st

def init_firebase():
    if not firebase_admin._apps:  # Nếu chưa có app nào được khởi tạo
        firebase_config = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
        print("Firebase initialized.")
    else:
        print("Firebase already initialized.")


# Load biến môi trường
load_dotenv()

# Gọi hàm khởi tạo
init_firebase()
db = firestore.client()

llm = ChatOpenAI(
    base_url=st.secrets["OPENAI_ENDPOINT"],
    api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["OPENAI_MODEL"],
    temperature=0
)

# OCR tiếng Việt
def image_to_text(uploaded_file):
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    reader = easyocr.Reader(['vi'], gpu=False)
    results = reader.readtext(image_np)
    texts = [{"text": text, "confidence": conf} for _, text, conf in results]
    text_for_prompt = "\n".join(
        f"{item['text']} (độ tin cậy: {item['confidence']:.2f})"
        for item in texts
    )
    return text_for_prompt

# Prompt phân loại
classification_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Phân loại tài liệu sau thành 1 trong các loại: đơn thuốc, kết quả xét nghiệm, unknown. 
Nội dung dưới đây là kết quả OCR từ ảnh, mỗi dòng gồm nội dung và độ tin cậy:
{text}
Chỉ trả về duy nhất tên loại.
"""
)

def classify_doc_type(text: str) -> str:
    try:
        chain = classification_prompt | llm
        result = chain.invoke({"text": text})
        print("Phân loại LLM trả về:", result)
        return result.content.strip().lower()
    except Exception as e:
        print("Lỗi khi gọi OpenAI:", e)
        return "unknown"

def analyze_medicine_with_knowledge(ocr_text: str, latest_prescription, latest_test_result) -> BaseMessage:
    """
    Dùng OpenAI để tìm hiểu thông tin thuốc từ nội dung OCR của đơn thuốc.
    """
    knowledge_prompt = PromptTemplate(
        input_variables=["ocr_text", "latest_prescription", "latest_test_result"],
        template="""
        Bạn là bác sĩ y khoa.
        
        Dưới đây là nội dung OCR từ ảnh đơn thuốc mới nhất mà bệnh nhân upload (có thể chứa nhiều tên thuốc, liều lượng, cách dùng):
        
        {ocr_text}
        
        Dưới đây là đơn thuốc gần nhất của bệnh nhân trong kho dữ liệu bệnh viện:
        
        {latest_prescription}
        
        Dưới đây là kết quả xét nghiệm gần nhất của bệnh nhân trong kho dữ liệu bệnh viện:
        
        {latest_test_result}
        
        Yêu cầu:
        1. Dựa theo nội dung OCR từ ảnh đơn thuốc mới nhất mà bệnh nhân vừa upload, trả về tên thuốc, liều lượng, cách dùng, tác dụng phụ
        2. So sánh nội dung OCR từ ảnh đơn thuốc mới nhất mà bệnh nhân upload với đơn thuốc gần nhất của bệnh nhân trong kho dữ liệu bệnh viện, kiểm tra tương tác thuốc có thể xảy ra giữa 2 đơn thuốc.
        3. Dựa vào kết quả xét nghiệm gần nhất của bệnh nhân trong kho dữ liệu bệnh viện, kiểm tra các thuốc trong nội dung OCR từ ảnh đơn thuốc mới nhất mà bệnh nhân upload có thể gây ảnh hưởng gì không
        """
    )


    chain = knowledge_prompt | llm
    result = chain.invoke({"ocr_text": ocr_text, "latest_prescription": latest_prescription, "latest_test_result": latest_test_result})
    return result


def analyze_lab_with_knowledge(lab_text: str, latest_prescription, latest_test_result) -> BaseMessage:
    """
    Dùng OpenAI để giải thích ý nghĩa cho danh sách kết quả xét nghiệm.
    lab_text: nội dung OCR của bảng kết quả xét nghiệm
    """
    knowledge_prompt = PromptTemplate(
        input_variables=["lab_text", "latest_prescription", "latest_test_result"],
        template="""
            Bạn là chuyên gia xét nghiệm y khoa.
            
            Dưới đây là nội dung OCR từ bảng kết quả xét nghiệm y tế:
            
            {lab_text}
            
            Dưới đây là đơn thuốc gần nhất của bệnh nhân trong kho dữ liệu bệnh viện:
                
            {latest_prescription}
                
            Dưới đây là kết quả xét nghiệm gần nhất của bệnh nhân trong kho dữ liệu bệnh viện:
                
            {latest_test_result}
                
            Yêu cầu:
            1. Dựa theo nội dung OCR từ bảng kết quả xét nghiệm y tế, phân tích và giải thích về tất cả các chỉ số.
            2. So sánh nội dung OCR từ bảng kết quả xét nghiệm y tế với kết quả xét nghiệm gần nhất của bệnh nhân trong kho dữ liệu bệnh viện, phân tích và nhận xét sự thay đổi giữa hai lần.
            3. Dựa vào đơn thuốc gần nhất của bệnh nhân trong kho dữ liệu bệnh viện, kiểm tra xem có loại thuốc nào cần chú ý hoặc có thể gây ảnh hưởng xấu đến bệnh nhân không?
            """
    )


    chain = knowledge_prompt | llm
    return chain.invoke({"lab_text": lab_text, "latest_prescription": latest_prescription, "latest_test_result": latest_test_result})

def process_image_pipeline(image_path: str, latest_prescription, latest_test_result):
    print("🔍 Bắt đầu OCR...")
    text = image_to_text(image_path)
    print("📄 Kết quả OCR:\n", text)

    print("📂 Đang phân loại tài liệu...")
    doc_type = classify_doc_type(text)
    print("📌 Loại tài liệu:", doc_type)

    item = None
    if doc_type == "đơn thuốc":
        item = analyze_medicine_with_knowledge(text, latest_prescription, latest_test_result)
        print("item1", item.content)
    elif doc_type == "kết quả xét nghiệm":
        item = analyze_lab_with_knowledge(text, latest_prescription, latest_test_result)
        print('item2', item.content)

    return {
        "doc_type": doc_type,
        "data": item.content,
    }
