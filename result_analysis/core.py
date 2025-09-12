import os
import firebase_admin
import openai
from dotenv import load_dotenv
from firebase_admin import firestore, credentials
from google.cloud.firestore_v1 import FieldFilter
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

client = openai.OpenAI(
    base_url=st.secrets["OPENAI_ENDPOINT"],
    api_key=st.secrets["OPENAI_API_KEY"],
)


def get_latest_data(user_id, data_type, limit=1):
    """Lấy kết quả xét nghiệm gần nhất (mặc định 1 bản ghi)."""

    col_name = "lab_results_grouped" if data_type == "lab" else "medicine_lists_grouped"
    results_ref = (
        db.collection(col_name)
        .where(filter=FieldFilter("user_id", "==", user_id))
        .order_by("document_date", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )

    docs = results_ref.stream()
    lab_results = [doc.to_dict() for doc in docs]
    return lab_results


# Các hàm tương ứng gọi khi nhận function calling
def handle_get_result(data_type, limit=1, user_id='A12345'):
    latest_data = get_latest_data(
        user_id=user_id,
        data_type=data_type,
        limit=limit  # Chỉ lấy 1 bản ghi mới nhất
    )

    if not latest_data:
        return None

    return latest_data

def summarize_user_result(system_prompt, user_result, previous_user_result):
    if user_result is not None:
        fileUrl = user_result.get("fileUrl", "Hiện chưa có thông tin") if isinstance(user_result, dict) else "Hiện chưa có thông tin"

        if isinstance(previous_user_result, list) and previous_user_result and isinstance(previous_user_result[0], dict):
            previousFileUrl = previous_user_result[0].get("fileUrl", "Hiện chưa có thông tin")
        else:
            previousFileUrl = "Hiện chưa có thông tin"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""
                    Dưới đây là danh sách những kết quả xét nghiệm của người dùng
                    - Kết quả xét nghiệm gần nhất: "{user_result or "Hiện chưa có thông tin"}"
                    
                    - Kết quả xét nghiệm trước đó: "{previous_user_result if previous_user_result else "Hiện chưa có thông tin"}"
                    
                    Yêu cầu:
                    1. Tóm tắt kết quả xét nghiệm gần nhất và đưa ra tư vấn về kết quả, lối sống, dinh dưỡng cho người dùng
                    2. So sánh kết quả xét nghiệm gần nhất và kết quả xét nghiệm trước đó, đưa ra những chỉ số có sự thay đổi và nhận xét chi tiết
                    - Ví dụ "8.0 -> 5.0 (giảm) -> Tốt"
                    - Nếu không có kêt quả xét nghiệm trước đó thì bỏ qua so sánh
                    3. Trả lời với giọng văn chuyên nghiệp, dễ hiểu
                    4. Kết thúc bằng link file gốc của kết quả xét nghiệm gần nhất(nếu có): {fileUrl}
                    5. Kết thúc bằng link file gốc của kết quả xét nghiệm trước đó(nếu có): {previousFileUrl}
                    """
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content
    else:
        return "Hiện tại bạn chưa có kết quả xét nghiệm nào"


def summarize_prescription(system_prompt, prescription, previous_prescription):
    if prescription is not None:
        # Lấy fileUrl của đơn thuốc hiện tại
        fileUrl = (
            prescription.get("fileUrl", "Hiện chưa có thông tin")
            if isinstance(prescription, dict)
            else "Hiện chưa có thông tin"
        )

        # Lấy fileUrl của đơn thuốc trước đó
        if isinstance(previous_prescription, list) and previous_prescription and isinstance(previous_prescription[0], dict):
            previousFileUrl = previous_prescription[0].get("fileUrl", "Hiện chưa có thông tin")
        else:
            previousFileUrl = "Hiện chưa có thông tin"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""
                    Đây là thông tin về đơn thuốc của người dùng:
                    - Đơn thuốc gần nhất: "{prescription or "Hiện chưa có thông tin"}"
                    - Đơn thuốc trước đó: "{previous_prescription if previous_prescription else "Hiện chưa có thông tin"}"

                    Yêu cầu:
                    1. Tóm tắt các thuốc trong đơn thuốc gần nhất, công dụng chính, lưu ý về liều dùng và tác dụng phụ thường gặp.
                    2. So sánh với đơn thuốc trước đó (nếu có):
                       - Kiểm tra có tương tác thuốc giữa các thuốc.
                    3. Đưa ra gợi ý cho người dùng về việc tuân thủ điều trị, theo dõi sức khỏe và khi nào nên tái khám.
                    4. Trả lời bằng văn phong chuyên nghiệp, dễ hiểu.
                    5. Cuối phần trả lời, bổ sung:
                       - Link file gốc đơn thuốc gần nhất: {fileUrl}
                       - Link file gốc đơn thuốc trước đó: {previousFileUrl}
                    """
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content
    else:
        return "Hiện tại bạn chưa có đơn thuốc nào"



def handle_compare_list_medicines(results):
    # Gửi dữ liệu này lên OpenAI để so sánh (ví dụ)
    if results is not None:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là một chuyên gia y dược có nhiệm vụ kiểm tra các tương tác giữa các loại thuốc do người dùng cung cấp"
                },
                {
                    "role": "user",
                    "content": f"""So sánh 2 đơn thuốc và kiểm tra các tương tác thuốc có thể xảy ra giữa các loại thuốc trong 2 đơn này.
                    Trả về kết quả bao gồm:
                     - Danh sách thuốc có thể tương tác với nhau
                     - Mức độ tương tác
                     - Triệu chứng, hiện tượng, tác hại có thể xảy ra khi tương tác thuốc
                     - Cách xử lý khi bị tương tác thuốc
                     Luôn kết thúc bằng: "Thông tin trên chỉ dành để tham khảo. Hãy liên hệ bác sỹ của bạn để có những thông tin chính xác nhất"
                     
                    Đây là dữ liệu 2 đơn thuốc:
                    {results}"""
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content
    else:
        return None
