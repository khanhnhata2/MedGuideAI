import os
import firebase_admin
import openai
from dotenv import load_dotenv
from firebase_admin import firestore, credentials
from google.cloud.firestore_v1 import FieldFilter


def init_firebase():
    if not firebase_admin._apps:  # Nếu chưa có app nào được khởi tạo
        cred = credentials.Certificate("baymax-a7a0d-firebase-adminsdk-fbsvc-f00628f505.json")
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
    base_url=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
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


def handle_compare_list_result(results):
    # Gửi dữ liệu này lên OpenAI để so sánh (ví dụ)
    if results is not None:
        print("---results", results)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là một bác sĩ xét nghiệm giỏi, có khả năng so sánh và phân tích kết quả xét nghiệm y khoa."
                },
                {
                    "role": "user",
                    "content": f"""So sánh 2 kết quả xét nghiệm của cùng một bệnh nhân (lần 1 là kết quả cũ hơn, lần 2 là kết quả mới hơn)
                    Dữ liệu:
                    Lần 1 (kết quả cũ): {results[1]}
                    Lần 2 (kết quả mới): {results[0]}
                    
                    - Nếu một trong hai lần xét nghiệm thiếu giá trị, Ghi "chưa rõ" một lần.
                    Ví dụ: PLT: 369 G/L ➡️ chưa rõ.
                    
                    Phân tích bao gồm:
                    1. Thông tin chung của từng lần xét nghiệm.
                    2. Bảng so sánh từng chỉ số:
                    - Tên chỉ số
                    - Giá trị lần 1
                    - Giá trị lần 2
                    - Mức thay đổi (tăng/giảm, %)
                    - Đánh giá sự thay đổi chỉ số đó là tốt hơn hay xấu hơn (tốt hơn, xấu hơn, ổn định)
                    3. Nhận xét tổng quan về tình hình sức khỏe.
                    4. Khuyến nghị theo dõi hoặc điều trị.
                    Hãy trả về kết quả dưới dạng JSON có 3 phần: 'summary', 'table', 'recommendations'."""
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content
    else:
        return None

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

        print("tuong tac", response.choices[0].message.content)

        return response.choices[0].message.content
    else:
        return None
