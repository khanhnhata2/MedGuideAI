# sched_appointment.py
"""
Xử lý đặt lịch khám bệnh qua OpenAI Function Calling và lưu vào Firebase Firestore.
"""

import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from openai import AzureOpenAI

# ==== Firebase Initialization ====
FIREBASE_KEY_PATH = "baymax-a7a0d-firebase-adminsdk-fbsvc-f00628f505.json"  # file service account
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)
    print("<duypv10 log> Firebase initialized.")
else:
    print("<duypv10 log> Firebase already initialized.")
db = firestore.client()


# ==== Core function: Lưu lịch hẹn ====
def schedule_appointment(date: str, time: str, patient_name: str, note: str = "") -> str:
    """
    Lưu lịch hẹn vào Firestore.
    """
    try:
        datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%d/%m/%Y")
        db.collection("appointments").add({
            "date": date,
            "time": time,
            "patientName": patient_name,
            "note": note,
            "status": "pending",
            "createdAt": firestore.SERVER_TIMESTAMP
        })
        return (
            "✅ Đặt lịch thành công cho:\n"
            f"- Người đặt lịch: {patient_name}\n"
            f"- Ngày khám: {formatted_date}\n"
            f"- Giờ khám: {time}\n"
            f"- Mục khám: {note}\n"
            "\nThông tin liên hệ:\n"
            "\n📍 123 Thụy Khuê, Phường Tây Hồ, TP Hà Nội\n"
            "\n📞 Hotline (24/24): 0912345678\n"
        )
    except ValueError:
        return "❌ Sai định dạng ngày hoặc giờ. Dùng YYYY-MM-DD và HH:MM."


# ==== Function để lấy ngày hiện tại ====
def get_current_datetime() -> str:
    """Lấy ngày giờ hiện tại theo múi giờ Việt Nam"""
    from datetime import datetime, timezone, timedelta
    
    vietnam_tz = timezone(timedelta(hours=7))
    current_time = datetime.now(vietnam_tz)
    
    result = {
        "current_datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "date": current_time.strftime("%Y-%m-%d"),
        "time": current_time.strftime("%H:%M:%S"),
        "day_of_week": current_time.strftime("%A"),
        "timezone": "UTC+7 (Vietnam)"
    }
    
    return json.dumps(result, ensure_ascii=False, indent=2)

# ==== Schema cho OpenAI Function Calling ====
get_current_datetime_schema = {
    "name": "get_current_datetime",
    "description": "Lấy ngày giờ hiện tại theo múi giờ Việt Nam để xử lý các ngày tương đối như 'ngày mai', 'tuần sau'",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

schedule_appointment_schema = {
    "name": "schedule_appointment",
    "description": "Đặt lịch khám và lưu vào Firebase Firestore",
    "parameters": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Ngày (YYYY-MM-DD)"},
            "time": {"type": "string", "description": "Giờ (HH:MM)"},
            "patient_name": {"type": "string", "description": "Tên bệnh nhân"},
            "note": {"type": "string", "description": "Đăng ký mục khám"}
        },
        "required": ["date", "time", "patient_name", "note"]
    }
}


# ==== Hàm xử lý function_call từ GPT ====
def handle_function_call(tool_call):
    """
    Nhận tool_call từ OpenAI, parse argument và gọi hàm tương ứng.
    """
    func_name = tool_call.function.name
    
    if func_name == "get_current_datetime":
        return get_current_datetime()
    elif func_name == "schedule_appointment":
        args = json.loads(tool_call.function.arguments)
        return schedule_appointment(**args)

    return f"❌ Function '{func_name}' chưa được hỗ trợ."


# Biến lưu context hội thoại
session_data = {"date": None, "time": None, "patient_name": None, "note": None}
chat_history = []

# ==== Hàm xử lý request đặt lịch ====
class AppointmentProcessor:
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str):
        """
        Truyền API key và endpoint từ main.py vào để dùng AzureOpenAI.
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

    def process_with_function_calling(self, user_input: str):
        global session_data, chat_history

        # Lưu user_input
        chat_history.append({"role": "user", "content": user_input})

        # Nếu là lần đầu tiên thì thêm system prompt
        if len(chat_history) == 1:
            chat_history.insert(0, {
                "role": "system",
                "content": "Bạn là trợ lý y tế, giúp đặt lịch khám bệnh. Khi người dùng nói 'ngày mai', 'tuần sau', hãy gọi get_current_datetime để biết ngày hiện tại rồi tính toán ngày chính xác. Nếu thiếu thông tin, hãy hỏi tiếp lịch sự, rõ ràng."
            })

        # Gửi sang GPT sử dụng function-calling với cả 2 functions
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history,
            tools=[
                {"type": "function", "function": get_current_datetime_schema},
                {"type": "function", "function": schedule_appointment_schema}
            ],
            tool_choice="auto",
            temperature=0
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            func_name = tool_call.function.name
            
            if func_name == "get_current_datetime":
                # Lấy thông tin ngày hiện tại
                datetime_result = get_current_datetime()
                
                # Thêm kết quả vào chat history và gọi lại GPT
                chat_history.append(choice.message)
                chat_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": datetime_result
                })
                
                # Gọi lại GPT với thông tin ngày hiện tại
                response2 = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=chat_history,
                    tools=[
                        {"type": "function", "function": get_current_datetime_schema},
                        {"type": "function", "function": schedule_appointment_schema}
                    ],
                    tool_choice="auto",
                    temperature=0
                )
                
                choice2 = response2.choices[0]
                if choice2.finish_reason == "tool_calls" and choice2.message.tool_calls:
                    # Xử lý schedule_appointment call
                    tool_call2 = choice2.message.tool_calls[0]
                    if tool_call2.function.name == "schedule_appointment":
                        args = json.loads(tool_call2.function.arguments)
                        session_data.update({k: v for k, v in args.items() if v})
                        
                        if all(session_data.values()):
                            result = schedule_appointment(**session_data)
                            ai_response = result
                            session_data = {"date": None, "time": None, "patient_name": None, "note": None}
                            chat_history = []
                        else:
                            ai_response = f"Hiện có: {session_data}. Bạn vui lòng cung cấp thông tin còn thiếu."
                    else:
                        ai_response = choice2.message.content or "❌ Không nhận dạng được yêu cầu."
                else:
                    ai_response = choice2.message.content or "❌ Không nhận dạng được yêu cầu."
                    
            elif func_name == "schedule_appointment":
                args = json.loads(tool_call.function.arguments)
                session_data.update({k: v for k, v in args.items() if v})

                if all(session_data.values()):
                    result = schedule_appointment(**session_data)
                    ai_response = result
                    session_data = {"date": None, "time": None, "patient_name": None, "note": None}
                    chat_history = []
                else:
                    ai_response = f"Hiện có: {session_data}. Bạn vui lòng cung cấp thông tin còn thiếu."
            else:
                ai_response = "❌ Function không được hỗ trợ."
        else:
            ai_response = choice.message.content or "❌ Không nhận dạng được yêu cầu."

        # Lưu phản hồi AI
        chat_history.append({"role": "assistant", "content": ai_response})

        return {"ai_response": ai_response}


