# login.py
# Chức năng đăng nhập với 2 loại user: admin và normal

import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

# Khởi tạo Firestore

firebase_config = dict(st.secrets["firebase"])
cred = credentials.Certificate(firebase_config)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Hàm tạo user mẫu vào Firestore
def create_sample_users():
    users_ref = db.collection('users')
    users_ref.document('admin').set({
        'password': 'admin123',
        'role': 'admin'
    })
    users_ref.document('user').set({
        'password': 'user123',
        'role': 'normal'
    })
    users_ref.document('120007313547').set({
        'password': 'user123',
        'role': 'normal'
    })

    users_ref.document('120007313545').set({
        'password': 'user123',
        'role': 'normal'
    })

def get_latest_record(collection_name, user_id):
    """Lấy bản ghi mới nhất, nếu chưa có trả None"""
    # Kiểm tra collection có document không
    check_exists = (
        db.collection(collection_name)
        .where("user_id", "==", user_id)
        .limit(1)
        .stream()
    )
    if not list(check_exists):
        return None  # Chưa có dữ liệu

    # Nếu có thì query bản mới nhất
    query = (
        db.collection(collection_name)
        .where("user_id", "==", user_id)
        .order_by("examDate", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )

    records = list(query)
    if not records:
        return None

    return records[0].to_dict()

def get_records_in_range(collection_name, user_id, start: int = 0, end: int = None):
    """Lấy các bản ghi từ vị trí start đến end (theo examDate DESC).
    Nếu end=None thì lấy hết từ start đến cuối."""

    query = (
        db.collection(collection_name)
        .where("user_id", "==", user_id)
        .order_by("examDate", direction=firestore.Query.DESCENDING)
        .stream()
    )

    records = [doc.to_dict() for doc in query]

    # Nếu end = None thì Python slice sẽ tự lấy đến hết danh sách
    filtered_results = records[start:end]

    return filtered_results if filtered_results else None



def login(username, password):
    user_ref = db.collection('users').document(username)
    user_doc = user_ref.get()

    if user_doc.exists:
        user = user_doc.to_dict()
        if user.get('password') == password:
            return {
                'username': username,
                'role': user.get('role')
            }

    return None

if __name__ == "__main__":
    # Tạo user mẫu nếu chưa có
    create_sample_users()
    username = input("Username: ")
    password = input("Password: ")
    result = login(username, password)
    if result:
        print(f"Login successful! Role: {result['role']}")
    else:
        print("Login failed!")
