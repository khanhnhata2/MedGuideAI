# login.py
# Chức năng đăng nhập với 2 loại user: admin và normal

import firebase_admin
from firebase_admin import credentials, firestore

# Khởi tạo Firestore
cred = credentials.Certificate('baymax-a7a0d-firebase-adminsdk-fbsvc-cf2ffd7165.json')
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
