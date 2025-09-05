# login.py
# Chức năng đăng nhập với 2 loại user: admin và normal

import firebase_admin
from firebase_admin import credentials, firestore

# Khởi tạo Firestore
cred = credentials.Certificate('baymax-a7a0d-firebase-adminsdk-fbsvc-f70ba3918d.json')
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

# Hàm đăng nhập từ Firestore
def login(username, password):
    user_ref = db.collection('users').document(username)
    user_doc = user_ref.get()
    if user_doc.exists:
        user = user_doc.to_dict()
        if user['password'] == password:
            return {'username': username, 'role': user['role']}
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
