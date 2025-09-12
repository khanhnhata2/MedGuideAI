# 🩺 MedGuideAI
**MedGuideAI** là chatbot y tế thông minh hỗ trợ người dùng trong việc:
- Upload tài liệu y tế
- Tư vấn về các loại thuốc.
- Phân tích, so sánh đơn thuốc/ kết quả xét nghiệm (bao gồm cả hình ảnh).
- Cho lời khuyên y tế
- Đặt lịch khám.

## 🚀 Công nghệ sử dụng
- **Software**: Python + Streamlit
- **Database & Storage**: Firestore + Pinecone + Cloudinary
- **AI**: OpenAI + LangChain
- **Tools**: GitHub Copilot

## Cài đặt & Chạy dự án
- Tạo file secrets.toml trong thư mục .streamlit/
```
  OPENAI_API_KEY = "your_openai_api_key"
  OPENAI_ENDPOINT = "your_openai_endpoint"
  OPENAI_MODEL = "gpt-4o-mini"
  OPENAI_EMBEDDING_KEY = "your_openai_embedding_key"
  OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
  PINECONE_API_KEY = "your_pinecone_key"
  CLOUDINARY_URL = "your_cloudinary_url"
  [firebase]
  firebase config...
```

- Install required library
```
  pip install -r requirements.txt
  pip install torch torchvision torchaudio
  FFmpeg binary installed and available in your system PATH
```

- How to run: 
```
streamlit run simple_app.py
```
