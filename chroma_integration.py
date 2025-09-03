import chromadb
from chromadb.config import Settings
import uuid
import re
from typing import List, Dict, Tuple
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class MedicalChromaDB:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.setup_collections()
        
    def setup_collections(self):
        """Create 3 collections and insert mock data"""
        # Collection 1: Symptoms
        self.symptoms_collection = self.client.get_or_create_collection(
            name="symptoms",
            metadata={"description": "Medical symptoms"}
        )
        
        # Collection 2: Drug groups
        self.drug_groups_collection = self.client.get_or_create_collection(
            name="drug_groups", 
            metadata={"description": "Drug group information"}
        )
        
        # Collection 3: Lab results
        self.lab_results_collection = self.client.get_or_create_collection(
            name="lab_results",
            metadata={"description": "Laboratory test results"}
        )
        
        # Only insert mock data if collections are empty
        if self.symptoms_collection.count() == 0:
            self.insert_mock_data()
    
    def insert_mock_data(self):
        """Insert mock data into collections"""
        
        # Mock data for symptoms
        symptoms_data = [
            {"id": str(uuid.uuid4()), "text": "Đau đầu kéo dài, chóng mặt, buồn nôn", "category": "neurological", "severity": "moderate"},
            {"id": str(uuid.uuid4()), "text": "Ho khan, khó thở, đau ngực", "category": "respiratory", "severity": "severe"},
            {"id": str(uuid.uuid4()), "text": "Đau bụng, tiêu chảy, buồn nôn", "category": "gastrointestinal", "severity": "mild"},
            {"id": str(uuid.uuid4()), "text": "Sốt cao, ớn lạnh, đau cơ", "category": "infectious", "severity": "severe"},
            {"id": str(uuid.uuid4()), "text": "Mệt mỏi, chán ăn, sụt cân", "category": "systemic", "severity": "moderate"},
            {"id": str(uuid.uuid4()), "text": "Đau khớp, sưng khớp, cứng khớp buổi sáng", "category": "musculoskeletal", "severity": "moderate"},
            {"id": str(uuid.uuid4()), "text": "Đau ngực trái, hồi hộp, khó thở", "category": "cardiovascular", "severity": "severe"},
            {"id": str(uuid.uuid4()), "text": "Nổi mẩn đỏ, ngứa, sưng", "category": "dermatological", "severity": "mild"}
        ]
        
        self.symptoms_collection.add(
            documents=[item["text"] for item in symptoms_data],
            metadatas=[{"category": item["category"], "severity": item["severity"]} for item in symptoms_data],
            ids=[item["id"] for item in symptoms_data]
        )
        
        # Mock data for drug groups
        drug_groups_data = [
            {"id": str(uuid.uuid4()), "text": "Paracetamol - Thuốc giảm đau, hạ sốt. Liều dùng: 500mg x 3 lần/ngày", "group": "analgesic_antipyretic", "usage": "take_after_meals"},
            {"id": str(uuid.uuid4()), "text": "Amoxicillin - Kháng sinh nhóm penicillin. Liều dùng: 500mg x 3 lần/ngày", "group": "antibiotic", "usage": "take_1h_before_meals"},
            {"id": str(uuid.uuid4()), "text": "Omeprazole - Thuốc ức chế bơm proton. Liều dùng: 20mg x 1 lần/ngày", "group": "gastric", "usage": "take_morning_empty_stomach"},
            {"id": str(uuid.uuid4()), "text": "Metformin - Thuốc điều trị tiểu đường type 2. Liều dùng: 500mg x 2 lần/ngày", "group": "diabetes", "usage": "take_with_meals"},
            {"id": str(uuid.uuid4()), "text": "Amlodipine - Thuốc hạ huyết áp nhóm CCB. Liều dùng: 5mg x 1 lần/ngày", "group": "hypertension", "usage": "take_morning"},
            {"id": str(uuid.uuid4()), "text": "Cetirizine - Thuốc kháng histamin H1. Liều dùng: 10mg x 1 lần/ngày", "group": "allergy", "usage": "take_evening"},
            {"id": str(uuid.uuid4()), "text": "Ibuprofen - Thuốc chống viêm không steroid. Liều dùng: 400mg x 3 lần/ngày", "group": "anti_inflammatory", "usage": "take_after_meals"},
            {"id": str(uuid.uuid4()), "text": "Simvastatin - Thuốc hạ cholesterol nhóm statin. Liều dùng: 20mg x 1 lần/ngày", "group": "blood_lipid", "usage": "take_evening"}
        ]
        
        self.drug_groups_collection.add(
            documents=[item["text"] for item in drug_groups_data],
            metadatas=[{"group": item["group"], "usage": item["usage"]} for item in drug_groups_data],
            ids=[item["id"] for item in drug_groups_data]
        )
        
        # Mock data for lab results
        lab_results_data = [
            {"id": str(uuid.uuid4()), "text": "Glucose máu đói: 126 mg/dL (bình thường: 70-100). Chỉ số cao, nghi ngờ tiểu đường", "test_type": "blood_chemistry", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "Cholesterol toàn phần: 240 mg/dL (bình thường: <200). Nguy cơ tim mạch", "test_type": "blood_lipid", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "Hemoglobin: 9.5 g/dL (bình thường: 12-15). Thiếu máu nhẹ", "test_type": "complete_blood_count", "status": "low"},
            {"id": str(uuid.uuid4()), "text": "ALT: 65 U/L (bình thường: <40). Chức năng gan bất thường", "test_type": "liver_function", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "Creatinine: 1.8 mg/dL (bình thường: 0.6-1.2). Chức năng thận giảm", "test_type": "kidney_function", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "TSH: 8.5 mIU/L (bình thường: 0.4-4.0). Suy giáp", "test_type": "thyroid_hormone", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "HbA1c: 8.2% (bình thường: <5.7%). Kiểm soát đường huyết kém", "test_type": "diabetes", "status": "high"},
            {"id": str(uuid.uuid4()), "text": "CRP: 15 mg/L (bình thường: <3). Viêm nhiễm cấp tính", "test_type": "inflammation", "status": "high"}
        ]
        
        self.lab_results_collection.add(
            documents=[item["text"] for item in lab_results_data],
            metadatas=[{"test_type": item["test_type"], "status": item["status"]} for item in lab_results_data],
            ids=[item["id"] for item in lab_results_data]
        )
    
    def search_symptoms(self, query, n_results=5):
        """Search for similar symptoms"""
        return self.symptoms_collection.query(
            query_texts=[query],
            n_results=n_results
        )
    
    def search_drug_groups(self, query, n_results=5):
        """Search for drug information"""
        return self.drug_groups_collection.query(
            query_texts=[query],
            n_results=n_results
        )
    
    def search_lab_results(self, query, n_results=5):
        """Search for similar lab results"""
        return self.lab_results_collection.query(
            query_texts=[query],
            n_results=n_results
        )
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into smaller chunks for processing"""
        # Split by sentences first
        sentences = re.split(r'[.!?]\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def classify_text_chunk(self, text_chunk: str) -> str:
        """Classify text chunk into symptoms, drug_groups, or lab_results"""
        try:
            client = AzureOpenAI(
                api_version="2024-07-01-preview",
                azure_endpoint="https://aiportalapi.stu-platform.live/jpe",
                api_key="sk-dEyinSJuZ8V_u8gKuPksuA",
            )
            
            classification_prompt = f"""
Classify the following medical text into one of these categories:
- symptoms: Text about symptoms, signs, medical conditions, patient complaints
- drug_groups: Text about medications, drugs, prescriptions, dosages, drug information
- lab_results: Text about test results, lab values, medical examinations, diagnostic results

Text: "{text_chunk}"

Return only the category name (symptoms/drug_groups/lab_results).
"""
            
            response = client.chat.completions.create(
                model="GPT-4o-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=50,
                temperature=0
            )
            
            category = response.choices[0].message.content.strip().lower()
            return category if category in ['symptoms', 'drug_groups', 'lab_results'] else 'symptoms'
            
        except Exception as e:
            print(f"Classification error: {e}")
            return 'symptoms'  # Default fallback
    
    def add_file_content_to_db(self, file_content: str, source_filename: str = "user_upload") -> Dict[str, int]:
        """Process file content and add to appropriate collections"""
        # Split text into chunks
        chunks = self.split_text_into_chunks(file_content)
        
        # Track additions
        additions = {"symptoms": 0, "drug_groups": 0, "lab_results": 0}
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:  # Skip very short chunks
                continue
                
            # Classify chunk
            category = self.classify_text_chunk(chunk)
            
            # Generate metadata
            metadata = {
                "source": source_filename,
                "chunk_index": i,
                "category": "user_uploaded"
            }
            
            # Add to appropriate collection
            chunk_id = str(uuid.uuid4())
            
            if category == "symptoms":
                self.symptoms_collection.add(
                    documents=[chunk],
                    metadatas=[{**metadata, "severity": "unknown"}],
                    ids=[chunk_id]
                )
                additions["symptoms"] += 1
                
            elif category == "drug_groups":
                self.drug_groups_collection.add(
                    documents=[chunk],
                    metadatas=[{**metadata, "group": "user_uploaded", "usage": "unknown"}],
                    ids=[chunk_id]
                )
                additions["drug_groups"] += 1
                
            else:  # lab_results
                self.lab_results_collection.add(
                    documents=[chunk],
                    metadatas=[{**metadata, "test_type": "user_uploaded", "status": "unknown"}],
                    ids=[chunk_id]
                )
                additions["lab_results"] += 1
        
        return additions
    
    def process_uploaded_file(self, file_path: str) -> Dict[str, int]:
        """Process uploaded file and add content to ChromaDB"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            filename = os.path.basename(file_path)
            return self.add_file_content_to_db(content, filename)
            
        except Exception as e:
            print(f"Error processing file: {e}")
            return {"symptoms": 0, "drug_groups": 0, "lab_results": 0}
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics of all collections"""
        return {
            "symptoms": self.symptoms_collection.count(),
            "drug_groups": self.drug_groups_collection.count(),
            "lab_results": self.lab_results_collection.count()
        }
    
    def add_to_specific_collection(self, content: str, filename: str, collection_type: str) -> Dict[str, int]:
        """Add content directly to specific collection without AI classification"""
        chunks = self.split_text_into_chunks(content)
        additions = {"symptoms": 0, "drug_groups": 0, "lab_results": 0}
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:
                continue
                
            metadata = {
                "source": filename,
                "chunk_index": i,
                "category": "manual_upload"
            }
            
            chunk_id = str(uuid.uuid4())
            
            if collection_type == "symptoms":
                self.symptoms_collection.add(
                    documents=[chunk],
                    metadatas=[{**metadata, "severity": "unknown"}],
                    ids=[chunk_id]
                )
                additions["symptoms"] += 1
                
            elif collection_type == "drug_groups":
                self.drug_groups_collection.add(
                    documents=[chunk],
                    metadatas=[{**metadata, "group": "manual_upload", "usage": "unknown"}],
                    ids=[chunk_id]
                )
                additions["drug_groups"] += 1
                
            elif collection_type == "lab_results":
                self.lab_results_collection.add(
                    documents=[chunk],
                    metadatas=[{**metadata, "test_type": "manual_upload", "status": "unknown"}],
                    ids=[chunk_id]
                )
                additions["lab_results"] += 1
        
        return additions

# Test function
if __name__ == "__main__":
    db = MedicalChromaDB()
    
    # Test search
    print("=== Test symptoms search ===")
    results = db.search_symptoms("đau đầu chóng mặt")
    print(results)
    
    print("\n=== Test drug search ===")
    results = db.search_drug_groups("thuốc giảm đau")
    print(results)
    
    print("\n=== Test lab results search ===")
    results = db.search_lab_results("đường huyết cao")
    print(results)
    
    # Test file processing
    print("\n=== Test file processing ===")
    sample_text = """
    Bệnh nhân than phiền đau đầu kéo dài 3 ngày, kèm theo chóng mặt và buồn nôn.
    Đã sử dụng Paracetamol 500mg x 2 lần/ngày nhưng chưa thấy cải thiện.
    Kết quả xét nghiệm cho thấy glucose máu đói: 140 mg/dL, cao hơn bình thường.
    """
    
    additions = db.add_file_content_to_db(sample_text, "test_file.txt")
    print(f"Added to collections: {additions}")
    
    print("\n=== Collection stats ===")
    stats = db.get_collection_stats()
    print(f"Collection sizes: {stats}")