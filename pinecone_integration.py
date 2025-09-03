import os
import uuid
import re
from typing import List, Dict
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time
 
load_dotenv()
 
class MedicalPineconeDB:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY", "pcsk_2LbbcE_JfeW6YB5qG8d599CoNSvQVKPirsLdF1KCudYSz5o4tnrsUbs9FFggrRv6JoA145")
        )
       
        # Initialize OpenAI for embeddings (separate from main client)
        self.embedding_client = openai.OpenAI(
            base_url=os.getenv("OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_EMBEDDING_KEY"),
        )
       
        # Initialize OpenAI for classification (main client)
        self.openai_client = openai.OpenAI(
            base_url=os.getenv("OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
       
        # Index configuration
        self.index_name = "medical-rag-index"
        self.embedding_dimension = 1536  # text-embedding-3-small dimension
       
        # Try to get actual dimension from a test embedding
        try:
            # Try to get actual dimension from a test embedding
            test_response = self.embedding_client.embeddings.create(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                input="test"
            )
            actual_dimension = len(test_response.data[0].embedding)
            if actual_dimension != self.embedding_dimension:
                print(f"Adjusting embedding dimension from {self.embedding_dimension} to {actual_dimension}")
                self.embedding_dimension = actual_dimension
        except Exception as e:
            print(f"Could not determine embedding dimension, using default {self.embedding_dimension}: {e}")
       
        self.setup_index()
        self.insert_mock_data()
   
    def setup_index(self):
        """Create or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
           
            if self.index_name not in existing_indexes:
                # Create index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
           
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
           
        except Exception as e:
            print(f"Error setting up Pinecone index: {e}")
            # Fallback: use a mock index for development
            self.index = None
   
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI text-embedding-3-small"""
        try:
            # Try OpenAI embedding first
            response = self.embedding_client.embeddings.create(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                input=text
            )
            embedding = response.data[0].embedding
           
            # Ensure it's not all zeros and has correct dimension
            if embedding and len(embedding) > 0:
                # Adjust dimension if needed (text-embedding-3-small is 1536 dimensions)
                if len(embedding) != self.embedding_dimension:
                    if len(embedding) < self.embedding_dimension:
                        # Pad with zeros if smaller
                        embedding.extend([0.0] * (self.embedding_dimension - len(embedding)))
                    else:
                        # Truncate if larger
                        embedding = embedding[:self.embedding_dimension]
               
                return embedding
            else:
                raise Exception("Empty embedding returned")
               
        except Exception as e:
            print(f"OpenAI embedding failed: {e}")
            print("Falling back to simple hash-based embedding...")
           
            # Fallback to simple embedding if openAI fails
            return self._get_simple_embedding(text)
   
    def _get_simple_embedding(self, text: str) -> List[float]:
        """Fallback simple embedding based on text characteristics"""
        try:
            import hashlib
           
            # Create a simple embedding based on text characteristics
            words = text.lower().split()
            char_count = len(text)
            word_count = len(words)
           
            # Create a simple feature vector
            embedding = [0.0] * self.embedding_dimension
           
            # Fill embedding with features based on text content
            if words:
                # Use hash of text to create pseudo-embedding
                text_hash = hashlib.md5(text.encode()).hexdigest()
               
                # Convert hash to numbers and normalize
                for i, char in enumerate(text_hash[:min(len(text_hash), self.embedding_dimension//32)]):
                    idx = i * 32
                    if idx < self.embedding_dimension:
                        # Convert hex char to float between -1 and 1
                        val = (int(char, 16) - 7.5) / 7.5
                        embedding[idx] = val
               
                # Add some text-based features
                if len(embedding) > 10:
                    embedding[0] = min(word_count / 50.0, 1.0)  # Word count feature
                    embedding[1] = min(char_count / 500.0, 1.0)  # Char count feature
                   
                    # Add keyword-based features
                    medical_keywords = ['thuốc', 'bệnh', 'đau', 'triệu chứng', 'xét nghiệm', 'glucose', 'cholesterol']
                    for i, keyword in enumerate(medical_keywords):
                        if i + 2 < len(embedding) and keyword in text.lower():
                            embedding[i + 2] = 1.0
           
            # Ensure it's not all zeros
            if all(x == 0.0 for x in embedding):
                embedding[0] = 0.1  # Make sure at least one value is non-zero
           
            return embedding
           
        except Exception as e:
            print(f"Error generating simple embedding: {e}")
            # Return a minimal non-zero embedding
            embedding = [0.0] * self.embedding_dimension
            embedding[0] = 0.1
            return embedding
   
    def insert_mock_data(self):
        """Insert mock data into Pinecone index"""
        if not self.index:
            print("Index not available, skipping mock data insertion")
            return
           
        try:
            # Check if data already exists
            stats = self.index.describe_index_stats()
            if stats.total_vector_count > 0:
                print("Mock data already exists in index")
                return
               
            # Mock data for symptoms
            symptoms_data = [
                {"text": "Đau đầu kéo dài, chóng mặt, buồn nôn", "category": "symptoms", "severity": "moderate", "type": "neurological"},
                {"text": "Ho khan, khó thở, đau ngực", "category": "symptoms", "severity": "severe", "type": "respiratory"},
                {"text": "Đau bụng, tiêu chảy, buồn nôn", "category": "symptoms", "severity": "mild", "type": "gastrointestinal"},
                {"text": "Sốt cao, ớn lạnh, đau cơ", "category": "symptoms", "severity": "severe", "type": "infectious"},
                {"text": "Mệt mỏi, chán ăn, sụt cân", "category": "symptoms", "severity": "moderate", "type": "systemic"},
                {"text": "Đau khớp, sưng khớp, cứng khớp buổi sáng", "category": "symptoms", "severity": "moderate", "type": "musculoskeletal"},
                {"text": "Đau ngực trái, hồi hộp, khó thở", "category": "symptoms", "severity": "severe", "type": "cardiovascular"},
                {"text": "Nổi mẩn đỏ, ngứa, sưng", "category": "symptoms", "severity": "mild", "type": "dermatological"}
            ]
           
            # Mock data for drug groups
            drug_groups_data = [
                {"text": "Paracetamol - Thuốc giảm đau, hạ sốt. Liều dùng: 500mg x 3 lần/ngày", "category": "drug_groups", "group": "analgesic_antipyretic", "usage": "take_after_meals"},
                {"text": "Amoxicillin - Kháng sinh nhóm penicillin. Liều dùng: 500mg x 3 lần/ngày", "category": "drug_groups", "group": "antibiotic", "usage": "take_1h_before_meals"},
                {"text": "Omeprazole - Thuốc ức chế bơm proton. Liều dùng: 20mg x 1 lần/ngày", "category": "drug_groups", "group": "gastric", "usage": "take_morning_empty_stomach"},
                {"text": "Metformin - Thuốc điều trị tiểu đường type 2. Liều dùng: 500mg x 2 lần/ngày", "category": "drug_groups", "group": "diabetes", "usage": "take_with_meals"},
                {"text": "Amlodipine - Thuốc hạ huyết áp nhóm CCB. Liều dùng: 5mg x 1 lần/ngày", "category": "drug_groups", "group": "hypertension", "usage": "take_morning"},
                {"text": "Cetirizine - Thuốc kháng histamin H1. Liều dùng: 10mg x 1 lần/ngày", "category": "drug_groups", "group": "allergy", "usage": "take_evening"},
                {"text": "Ibuprofen - Thuốc chống viêm không steroid. Liều dùng: 400mg x 3 lần/ngày", "category": "drug_groups", "group": "anti_inflammatory", "usage": "take_after_meals"},
                {"text": "Simvastatin - Thuốc hạ cholesterol nhóm statin. Liều dùng: 20mg x 1 lần/ngày", "category": "drug_groups", "group": "blood_lipid", "usage": "take_evening"}
            ]
           
            # Mock data for lab results
            lab_results_data = [
                {"text": "Glucose máu đói: 126 mg/dL (bình thường: 70-100). Chỉ số cao, nghi ngờ tiểu đường", "category": "lab_results", "test_type": "blood_chemistry", "status": "high"},
                {"text": "Cholesterol toàn phần: 240 mg/dL (bình thường: <200). Nguy cơ tim mạch", "category": "lab_results", "test_type": "blood_lipid", "status": "high"},
                {"text": "Hemoglobin: 9.5 g/dL (bình thường: 12-15). Thiếu máu nhẹ", "category": "lab_results", "test_type": "complete_blood_count", "status": "low"},
                {"text": "ALT: 65 U/L (bình thường: <40). Chức năng gan bất thường", "category": "lab_results", "test_type": "liver_function", "status": "high"},
                {"text": "Creatinine: 1.8 mg/dL (bình thường: 0.6-1.2). Chức năng thận giảm", "category": "lab_results", "test_type": "kidney_function", "status": "high"},
                {"text": "TSH: 8.5 mIU/L (bình thường: 0.4-4.0). Suy giáp", "category": "lab_results", "test_type": "thyroid_hormone", "status": "high"},
                {"text": "HbA1c: 8.2% (bình thường: <5.7%). Kiểm soát đường huyết kém", "category": "lab_results", "test_type": "diabetes", "status": "high"},
                {"text": "CRP: 15 mg/L (bình thường: <3). Viêm nhiễm cấp tính", "category": "lab_results", "test_type": "inflammation", "status": "high"}
            ]
           
            # Combine all data
            all_data = symptoms_data + drug_groups_data + lab_results_data
           
            # Prepare vectors for upsert
            vectors = []
            batch_size = 50  # Pinecone batch size limit
           
            for i, item in enumerate(all_data):
                vector_id = str(uuid.uuid4())
                embedding = self.get_embedding(item["text"])
               
                # Prepare metadata
                metadata = {
                    "text": item["text"],
                    "category": item["category"],
                    "source": "mock_data"
                }
               
                # Add category-specific metadata
                if item["category"] == "symptoms":
                    metadata.update({
                        "severity": item.get("severity", "unknown"),
                        "type": item.get("type", "unknown")
                    })
                elif item["category"] == "drug_groups":
                    metadata.update({
                        "group": item.get("group", "unknown"),
                        "usage": item.get("usage", "unknown")
                    })
                elif item["category"] == "lab_results":
                    metadata.update({
                        "test_type": item.get("test_type", "unknown"),
                        "status": item.get("status", "unknown")
                    })
               
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
               
                # Upsert in batches
                if len(vectors) >= batch_size or i == len(all_data) - 1:
                    self.index.upsert(vectors=vectors)
                    vectors = []
           
            print(f"Inserted {len(all_data)} mock data items into Pinecone")
           
        except Exception as e:
            print(f"Error inserting mock data: {e}")
   
    def search_symptoms(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar symptoms"""
        return self._search_by_category(query, "symptoms", n_results)
   
    def search_drug_groups(self, query: str, n_results: int = 5) -> Dict:
        """Search for drug information"""
        return self._search_by_category(query, "drug_groups", n_results)
   
    def search_lab_results(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar lab results"""
        return self._search_by_category(query, "lab_results", n_results)
   
    def _search_by_category(self, query: str, category: str, n_results: int = 5) -> Dict:
        """Internal method to search by category"""
        if not self.index:
            # Return mock results for development
            return {
                "documents": [["Mock result for development - Pinecone not configured"]],
                "metadatas": [[{"category": category, "source": "mock"}]],
                "distances": [[0.5]]
            }
           
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
           
            # Search with category filter
            results = self.index.query(
                vector=query_embedding,
                top_k=n_results,
                filter={"category": {"$eq": category}},
                include_metadata=True
            )
           
            # Convert to ChromaDB-like format for compatibility
            documents = []
            metadatas = []
            distances = []
           
            for match in results.matches:
                documents.append(match.metadata.get("text", ""))
                metadatas.append(match.metadata)
                distances.append(match.score)
           
            return {
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances]
            }
           
        except Exception as e:
            print(f"Error searching Pinecone: {e}")
            # Return empty results
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
   
    def convert_txt_to_markdown(self, txt_content: str, filename: str) -> str:
        """Convert plain text to structured markdown format for better processing"""
        try:
            # Clean and normalize the text
            content = txt_content.strip()
           
            # Split content into lines
            lines = content.split('\n')
           
            # Process and structure the content
            markdown_lines = []
            markdown_lines.append(f"# Medical Document: {filename}")
            markdown_lines.append("")
           
            current_section = ""
            current_content = []
           
            for line in lines:
                line = line.strip()
                if not line:
                    continue
               
                # Detect potential headers/sections
                if self._is_section_header(line):
                    # Save previous section
                    if current_section and current_content:
                        markdown_lines.append(f"## {current_section}")
                        markdown_lines.append("")
                        for content_line in current_content:
                            markdown_lines.append(f"- {content_line}")
                        markdown_lines.append("")
                   
                    # Start new section
                    current_section = line
                    current_content = []
                else:
                    # Regular content
                    current_content.append(line)
           
            # Add the last section
            if current_section and current_content:
                markdown_lines.append(f"## {current_section}")
                markdown_lines.append("")
                for content_line in current_content:
                    markdown_lines.append(f"- {content_line}")
            elif current_content:
                # No sections detected, treat as general content
                markdown_lines.append("## Nội dung chính")
                markdown_lines.append("")
                for content_line in current_content:
                    markdown_lines.append(f"- {content_line}")
           
            # Join all lines
            markdown_content = '\n'.join(markdown_lines)
           
            print(f"📝 Converted TXT to Markdown: {len(txt_content)} chars → {len(markdown_content)} chars")
            return markdown_content
           
        except Exception as e:
            print(f"⚠️ Error converting TXT to Markdown: {e}")
            # Return original content with basic markdown structure
            return f"# Medical Document: {filename}\n\n## Nội dung\n\n{txt_content}"
   
    def _is_section_header(self, line: str) -> bool:
        """Detect if a line is likely a section header"""
        line_lower = line.lower()
       
        # Medical section keywords
        section_keywords = [
            'triệu chứng', 'symptoms', 'dấu hiệu',
            'thuốc', 'medication', 'drug', 'điều trị',
            'xét nghiệm', 'test', 'lab', 'kết quả',
            'chẩn đoán', 'diagnosis', 'bệnh',
            'tiền sử', 'history', 'khám',
            'tóm tắt', 'summary', 'kết luận',
            'hướng dẫn', 'instruction', 'lưu ý',
            'tác dụng phụ', 'side effect', 'chống chỉ định'
        ]
       
        # Check for keywords
        for keyword in section_keywords:
            if keyword in line_lower:
                return True
       
        # Check for patterns like "1.", "A.", "I.", etc.
        import re
        if re.match(r'^[0-9]+[\.\)]', line) or re.match(r'^[A-Z][\.\)]', line):
            return True
       
        # Check if line is short and might be a header
        if len(line) < 50 and line.endswith(':'):
            return True
       
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
            classification_prompt = f"""
Classify the following medical text into one of these categories:
- symptoms: Text about symptoms, signs, medical conditions, patient complaints
- drug_groups: Text about medications, drugs, prescriptions, dosages, drug information
- lab_results: Text about test results, lab values, medical examinations, diagnostic results
 
Text: "{text_chunk}"
 
Return only the category name (symptoms/drug_groups/lab_results).
"""
           
            response = self.openai_client.chat.completions.create(
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
        if not self.index:
            print("❌ Index not available - check your Pinecone API key configuration")
            return {"symptoms": 0, "drug_groups": 0, "lab_results": 0, "error": "No Pinecone connection"}
       
        print(f"📄 Processing file: {source_filename}")
       
        # Step 1: Convert TXT to Markdown (background processing)
        if source_filename.lower().endswith('.txt'):
            print("🔄 Converting TXT to Markdown format...")
            processed_content = self.convert_txt_to_markdown(file_content, source_filename)
        else:
            processed_content = file_content
       
        # Step 2: Split text into chunks
        chunks = self.split_text_into_chunks(processed_content)
       
        if not chunks:
            print("❌ No valid text chunks found in the uploaded content")
            return {"symptoms": 0, "drug_groups": 0, "lab_results": 0, "error": "No content to process"}
       
        print(f"📋 Created {len(chunks)} chunks for processing")
       
        # Track additions
        additions = {"symptoms": 0, "drug_groups": 0, "lab_results": 0}
       
        vectors = []
        batch_size = 50
       
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:  # Skip very short chunks
                continue
               
            # Classify chunk
            category = self.classify_text_chunk(chunk)
            print(f"📋 Chunk {i+1}/{len(chunks)}: '{chunk[:50]}...' → {category}")
           
            # Generate embedding
            embedding = self.get_embedding(chunk)
            if not embedding or all(x == 0.0 for x in embedding):
                print(f"⚠️ Failed to generate embedding for chunk {i+1}")
                continue
           
            # Generate metadata
            metadata = {
                "text": chunk,
                "category": category,
                "source": source_filename,
                "chunk_index": i,
                "upload_type": "user_uploaded",
                "processed_format": "markdown" if source_filename.lower().endswith('.txt') else "original"
            }
           
            # Add category-specific metadata
            if category == "symptoms":
                metadata.update({"severity": "unknown", "type": "user_uploaded"})
                additions["symptoms"] += 1
            elif category == "drug_groups":
                metadata.update({"group": "user_uploaded", "usage": "unknown"})
                additions["drug_groups"] += 1
            else:  # lab_results
                metadata.update({"test_type": "user_uploaded", "status": "unknown"})
                additions["lab_results"] += 1
           
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": metadata
            })
           
            # Upsert in batches
            if len(vectors) >= batch_size:
                try:
                    result = self.index.upsert(vectors=vectors)
                    print(f"✅ Uploaded batch of {len(vectors)} vectors")
                    vectors = []
                except Exception as e:
                    print(f"❌ Error upserting batch: {e}")
                    return {"symptoms": 0, "drug_groups": 0, "lab_results": 0, "error": str(e)}
       
        # Upsert remaining vectors
        if vectors:
            try:
                result = self.index.upsert(vectors=vectors)
                print(f"✅ Uploaded final batch of {len(vectors)} vectors")
            except Exception as e:
                print(f"❌ Error upserting final batch: {e}")
                return {"symptoms": 0, "drug_groups": 0, "lab_results": 0, "error": str(e)}
       
        print(f"🎉 Successfully processed {sum(additions.values())} chunks: {additions}")
        return additions
   
    def process_uploaded_file(self, file_path: str) -> Dict[str, int]:
        """Process uploaded file and add content to Pinecone"""
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
        if not self.index:
            return {"symptoms": 0, "drug_groups": 0, "lab_results": 0}
           
        try:
            stats = self.index.describe_index_stats()
           
            # Count by category (this is an approximation since Pinecone doesn't provide exact counts by metadata)
            # You might want to implement a more precise counting mechanism
            total_count = stats.total_vector_count
           
            # For now, return equal distribution as approximation
            # In a real implementation, you might want to store these counts separately
            avg_count = total_count // 3 if total_count > 0 else 0
           
            return {
                "symptoms": avg_count,
                "drug_groups": avg_count,
                "lab_results": avg_count
            }
           
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"symptoms": 0, "drug_groups": 0, "lab_results": 0}
   
    def add_to_specific_collection(self, content: str, filename: str, collection_type: str) -> Dict[str, int]:
        """Add content directly to specific collection without AI classification"""
        if not self.index:
            print("Index not available")
            return {"symptoms": 0, "drug_groups": 0, "lab_results": 0}
           
        chunks = self.split_text_into_chunks(content)
        additions = {"symptoms": 0, "drug_groups": 0, "lab_results": 0}
       
        vectors = []
        batch_size = 50
       
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:
                continue
               
            # Generate embedding
            embedding = self.get_embedding(chunk)
           
            metadata = {
                "text": chunk,
                "category": collection_type,
                "source": filename,
                "chunk_index": i,
                "upload_type": "manual_upload"
            }
           
            # Add category-specific metadata
            if collection_type == "symptoms":
                metadata.update({"severity": "unknown", "type": "manual_upload"})
                additions["symptoms"] += 1
            elif collection_type == "drug_groups":
                metadata.update({"group": "manual_upload", "usage": "unknown"})
                additions["drug_groups"] += 1
            elif collection_type == "lab_results":
                metadata.update({"test_type": "manual_upload", "status": "unknown"})
                additions["lab_results"] += 1
           
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": metadata
            })
           
            # Upsert in batches
            if len(vectors) >= batch_size:
                try:
                    self.index.upsert(vectors=vectors)
                    vectors = []
                except Exception as e:
                    print(f"Error upserting batch: {e}")
       
        # Upsert remaining vectors
        if vectors:
            try:
                self.index.upsert(vectors=vectors)
            except Exception as e:
                print(f"Error upserting final batch: {e}")
       
        return additions
 
# Test function
if __name__ == "__main__":
    db = MedicalPineconeDB()
   
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