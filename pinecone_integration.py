import uuid
import re
from typing import List, Dict
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time
import streamlit as st
 
load_dotenv()
 
class MedicalPineconeDB:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=st.secrets["PINECONE_API_KEY"]
        )
       
        # Initialize OpenAI for embeddings (separate from main client)
        self.embedding_client = openai.OpenAI(
            base_url=st.secrets["OPENAI_ENDPOINT"],
            api_key=st.secrets["OPENAI_EMBEDDING_KEY"],
        )
       
        # Initialize OpenAI for classification (main client)
        self.openai_client = openai.OpenAI(
            base_url=st.secrets["OPENAI_ENDPOINT"],
            api_key=st.secrets["OPENAI_API_KEY"],
        )
       
        # Index configuration
        self.index_name = "medical-rag-index"
        self.embedding_dimension = 1536  # text-embedding-3-small dimension
       
        # Try to get actual dimension from a test embedding
        try:
            # Try to get actual dimension from a test embedding
            test_response = self.embedding_client.embeddings.create(
                model=st.secrets["OPENAI_EMBEDDING_MODEL"],
                input="test"
            )
            actual_dimension = len(test_response.data[0].embedding)
            if actual_dimension != self.embedding_dimension:
                print(f"Adjusting embedding dimension from {self.embedding_dimension} to {actual_dimension}")
                self.embedding_dimension = actual_dimension
        except Exception as e:
            print(f"Could not determine embedding dimension, using default {self.embedding_dimension}: {e}")
       
        self.setup_index()
   
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
                model=st.secrets["OPENAI_EMBEDDING_MODEL"],
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
   
    def search_symptoms(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar symptoms"""
        return self._search_by_category(query, "symptoms", n_results)
   
    def search_drug_groups(self, query: str, n_results: int = 5) -> Dict:
        """Search for drug information"""
        return self._search_by_category(query, "drug_groups", n_results)

    def _search_by_category(self, query: str, category: str, n_results: int = 5, min_score: float = 0.5) -> Dict:
        """Internal method to search by category with similarity threshold"""
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

            print("---results", results)

            # Filter by similarity score threshold
            filtered_matches = [
                match for match in results.matches
                if match.score >= min_score
            ]

            # Nếu không có match nào đạt ngưỡng -> trả về rỗng
            if not filtered_matches:
                return {
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]]
                }

            # Convert to ChromaDB-like format for compatibility
            documents = [match.metadata.get("text", "") for match in filtered_matches]
            metadatas = [match.metadata for match in filtered_matches]
            distances = [match.score for match in filtered_matches]

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
        """
        Split text into chunks by blank lines, and further split if a chunk is too long
        """
        raw_chunks = re.split(r'\n\s*\n', text.strip())
        clean_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

        final_chunks = []
        for chunk in clean_chunks:
            if len(chunk) > chunk_size:
                # Nếu chunk quá dài, cắt tiếp theo câu
                sentences = re.split(r'(?<=[.!?]) +', chunk)
                temp = ""
                for sentence in sentences:
                    if len(temp + sentence) <= chunk_size:
                        temp += sentence + " "
                    else:
                        final_chunks.append(temp.strip())
                        temp = sentence + " "
                if temp:
                    final_chunks.append(temp.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks

   
    def classify_text_chunk(self, text_chunk: str) -> str:
        """Classify text chunk into symptoms, drug_groups"""
        try:
            classification_prompt = f"""
Classify the following medical text into one of these categories:
- symptoms: Text about symptoms, signs, medical conditions, patient complaints
- drug_groups: Text about medications, drugs, prescriptions, dosages, drug information
 
Text: "{text_chunk}"
 
Return only the category name (symptoms/drug_groups).
"""
           
            response = self.openai_client.chat.completions.create(
                model="GPT-4o-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=50,
                temperature=0
            )
           
            category = response.choices[0].message.content.strip().lower()
            return category if category in ['symptoms', 'drug_groups'] else 'symptoms'
           
        except Exception as e:
            print(f"Classification error: {e}")
            return 'symptoms'  # Default fallback
   
    def add_file_content_to_db(self, file_content: str, source_filename: str = "user_upload") -> Dict[str, int]:
        """Process file content and add to appropriate collections"""
        if not self.index:
            print("❌ Index not available - check your Pinecone API key configuration")
            return {"symptoms": 0, "drug_groups": 0, "error": "No Pinecone connection"}
       
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
            return {"symptoms": 0, "drug_groups": 0, "error": "No content to process"}
       
        print(f"📋 Created {len(chunks)} chunks for processing")
       
        # Track additions
        additions = {"symptoms": 0, "drug_groups": 0}
       
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
                "upload_type": "admin_uploaded",
                "processed_format": "markdown" if source_filename.lower().endswith('.txt') else "original"
            }
           
            # Add category-specific metadata
            if category == "symptoms":
                additions["symptoms"] += 1
            elif category == "drug_groups":
                additions["drug_groups"] += 1
           
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
                    return {"symptoms": 0, "drug_groups": 0, "error": str(e)}
       
        # Upsert remaining vectors
        if vectors:
            try:
                result = self.index.upsert(vectors=vectors)
                print(f"✅ Uploaded final batch of {len(vectors)} vectors")
            except Exception as e:
                print(f"❌ Error upserting final batch: {e}")
                return {"symptoms": 0, "drug_groups": 0, "error": str(e)}
       
        print(f"🎉 Successfully processed {sum(additions.values())} chunks: {additions}")
        return additions
   
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics of all collections"""
        if not self.index:
            return {"symptoms": 0, "drug_groups": 0}
           
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
                "drug_groups": avg_count
            }
           
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"symptoms": 0, "drug_groups": 0}
   
    def add_to_specific_collection(self, content: str, filename: str, collection_type: str) -> Dict[str, int]:
        """Add content directly to specific collection without AI classification"""
        if not self.index:
            print("Index not available")
            return {"patient_prescriptions": 0, "drug_groups": 0, "patient_test_results": 0}
           
        chunks = self.split_text_into_chunks(content)
        additions = {"symptoms": 0, "drug_groups": 0}
       
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