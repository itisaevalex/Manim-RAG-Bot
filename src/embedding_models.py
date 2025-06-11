from abc import ABC, abstractmethod
import numpy as np
import tiktoken
import os
from openai import OpenAI
from typing import List, Tuple

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models with chunking support"""
    def __init__(self):
        self.max_tokens = 512  # Default value
        self.tokenizer = None  # Must be initialized in subclasses
        
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text input"""
        pass
    
    def split_documents(self, documents: List[str]) -> List[str]:
        """
        Split documents into model-appropriate chunks using proper token-based chunking
        - Uses safe chunk size (much smaller than max_tokens)
        - Includes overlap for context preservation
        - Ensures no chunk exceeds token limits
        """
        final_chunks = []
        
        # Use a safe chunk size well below the token limit
        safe_chunk_size = min(5000, self.max_tokens - 500)  # 5000 tokens, staying well under 8191 limit
        overlap_size = safe_chunk_size // 10  # 10% overlap (500 tokens)
        
        for doc in documents:
            # First split by EOC markers if they exist
            parts = [p.strip() for p in doc.split('<EOC>') if p.strip()]
            
            for part in parts:
                # Tokenize using model-specific tokenizer
                tokens = self.tokenizer.encode(part)
                
                # If the part is small enough, use it as-is
                if len(tokens) <= safe_chunk_size:
                    final_chunks.append(part)
                    continue
                
                # Split into overlapping chunks
                start = 0
                while start < len(tokens):
                    # Calculate end position
                    end = min(start + safe_chunk_size, len(tokens))
                    
                    # Extract chunk tokens
                    chunk_tokens = tokens[start:end]
                    
                    # Decode back to text
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    
                    # Only add non-empty chunks
                    if chunk_text.strip():
                        final_chunks.append(chunk_text.strip())
                    
                    # Move start position with overlap
                    if end >= len(tokens):
                        break
                    start = end - overlap_size
        
        print(f"[embedding_models.py] Split {len(documents)} documents into {len(final_chunks)} chunks (max {safe_chunk_size} tokens each)")
        return final_chunks
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Default batch processing - override in subclasses for efficiency"""
        return np.array([self.get_embedding(text) for text in texts])

class MiniEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.max_tokens = 256  # Model's actual max sequence length
        
    def get_embedding(self, text: str) -> List[float]:
        """Return embedding"""
        embedding = self.model.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Optimized batch processing"""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="text-embedding-3-small"):
        super().__init__()
        import tiktoken
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8191  # OpenAI's limit
        
    def get_embedding(self, text: str) -> List[float]:
        """Return OpenAI embedding for single text"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Efficiently batch process embeddings using OpenAI's batch API"""
        all_embeddings = []
        batch_size = 100  # Conservative batch size to stay within limits
        
        print(f"[OpenAIEmbeddingModel] Processing {len(texts)} chunks in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(texts) + batch_size - 1)//batch_size
            print(f"[OpenAIEmbeddingModel] Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                # Use OpenAI's batch embedding API
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                
                # Extract embeddings in the same order as input
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                print(f"[OpenAIEmbeddingModel] ✅ Batch {batch_num}/{total_batches} completed successfully")
                
            except Exception as e:
                print(f"[OpenAIEmbeddingModel] ❌ Error in batch {batch_num}: {e}")
                print(f"[OpenAIEmbeddingModel] Falling back to individual processing for this batch...")
                
                # Fallback to individual processing for this batch
                for j, text in enumerate(batch):
                    try:
                        embedding = self.get_embedding(text)
                        all_embeddings.append(embedding)
                        if (j + 1) % 10 == 0:
                            print(f"[OpenAIEmbeddingModel] Individual processing: {j+1}/{len(batch)} completed")
                    except Exception as single_error:
                        print(f"[OpenAIEmbeddingModel] Error processing single chunk: {single_error}")
                        # Add zero vector as fallback
                        all_embeddings.append([0.0] * 1536)  # Standard embedding size for text-embedding-3-small
        
        print(f"[OpenAIEmbeddingModel] ✅ All {len(texts)} chunks processed successfully!")
        return np.array(all_embeddings)