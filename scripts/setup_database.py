#!/usr/bin/env python3
"""
Setup script for Ask Manim RAG System
Generates the embeddings database with the proper directory structure.
"""

import os
import sys

# Add the parent directory to Python path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding_models import OpenAIEmbeddingModel
from src.embedding_db import VectorDB

def main():
    print("🎬 Ask Manim RAG System - Database Setup")
    print("=" * 50)
    
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"✅ Created {data_dir} directory")
    
    # Ensure documents directory exists
    documents_dir = "documents"
    if not os.path.exists(documents_dir):
        print(f"❌ Error: {documents_dir} directory not found!")
        print("Please run the documentation scraper first:")
        print("python src/scrape_manim_docs.py")
        sys.exit(1)
    
    # Check if documents directory has content
    doc_files = [f for f in os.listdir(documents_dir) if f.endswith('.txt')]
    if not doc_files:
        print(f"❌ Error: No .txt files found in {documents_dir}!")
        print("Please run the documentation scraper first:")
        print("python src/scrape_manim_docs.py")
        sys.exit(1)
    
    print(f"📁 Found {len(doc_files)} documentation files")
    
    # Initialize embedding model
    print("🔄 Initializing OpenAI embedding model...")
    openai_embedding_model = OpenAIEmbeddingModel()
    
    # Name of the file that stores the embeddings in memory
    embedding_database_file = os.path.join(data_dir, "database.npy")
    
    # Create vector database
    print("🔄 Creating vector database...")
    vector_db = VectorDB(
        directory=documents_dir, 
        vector_file=embedding_database_file, 
        max_words_per_chunk=1000,  # Reduced from 4000 to avoid token limit issues
        embedding_model=openai_embedding_model
    )
    
    print("✅ Database setup complete!")
    print(f"📊 Generated embeddings saved to: {embedding_database_file}")
    print(f"📝 Text chunks saved to: {os.path.join(data_dir, 'database_chunks.pkl')}")
    print("\n🚀 You can now run the applications:")
    print("   • CLI: python ask_manim_cli.py")
    print("   • Web: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 