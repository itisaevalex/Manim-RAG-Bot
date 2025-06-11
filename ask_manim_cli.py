#!/usr/bin/env python3
"""
Ask Manim CLI - Command Line Interface for Manim RAG System
A simple command-line version of the "Ask Manim" RAG application.
"""

import os
import numpy as np
import pickle
from openai import OpenAI
from dotenv import load_dotenv
from src.embedding_models import OpenAIEmbeddingModel
from src.embedding_db import VectorDB
import sys

# Load environment variables
load_dotenv()

class AskManimCLI:
    def __init__(self, model_name="gpt-4.1-2025-04-14"):
        """Initialize the Ask Manim CLI system"""
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = None
        self.embeddings = None
        self.chunks = None
        
        print("🎬 Ask Manim CLI - Your AI Assistant for Mathematical Animation Engine")
        print("=" * 70)
        
        self.load_rag_system()
    
    def load_rag_system(self):
        """Load the RAG system components"""
        try:
            print("🔄 Loading RAG system...")
            
            # Initialize embedding model
            self.embedding_model = OpenAIEmbeddingModel()
            
            # Load pre-computed embeddings and chunks
            embeddings_file = "data/database.npy"
            chunks_file = "data/database_chunks.pkl"
            
            if not os.path.exists(embeddings_file) or not os.path.exists(chunks_file):
                print("❌ Error: Embeddings database not found!")
                print("Please run `python embedding_db.py` first to generate the embeddings.")
                sys.exit(1)
            
            # Load embeddings and chunks
            print("📁 Loading embeddings...")
            self.embeddings = np.load(embeddings_file)
            
            print("📁 Loading text chunks...")
            with open(chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            
            print(f"✅ Successfully loaded {len(self.chunks)} chunks with {self.embeddings.shape[1]}-dimensional embeddings")
            print(f"🤖 Using model: {self.model_name}")
            print()
            
        except Exception as e:
            print(f"❌ Error loading RAG system: {str(e)}")
            sys.exit(1)
    
    def get_relevant_chunks(self, query: str, k: int = 5):
        """Retrieve top-k most relevant chunks for the query"""
        try:
            print(f"🔍 Searching for relevant information about: '{query}'")
            
            # Generate query embedding
            query_embedding = self.embedding_model.get_embedding(query)
            query_embedding = np.array(query_embedding)
            
            # Calculate cosine similarities
            similarities = []
            for embedding in self.embeddings:
                similarity = VectorDB.cosine_similarity(query_embedding, embedding)
                similarities.append(similarity)
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # Return top chunks with scores
            results = []
            for idx in top_k_indices:
                results.append({
                    'chunk': self.chunks[idx],
                    'score': similarities[idx],
                    'index': idx
                })
            
            print(f"📊 Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            print(f"❌ Error retrieving relevant chunks: {str(e)}")
            return []
    
    def generate_response(self, query: str, relevant_chunks: list):
        """Generate response using GPT model with retrieved context"""
        try:
            print(f"🤖 Generating response using {self.model_name}...")
            
            # Prepare context from relevant chunks
            context = "\n\n".join([
                f"[Chunk {i+1}]:\n{chunk['chunk'][:1500]}..." if len(chunk['chunk']) > 1500 
                else f"[Chunk {i+1}]:\n{chunk['chunk']}"
                for i, chunk in enumerate(relevant_chunks)
            ])
            
            # Construct prompt
            system_prompt = """You are an expert assistant specializing in Manim (Mathematical Animation Engine). 
Your task is to provide helpful, accurate, and practical answers about Manim based on the provided documentation.

Guidelines:
1. Base your answers primarily on the provided context chunks
2. If the context doesn't contain enough information, say so honestly
3. Provide code examples when relevant and possible
4. Be concise but comprehensive
5. Reference specific Manim classes, methods, and concepts when applicable
6. If asked about animations, explain both the concept and practical implementation
7. Format code blocks properly for readability"""

            user_prompt = f"""Based on the following Manim documentation context, please answer the user's question:

CONTEXT:
{context}

USER QUESTION: {query}

Please provide a helpful and accurate answer based on the documentation provided above."""

            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.3  # Lower temperature for more consistent answers
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return None
    
    def display_sources(self, relevant_chunks):
        """Display source information for the retrieved chunks"""
        print("\n📚 Sources used:")
        print("-" * 50)
        
        for i, chunk in enumerate(relevant_chunks):
            score_percentage = chunk['score'] * 100
            preview = chunk['chunk'][:150].replace('\n', ' ')
            print(f"{i+1}. Relevance: {score_percentage:.1f}%")
            print(f"   Preview: {preview}...")
            print()
    
    def ask_question(self, query: str, k_chunks: int = 5, show_sources: bool = True):
        """Main method to ask a question and get an answer"""
        print("=" * 70)
        print(f"❓ Question: {query}")
        print("=" * 70)
        
        # Retrieve relevant chunks
        relevant_chunks = self.get_relevant_chunks(query, k=k_chunks)
        
        if not relevant_chunks:
            print("❌ Could not retrieve relevant documentation. Please try rephrasing your question.")
            return
        
        # Generate response
        response = self.generate_response(query, relevant_chunks)
        
        if response:
            print("\n🤖 Answer:")
            print("-" * 30)
            print(response)
            
            if show_sources:
                self.display_sources(relevant_chunks)
        else:
            print("❌ Could not generate a response. Please try again.")
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("💬 Interactive Mode - Type 'quit' or 'exit' to stop")
        print("💡 Try asking: 'How do I create a circle animation?'")
        print()
        
        while True:
            try:
                query = input("🎬 Ask Manim: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Thanks for using Ask Manim CLI!")
                    break
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                self.ask_question(query)
                print("\n" + "="*70 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thanks for using Ask Manim CLI!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def demo_mode(self):
        """Run a demo with sample questions"""
        print("🎯 Demo Mode - Sample Questions")
        print()
        
        demo_questions = [
            "How do I create a simple circle animation?",
            "What's the difference between Scene and ThreeDScene?",
            "How do I add text to my animations?",
            "How do I animate mathematical equations?",
            "What are the basic animation types in Manim?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n🎯 Demo Question {i}/{len(demo_questions)}:")
            self.ask_question(question, show_sources=False)
            
            if i < len(demo_questions):
                input("\nPress Enter to continue to next question...")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ask Manim CLI - AI Assistant for Manim")
    parser.add_argument("--demo", action="store_true", help="Run demo mode with sample questions")
    parser.add_argument("--model", default="gpt-4.1-2025-04-14", help="OpenAI model to use")
    parser.add_argument("--question", "-q", help="Ask a single question and exit")
    parser.add_argument("--chunks", "-k", type=int, default=5, help="Number of context chunks to retrieve")
    
    args = parser.parse_args()
    
    try:
        # Initialize the CLI system
        cli = AskManimCLI(model_name=args.model)
        
        if args.demo:
            # Run demo mode
            cli.demo_mode()
        elif args.question:
            # Ask single question
            cli.ask_question(args.question, k_chunks=args.chunks)
        else:
            # Run interactive mode
            cli.interactive_mode()
            
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 