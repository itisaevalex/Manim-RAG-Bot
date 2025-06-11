import streamlit as st
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from src.embedding_models import OpenAIEmbeddingModel
from src.embedding_db import VectorDB
import time
import subprocess
import tempfile
import shutil
import ast

# Load environment variables
load_dotenv()

# Directory for user-rendered videos
USER_RENDERS_DIR = "user_renders"
os.makedirs(USER_RENDERS_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Ask Manim - AI Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    
    .message-avatar {
        font-size: 1.5rem;
        margin-right: 1rem;
        margin-top: 0.2rem;
    }
    
    .message-content {
        flex: 1;
    }
    
    .source-citation {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load the RAG system components (cached for performance)"""
    try:
        # Initialize embedding model
        embedding_model = OpenAIEmbeddingModel()
        
        # Load pre-computed embeddings and chunks
        embeddings_file = "data/database.npy"
        chunks_file = "data/database_chunks.pkl"
        
        if not os.path.exists(embeddings_file) or not os.path.exists(chunks_file):
            st.error("❌ Embeddings database not found! Please run `python embedding_db.py` first.")
            st.stop()
        
        # Load embeddings and chunks
        embeddings = np.load(embeddings_file)
        
        import pickle
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
        
        st.success(f"✅ Loaded {len(chunks)} chunks with {embeddings.shape[1]}-dimensional embeddings")
        
        return embedding_model, embeddings, chunks
        
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {str(e)}")
        st.stop()

def extract_scene_name_from_code(code: str) -> str:
    """Extracts the first Manim Scene class name from a code string."""
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and 'Scene' in base.id:
                        return node.name
    except Exception:
        return "" # Fallback
    return "" # Fallback

def run_manim_render_sync(scene_code: str, scene_name: str):
    """
    Synchronously renders a Manim scene from a string of code.
    Returns (success: bool, message: str, video_path: str or None)
    """
    if not scene_name:
        return False, "Error: Scene name is missing. Please specify which scene to render.", None
        
    manim_cwd = os.getcwd()

    with tempfile.TemporaryDirectory(prefix="manim_render_", dir=manim_cwd) as temp_run_dir:
        temp_script_filename = "temp_scene.py"
        temp_script_path = os.path.join(temp_run_dir, temp_script_filename)

        full_code_for_manim = f"from manim import *\n\n{scene_code}"

        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(full_code_for_manim)

        temp_media_dir = os.path.join(temp_run_dir, "media")
        os.makedirs(temp_media_dir, exist_ok=True)

        command = [
            "manim", temp_script_path, scene_name,
            "-qm",
            "--media_dir", temp_media_dir
        ]

        try:
            process = subprocess.run(
                command,
                cwd=manim_cwd,
                timeout=120,
                capture_output=True,
                text=True,
                check=False
            )

            if process.returncode == 0:
                script_name_no_ext = os.path.splitext(temp_script_filename)[0]
                expected_video_base_dir = os.path.join(temp_media_dir, "videos", script_name_no_ext)
                
                found_video_path = None
                if os.path.isdir(expected_video_base_dir):
                    quality_dirs = os.listdir(expected_video_base_dir)
                    if quality_dirs:
                        potential_video_file = os.path.join(expected_video_base_dir, quality_dirs[0], f"{scene_name}.mp4")
                        if os.path.exists(potential_video_file):
                            found_video_path = potential_video_file

                if found_video_path:
                    timestamp = int(time.time())
                    user_video_filename = f"{scene_name}_{timestamp}.mp4"
                    final_video_path = os.path.join(USER_RENDERS_DIR, user_video_filename)
                    
                    shutil.move(found_video_path, final_video_path)
                    
                    return True, f"Video ready: {user_video_filename}", final_video_path
                else:
                    return False, "Manim ran successfully, but the output video was not found.", None
            else:
                error_message = f"Manim render failed (code {process.returncode}).\n\n**Stderr:**\n```\n{process.stderr[-1000:]}\n```"
                return False, error_message, None

        except subprocess.TimeoutExpired:
            return False, "Manim rendering timed out (120 seconds). The scene may be too complex.", None
        except FileNotFoundError:
            return False, "Error: `manim` command not found. Is Manim installed and in your system's PATH?", None
        except Exception as e:
            return False, f"An error occurred during rendering: {str(e)}", None

def get_relevant_chunks(query: str, embedding_model, embeddings, chunks, k: int = 5):
    """Retrieve top-k most relevant chunks for the query"""
    try:
        # Generate query embedding
        query_embedding = embedding_model.get_embedding(query)
        query_embedding = np.array(query_embedding)
        
        # Calculate cosine similarities
        similarities = []
        for embedding in embeddings:
            similarity = VectorDB.cosine_similarity(query_embedding, embedding)
            similarities.append(similarity)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return top chunks with scores
        results = []
        for idx in top_k_indices:
            results.append({
                'chunk': chunks[idx],
                'score': similarities[idx],
                'index': idx
            })
        
        return results
        
    except Exception as e:
        st.error(f"❌ Error retrieving relevant chunks: {str(e)}")
        return []

def generate_response(query: str, relevant_chunks: list, model_name: str = "gpt-4.1-2025-04-14"):
    """Generate response using GPT model with retrieved context"""
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Prepare context from relevant chunks
        context = "\n\n".join([
            f"[Chunk {i+1}]:\n{chunk['chunk'][:1000]}..." if len(chunk['chunk']) > 1000 
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
7. When providing code examples, ALWAYS format them in a proper Python code block with correct indentation."""

        user_prompt = f"""Based on the following Manim documentation context, please answer the user's question:

CONTEXT:
{context}

USER QUESTION: {query}

Please provide a helpful and accurate answer based on the documentation provided above."""

        # Generate response
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.3,  # Lower temperature for more consistent answers
            stream=True  # Enable streaming for better UX
        )
        
        return response
        
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🎬 Ask Manim</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your AI assistant for Mathematical Animation Engine</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "Select AI Model:",
            ["gpt-4.1-2025-04-14", "gpt-4o", "gpt-4-turbo"],
            index=0,
            help="Choose the OpenAI model for generating responses"
        )
        
        # Number of chunks to retrieve
        k_chunks = st.slider(
            "Context Chunks:",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of relevant documentation chunks to use as context"
        )
        
        st.markdown("---")
        
        # System info
        st.header("📊 System Info")
        if st.button("🔄 Reload RAG System"):
            st.cache_resource.clear()
            st.rerun()
        
        # Example queries
        st.header("💡 Example Questions")
        example_queries = [
            "How do I create a simple circle animation?",
            "What's the difference between Scene and ThreeDScene?",
            "How to add text to my animations?",
            "How do I animate mathematical equations?",
            "What are the basic animation types in Manim?",
            "How do I change colors in Manim objects?"
        ]
        
        for query in example_queries:
            if st.button(f"📝 {query}", key=f"example_{hash(query)}"):
                st.session_state['pending_query'] = query
    
    # Load RAG system
    with st.spinner("🔄 Loading RAG system..."):
        embedding_model, embeddings, chunks = load_rag_system()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    st.header("💬 Chat with Manim AI")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-avatar">👤</div>
                <div class="message-content">
                    <strong>You:</strong><br>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <strong>Manim AI:</strong><br>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Check for new sources format
            if 'sources' in message and isinstance(message['sources'], list):
                with st.container():
                    st.markdown("**📚 Sources**")
                    for source in message['sources']:
                        relevance = source.get('relevance', 0)
                        index = source.get('index', '')
                        url = source.get('url', '')
                        is_link = source.get('is_link', False)
                        
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            if is_link:
                                st.markdown(f"**{index}.** <a href='{url}' target='_blank' style='text-decoration: none;'>{url}</a>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"**{index}.** {url}")
                            st.caption(f"Relevance: {relevance:.1f}%")
                        with col2:
                             with st.popover("Peek 🧐"):
                                st.text(source.get('text', '')[:1000])
                        st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)

            # Handle old sources format for backward compatibility
            elif 'sources' in message and isinstance(message['sources'], str):
                 st.markdown(message['sources'], unsafe_allow_html=True)
    
    # Handle pending query from sidebar buttons
    if 'pending_query' in st.session_state:
        pending_query = st.session_state['pending_query']
        del st.session_state['pending_query']
        
        # Add user message to chat immediately
        st.session_state.chat_history.append({'role': 'user', 'content': pending_query})
        
        # Process the query
        with st.spinner("🔍 Searching documentation..."):
            relevant_chunks = get_relevant_chunks(pending_query, embedding_model, embeddings, chunks, k=k_chunks)
        
        if relevant_chunks:
            with st.spinner("🤖 Generating response..."):
                response_stream = generate_response(pending_query, relevant_chunks, model_choice)
                
                if response_stream:
                    # Stream the response
                    full_response = ""
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                    
                    # Create sources section
                    sources_data = []
                    for i, chunk_data in enumerate(relevant_chunks):
                        chunk_text = chunk_data['chunk']
                        source_url = "Manim Documentation (local chunk)"
                        is_link = False
                        
                        lines = chunk_text.splitlines()
                        if lines and (lines[0].startswith('http://') or lines[0].startswith('https://')):
                            source_url = lines[0]
                            is_link = True
                        
                        sources_data.append({
                            'url': source_url,
                            'is_link': is_link,
                            'text': chunk_text,
                            'relevance': chunk_data['score'] * 100,
                            'index': i + 1
                        })
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant', 
                        'content': full_response,
                        'sources': sources_data
                    })
                    
                    st.rerun()
        else:
            st.error("❌ Could not retrieve relevant documentation. Please try rephrasing your question.")

    # User input
    user_input = st.text_input(
        "Ask me anything about Manim:",
        placeholder="e.g., How do I create an animation of a circle?",
        key="query_input"
    )
    
    col1, col2 = st.columns([4, 1])
    with col2:
        send_button = st.button("🚀 Send", type="primary")
    
    if send_button and user_input.strip():
        # Add user message to chat
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        with st.spinner("🔍 Searching documentation..."):
            # Retrieve relevant chunks
            relevant_chunks = get_relevant_chunks(user_input, embedding_model, embeddings, chunks, k=k_chunks)
        
        if relevant_chunks:
            with st.spinner("🤖 Generating response..."):
                # Generate response
                response_stream = generate_response(user_input, relevant_chunks, model_choice)
                
                if response_stream:
                    # Create placeholder for streaming response
                    response_placeholder = st.empty()
                    
                    # Stream the response
                    full_response = ""
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(f"""
                            <div class="chat-message assistant-message">
                                <div class="message-avatar">🤖</div>
                                <div class="message-content">
                                    <strong>Manim AI:</strong><br>
                                    {full_response}▊
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Create sources section
                    sources_data = []
                    for i, chunk_data in enumerate(relevant_chunks):
                        chunk_text = chunk_data['chunk']
                        source_url = "Manim Documentation (local chunk)"
                        is_link = False
                        
                        lines = chunk_text.splitlines()
                        if lines and (lines[0].startswith('http://') or lines[0].startswith('https://')):
                            source_url = lines[0]
                            is_link = True
                        
                        sources_data.append({
                            'url': source_url,
                            'is_link': is_link,
                            'text': chunk_text,
                            'relevance': chunk_data['score'] * 100,
                            'index': i + 1
                        })
                    
                    # Add complete response to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant', 
                        'content': full_response,
                        'sources': sources_data
                    })
                    
                    # Clear the response placeholder and refresh
                    response_placeholder.empty()
                    st.rerun()
        else:
            st.error("❌ Could not retrieve relevant documentation. Please try rephrasing your question.")
    
    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    
    with st.expander("🧪 Live Manim Editor", expanded=True):
        st.info("💡 Take advice from the chat and try it out here! The AI can help you debug your code.")
        
        if 'manim_code' not in st.session_state:
            st.session_state.manim_code = """class MyScene(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        
        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
        self.wait()
"""
        if 'render_output' not in st.session_state:
            st.session_state.render_output = None

        code = st.text_area(
            "Manim Scene Code",
            value=st.session_state.manim_code,
            height=300,
            key="manim_code_editor"
        )
        st.session_state.manim_code = code

        detected_scene_name = extract_scene_name_from_code(code)

        col1, col2 = st.columns([1, 3])
        with col1:
            scene_name = st.text_input("Scene Name", value=detected_scene_name, help="The name of the class to render.")
            run_button = st.button("🚀 Render Video", type="primary")

        if run_button:
            with st.spinner("⚙️ Rendering with Manim... This can take a minute."):
                st.session_state.render_output = run_manim_render_sync(code, scene_name)
            st.rerun()

        if st.session_state.render_output:
            success, message, video_path = st.session_state.render_output
            with col2:
                if success:
                    st.success("✅ " + message)
                    if video_path and os.path.exists(video_path):
                        with open(video_path, "rb") as f:
                            st.video(f.read())
                    else:
                        st.warning("Video file not found, but Manim reported success.")
                else:
                    st.error("❌ Render Failed")
                    st.markdown(message)

if __name__ == "__main__":
    main() 