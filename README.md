# 🎬 Ask Manim - RAG System

An advanced Retrieval-Augmented Generation (RAG) system that provides AI-powered assistance for Manim (Mathematical Animation Engine) using OpenAI's latest GPT models and comprehensive documentation embedding.

## 🌟 Features

### Core Functionality
- **Comprehensive Documentation Coverage**: Complete scraping of Manim documentation (1,061 pages)
- **Advanced Chunking**: Token-based chunking with 5,000 tokens per chunk and 500-token overlap
- **Efficient Embedding**: Batch processing of embeddings using OpenAI's `text-embedding-3-small`
- **Semantic Search**: Cosine similarity-based retrieval of relevant documentation
- **State-of-the-Art AI**: Powered by GPT-4.1-2025-04-14 (latest OpenAI model)

### User Interfaces
- **Command Line Interface (CLI)**: Full-featured terminal-based interaction
- **Streamlit Web App**: Modern web interface (ready for deployment)
- **Interactive Modes**: Single question, demo mode, and continuous chat

### Performance Optimizations
- **Batched API Calls**: Process 100 chunks per API call (99.5% reduction in API calls)
- **Cached Embeddings**: Pre-computed embeddings for instant query processing
- **Smart Context Management**: Automatic truncation and relevance-based chunk selection

## 📁 Project Structure

```
RAG Portion/
├── README.md                  # This file
├── CHANGELOG.md              # Project history and updates
├── .env                      # OpenAI API key configuration
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore patterns
│
├── Main Applications:
├── streamlit_app.py          # Web interface (Streamlit) - Main App
├── ask_manim_cli.py          # Command-line interface
│
├── src/                      # Core RAG System Components
│   ├── __init__.py           # Package initialization
│   ├── embedding_models.py   # Embedding model implementations
│   ├── embedding_db.py       # Vector database and document processing
│   └── scrape_manim_docs.py  # Documentation scraper
│
├── tests/                    # Testing utilities
│   └── test_gpt4_model.py    # Model testing utility
│
├── data/                     # Generated data files
│   ├── database.npy          # Computed embeddings (1234 x 1536)
│   ├── database_chunks.pkl   # Text chunks from documentation
│   └── urls_to_scrape.txt    # URLs for documentation scraping
│
├── documents/                # Scraped documentation files (1,061 files)
├── files/                    # Screenshots and media assets
└── user_renders/             # User-generated Manim videos
```

## 🚀 Installation

### Prerequisites
- Python 3.11+
- OpenAI API key
- Conda environment (recommended)

### Setup Steps

1. **Clone and Navigate**
   ```bash
   cd "RAG Portion"
   ```

2. **Install Dependencies**
   ```bash
   conda install -c conda-forge tiktoken sentence-transformers pypdf2 numpy python-dotenv -y
   pip install streamlit  # For web interface
   ```

3. **Configure API Key**
   Create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Generate Embeddings** (if not already done)
   ```bash
   python src/embedding_db.py
   ```

## 💡 Usage

### Command Line Interface (Recommended)

**Single Question:**
```bash
python ask_manim_cli.py --question "How do I create a circle animation?"
```

**Interactive Mode:**
```bash
python ask_manim_cli.py
```

**Demo Mode:**
```bash
python ask_manim_cli.py --demo
```

**Advanced Options:**
```bash
python ask_manim_cli.py --question "Your question" --chunks 3 --model gpt-4o
```

### Web Interface (Streamlit)

```bash
streamlit run streamlit_app.py
```

### Python API

```python
from ask_manim_cli import AskManimCLI

# Initialize the system
cli = AskManimCLI(model_name="gpt-4.1-2025-04-14")

# Ask a question
cli.ask_question("How do I animate mathematical equations?")
```

## 🎯 Example Queries

Try asking these questions:

- **Basic Animations**: "How do I create a simple circle animation?"
- **Scene Types**: "What's the difference between Scene and ThreeDScene?"
- **Text Handling**: "How do I add and animate text in Manim?"
- **Mathematical Content**: "How do I animate mathematical equations with MathTex?"
- **Advanced Features**: "How do I create custom animations with updaters?"
- **3D Animations**: "How do I create 3D objects and rotate them?"

## 🔧 Technical Specifications

### Performance Metrics
- **Documentation Coverage**: 1,061 pages from docs.manim.community
- **Total Chunks**: 1,234 chunks (5,000 tokens each)
- **Embedding Dimensions**: 1,536 (OpenAI text-embedding-3-small)
- **Processing Time**: ~1 minute for full corpus (vs. 30+ minutes before optimization)
- **API Efficiency**: 13 batch calls vs. 1,234+ individual calls

### Architecture
- **Embedding Model**: OpenAI `text-embedding-3-small`
- **LLM**: GPT-4.1-2025-04-14 (configurable)
- **Chunking Strategy**: Token-based with overlap
- **Similarity Search**: Cosine similarity
- **Vector Storage**: NumPy arrays (.npy format)

### Optimization Features
- **Batch Processing**: 100 chunks per API call
- **Token Management**: Precise token counting with tiktoken
- **Error Handling**: Graceful fallbacks and retry logic
- **Caching**: Pre-computed embeddings for instant retrieval

## 🔍 How It Works

1. **Documentation Scraping**: Comprehensive crawling of Manim docs
2. **Intelligent Chunking**: Split documents into 5,000-token chunks with overlap
3. **Embedding Generation**: Batch process chunks into high-dimensional vectors
4. **Query Processing**: Convert user questions into embeddings
5. **Similarity Search**: Find most relevant documentation chunks
6. **Context Assembly**: Combine relevant chunks for LLM context
7. **Response Generation**: Generate accurate answers using GPT-4.1
8. **Source Citation**: Provide transparency with source references

## 🎨 Sample Output

```
🎬 Ask Manim CLI - Your AI Assistant for Mathematical Animation Engine
======================================================================
❓ Question: How do I create a simple circle animation?
======================================================================
🔍 Searching for relevant information...
📊 Found 5 relevant chunks
🤖 Generating response using gpt-4.1-2025-04-14...

🤖 Answer:
------------------------------
To create a simple circle animation in Manim, you need to define a Scene 
and add animation instructions for a Circle mobject. Here's a step-by-step guide:

1. Create a Python file (e.g., circle_animation.py):

from manim import *

class CircleAnimation(Scene):
    def construct(self):
        circle = Circle()                  # Create a circle
        self.play(Create(circle))          # Animate drawing the circle
        self.wait(1)                       # Pause for a second

2. Render the animation:
manim -pql circle_animation.py CircleAnimation

📚 Sources used:
- Manim Tutorials (Relevance: 49.5%)
- Quickstart Guide (Relevance: 45.2%)
```

## 📸 Screenshots

### Web Interface (Streamlit App)

#### Landing Page
![Landing Page](files/Landing%20window%20showing%20chunk%20number%20selection%20and%20example%20questions.png)
*The main interface showing chunk selection and example questions to get you started*

#### Code Editor
![Code Editor](files/Example%20Code%20Editor.png)
*Interactive code editor with syntax highlighting for Manim animations*

#### Code Execution
![Code Execution](files/Example%20Code%20executed%20and%20generated.png)
*Live code execution with video generation and preview capabilities*

#### AI Assistant Response
![AI Response](files/Sample%20AI%20output.png)
*Comprehensive AI responses with source citations and practical examples*

## 🤝 Contributing

### Updating Documentation
To refresh the documentation database:
```bash
python scrape_manim_docs.py  # Re-scrape docs
python embedding_db.py       # Regenerate embeddings
```

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## 📊 Performance Comparison

| Metric | Before Optimization | After Optimization | Improvement |
|--------|-------------------|------------------|-------------|
| Chunk Size | 1,000 words | 5,000 tokens | 5x larger, better context |
| Total Chunks | 2,692+ | 1,234 | 54% reduction |
| API Calls | 2,692+ individual | 13 batched | 99.5% reduction |
| Processing Time | 30-60 minutes | ~1 minute | 60x faster |
| Token Errors | Many warnings | Zero errors | ✅ Fixed |

## 🔧 Troubleshooting

### Common Issues

**"No module named streamlit"**
```bash
pip install streamlit
```

**"Embeddings database not found"**
```bash
python embedding_db.py
```

**"API key not found"**
- Ensure `.env` file exists with `OPENAI_API_KEY=your_key`

**"Rate limit exceeded"**
- The system uses conservative batching to avoid rate limits
- If issues persist, reduce batch size in `embedding_models.py`

## 📜 License

This project is built for educational and development purposes. Please ensure compliance with OpenAI's usage policies and Manim's documentation license.

## 🙏 Acknowledgments

- **Manim Community**: For the excellent documentation
- **OpenAI**: For powerful embedding and language models
- **Python Community**: For amazing libraries (numpy, tiktoken, etc.)

---

*Built with ❤️ for the Manim community* 