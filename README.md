# 📄 DocuMind — AI Document Q&A Assistant

DocuMind is a Retrieval-Augmented Generation (RAG) application that lets you 
upload a PDF document and ask questions about it in natural language. It 
retrieves the most relevant sections of the document and generates accurate, 
grounded answers using an LLM — instead of relying on the model's general 
knowledge, which can hallucinate.

## Features

- 📄 Load and process PDF documents
- 🔍 Semantic search using vector embeddings (finds relevant content by meaning, not just keywords)
- 🤖 LLM-powered answers grounded in the actual document content
- 💬 Interactive chat interface built with Streamlit
- 🚫 Refuses to answer when the document doesn't contain the relevant information (reduces hallucination)

## Tech Stack

- **Python**
- **LangChain** — orchestrates the retrieval pipeline
- **ChromaDB** — vector database for storing document embeddings
- **HuggingFace Embeddings** (`BAAI/bge-base-en-v1.5`) — converts text into searchable vectors
- **Groq API** (`openai/gpt-oss-120b`) — fast LLM inference for answer generation
- **Streamlit** — web-based chat interface

## How It Works

1. **Document Loading** — PDF is loaded and split into pages
2. **Chunking** — pages are split into smaller, overlapping text chunks for better retrieval accuracy
3. **Embedding** — each chunk is converted into a vector representation capturing its meaning
4. **Storage** — vectors are stored in a local Chroma vector database
5. **Retrieval** — when a question is asked, the most semantically similar chunks are retrieved
6. **Generation** — retrieved context + the question are sent to an LLM, which generates a grounded answer

## Setup & Installation

1. Clone this repository
2. Create a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Create a `.env` file in the project root:
5. Add your PDF file as `sample.pdf` in the project folder
6. Build the vector database:
```bash
   python build_vectorstore.py
```
7. Run the chat interface:
```bash
   streamlit run app.py
```

## Project Structure


## Future Improvements

- Support for multiple document uploads
- Source citation (show which page an answer came from)
- Support for other file types (Word, TXT)
- Hybrid search (combining keyword + semantic search) for better retrieval on technical documents

## Author

Built by [Muhammad Zeeshan Bajwa] as a hands-on project to learn RAG (Retrieval-Augmented Generation) systems.