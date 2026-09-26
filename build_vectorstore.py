import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

# Load environment variables (needed for HF_TOKEN)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Step 1: Load the PDF
loader = PyPDFLoader("sample.pdf")
pages = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
)
chunks = text_splitter.split_documents(pages)
print(f"Number of chunks created: {len(chunks)}")

# Filter out very short/low-content chunks (like bare chapter titles)
chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 100]
print(f"Number of chunks after filtering short ones: {len(chunks)}")

# Step 3: Load the embedding model (using HuggingFace's hosted API, not local)
print("Connecting to HuggingFace embedding API...")
embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

# Step 4: Create the vector database and store our chunks in it
print("Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Vector store created and saved successfully!")
print(f"Total chunks stored: {vectorstore._collection.count()}")