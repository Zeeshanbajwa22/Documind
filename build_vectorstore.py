from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Step 1: Load the PDF (same as before)
loader = PyPDFLoader("sample.pdf")
pages = loader.load()

# Step 2: Split into chunks (same as before)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
)
chunks = text_splitter.split_documents(pages)
print(f"Number of chunks created: {len(chunks)}")

chunks = text_splitter.split_documents(pages)
print(f"Number of chunks created: {len(chunks)}")

# ↓↓↓ ADD THESE TWO NEW LINES RIGHT HERE ↓↓↓
chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 100]
print(f"Number of chunks after filtering short ones: {len(chunks)}")
# ↑↑↑ NEW LINES END HERE ↑↑↑

# Step 3: Load the embedding model
print("Loading embedding model... (first time takes a bit longer)")

# Step 3: Load the embedding model
print("Loading embedding model... (first time takes a bit longer)")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Step 4: Create the vector database and store our chunks in it
print("Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Vector store created and saved successfully!")
print(f"Total chunks stored: {vectorstore._collection.count()}")