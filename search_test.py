from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the same embedding model we used before
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Reload the vector store we already built (no need to re-embed anything)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Ask a question
query = "What is the hardware requirement of this project?"

# Search for the most relevant chunks
results = vectorstore.similarity_search_with_score(query, k=3)

print(f"Query: {query}")
print(f"Found {len(results)} relevant chunks:\n")

for i, (doc, score) in enumerate(results):
    print(f"--- Result {i+1} (Page {doc.metadata.get('page_label')}) | Score: {score:.4f} ---")
    print(doc.page_content)
    print()  # k=3 means "give me top 3 matches"


