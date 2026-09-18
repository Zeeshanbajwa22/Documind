import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load API key from .env
load_dotenv()
api_key = os.getenv("GROK_API_KEY")
print(f"API key loaded: {api_key}")

# Set up Grok client (OpenAI-compatible endpoint)
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Load the same embedding model and vector store as before
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Ask a question
query = "what are the hardware requirement of this project?"

# Step A: Retrieve relevant chunks (same as Step 6)
results = vectorstore.similarity_search(query, k=3)

# Step B: Combine the retrieved chunks into one context block
context = "\n\n".join([doc.page_content for doc in results])

# Step C: Build the prompt
prompt = f"""You are a helpful assistant answering questions about a research document.
Use ONLY the context below to answer the question. If the context doesn't contain 
the answer, say "I don't have enough information to answer that" — do not make up information.

Context:
{context}

Question: {query}

Answer:"""

# Step D: Send it to Grok
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Step E: Print the answer
print(f"Question: {query}\n")
print(f"Answer: {response.choices[0].message.content}")