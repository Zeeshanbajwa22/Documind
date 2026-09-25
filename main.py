import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load API key
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
api_key = os.getenv("GROQ_API_KEY")

# Set up Groq client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Load embedding model + vector store once, when the API starts
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Create the FastAPI app
app = FastAPI(title="DocuMind API")

# Define what a request looks like (a question, as JSON)
class Question(BaseModel):
    query: str

# Define what a response looks like
class Answer(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "DocuMind API is running"}

@app.post("/ask", response_model=Answer)
def ask(question: Question):
    # Retrieve relevant chunks
    results = vectorstore.similarity_search(question.query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Build the prompt
    prompt = f"""You are a helpful assistant answering questions about a research document.
Use ONLY the context below to answer the question. If the context doesn't contain 
the answer, say "I don't have enough information to answer that" — do not make up information.

Context:
{context}

Question: {question.query}

Answer:"""

    # Call Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return Answer(answer=response.choices[0].message.content)