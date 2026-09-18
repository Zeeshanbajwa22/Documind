import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load API key
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
api_key = os.getenv("GROQ_API_KEY")


# Set up Groq client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Cache the embedding model and vector store so they only load once,
# not on every single question
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    return vectorstore

vectorstore = load_vectorstore()

# Page setup
st.set_page_config(page_title="DocuMind", page_icon="📄")
st.title("📄 DocuMind — Ask Your Document")
st.write("Ask questions about your uploaded document, powered by RAG.")

# Keep chat history across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box
user_question = st.chat_input("Ask a question about the document...")

if user_question:
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(user_question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Build the prompt
    prompt = f"""You are a helpful assistant answering questions about a research document.
Use ONLY the context below to answer the question. If the context doesn't contain 
the answer, say "I don't have enough information to answer that" — do not make up information.

Context:
{context}

Question: {user_question}

Answer:"""

    # Get answer from Groq
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})