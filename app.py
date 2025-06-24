
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
import time
from dotenv import load_dotenv

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']
client = Groq(api_key=groq_api_key)

if not groq_api_key:
    st.error("⚠️ Please set the GROQ_API_KEY environment variable.")
    st.stop()

# # Example usage in a web request (if needed)
# headers = {"User-Agent": user_agent}

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load Groq API Key
from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']
client = Groq(api_key=groq_api_key)

if not groq_api_key:
    st.error("⚠️ Please set the GROQ_API_KEY environment variable.")
    st.stop()


# Sidebar with Introduction and Instructions
st.sidebar.title("✨ Welcome to WebGenie ✨")
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <h1>🤖 Meet WebGenie!</h1>
        <p>Your smart assistant to explore and understand website content.</p>
        <hr>
        <h3>🚀 How to Use?</h3>
        <ol style="text-align: left;">
            <li>🔗 <b>Enter</b> a website URL in the input box.</li>
            <li>⏳ <b>Wait</b> while WebGenie processes the webpage.</li>
            <li>📜 <b>Read</b> the generated summary.</li>
            <li>❓ <b>Ask</b> questions to dive deeper into the content.</li>
        </ol>
        <hr>
        <p>⚡ <b>Unlock knowledge from any website instantly!</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Custom CSS to center main content
st.markdown("""
    <style>
        .main-content {
            max-width: 800px;
            margin: auto;
        }
        .user-message {
            text-align: left;
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
            margin-right: 30px;
        }
        .ai-message {
            text-align: left;
            background-color: #EAEAEA;
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
            margin-right: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

## Centered Content
st.markdown("<div class='main-content'>", unsafe_allow_html=True)
st.markdown("""
    <h1 style='text-align: center; color: #1E90FF; font-size: 36px;'>🌍 WebGenie</h1>
    <h2 style='text-align: center; color: #FF5733; font-weight: bold;'>Your AI-Powered Website Assistant</h2>
    <p style='text-align: center; font-size: 18px; color: #555;'>Analyze, Summarize, and Interact with Website Content Seamlessly!</p>
    <hr>
""", unsafe_allow_html=True)

website = st.text_input("Enter a website URL:")
if website:
    if "vectors" not in st.session_state:
        with st.spinner("Processing website..."):
            try:
                st.session_state.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
                st.session_state.loader = WebBaseLoader(website)
                st.session_state.docs = st.session_state.loader.load()
                
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

            except Exception as e:
                st.error(f"❌ Error processing website: {e}")
                st.stop()
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    
    # Summarizing the website
    query = "Summarize the main points of this webpage."
    docs = st.session_state.vectors.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])
    summary_prompt = ChatPromptTemplate.from_template(
        "Given the following webpage content:\n\n{context}\n\nGenerate a concise summary."
    )
    
    chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = chain.run(context=context)
    st.markdown(f"<div class='ai-message'>🤖 <strong>WebGenie:</strong> {summary}</div>", unsafe_allow_html=True)

    # Q&A for website
    prompt = ChatPromptTemplate.from_template(
        """
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_query = st.chat_input("Ask anything about this website...")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.markdown(f"<div class='user-message'>🥷 <strong>You:</strong> {user_query}</div>", unsafe_allow_html=True)

        response = retrieval_chain.invoke({"input": user_query})
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        st.markdown(f"<div class='ai-message'>🤖 <strong>WebGenie:</strong> {response['answer']}</div>", unsafe_allow_html=True)
        
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
