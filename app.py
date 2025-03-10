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

# Set API Key from Streamlit Secrets or .env
groq_api_key = st.secrets.get("groq_api_key", os.getenv("groq_api_key"))
user_agent = st.secrets.get("USER_AGENT", "Mozilla/5.0")  # Use default if not found

if not groq_api_key:
    st.error("⚠️ API Key not found! Please add it in Streamlit Secrets.")
    st.stop()

# Example usage in a web request (if needed)
headers = {"User-Agent": user_agent}

# Streamlit App UI
import streamlit as st

st.markdown(
    "<h1 style='text-align: center; color: #1E90FF;'>🌍 WebGenie</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center; color: #FF5733;'>|--Your Website Chatbot--|</h3>", 
    unsafe_allow_html=True
)

website = st.text_input("Enter a website URL:")

if website:
    if "vectors" not in st.session_state:  # Use "vectors" instead of "vector"
        with st.spinner("Processing website..."):
            try:
                st.session_state.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
                st.session_state.loader=WebBaseLoader(website)
                st.session_state.docs=st.session_state.loader.load()
            
                st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
                st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
                st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

            except Exception as e:
                st.error(f"❌ Error processing website: {e}")
                st.stop()                
    
    llm=ChatGroq(groq_api_key=groq_api_key,
                 model_name="mixtral-8x7b-32768")
    
# Custom CSS for chat styling
    st.markdown("""
        <style>
        .user-message {
            text-align: left;
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
            margin-right: 30px; /* Indent the user message */
        }
        .ai-message {
            text-align: left;
            background-color: #EAEAEA;
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
            margin-right: 30px; /* Indent the AI message */
        }
        .message-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
 ## Summarizing the website
    # Retrieve most relevant chunks
    query = "Summarize the main points of this webpage."
    docs =st.session_state.vectors.similarity_search(query, k=3)
    
    # Prepare prompt
    context = " ".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(
        "Given the following webpage content:\n\n{context}\n\nGenerate a concise summary."
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(context=context)
    st.markdown(f"<div class='ai-message'>🤖 <strong>WebGenie:</strong> {summary}</div>", unsafe_allow_html=True)

  ## Creating Q&A for website

    prompt=ChatPromptTemplate.from_template(
    """
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    At the end of the answer, ask if the user want to ask more question and you are happy to help!
    """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    prompt = st.chat_input("Ask anything about this website...")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='user-message'>🥷 <strong>You:</strong> {prompt}</div>", unsafe_allow_html=True)

        response = retrieval_chain.invoke({"input": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        st.markdown(f"<div class='ai-message'>🤖 <strong>WebGenie:</strong> {response['answer']}</div>", unsafe_allow_html=True)
        
        st.rerun()

    
   
