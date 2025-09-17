# --- pages/3_📄_Document_Q&A.py ---

import streamlit as st
import os
import tempfile

# LangChain Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai


# --- Add this "gate" to the top of each protected page ---

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is correct, otherwise prompt for the password.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Enter Password to Access", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("😕 Password incorrect")
    return False


if check_password():


    # ----------------------------------
    # RAG Document Q&A Section
    # ----------------------------------
    st.set_page_config(page_title="Document Q&A", layout="wide") # Added for this page
    st.title("📄 Chat with a Document (RAG)")
    st.markdown("---")

    # Configure API Key at the start of the script
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.warning("Google API Key not found in secrets. Please add it to continue.")
        st.stop()


    # --- NEW: Cached function to process the PDF ---
    @st.cache_resource
    def create_retriever_from_pdf(file_bytes):
        """
        This function takes the bytes of an uploaded PDF, processes it,
        and returns a LangChain retriever object. The @st.cache_resource
        decorator ensures this expensive operation is only run once per file.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # Note: The first time this runs, it will download the model.
            embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
            
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            
            return vectorstore.as_retriever()
        finally:
            # Clean up the temporary file
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)


    # --- Main App Logic ---
    uploaded_file = st.file_uploader("Upload a PDF document to ask questions about:", type="pdf")

    if uploaded_file:
        # Get the bytes of the file
        file_bytes = uploaded_file.getvalue()
        
        # Process the document and create the retriever using the cached function
        # The spinner will only show the first time a new document is processed.
        with st.spinner("Processing document... This may take a few minutes for new documents."):
            retriever = create_retriever_from_pdf(file_bytes)
        
        st.success(f"Document '{uploaded_file.name}' processed successfully!")

        # --- Chat interface ---
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
        
        template = """
        You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Be concise.

        Context: {context}

        Question: {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_query = st.text_input("Ask a question about the processed document:")

        if st.button("Get RAG Answer"):
            if rag_query:
                with st.spinner("Searching the document and generating an answer..."):
                    answer = rag_chain.invoke(rag_query)
                    st.markdown(answer)
            else:
                st.warning("Please enter a question.")