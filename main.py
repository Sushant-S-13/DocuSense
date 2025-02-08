import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. 

    1. If the user greets you (e.g., "Hi", "Hello"), respond warmly and conversationally. 
    2. If the user asks you to summarize something, provide a summary based on the provided context. 
    3. If the user asks a question, answer it based on the context. If the answer is not in the context, say, "The answer is not available in the provided context."
    4. If the user input doesn't match any of the above, try your best to help, or let the user know how you can assist.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="DocuSense", page_icon="📄", layout="wide")

    # Customizing Header
    st.markdown(
        """
        <style>
            .header-container {
                background: linear-gradient(to right, #FF0000, #000000); /* Red to White Gradient */
                border-radius: 15px;
                padding: 20px;
                text-align: left;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
                color: white;
                margin-bottom: 30px;
            }
            .header-title {
                
                color: white;
                font-size: 50px;
                font-weight: bold;
                
                margin-bottom: 0;
            }
            .subheader-title {
                font-size: 20px;
                font-weight: lighter;
                color: #718096; /* Neutral gray */
                margin-top: -10px;
            }
            .question-label {
                font-size: 18px;
                font-weight: bold;
                color: #3182CE; /* Blue accent for question input */
            }
            .text-input {
                background-color: #EDF2F7; /* Light gray background */
                color: #2D3748; /* Dark text */
                border: 1px solid #CBD5E0; /* Subtle border */
                font-size: 16px;
                padding: 10px;
            }
            .gradient-sidebar {
                background: linear-gradient(to bottom, #FF0000, #000000); /* Red to White Gradient */

                padding: 15px;
                border-radius: 10px;
                color: white;
            }
            .sidebar-title {
                font-size: 30px;
                font-weight: bold;
                text-align: left;
                color: white;
            }
            .tips {
                font-size: 16px;
                margin-top: 20px;
                font-weight: lighter;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Displaying Header and Subheader
    st.markdown(
        """
        <div class='header-container'>
            <div class='header-title'>DocuSense 💡</div>
            <div class='subheader-title'>Your Intelligent PDF Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chat Section
    st.markdown("<p class='question-label'>Ask a Question from the PDF Files:</p>", unsafe_allow_html=True)
    user_question = st.text_input("", placeholder="Type your question here...", key="question")

    if user_question:
        st.markdown(
            f"<p style='color: #3182CE; font-size: 18px; font-weight: bold;'>🤔 Processing your question...</p>",
            unsafe_allow_html=True,
        )
        user_input(user_question)

    # Sidebar with Gradient Background
    with st.sidebar:
        st.markdown(
            """
            <div class='gradient-sidebar'>
                <h2 class='sidebar-title'>📄 DocuSense</h2>
                <p>Navigate your PDFs with AI!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Menu Section
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("All set! Start asking your questions.")

        # Quick Tips
        st.markdown(
            """
            <div class='gradient-sidebar tips'>
                <h3>Quick Tips:</h3>
                <ul>
                    <li>📁 Upload multiple PDFs for combined insights.</li>
                    <li>💡 Ask specific, detailed questions for better results.</li>
                    <li>🚀 Explore the power of AI with DocuSense!</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        "<p style='text-align: center; font-size: 16px; color: #808080;'>Made by Sushant | DocuSense 2025</p>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()