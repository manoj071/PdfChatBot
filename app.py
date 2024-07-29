# importing dependencies
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from langchain_groq import ChatGroq

# creating custom template
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:

"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# extracting text from pdf
def get_pdf_text(docs):
    text=""
    for pdf in docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# converting text to chunks
def get_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len)   
    chunks=text_splitter.split_text(raw_text)
    return chunks

# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunks):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    vectorstore=faiss.FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vectorstore

# generating conversation chain  
def get_conversationchain(vectorstore):
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    
    llm = ChatGroq(model="llama3-8b-8192")
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory)
    return conversation_chain


def handle_question(question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': question})
        
        st.session_state.messages.append({'role': 'user', 'content': question})

        st.session_state.messages.append({'role': 'assistant', 'content': response['answer']})


st.set_page_config(page_title="PDF ChatBot", page_icon=":bar_chart:")

# Custom styles for user and bot messages
user_message_style = "background-color: #FFD700; padding: 10px; border-radius: 10px; margin-bottom: 10px;"
bot_message_style = "background-color: #F4EBDC; padding: 10px; border-radius: 10px; margin-bottom: 10px;"

# Main function to run the Streamlit app
def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("""
    <div style='display: flex; align-items: center; justify-content: center;'>
        <h1 style='text-align: center; color: #6f3c2f;'>PDF ChatBot</h1>
    </div>
    """, unsafe_allow_html=True)

    docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)

    if "query" not in st.session_state:
        st.session_state.query = False

    if docs:
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(docs)
                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversationchain(vectorstore)
                st.session_state.query = True

    if st.session_state.query:
        question = st.chat_input("Ask a question related to your document:")
        if question:
            handle_question(question)
            for message in st.session_state.messages:
                if message['role'] == 'user':
                    st.markdown(f"<div style='{user_message_style}'><strong>User:</strong> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='{bot_message_style}'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Steps Should be followed:")
        
        st.write("step 1: Upload your documents...")
        
        st.write("step 2: click the process button for accessing your documents")
        
        st.write("step 3: ask your questions and see the responses from your documents")
        
        st.subheader("Key points:")
        
        st.write("1. Ensure that the documents are in English language")
        
        st.write("2. Avoid asking questions that are too broad or vague")
        
        st.write("3. Be patient and wait for the responses")
        
        st.write("4. multiple files can be uploaded")

if __name__ == '__main__':
    main()
