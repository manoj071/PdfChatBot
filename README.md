# PDF ChatBot

## Overview

The PDF ChatBot is a web application built using Streamlit that allows users to interact with PDF documents through a conversational interface. It processes uploaded PDFs, extracts and indexes their content, and provides a way to ask questions about the document. The application leverages language models to understand and respond to user queries based on the content of the PDFs.

## Technologies Used

   **Streamlit**: For creating the web application interface.

   **PyPDF2**: For extracting text from PDF files.

   **LangChain**: For text processing, embeddings, and creating the conversational retrieval chain.

   **CharacterTextSplitter**: For splitting text into manageable chunks.

   **HuggingFaceEmbeddings**: For generating embeddings from text using the all-MiniLM model.

   **FAISS**: For creating and querying the vector store.

   **ConversationalRetrievalChain**: For managing conversation flow and integrating with the language model.

   **ChatGroq**: For generating responses based on user queries.

   **Python Dotenv**: For loading environment variables.

## How It Works

  # Upload PDF Documents:

      Users can upload one or more PDF documents using the file uploader in the Streamlit app.

  # Process Documents:

      Once the PDFs are uploaded, clicking the "Process" button triggers the processing of the documents. The text from the PDFs is extracted and split into chunks.

  # Create Vector Store:

      The extracted text chunks are converted into embeddings using the HuggingFaceEmbeddings model. These embeddings are then stored in a FAISS vector store for efficient retrieval.

  # Generate Conversation Chain:

      A conversational retrieval chain is created using the ChatGroq model and the vector store. This chain enables the chatbot to respond to user questions based on the content of the PDFs.

  # Ask Questions:

      Users can input questions related to the content of the uploaded PDFs. The chatbot will generate responses based on the document content and the conversation history.

  # Display Responses:

      The conversation history is displayed in the chat interface, with user and assistant messages styled for clarity.



## Running the Application

To use the PDF Intelligence System:

   1. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

   2. Set Up Environment Variables
   ```bash
   GROQ_API_KEY=<your_groq_api_key>
   ```

   3. Run the application.
   ```bash
   streamlit run app.py
   ```

   
   





