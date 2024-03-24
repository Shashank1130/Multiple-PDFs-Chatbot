import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma

# Loading dotenv
load_dotenv()

#  Settign up the google api key
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to get read the PDFs content
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text


# Function for splitting the text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n", "\n\n"],chunk_size=10000, 
                                                   chunk_overlap=100)
    chunks = text_splitter.split_text(text=text)
    return chunks


# Function to store the chunks into vector store
def get_vector_store(text_chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks,embedding=embedding)
    vector_store.save_local("faiss_index")


# Function for doing conversation and making prompts
def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provides context, make sure to 
        provide all the details, id the answer is not in provided context jaust say, 
        "answer is not availabel in the context", dont's provide the wrong answer\n\n

        Context:\n{context}\n
        Question:\n{question}\n

        Answer:
        """
    # Creating llm model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # Setting up the prompt template
    prompt = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "question"])
    # Creating the chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to get the user question and perform searching
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Calling "get_conversational_chain()" function
    chain = get_conversational_chain()

    # Response
    response = chain({"input_documents":docs,
                      "question":user_question,},
                       return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])





# Creating streamlit app
def main():
    st.set_page_config("Chat with multiple PDFs")
    st.header("Chat with multiple PDFs :books:")

    # Taking the question from the user
    user_question = st.text_input("Ask a Question from teh PDF Files: ")
    
    # Answering the question
    if user_question:
        user_input(user_question)

    # Side bar for uploading the PDFs and convert them into vector
    with st.sidebar:
        st.title("Your Documents:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process button", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Reading the pdfs
                raw_text = get_pdf_text(pdf_docs)

                # Splitting into chunks
                text_chunks = get_text_chunks(raw_text)

                # Saving into vector store
                get_vector_store(text_chunks)
                st.success("Done")










if __name__ == "__main__":
    main()