from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def process_pdf(file:str):
    """Process  file Chunk it and save into faiss vector store"""
    pdf_loader = PyPDFLoader(file)
    document = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,add_start_index=True)
    chunks = text_splitter.split_documents(document)
    embedding = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local("faiss_index")
    print("File Processed and saved into faiss_index folder")
    return "Processing Complete"