from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import base64
import os
import json


INGESTED_FILES_METADATA_PATH = "ingested_files_metadata.json"   

def get_ingested_files():
    """Returns a list of all files tracked as ingested."""
    if os.path.exists(INGESTED_FILES_METADATA_PATH):
        with open(INGESTED_FILES_METADATA_PATH, 'r') as f:
            return json.load(f)
    return {"pdfs": [], "images": []}

def process_pdf(file_path:str):
    """Process  file Chunk it and save into faiss vector store"""
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    pdf_loader = PyPDFLoader(file_path)
    document_pages = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    documents_to_add = text_splitter.split_documents(document_pages)
    
    if documents_to_add:
        if os.path.exists("faiss_index"):
            vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
            vectorstore.add_documents(documents_to_add)
        else:
            vectorstore = FAISS.from_documents(documents_to_add, embedding_model)
        vectorstore.save_local("faiss_index")
        print("PDF content added to faiss_index folder.")
    
    metadata = get_ingested_files()
    if file_path not in metadata["pdfs"]:
        metadata["pdfs"].append(file_path)
    with open(INGESTED_FILES_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    return "Processing Complete"


def process_image(file_path: str):
    """
    Processes an image file (placeholder description), and saves/updates it
    into the FAISS vector store.
    """
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-2.5-pro")
    multimodal_llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.3)
    try:
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        return f"Error reading image: {e}"
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe this image in detail, focusing on objects, colors, and the overall scene. make it detailed and comprehensive."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    )

    try:
        response = multimodal_llm.invoke([message])
        image_description = response.content
    except Exception as e:
        return f"Error reading image: {e}"

    image_document = Document(page_content=image_description, metadata={"source": file_path, "type": "image"})
    documents_to_add = [image_document]

    if documents_to_add:
        if os.path.exists("faiss_index"):
            vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
            vectorstore.add_documents(documents_to_add)
        else:
            vectorstore = FAISS.from_documents(documents_to_add, embedding_model)
        vectorstore.save_local("faiss_index")

    metadata = get_ingested_files()
    if file_path not in metadata["images"]:
        metadata["images"].append(file_path)
    with open(INGESTED_FILES_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    return "Processing Complete"