from fastapi import FastAPI, UploadFile,HTTPException
from dotenv import load_dotenv
load_dotenv()
from fastapi import UploadFile 
from rag.rag_query import respond_to_query
from rag.rag_ingestion import get_ingested_files, process_image,process_pdf
import os

app = FastAPI()

UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.post('/ingest')
async def ingest_file(file: UploadFile):
    """Endpoint to ingest file and store according to there type """
    """
    Endpoint to ingest file and store according to its type (PDF or JPEG).
    """
    file_type = file.content_type
    file_location = os.path.join(UPLOADS_DIR, file.filename)

    try:
        # Save the file locally
        with open(file_location, "wb") as f:
            f.write(await file.read())

        if file_type == "application/pdf":
            # Process PDF file
            process_pdf(file_location)
        elif file_type.startswith("image/"):
            # Process image file
            process_image(file_location)
        else:
            os.remove(file_location) 
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and image files are supported.")
        
        return {"Success": f"File '{file.filename}' ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.get("/query")
async def get_response_to_query(query: str):
    """
    Endpoint to respond to user query using the RAG system.
    """
    if not os.path.exists("faiss_index"):
        raise HTTPException(status_code=404, detail="Vector store not found. Please ingest documents first.")
    
    response = respond_to_query(query)
    return {"Response": response}


@app.get("/ingested-files")
async def list_ingested_files():
    """
    Endpoint to list all files that have been ingested.
    """
    files = get_ingested_files()
    return {"ingested_files": files}


@app.get("/")
def read_root():
    return {"Response": "Multimodel Rag App is running!"}
