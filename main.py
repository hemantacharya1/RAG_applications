from fastapi import FastAPI, UploadFile
from dotenv import load_dotenv
load_dotenv()
from fastapi import UploadFile
from rag.rag_ingestion import process_pdf
from rag.rag_query import respond_to_query

app = FastAPI()


@app.post('/ingest')
async def ingest_file(file: UploadFile):
    """Endpoint to ingest file and store according to there type """
    file_type = file.content_type
    if file_type == "application/pdf":
       """ PDF file and save it locally"""
       with open(f"uploads/{file.filename}", "wb") as f:
           f.write(await file.read())
       process_pdf(f"uploads/{file.filename}")
    elif file_type == "image/jpeg":
        # Process JPEG image
        pass
    else:
        return {"Error": "Unsupported file type"}
    return {"Success": "File ingested successfully"}

@app.get("/query")
async def respond_to_query(query: str):
    """Endpoint to respond to user query"""
    return respond_to_query(query)


@app.get("/")
def read_root():
    return {"Response": "Multimodel Rag App is running!"}
