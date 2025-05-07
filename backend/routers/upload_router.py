# backend/routers/upload_router.py
import os
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader # Ensure this is the correct import
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import schemas, config, and RAG components using direct absolute paths
from schemas import UploadResponse
from config import logger
from rag_components import vectorstore, embedding_function # Get access to global vectorstore


router = APIRouter(
    prefix="/upload", # Prefix relative to /api -> /api/upload
    tags=["upload"],
)

@router.post("", response_model=UploadResponse) # Path relative to prefix -> /api/upload
async def upload_document_endpoint(file: UploadFile = File(...)):
    if not vectorstore:
        logger.error("Upload endpoint called but vector store is unavailable.")
        raise HTTPException(status_code=503, detail="Vector store unavailable. Cannot process upload.")
    if not embedding_function: # Though vectorstore implies embedding_function, good to check
        logger.error("Upload endpoint called but embedding function is unavailable.")
        raise HTTPException(status_code=503, detail="Embedding function unavailable. Cannot process upload.")

    logger.info(f"Upload request: {file.filename}, Content-Type: {file.content_type}")
    
    temp_file_path = f"/tmp/{file.filename}" 

    try:
        file_content = await file.read()
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)
        logger.info(f"File '{file.filename}' saved temporarily to {temp_file_path}")

        if file.content_type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        else:
            logger.warning(f"Unsupported file type for upload: {file.content_type}")
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Only PDF is currently supported.")
        
        docs = loader.load()
        if not docs:
            logger.warning(f"No documents were loaded from file: {file.filename}")
            return UploadResponse(message="No content found in the document.", filename=file.filename, chunks_added=0)

        logger.info(f"Loaded {len(docs)} document sections from '{file.filename}'.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split content from '{file.filename}' into {len(chunks)} chunks.")

        if not chunks:
            logger.warning(f"No text chunks were generated from file: {file.filename}")
            return UploadResponse(message="No text content could be extracted for vectorization.", filename=file.filename, chunks_added=0)
        
        vectorstore.add_documents(chunks)
        logger.info(f"Successfully added {len(chunks)} chunks from '{file.filename}' to the vector store.")
        
        return UploadResponse(
            message="Document processed and added to vector store successfully.", 
            filename=file.filename, 
            chunks_added=len(chunks)
        )

    except HTTPException as e_http:
        raise e_http # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Error processing upload for file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Temporary file '{temp_file_path}' removed successfully.")
            except OSError as e_os:
                logger.error(f"Error removing temporary file '{temp_file_path}': {e_os}")
        await file.close() 
