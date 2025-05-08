# backend/routers/upload_router.py
import os
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends # Import Depends
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma as LangchainChroma # For type hinting
# from langchain_huggingface import HuggingFaceEmbeddings # No longer needed directly

from schemas import UploadResponse
from config import logger
# Import the dependency function
from rag_components import get_vectorstore

router = APIRouter(
    prefix="/upload",
    tags=["upload"],
)

@router.post("", response_model=UploadResponse)
async def upload_document_endpoint(
    file: UploadFile = File(...),
    # Inject dependencies using Depends
    vectorstore: LangchainChroma = Depends(get_vectorstore)
    # embedding_function: HuggingFaceEmbeddings = Depends(get_embedding_function) # If needed later
):
    # Dependencies are now guaranteed to be available here by FastAPI

    # Get the original filename BEFORE any processing
    original_filename = file.filename
    logger.info(f"Upload request received for: {original_filename}, Content-Type: {file.content_type}")

    temp_dir = "/app/temp_uploads" # Use a directory inside the container
    os.makedirs(temp_dir, exist_ok=True) # Ensure the directory exists
    temp_file_path = os.path.join(temp_dir, original_filename)

    try:
        # Read file content
        try:
            file_content = await file.read()
            if not file_content:
                 logger.warning(f"Uploaded file '{original_filename}' is empty.")
                 raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        except Exception as read_err:
            logger.error(f"Error reading uploaded file '{original_filename}': {read_err}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {str(read_err)}")

        # Save file temporarily
        try:
            with open(temp_file_path, "wb") as buffer:
                buffer.write(file_content)
            logger.info(f"File '{original_filename}' saved temporarily to {temp_file_path}")
        except IOError as save_err:
             logger.error(f"Error saving temporary file '{temp_file_path}': {save_err}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Could not save uploaded file temporarily: {str(save_err)}")

        # --- Document Loading ---
        if file.content_type == "application/pdf":
            try:
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
            except Exception as pdf_err:
                 logger.error(f"Error loading PDF '{original_filename}' with PyPDFLoader: {pdf_err}", exc_info=True)
                 # Clean up temp file on loading error
                 if os.path.exists(temp_file_path): os.remove(temp_file_path)
                 raise HTTPException(status_code=400, detail=f"Failed to load PDF content. File might be corrupted or invalid: {str(pdf_err)}")
        else:
            if os.path.exists(temp_file_path): os.remove(temp_file_path)
            logger.warning(f"Unsupported file type for upload: {file.content_type}")
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Only PDF is currently supported.")

        if not docs:
            if os.path.exists(temp_file_path): os.remove(temp_file_path)
            logger.warning(f"No documents were loaded from file: {original_filename}")
            raise HTTPException(status_code=400, detail="No content could be extracted from the document.")

        logger.info(f"Loaded {len(docs)} document sections from '{original_filename}'.")

        # --- Add original filename to metadata BEFORE splitting ---
        for doc in docs:
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata['original_filename'] = original_filename
            # Optionally remove or keep the temporary 'source' path
            # del doc.metadata['source'] # Example: remove temporary path

        # --- Text Splitting ---
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            logger.info(f"Split content from '{original_filename}' into {len(chunks)} chunks.")
        except Exception as split_err:
            logger.error(f"Error splitting document '{original_filename}': {split_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to split document content: {str(split_err)}")


        if not chunks:
            logger.warning(f"No text chunks were generated from file: {original_filename}")
            raise HTTPException(status_code=400, detail="No text content could be extracted for vectorization.")

        # --- Adding to Vector Store ---
        try:
            # Use the injected vectorstore instance
            vectorstore.add_documents(chunks)
            logger.info(f"Successfully added {len(chunks)} chunks from '{original_filename}' to the vector store.")
        except Exception as vs_err:
             logger.error(f"Error adding document chunks from '{original_filename}' to vector store: {vs_err}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Failed to add document to vector store: {str(vs_err)}")

        # --- Success Response ---
        return UploadResponse(
            message="Document processed and added to vector store successfully.",
            filename=original_filename, # Return original filename in response
            chunks_added=len(chunks)
        )

    except HTTPException as e_http:
        logger.error(f"HTTPException during upload of '{original_filename}': Status={e_http.status_code}, Detail={e_http.detail}")
        raise e_http
    except Exception as e:
        logger.error(f"Unexpected error processing upload for file '{original_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred while processing the file: {str(e)}")
    finally:
        # --- Cleanup Temporary File ---
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Temporary file '{temp_file_path}' removed successfully.")
            except OSError as e_os:
                logger.error(f"Error removing temporary file '{temp_file_path}': {e_os}")
        await file.close()
        logger.info(f"Closed file handle for '{original_filename}'.")
