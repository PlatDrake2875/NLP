# backend/routers/document_router.py
import logging
from fastapi import APIRouter, Depends, HTTPException # Import Depends
from typing import List, Dict, Any, Optional

# Import schemas and dependency functions
from schemas import DocumentListResponse, DocumentChunk
from config import logger, COLLECTION_NAME
# Import the specific client dependency function
from rag_components import get_chroma_client # Import the dependency function
import chromadb # Import base chromadb client library

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

@router.get("", response_model=DocumentListResponse)
async def get_all_documents(
    # Inject the ChromaDB client using Depends
    client: chromadb.HttpClient = Depends(get_chroma_client)
):
    """
    Retrieves all document chunks currently stored in the vector database.
    """
    logger.info("Received request to get all documents from vector store.")

    # The 'client' variable is now guaranteed to be initialized if this code runs
    # because get_chroma_client would have raised an HTTPException otherwise.

    try:
        collection_name: str = COLLECTION_NAME

        # Get the collection object using the injected client
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Accessed ChromaDB collection: {collection_name}")
        except Exception as e_coll:
             logger.error(f"Failed to get ChromaDB collection '{collection_name}': {e_coll}", exc_info=True)
             # Provide a more specific error if possible, e.g., collection not found?
             raise HTTPException(status_code=500, detail=f"Could not access vector store collection '{collection_name}': {str(e_coll)}")

        # Retrieve all documents
        try:
            results = collection.get(include=["metadatas", "documents"])
            logger.debug(f"ChromaDB get results: {results}") # Log the raw results
        except Exception as e_get:
             logger.error(f"Failed to retrieve documents from collection '{collection_name}': {e_get}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Could not retrieve documents from vector store: {str(e_get)}")


        if not results or not results.get("ids"):
            logger.info("No documents found in the collection.")
            return DocumentListResponse(count=0, documents=[])

        # Process results
        doc_chunks = []
        ids = results.get("ids", [])
        contents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not (len(ids) == len(contents) == len(metadatas)):
            logger.error(f"Mismatch in lengths returned by ChromaDB get: ids({len(ids)}), contents({len(contents)}), metadatas({len(metadatas)})")
            raise HTTPException(status_code=500, detail="Inconsistent data received from vector store.")

        for i in range(len(ids)):
            doc_id = ids[i]
            content = contents[i] if contents and i < len(contents) else ""
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            if metadata is None: metadata = {}
            if content is None: content = ""

            doc_chunks.append(
                DocumentChunk(
                    id=doc_id,
                    page_content=content, # Pydantic model uses alias 'content'
                    metadata=metadata
                )
            )

        logger.info(f"Successfully processed {len(doc_chunks)} document chunks.")
        return DocumentListResponse(count=len(doc_chunks), documents=doc_chunks)

    except HTTPException as e_http:
        # Re-raise specific HTTP exceptions
        raise e_http
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(f"Unexpected error retrieving documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred while retrieving documents: {str(e)}")

