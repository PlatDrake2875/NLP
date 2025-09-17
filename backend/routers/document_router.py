# backend/routers/document_router.py
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from langchain_chroma import Chroma as LangchainChroma  # For type hinting

from config import COLLECTION_NAME, logger

# Import the vectorstore dependency function
from rag_components import get_vectorstore  # Use the vectorstore dependency

# Import schemas and dependency functions
from schemas import DocumentChunk, DocumentListResponse

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.get("", response_model=DocumentListResponse)
async def get_all_documents(
    # Inject the LangchainChroma vectorstore object
    vectorstore: LangchainChroma = Depends(get_vectorstore),
):
    """
    Retrieves all document chunks currently stored in the vector database.
    """
    logger.info("Received request to get all documents from vector store.")

    try:
        # --- Attempt to access the underlying Chroma client ---
        client: Optional[Any] = None  # Use Any type initially
        client_type_found = "None"  # For logging
        possible_client_attrs = ["client", "_client"]  # Common attribute names
        for attr_name in possible_client_attrs:
            if hasattr(vectorstore, attr_name):
                potential_client = getattr(vectorstore, attr_name)
                # logger.info(f"Found attribute '{attr_name}'. Type: {type(potential_client)}") # Keep commented for potential future debug
                # Check if it has the expected method instead of using isinstance
                if potential_client is not None and hasattr(
                    potential_client, "get_collection"
                ):
                    client = potential_client
                    client_type_found = str(type(client))
                    logger.info(
                        f"Using object from attribute '{attr_name}' as the client."
                    )
                    break  # Use the first found attribute
                # else: # Keep commented for potential future debug
                # logger.warning(f"Attribute '{attr_name}' found but is not a valid client object (missing get_collection or is None).")
            # else: # Keep commented for potential future debug
            # logger.debug(f"Attribute '{attr_name}' not found on vectorstore object.")

        if client is None:
            available_attrs = [
                attr
                for attr in dir(vectorstore)
                if not callable(getattr(vectorstore, attr))
                and not attr.startswith("__")
            ]
            logger.error(
                f"Could not access the underlying ChromaDB client from the vectorstore object. Available attributes: {available_attrs}"
            )
            raise HTTPException(
                status_code=503,
                detail="Internal configuration error: Could not obtain ChromaDB client from vectorstore.",
            )
        # --- End Client Access ---

        logger.info(f"Proceeding with client object of type: {client_type_found}")

        collection_name: str = getattr(vectorstore, "_collection_name", COLLECTION_NAME)

        # Get the collection object using the found client
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Accessed ChromaDB collection: {collection_name}")
        except Exception as e_coll:
            logger.error(
                f"Failed to get ChromaDB collection '{collection_name}': {e_coll}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Could not access vector store collection '{collection_name}': {str(e_coll)}",
            ) from e_coll

        # Retrieve all documents
        try:
            # Retrieve only necessary fields
            results = collection.get(include=["metadatas", "documents"])
            # --- REMOVED VERBOSE LOGGING ---
            # logger.info(f"Raw results from ChromaDB collection.get: {json.dumps(results, indent=2)}")
            # --- END REMOVED LOGGING ---
        except Exception as e_get:
            logger.error(
                f"Failed to retrieve documents from collection '{collection_name}': {e_get}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Could not retrieve documents from vector store: {str(e_get)}",
            ) from e_get

        if not results or not results.get("ids"):
            logger.info("No documents found in the collection.")
            return DocumentListResponse(count=0, documents=[])

        # Process results
        doc_chunks = []
        ids = results.get("ids", [])
        contents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        logger.info(
            f"Processing results: {len(ids) if ids else 0} IDs, {len(contents) if contents else 0} contents, {len(metadatas) if metadatas else 0} metadatas."
        )

        if (
            ids is None
            or contents is None
            or metadatas is None
            or not (len(ids) == len(contents) == len(metadatas))
        ):
            logger.error(
                f"Mismatch or None in lists returned by ChromaDB get: ids({len(ids) if ids else 'None'}), contents({len(contents) if contents else 'None'}), metadatas({len(metadatas) if metadatas else 'None'})"
            )
            logger.error(f"Problematic ChromaDB results object: {results}")
            raise HTTPException(
                status_code=500, detail="Inconsistent data received from vector store."
            )

        for i in range(len(ids)):
            doc_id = ids[i]
            content = contents[i] if i < len(contents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}
            if metadata is None:
                metadata = {}
            if content is None:
                content = ""

            # --- REMOVED PER-CHUNK DEBUG LOGGING ---
            # logger.debug(f"Processing chunk {i}: ID={doc_id}, Metadata={metadata}, Content Length={len(content)}")
            # if len(content) < 50:
            #      logger.debug(f"  Content (short/empty): '{content}'")
            # --- END REMOVED LOGGING ---

            doc_chunks.append(
                DocumentChunk(
                    id=doc_id,
                    page_content=content,  # Pydantic model uses alias 'content'
                    metadata=metadata,
                )
            )

        logger.info(
            f"Successfully processed {len(doc_chunks)} document chunks for response."
        )
        return DocumentListResponse(count=len(doc_chunks), documents=doc_chunks)

    except HTTPException as e_http:
        raise e_http
    except Exception as e:
        logger.error(f"Unexpected error retrieving documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while retrieving documents: {str(e)}",
        ) from e
