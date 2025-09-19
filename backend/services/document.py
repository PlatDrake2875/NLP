"""
Document service layer containing all document retrieval business logic.
Separated from the web layer for better testability and maintainability.
"""

from typing import Any, Optional

from fastapi import HTTPException
from langchain_chroma import Chroma as LangchainChroma

from backend.config import COLLECTION_NAME
from backend.schemas import DocumentChunk, DocumentListResponse


class DocumentService:
    """Service class handling all document retrieval business logic."""

    def __init__(self):
        self.collection_name = COLLECTION_NAME

    async def get_all_documents(
        self, vectorstore: LangchainChroma
    ) -> DocumentListResponse:
        try:
            # Get the underlying ChromaDB client
            client = self._get_chroma_client(vectorstore)

            # Get the collection
            collection = self._get_collection(client)

            # Retrieve all documents
            results = self._retrieve_documents(collection)

            if not results or not results.get("ids"):
                return DocumentListResponse(count=0, documents=[])

            # Process and format results
            doc_chunks = self._process_results(results)

            return DocumentListResponse(count=len(doc_chunks), documents=doc_chunks)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred while retrieving documents: {str(e)}",
            ) from e

    def _get_chroma_client(self, vectorstore: LangchainChroma) -> Any:
        """Get the underlying ChromaDB client from the vectorstore."""
        client: Optional[Any] = None
        possible_client_attrs = ["client", "_client"]

        for attr_name in possible_client_attrs:
            if hasattr(vectorstore, attr_name):
                potential_client = getattr(vectorstore, attr_name)
                if potential_client is not None and hasattr(
                    potential_client, "get_collection"
                ):
                    client = potential_client
                    break

        if client is None:
            raise HTTPException(
                status_code=503,
                detail="Internal configuration error: Could not obtain ChromaDB client from vectorstore.",
            )

        return client

    def _get_collection(self, client: Any) -> Any:
        """Get the collection from the ChromaDB client."""
        collection_name = getattr(client, "_collection_name", self.collection_name)

        try:
            return client.get_collection(name=collection_name)
        except Exception as e_coll:
            raise HTTPException(
                status_code=500,
                detail=f"Could not access vector store collection '{collection_name}': {str(e_coll)}",
            ) from e_coll

    def _retrieve_documents(self, collection: Any) -> dict:
        """Retrieve documents from the collection."""
        try:
            return collection.get(include=["metadatas", "documents"])
        except Exception as e_get:
            raise HTTPException(
                status_code=500,
                detail=f"Could not retrieve documents from vector store: {str(e_get)}",
            ) from e_get

    def _process_results(self, results: dict) -> list[DocumentChunk]:
        """Process the raw results into DocumentChunk objects."""
        ids = results.get("ids", [])
        contents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if (
            ids is None
            or contents is None
            or metadatas is None
            or not (len(ids) == len(contents) == len(metadatas))
        ):
            raise HTTPException(
                status_code=500, detail="Inconsistent data received from vector store."
            )

        doc_chunks = []
        for i in range(len(ids)):
            doc_id = ids[i]
            content = contents[i] if i < len(contents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}

            if metadata is None:
                metadata = {}
            if content is None:
                content = ""

            doc_chunks.append(
                DocumentChunk(
                    id=doc_id,
                    page_content=content,
                    metadata=metadata,
                )
            )

        return doc_chunks
