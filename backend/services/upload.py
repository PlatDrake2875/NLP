"""
Upload service layer containing all document upload business logic.
Separated from the web layer for better testability and maintainability.
"""

import os

from fastapi import HTTPException, UploadFile
from langchain_chroma import Chroma as LangchainChroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.schemas import UploadResponse


class UploadService:
    """Service class handling all document upload business logic."""

    def __init__(self):
        self.temp_dir = "/app/temp_uploads"
        self.supported_content_types = {"application/pdf"}
        self.chunk_size = 1000
        self.chunk_overlap = 200

    async def process_document_upload(
        self, file: UploadFile, vectorstore: LangchainChroma
    ) -> UploadResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        original_filename = file.filename
        temp_file_path = self._prepare_temp_file_path(original_filename)

        file_content = await self._read_and_validate_file(file)
        self._save_temp_file(file_content, temp_file_path)
        docs = self._load_document(file, temp_file_path, original_filename)
        chunks = self._split_document_into_chunks(docs)
        self._add_to_vector_store(chunks, vectorstore)

        self._cleanup_temp_file(temp_file_path)
        await file.close()

        return UploadResponse(
            message="Document processed and added to vector store successfully.",
            filename=original_filename,
            chunks_added=len(chunks),
        )

    def _prepare_temp_file_path(self, filename: str) -> str:
        """Prepare the temporary file path."""
        os.makedirs(self.temp_dir, exist_ok=True)
        return os.path.join(self.temp_dir, filename)

    async def _read_and_validate_file(self, file: UploadFile) -> bytes:
        """Read and validate the uploaded file content."""
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return file_content

    def _save_temp_file(self, file_content: bytes, temp_file_path: str) -> None:
        """Save the file content to a temporary file."""
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)

    def _load_document(
        self, file: UploadFile, temp_file_path: str, original_filename: str
    ) -> list:
        """Load the document based on its content type."""
        if file.content_type not in self.supported_content_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Only PDF is currently supported.",
            )

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the document.",
            )

        for doc in docs:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["original_filename"] = original_filename

        return docs

    def _split_document_into_chunks(self, docs: list) -> list:
        """Split the document into chunks for vectorization."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted for vectorization.",
            )

        return chunks

    def _add_to_vector_store(self, chunks: list, vectorstore: LangchainChroma) -> None:
        """Add the document chunks to the vector store."""
        vectorstore.add_documents(chunks)

    def _cleanup_temp_file(self, temp_file_path: str) -> None:
        """Clean up the temporary file."""
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def main():
    """Test the UploadService with a mock PDF document."""
    import shutil
    import tempfile
    from io import BytesIO

    from langchain_huggingface import HuggingFaceEmbeddings
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    test_temp_dir = tempfile.mkdtemp(prefix="upload_service_test_")

    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        story = [
            Paragraph("Test Document for Upload Service", styles["Title"]),
            Paragraph(
                "This is a test document created for testing the upload service functionality.",
                styles["Normal"],
            ),
            Paragraph(
                "It contains multiple paragraphs to test text splitting and chunking.",
                styles["Normal"],
            ),
            Paragraph(
                "The document should be processed, split into chunks, and added to a vector store.",
                styles["Normal"],
            ),
            Paragraph(
                "This paragraph contains some additional content to ensure we have enough text for meaningful chunks.",
                styles["Normal"],
            ),
        ]

        doc.build(story)
        pdf_content = pdf_buffer.getvalue()
        pdf_buffer.close()

        class MockUploadFile:
            def __init__(self, content: bytes, filename: str, content_type: str):
                self.content = content
                self.filename = filename
                self.content_type = content_type
                self._closed = False

            async def read(self) -> bytes:
                if self._closed:
                    raise ValueError("File is closed")
                return self.content

            async def close(self):
                self._closed = True

        mock_file = MockUploadFile(
            content=pdf_content,
            filename="test_document.pdf",
            content_type="application/pdf",
        )

        upload_service = UploadService()
        upload_service.temp_dir = test_temp_dir

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = LangchainChroma(embedding_function=embeddings)

        await upload_service.process_document_upload(mock_file, vectorstore)

        collection = vectorstore._collection
        count = collection.count()

        if count > 0:
            vectorstore.similarity_search("test document", k=2)

    except Exception:
        pass

    finally:
        try:
            shutil.rmtree(test_temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
