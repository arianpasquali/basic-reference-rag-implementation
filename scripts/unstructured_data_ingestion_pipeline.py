#!/usr/bin/env python3
"""
PDF Document Ingestion Pipeline for ChromaDB

Pipeline for ingesting PDF documents into ChromaDB with:
- Semantic-aware text chunking
- Rich metadata extraction and management
- Error handling and logging
- Batch processing capabilities
- Verification and testing functions
"""

from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

import chromadb
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Add src directory to path for importing settings
sys.path.append(str(Path(__file__).parent.parent / "src"))
from core.settings import settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chroma_ingestion.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress PyPDF warnings about deprecated features

warnings.filterwarnings("ignore", category=UserWarning, module="pypdf._reader")


class ChromaPDFIngestionPipeline:
    """
    Enhanced PDF ingestion pipeline for ChromaDB with comprehensive features.
    """

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        openai_api_key: Optional[str] = None,
        delete_existing: bool = False,
    ):
        """
        Initialize the ChromaDB PDF ingestion pipeline.

        Args:
            persist_directory: Directory to store ChromaDB data (uses settings default if None)
            collection_name: Name of the ChromaDB collection (uses settings default if None)
            embedding_model: OpenAI embedding model to use (uses settings default if None)
            chunk_size: Maximum characters per text chunk (uses settings default if None)
            chunk_overlap: Overlap between chunks for context preservation (uses settings default if None)
            openai_api_key: OpenAI API key (uses env var if not provided)
            delete_existing: Whether to delete existing database
        """
        # Use settings as defaults if parameters are not provided
        self.persist_directory = persist_directory or str(settings.CHROMA_DB_PATH)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Setup API key
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        # Initialize components
        self._setup_database(delete_existing)
        self._setup_text_splitter()
        self._setup_embeddings()

        logger.info(f"ChromaDB pipeline initialized - Collection: {collection_name}")

    def _setup_database(self, delete_existing: bool) -> None:
        """Setup ChromaDB database and handle existing data."""
        if delete_existing and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            logger.info(f"Deleted existing database at {self.persist_directory}")

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

    def _setup_text_splitter(self) -> None:
        """Initialize the text splitter with semantic-aware settings."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentence endings
                "! ",  # Exclamation sentences
                "? ",  # Question sentences
                "; ",  # Semicolon breaks
                ", ",  # Comma breaks
                " ",  # Word boundaries
                "",  # Character fallback
            ],
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(
            f"Text splitter configured - Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}"
        )

    def _setup_embeddings(self) -> None:
        """Initialize OpenAI embeddings."""
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. Embeddings will be initialized when needed.")
            self.embeddings = None
            return

        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=self.openai_api_key,
                show_progress_bar=True,
            )
            logger.info(f"Embeddings initialized - Model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def _get_vectorstore(self) -> Chroma:
        """Get or create ChromaDB vector store."""
        if not self.embeddings:
            if not self.openai_api_key:
                raise ValueError(
                    "OpenAI API key is required for vector store operations. Set OPENAI_API_KEY environment variable."
                )
            # Initialize embeddings if not done yet
            self._setup_embeddings()

        try:
            if settings.CHROMA_API_KEY and settings.CHROMA_API_KEY.strip():
                # Use ChromaDB Cloud
                logger.info("Initializing ChromaDB Cloud client...")
                chroma_client = chromadb.CloudClient(
                    api_key=settings.CHROMA_API_KEY,
                    tenant=settings.CHROMA_TENANT_ID,
                    database=settings.CHROMA_DATABASE_NAME,
                )
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                )
                logger.info(
                    f"ChromaDB Cloud client initialized with collection: {self.collection_name}"
                )
            else:
                # Use local ChromaDB
                logger.info(f"Initializing local ChromaDB client at: {settings.CHROMA_DB_PATH}")
                vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_directory),
                )
                logger.info(
                    f"Local ChromaDB client initialized with collection: {self.collection_name}"
                )

            return vectorstore
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            logger.error(
                f"Chroma parameters - collection: {self.collection_name}, persist_dir: {self.persist_directory}"
            )
            raise

    def _extract_pdf_metadata(self, file_path: str, document: Document) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from PDF document.

        Args:
            file_path: Path to the PDF file
            document: Loaded document object

        Returns:
            Dictionary containing metadata
        """
        file_stat = os.stat(file_path)
        file_hash = self._calculate_file_hash(file_path)

        metadata = {
            # File information
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_size": file_stat.st_size,
            "file_hash": file_hash,
            "ingestion_timestamp": datetime.now().isoformat(),
            # Document type and processing info
            "document_type": "pdf",
            "collection": self.collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
        }

        # Add original document metadata if available
        if hasattr(document, "metadata") and document.metadata:
            original_metadata = document.metadata
            metadata.update(
                {
                    "page_number": original_metadata.get("page", 0),
                    "total_pages": original_metadata.get("total_pages", 0),
                }
            )

        return metadata

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for deduplication."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _process_single_pdf(self, file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Process a single PDF file into chunks with metadata.

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (document chunks, processing stats)
        """
        try:
            logger.info(f"Processing PDF: {file_path}")

            # Load PDF document
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            if not documents:
                raise ValueError(f"No content extracted from {file_path}")

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Enhance metadata for each chunk
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                base_metadata = self._extract_pdf_metadata(file_path, chunk)

                # Add chunk-specific metadata
                chunk_metadata = {
                    **base_metadata,
                    "chunk_id": f"{base_metadata['file_hash']}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_text_length": len(chunk.page_content),
                }

                # Update chunk metadata
                chunk.metadata = chunk_metadata
                enhanced_chunks.append(chunk)

            processing_stats = {
                "filename": os.path.basename(file_path),
                "pages_processed": len(documents),
                "chunks_created": len(enhanced_chunks),
                "total_characters": sum(len(chunk.page_content) for chunk in enhanced_chunks),
                "status": "success",
            }

            logger.info(
                f"Successfully processed {file_path} - {len(enhanced_chunks)} chunks created"
            )
            return enhanced_chunks, processing_stats

        except Exception as e:
            error_stats = {
                "filename": os.path.basename(file_path),
                "status": "error",
                "error_message": str(e),
            }
            logger.error(f"Failed to process {file_path}: {e}")
            return [], error_stats

    def ingest_pdf_directory(
        self, folder_path: str, file_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest all PDF files from a directory.

        Args:
            folder_path: Path to directory containing PDFs
            file_patterns: List of file patterns to match (default: ["*.pdf"])

        Returns:
            Dictionary containing ingestion results and statistics
        """
        if file_patterns is None:
            file_patterns = ["*.pdf"]

        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Directory does not exist: {folder_path}")

        # Find PDF files
        pdf_files = []
        for pattern in file_patterns:
            pdf_files.extend(folder_path.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return {"status": "no_files", "files_processed": 0}

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Process files and collect results
        vectorstore = self._get_vectorstore()
        all_chunks = []
        processing_results = []

        for pdf_file in pdf_files:
            chunks, stats = self._process_single_pdf(str(pdf_file))
            all_chunks.extend(chunks)
            processing_results.append(stats)

        # Batch add documents to ChromaDB with size limits
        if all_chunks:
            try:
                # Generate unique IDs for chunks
                chunk_ids = [chunk.metadata["chunk_id"] for chunk in all_chunks]

                # Determine batch size based on ChromaDB type and quota limits
                if settings.CHROMA_API_KEY and settings.CHROMA_API_KEY.strip():
                    max_batch_size = 100  # ChromaDB Cloud conservative limit (respecting quota)
                    logger.warning(
                        "⚠️  Using ChromaDB Cloud with quota limits. Consider upgrading plan for larger datasets."
                    )
                else:
                    max_batch_size = 5000  # Local ChromaDB can handle larger batches

                total_chunks = len(all_chunks)
                logger.info(
                    f"Adding {total_chunks} chunks to ChromaDB in batches of {max_batch_size}"
                )

                # Process documents in batches
                for i in range(0, total_chunks, max_batch_size):
                    end_idx = min(i + max_batch_size, total_chunks)
                    batch_docs = all_chunks[i:end_idx]
                    batch_ids = chunk_ids[i:end_idx]

                    logger.info(
                        f"Processing batch {i // max_batch_size + 1}: documents {i + 1}-{end_idx}"
                    )

                    # Add batch to vector store
                    try:
                        vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                        logger.info(
                            f"Successfully added batch {i // max_batch_size + 1} ({len(batch_docs)} chunks)"
                        )
                    except Exception as batch_error:
                        if "Quota exceeded" in str(batch_error):
                            logger.error(
                                f"❌ ChromaDB Cloud quota exceeded in batch {i // max_batch_size + 1}"
                            )
                            logger.error("💡 Consider:")
                            logger.error("   - Upgrading your ChromaDB Cloud plan")
                            logger.error("   - Using local ChromaDB (unset CHROMA_API_KEY)")
                            logger.error("   - Processing fewer documents at a time")
                            raise
                        else:
                            raise batch_error

                logger.info(f"✅ Successfully added all {total_chunks} chunks to ChromaDB")

            except Exception as e:
                logger.error(f"Failed to add documents to ChromaDB: {e}")
                raise

        # Compile final results
        successful_files = [r for r in processing_results if r["status"] == "success"]
        failed_files = [r for r in processing_results if r["status"] == "error"]

        results = {
            "status": "completed",
            "files_processed": len(pdf_files),
            "successful_files": len(successful_files),
            "failed_files": len(failed_files),
            "total_chunks": len(all_chunks),
            "total_characters": sum(r.get("total_characters", 0) for r in successful_files),
            "processing_results": processing_results,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }

        logger.info(
            f"Ingestion completed - {results['successful_files']}/{results['files_processed']} files successful"
        )
        return results

    def verify_ingestion(self, sample_queries: List[str] = None) -> Dict[str, Any]:
        """
        Verify the ingestion by performing test searches.

        Args:
            sample_queries: List of test queries

        Returns:
            Dictionary containing verification results
        """
        if sample_queries is None:
            sample_queries = [
                "Toyota",
                "warranty",
                "contract terms",
                "Lexus",
                "vehicle maintenance",
            ]

        vectorstore = self._get_vectorstore()
        verification_results = {
            "total_documents": vectorstore._collection.count(),
            "search_tests": [],
        }

        for query in sample_queries:
            try:
                results = vectorstore.similarity_search(query=query, k=3)

                search_result = {
                    "query": query,
                    "results_found": len(results),
                    "status": "success",
                    "sample_content": results[0].page_content[:200] + "..." if results else None,
                }

            except Exception as e:
                search_result = {"query": query, "status": "error", "error_message": str(e)}

            verification_results["search_tests"].append(search_result)

        logger.info(
            f"Verification completed - {verification_results['total_documents']} documents in collection"
        )
        return verification_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            vectorstore = self._get_vectorstore()
            collection = vectorstore._collection

            stats = {
                "collection_name": self.collection_name,
                "total_documents": collection.count(),
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model,
                "chunk_configuration": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            }

            # Try to get a sample document for metadata analysis
            try:
                sample_docs = vectorstore.similarity_search("sample", k=1)
                if sample_docs:
                    stats["sample_metadata"] = sample_docs[0].metadata
            except Exception:
                logger.warning("No sample document found for metadata analysis")

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}


def create_pipeline_from_settings(delete_existing: bool = False) -> ChromaPDFIngestionPipeline:
    """
    Create a ChromaPDFIngestionPipeline using configuration from settings.

    Args:
        delete_existing: Whether to delete existing database

    Returns:
        Configured ChromaPDFIngestionPipeline instance
    """
    return ChromaPDFIngestionPipeline(
        persist_directory=str(settings.CHROMA_DB_PATH),
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        delete_existing=delete_existing,
    )


def main():
    """Main function demonstrating the pipeline usage."""
    # Configuration from settings
    DOCS_FOLDER = str(settings.INPUT_DOCS_PATH)

    # Display current configuration
    print("📋 Configuration from settings:")
    print(f"  - Input docs path: {DOCS_FOLDER}")
    print(f"  - ChromaDB path: {settings.CHROMA_DB_PATH}")
    print(f"  - Collection name: {settings.CHROMA_COLLECTION_NAME}")
    print(f"  - Embedding model: {settings.EMBEDDING_MODEL}")
    print(f"  - Chunk size: {settings.CHUNK_SIZE}")
    print(f"  - Chunk overlap: {settings.CHUNK_OVERLAP}")
    print()

    try:
        # Initialize pipeline using settings
        pipeline = create_pipeline_from_settings(delete_existing=True)

        # Ingest PDF documents
        print("🚀 Starting PDF ingestion...")
        results = pipeline.ingest_pdf_directory(DOCS_FOLDER)

        print("\n📊 Ingestion Results:")
        print(f"  - Files processed: {results['files_processed']}")
        print(f"  - Successful files: {results['successful_files']}")
        print(f"  - Failed files: {results['failed_files']}")
        print(f"  - Total chunks created: {results['total_chunks']}")
        print(f"  - Total characters: {results['total_characters']:,}")

        # Verify ingestion
        print("\n🔍 Verifying ingestion...")
        verification = pipeline.verify_ingestion(
            [
                "Toyota warranty policy",
                "Lexus contract terms",
                "vehicle maintenance requirements",
                "warranty coverage period",
            ]
        )

        print(f"  - Total documents in collection: {verification['total_documents']}")
        print("  - Search test results:")
        for test in verification["search_tests"]:
            status = "✅" if test["status"] == "success" else "❌"
            print(f"    {status} '{test['query']}': {test.get('results_found', 0)} results")

        # Show collection statistics
        print("\n📈 Collection Statistics:")
        stats = pipeline.get_collection_stats()
        print(f"  - Collection: {stats['collection_name']}")
        print(f"  - Total documents: {stats['total_documents']}")
        print(f"  - Storage location: {stats['persist_directory']}")
        print(f"  - Embedding model: {stats['embedding_model']}")

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
