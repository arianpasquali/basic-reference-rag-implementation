"""This module provides Toyota RAG tools for document search and SQL queries.

These tools are designed for Toyota/Lexus vehicle information and sales data analysis.
Uses ChromaDB for semantic document search and SQLite for sales data queries.
"""

import logging
import os
import sqlite3
from typing import Any, Callable, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from pydantic import BaseModel, Field

from core.settings import settings

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Schema for search results."""

    filename: str = Field(description="Name of the document file")
    page: int = Field(description="Page number in the document")
    chunk_index: int = Field(description="Chunk index within the document")
    content: str = Field(description="Text content of the chunk")
    relevance_score: float = Field(description="Relevance score for the search")
    chunk_id: str = Field(description="Unique chunk identifier")


# Global variables for lazy initialization
vectorstore = None


def _get_vectorstore() -> Optional[Chroma]:
    """Lazy initialization of ChromaDB vectorstore."""
    global vectorstore

    if vectorstore is not None:
        return vectorstore

    try:
        if not settings.CHROMA_DB_PATH.exists():
            logger.warning(f"ChromaDB not found at {settings.CHROMA_DB_PATH}")
            return None

        # Initialize ChromaDB connection
        embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL, openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(settings.CHROMA_DB_PATH),
        )

        logger.info(f"Connected to ChromaDB at {settings.CHROMA_DB_PATH}")
        logger.info(
            f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' has {vectorstore._collection.count()} documents"
        )
        return vectorstore

    except Exception as e:
        logger.warning(f"Failed to connect to ChromaDB: {e}")
        return None


def _get_sqlite_connection():
    """Create a new SQLite connection for each request to avoid threading issues.

    This function creates a fresh connection for every request to prevent
    'SQLite objects created in a thread can only be used in that same thread' errors
    that occur when the same connection is reused across different conversation turns.
    """
    try:
        # Create a new connection each time to avoid threading issues
        # check_same_thread=False allows the connection to be used across threads
        # mode=ro ensures read-only access for safety
        conn = sqlite3.connect(
            f"file:{settings.DEFAULT_SQLITE_PATH}?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=30.0,  # Add timeout to prevent hanging
        )
        logger.debug(f"Created new SQLite connection to {settings.DEFAULT_SQLITE_PATH}")
        return conn
    except Exception as e:
        logger.warning(f"Failed to connect to SQLite: {e}")
        return None


@tool(
    description="""Use this tool to answer questions about warranty terms, policy clauses, or owner's manual content by searching the document database.

IMPORTANT: When you use this tool to answer a question, you MUST:
1. Provide a comprehensive answer based on the retrieved content

Args:
    query: The search query string
    limit: Maximum number of results to return (default: 10)
Returns:
    List of relevant document chunks with metadata including filename and page numbers for citation"""
)
def search_documents(query: str, limit: int = settings.MAX_SEARCH_RESULTS) -> List[SearchResult]:
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return []
    try:
        # Use ChromaDB similarity search with scores
        results_with_scores = vectorstore.similarity_search_with_score(query=query, k=limit)
        search_results = []

        for doc, score in results_with_scores:
            # Extract metadata from the document
            metadata = doc.metadata
            result = SearchResult(
                filename=metadata.get("filename", "Unknown"),
                page=int(metadata.get("page_number", metadata.get("page", 0))),
                chunk_index=int(metadata.get("chunk_index", 0)),
                content=doc.page_content,
                relevance_score=float(1.0 - score),  # Convert distance to similarity score
                chunk_id=metadata.get("chunk_id", ""),
            )
            search_results.append(result)

        logger.info(f"ChromaDB search found {len(search_results)} results for query: '{query}'")
        return search_results

    except Exception as e:
        logger.error(f"ChromaDB search error: {e}")
        return []


@tool(
    description="""Use this tool to answer questions about sales, time, country/region, model, powertrain, or any data that can be retrieved via SQL queries from the sales database (SQLite).

DATABASE SCHEMA (Star Schema):
FACT TABLE:
- fact_sales(model_id, country_code, year, month, contracts) - Main sales data

DIMENSION TABLES:
- dim_country(country, country_code, region) - Country information
- dim_model(model_id, model_name, brand, segment, powertrain) - Vehicle model details
- dim_ordertype(ordertype_id, ordertype_name, description) - Order type info
- fact_sales_ordertype(model_id, country_code, year, month, contracts, ordertype_id) - Sales by order type

JOIN RELATIONSHIPS:
- fact_sales.model_id = dim_model.model_id
- fact_sales.country_code = dim_country.country_code
- fact_sales_ordertype.model_id = dim_model.model_id
- fact_sales_ordertype.country_code = dim_country.country_code
- fact_sales_ordertype.ordertype_id = dim_ordertype.ordertype_id

QUERY PATTERNS & EXAMPLES:

1. For model-specific queries, JOIN with dim_model:
SELECT fs.year, fs.month, SUM(fs.contracts) AS total_sales
FROM fact_sales fs
JOIN dim_model dm ON fs.model_id = dm.model_id
WHERE dm.model_name = 'RAV4 HEV' AND fs.year = 2024
GROUP BY fs.year, fs.month ORDER BY fs.month;

2. For country/region queries, JOIN with dim_country:
SELECT dc.country, SUM(fs.contracts) AS total_sales
FROM fact_sales fs
JOIN dim_country dc ON fs.country_code = dc.country_code
WHERE dc.region = 'Europe' AND fs.year = 2024
GROUP BY dc.country;

3. For combined model + country queries:
SELECT fs.year, fs.month, SUM(fs.contracts) AS total_sales
FROM fact_sales fs
JOIN dim_model dm ON fs.model_id = dm.model_id
JOIN dim_country dc ON fs.country_code = dc.country_code
WHERE dm.model_name LIKE 'RAV4%' AND dc.country = 'Germany' AND fs.year = 2024
GROUP BY fs.year, fs.month ORDER BY fs.month;

4. For powertrain analysis:
SELECT dm.powertrain, SUM(fs.contracts) AS total_sales
FROM fact_sales fs
JOIN dim_model dm ON fs.model_id = dm.model_id
WHERE fs.year = 2024
GROUP BY dm.powertrain;

IMPORTANT NOTES:
- Always use JOINs when you need model names, country names, or other dimension attributes
- Use LIKE 'RAV4%' for RAV4 variants (RAV4, RAV4 HEV, etc.)
- Country codes: DE=Germany, US=United States, etc.
- Use SUM(contracts) for total sales volumes
- Always GROUP BY when using aggregation functions"""
)
def execute_sql(query: str) -> str:
    """
    Use this tool to answer questions about sales, time, country/region, model, powertrain, or any data that can be retrieved via SQL queries from the sales database (SQLite).

    Args:
        query: The SQL query string
    Returns:
        Query result as plain text table, or error message
    """
    # This will be handled specially in the custom call_tools function to avoid blocking
    return _sync_execute_sql(query)


def _sync_execute_sql(query: str) -> str:
    """Synchronous SQL execution helper."""
    conn = None
    try:
        # TODO: This is a workaround. Force correct common table name mistakes. To be improved with proper test suite.
        corrected_query = query.replace("sales_data", "fact_sales")

        conn = _get_sqlite_connection()
        if not conn:
            return "SQL execution error: Could not connect to database"

        df = pd.read_sql_query(corrected_query, conn)
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL execution error: {e}"
    finally:
        # Always close the connection to prevent resource leaks
        if conn:
            conn.close()


@tool
def get_sql_schema() -> str:
    """
    Use this tool to get the schema (table and column names) of the sales database (SQLite). This helps you write correct SQL queries.
    Returns:
        A string listing all tables and their columns in the database.
    """
    # This will be handled specially in the custom call_tools function to avoid blocking
    return _sync_get_schema()


def _sync_get_schema() -> str:
    """Synchronous schema retrieval helper."""
    conn = None
    try:
        conn = _get_sqlite_connection()
        if not conn:
            return "Error retrieving schema: Could not connect to database"

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        schema_str = ""
        for table_name in tables:
            schema_str += f"\nTable: {table_name}\n"
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for col in columns:
                schema_str += f"  - {col[1]} ({col[2]})\n"
        return schema_str.strip()
    except Exception as e:
        return f"Error retrieving schema: {e}"
    finally:
        # Always close the connection to prevent resource leaks
        if conn:
            conn.close()


@tool
def list_available_documents() -> List[Dict[str, Any]]:
    """
    List all available documents in the database with statistics.
    Returns:
        List of document information including filename, pages, and chunks
    """
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return []
    try:
        # Get a sample of documents to analyze metadata
        sample_docs = vectorstore.similarity_search("document", k=100)  # Get larger sample

        # Organize by filename
        file_stats = {}
        for doc in sample_docs:
            metadata = doc.metadata
            filename = metadata.get("filename", "Unknown")
            page = metadata.get("page_number", metadata.get("page", 0))

            if filename not in file_stats:
                file_stats[filename] = {"pages": set(), "chunks": 0, "total_characters": 0}

            file_stats[filename]["pages"].add(page)
            file_stats[filename]["chunks"] += 1
            file_stats[filename]["total_characters"] += len(doc.page_content)

        # Convert to list format
        documents = []
        for filename, stats in file_stats.items():
            doc_info = {
                "filename": filename,
                "pages": len(stats["pages"]),
                "chunks": stats["chunks"],
                "total_characters": stats["total_characters"],
            }
            documents.append(doc_info)

        logger.info(f"Listed {len(documents)} available documents")
        return documents

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []


@tool(
    description="""Search within a specific document for relevant content.

IMPORTANT: When you use this tool to answer a question, you MUST:
1. Provide a comprehensive answer based on the retrieved content from the specified document

Args:
    filename: Name of the document to search in
    query: The search query string
    limit: Maximum number of results to return (default: 3)
Returns:
    List of relevant chunks from the specified document with page numbers for citation"""
)
def search_in_document(filename: str, query: str, limit: int = 3) -> List[SearchResult]:
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return []
    try:
        # Get more results to filter by filename
        results_with_scores = vectorstore.similarity_search_with_score(query=query, k=20)

        # Filter results by filename and take top matches
        filtered_results = []
        for doc, score in results_with_scores:
            doc_filename = doc.metadata.get("filename", "")
            if filename.lower() in doc_filename.lower() or doc_filename.lower() in filename.lower():
                filtered_results.append((doc, score))
                if len(filtered_results) >= limit:
                    break

        search_results = []
        for doc, score in filtered_results:
            metadata = doc.metadata
            result = SearchResult(
                filename=metadata.get("filename", "Unknown"),
                page=int(metadata.get("page_number", metadata.get("page", 0))),
                chunk_index=int(metadata.get("chunk_index", 0)),
                content=doc.page_content,
                relevance_score=float(1.0 - score),  # Convert distance to similarity score
                chunk_id=metadata.get("chunk_id", ""),
            )
            search_results.append(result)

        logger.info(
            f"🔍 ChromaDB search found {len(search_results)} results in {filename} for query: '{query}'"
        )
        return search_results

    except Exception as e:
        logger.error(f"ChromaDB search error in document {filename}: {e}")
        return []


# Expose tools for LangGraph
TOOLS: List[Callable[..., Any]] = [
    search_documents,
    execute_sql,
    get_sql_schema,
    list_available_documents,
    search_in_document,
]


# def create_toyota_tools(chroma_db_path: str = str(CHROMA_DB_PATH), sqlite_path: str = str(DEFAULT_SQLITE_PATH)):
#     """
#     Create Toyota-specific tools for document search and SQL queries.
#     This function is kept for backward compatibility.

#     Args:
#         chroma_db_path: Path to ChromaDB database directory
#         sqlite_path: Path to SQLite database

#     Returns:
#         List of tools for the agent
#     """
#     return TOOLS


def get_vectorstore() -> Optional[Chroma]:
    """
    Get the ChromaDB vectorstore instance.

    Returns:
        ChromaDB vectorstore instance or None if not available
    """
    return _get_vectorstore()


def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the ChromaDB collection.

    Returns:
        Dictionary containing collection statistics
    """
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return {"error": "ChromaDB not available"}

    try:
        collection = vectorstore._collection
        return {
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "total_documents": collection.count(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "database_path": str(settings.CHROMA_DB_PATH),
        }
    except Exception as e:
        return {"error": f"Failed to get collection stats: {e}"}
