"""This module provides Toyota RAG tools for document search and SQL queries.

These tools are designed for Toyota/Lexus vehicle information and sales data analysis.
"""

import logging
import sqlite3
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Callable

import lancedb
import pandas as pd
from langchain_core.tools import tool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default database paths
DEFAULT_DB_PATH = "./lancedb"
DEFAULT_SQLITE_PATH = "toyota_sales.db"


# Async helper functions for SQLite operations
async def _execute_sql_query_async(query: str, db_path: str = DEFAULT_SQLITE_PATH) -> str:
    """Execute SQL query asynchronously to avoid blocking the event loop."""
    def _sync_execute():
        try:
            # Auto-fix common table name mistakes
            corrected_query = query.replace("sales_data", "fact_sales")
            
            conn = sqlite3.connect(db_path, check_same_thread=False)
            df = pd.read_sql_query(corrected_query, conn)
            conn.close()
            return df.to_string(index=False)
        except Exception as e:
            return f"SQL execution error: {e}"
    
    # Run the blocking operation in a separate thread
    return await asyncio.to_thread(_sync_execute)


async def _get_sql_schema_async(db_path: str = DEFAULT_SQLITE_PATH) -> str:
    """Get SQL schema asynchronously to avoid blocking the event loop."""
    def _sync_get_schema():
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
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
            conn.close()
            return schema_str.strip()
        except Exception as e:
            return f"Error retrieving schema: {e}"
    
    # Run the blocking operation in a separate thread
    return await asyncio.to_thread(_sync_get_schema)

class SearchResult(BaseModel):
    """Schema for search results."""
    filename: str = Field(description="Name of the document file")
    page: int = Field(description="Page number in the document")
    chunk_index: int = Field(description="Chunk index within the document")
    content: str = Field(description="Text content of the chunk")
    relevance_score: float = Field(description="Relevance score for the search")


# Initialize database connections at module level
try:
    if Path(DEFAULT_DB_PATH).exists():
        db = lancedb.connect(DEFAULT_DB_PATH)
        table = db.open_table("documents")
        sqlite_conn = sqlite3.connect(DEFAULT_SQLITE_PATH)
        
        # Ensure FTS index exists for hybrid search
        try:
            # Try to create FTS index on text field (will skip if already exists)
            table.create_fts_index("text", replace=False)
            logger.info("✅ FTS index verified for hybrid search")
        except Exception as fts_error:
            if "already exists" in str(fts_error).lower():
                logger.info("✅ FTS index already exists")
            else:
                logger.warning(f"⚠️ Could not create FTS index: {fts_error}")
        
        logger.info(f"✅ Connected to databases: {DEFAULT_DB_PATH}, {DEFAULT_SQLITE_PATH}")
    else:
        # Create dummy connections for import-time compatibility
        db = None
        table = None
        sqlite_conn = None
        logger.warning(f"⚠️ Databases not found, tools will not function until databases are available")
except Exception as e:
    db = None
    table = None
    sqlite_conn = None
    logger.warning(f"⚠️ Failed to connect to databases: {e}")


@tool(description="""Use this tool to answer questions about warranty terms, policy clauses, or owner's manual content by searching the document database.

IMPORTANT: When you use this tool to answer a question, you MUST:
1. Provide a comprehensive answer based on the retrieved content
2. Always cite your sources at the end of your response using this format:

**Sources:**
- [Document Name, Page X]
- [Document Name, Page Y]

Example response format:
"According to the warranty policy, [your answer here based on retrieved content].

**Sources:**
- [Warranty_Policy_Appendix.pdf, Page 5]
- [Contract_Toyota_2023.pdf, Page 12]"

Args:
    query: The search query string
    limit: Maximum number of results to return (default: 5)
Returns:
    List of relevant document chunks with metadata including filename and page numbers for citation""")
def search_documents(query: str, limit: int = 5) -> List[SearchResult]:
    if not table:
        return []
    try:
        # Use hybrid search which combines vector (semantic) and full-text search
        # This uses both OpenAI embeddings and BM25 scoring with RRF ranking
        results = table.search(query, query_type="hybrid", vector_column_name="vector", fts_columns=["text"]).limit(limit).to_pandas()
        search_results = []
        for _, row in results.iterrows():
            result = SearchResult(
                filename=row['filename'],
                page=int(row['page']),
                chunk_index=int(row['chunk_index']),
                content=row['text'],
                relevance_score=float(row.get('_relevance_score', row.get('_distance', 0.0)))
            )
            search_results.append(result)
        logger.info(f"🔍 Hybrid search found {len(search_results)} results for query: '{query}'")
        return search_results
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        # Fallback to vector search if hybrid fails
        try:
            logger.info("Falling back to vector search...")
            results = table.search(query, query_type="fts", fts_columns=["text"]).limit(limit).to_pandas()
            search_results = []
            for _, row in results.iterrows():
                result = SearchResult(
                    filename=row['filename'],
                    page=int(row['page']),
                    chunk_index=int(row['chunk_index']),
                    content=row['text'],
                    relevance_score=float(row.get('_distance', 0.0))
                )
                search_results.append(result)
            logger.info(f"📊 Vector search found {len(search_results)} results as fallback")
            return search_results
        except Exception as fallback_error:
            logger.error(f"Vector search fallback also failed: {fallback_error}")
            return []


@tool(description="""Use this tool to answer questions about sales, time, country/region, model, powertrain, or any data that can be retrieved via SQL queries from the sales database (SQLite).

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
- Always GROUP BY when using aggregation functions""")
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
    try:
        # Auto-fix common table name mistakes
        corrected_query = query.replace("sales_data", "fact_sales")
        
        conn = sqlite3.connect(DEFAULT_SQLITE_PATH, check_same_thread=False)
        df = pd.read_sql_query(corrected_query, conn)
        conn.close()
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL execution error: {e}"


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
    try:
        conn = sqlite3.connect(DEFAULT_SQLITE_PATH, check_same_thread=False)
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
        conn.close()
        return schema_str.strip()
    except Exception as e:
        return f"Error retrieving schema: {e}"


@tool
def list_available_documents() -> List[Dict[str, Any]]:
    """
    List all available documents in the database with statistics.
    Returns:
        List of document information including filename, pages, and chunks
    """
    if not table:
        return []
    try:
        df = table.to_pandas()
        file_stats = df.groupby('filename').agg({
            'chunk_index': 'count',
            'char_count': 'sum',
            'page': 'nunique'
        }).rename(columns={'chunk_index': 'chunks', 'page': 'pages'})
        documents = []
        for filename, stats in file_stats.iterrows():
            doc_info = {
                'filename': filename,
                'pages': int(stats['pages']),
                'chunks': int(stats['chunks']),
                'total_characters': int(stats['char_count'])
            }
            documents.append(doc_info)
        logger.info(f"Listed {len(documents)} available documents")
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []


@tool(description="""Search within a specific document for relevant content.

IMPORTANT: When you use this tool to answer a question, you MUST:
1. Provide a comprehensive answer based on the retrieved content from the specified document
2. Always cite your source at the end of your response using this format:

**Source:**
- [Document Name, Page X]
- [Document Name, Page Y]

Example response format:
"Based on the Toyota contract, [your answer here based on retrieved content].

**Source:**
- [Contract_Toyota_2023.pdf, Page 8]
- [Contract_Toyota_2023.pdf, Page 15]"

Args:
    filename: Name of the document to search in
    query: The search query string  
    limit: Maximum number of results to return (default: 3)
Returns:
    List of relevant chunks from the specified document with page numbers for citation""")
def search_in_document(filename: str, query: str, limit: int = 3) -> List[SearchResult]:
    if not table:
        return []
    try:
        # Use hybrid search for better semantic + keyword matching within specific document
        all_results = table.search(query, query_type="hybrid").limit(50).to_pandas()
        filtered_results = all_results[all_results['filename'] == filename].head(limit)
        search_results = []
        for _, row in filtered_results.iterrows():
            result = SearchResult(
                filename=row['filename'],
                page=int(row['page']),
                chunk_index=int(row['chunk_index']),
                content=row['text'],
                relevance_score=float(row.get('_relevance_score', row.get('_distance', 0.0)))
            )
            search_results.append(result)
        logger.info(f"🔍 Hybrid search found {len(search_results)} results in {filename} for query: '{query}'")
        return search_results
    except Exception as e:
        logger.error(f"Hybrid search error in document {filename}: {e}")
        # Fallback to vector search if hybrid fails
        try:
            logger.info(f"Falling back to vector search for document {filename}...")
            all_results = table.search(query, query_type="vector").limit(50).to_pandas()
            filtered_results = all_results[all_results['filename'] == filename].head(limit)
            search_results = []
            for _, row in filtered_results.iterrows():
                result = SearchResult(
                    filename=row['filename'],
                    page=int(row['page']),
                    chunk_index=int(row['chunk_index']),
                    content=row['text'],
                    relevance_score=float(row.get('_distance', 0.0))
                )
                search_results.append(result)
            logger.info(f"📊 Vector search found {len(search_results)} results in {filename} as fallback")
            return search_results
        except Exception as fallback_error:
            logger.error(f"Vector search fallback also failed for {filename}: {fallback_error}")
            return []


# Expose tools for LangGraph
TOOLS: List[Callable[..., Any]] = [
    search_documents, 
    execute_sql, 
    get_sql_schema, 
    list_available_documents, 
    search_in_document
]


def create_toyota_tools(db_path: str = DEFAULT_DB_PATH, sqlite_path: str = DEFAULT_SQLITE_PATH):
    """
    Create Toyota-specific tools for document search and SQL queries.
    This function is kept for backward compatibility.
    
    Args:
        db_path: Path to LanceDB database
        sqlite_path: Path to SQLite database
    
    Returns:
        List of tools for the agent
    """
    return TOOLS
