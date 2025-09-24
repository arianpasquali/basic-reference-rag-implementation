"""Secure SQL tools for Toyota RAG Assistant.

This module provides secure replacements for the original SQL tools,
using parameterized queries and LLM-based intent parsing.
"""

import logging
from typing import Any, Callable, Dict, List

from langchain_core.tools import tool

from .query_parser import query_intent_parser
from .secure_sql import secure_sql_executor
from .sql_validator import sql_security_validator

logger = logging.getLogger(__name__)


@tool(
    description="""Use this tool to answer questions about Toyota/Lexus sales data using secure, parameterized queries.

This tool accepts natural language questions about:
- Sales by model (e.g., "RAV4 sales in Germany 2024")
- Sales by country/region (e.g., "Toyota sales in Europe 2024")
- Powertrain analysis (e.g., "hybrid vehicle sales 2024")
- Sales trends and comparisons (e.g., "monthly sales trends 2024")
- Top performers (e.g., "top 10 models by sales")
- Model comparisons (e.g., "Toyota vs Lexus sales")

The tool automatically converts your question into secure SQL queries with proper parameterization.
All queries use read-only database access and are protected against SQL injection.

Examples:
- "What were RAV4 sales in Germany in 2024?"
- "Show me Toyota sales by powertrain in 2024"
- "Compare Lexus sales across European countries"
- "What are the top 5 performing models this year?"
- "Show me monthly sales trends for hybrid vehicles"

The tool supports years 2020-2024 and includes data for major markets worldwide.
"""
)
def execute_sql_secure(question: str) -> str:
    """
    Execute secure SQL queries based on natural language questions.

    Args:
        question: Natural language question about sales data
    Returns:
        Query results or error message
    """
    try:
        # Step 1: Input validation and security screening
        is_valid, reason, warnings = sql_security_validator.validate_input(question)

        if not is_valid:
            logger.warning(f"Input validation failed: {reason} for question: {question[:100]}")
            return f"Input validation error: {reason}. Please rephrase your question."

        if warnings:
            logger.info(f"Input warnings: {warnings} for question: {question[:100]}")

        # Step 2: Sanitize input
        sanitized_question = sql_security_validator.sanitize_input(question)

        # Step 3: Calculate security score
        security_score = sql_security_validator.get_security_score(sanitized_question)

        if security_score < 0.3:
            logger.warning(f"Low security score: {security_score} for question: {question[:100]}")
            return (
                "Security validation failed: Question appears potentially unsafe. Please rephrase."
            )

        # Step 4: Parse the natural language question using LLM
        query_type, params = query_intent_parser.parse_query_intent(sanitized_question)

        # Step 5: Execute secure parameterized query
        result = secure_sql_executor.execute_secure_query(query_type, params)

        # Log for security monitoring and debugging
        logger.info(
            f"Secure SQL executed - Type: {query_type.value}, Security Score: {security_score:.2f}, Question: {question[:100]}"
        )

        return result

    except Exception as e:
        logger.error(f"Secure SQL execution failed: {e}")
        return "SQL execution error: Unable to process query securely. Please try rephrasing your question."


@tool(
    description="""Get database schema information securely.

This tool provides information about the available tables and their structure in the Toyota sales database.
Use this when you need to understand what data is available or how the database is organized.

The database contains:
- fact_sales: Main sales data with contracts by model, country, and time
- dim_model: Vehicle model information (names, brands, powertrains)
- dim_country: Country and region information
- dim_ordertype: Order type classifications
- fact_sales_ordertype: Sales data with order type details

All schema queries are executed securely without exposing sensitive system information.
"""
)
def get_sql_schema_secure() -> str:
    """Get database schema information securely."""
    try:
        from .secure_sql import QueryParameters, QueryType

        params = QueryParameters()
        result = secure_sql_executor.execute_secure_query(QueryType.SCHEMA_INFO, params)

        logger.info("Schema information requested")
        return result

    except Exception as e:
        logger.error(f"Schema query failed: {e}")
        return "Error retrieving schema information securely."


@tool(
    description="""Get information about available secure query types.

This tool returns information about the types of queries that can be executed securely,
along with examples of how to phrase questions for each query type.

Use this tool when you want to understand what kinds of sales analysis questions
can be answered with the secure SQL system.
"""
)
def get_available_query_types() -> str:
    """Get information about available secure query types."""
    try:
        query_types = secure_sql_executor.get_available_query_types()

        info = "Available Secure Query Types:\n\n"

        type_descriptions = {
            "sales_by_model": "Sales data for specific vehicle models (e.g., 'RAV4 sales in 2024')",
            "sales_by_country": "Sales data for specific countries (e.g., 'Toyota sales in Germany')",
            "sales_by_region": "Sales data by geographic regions (e.g., 'European market performance')",
            "sales_trend": "Time-based sales trends (e.g., 'monthly sales trends 2024')",
            "model_comparison": "Compare different models (e.g., 'Toyota vs Lexus sales')",
            "powertrain_analysis": "Analysis by powertrain type (e.g., 'hybrid vehicle performance')",
            "top_performers": "Top performing models/countries (e.g., 'top 10 models by sales')",
            "schema_info": "Database structure information",
        }

        for query_type in query_types:
            description = type_descriptions.get(query_type, "Sales data analysis")
            info += f"• {query_type}: {description}\n"

        info += "\nAll queries use parameterized execution for security and support natural language input."

        logger.info("Query types information requested")
        return info

    except Exception as e:
        logger.error(f"Failed to get query types: {e}")
        return "Error retrieving query type information."


# Secure tools list for export
SECURE_SQL_TOOLS: List[Callable[..., Any]] = [
    execute_sql_secure,
    get_sql_schema_secure,
    get_available_query_types,
]


def get_tool_usage_stats() -> Dict[str, Any]:
    """Get usage statistics for secure SQL tools (for monitoring)."""
    return {
        "available_tools": len(SECURE_SQL_TOOLS),
        "tool_names": [tool.name for tool in SECURE_SQL_TOOLS],
        "security_level": "HIGH - Parameterized queries with LLM intent parsing",
        "supported_query_types": secure_sql_executor.get_available_query_types(),
    }
