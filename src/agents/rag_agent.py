#!/usr/bin/env python3
"""
LangGraph-based RAG agent for Toyota/Lexus document search.
Decoupled from UI for independent testing and debugging.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Annotated
import lancedb
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
import sqlite3

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# State definition for the agent
class AgentState(MessagesState):
    """State for the RAG agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    search_results: List[Dict[str, Any]] = []
    query: str = ""


class RAGAgent:
    """LangGraph-based RAG agent for document search and question answering.
    Uses LanceDB for vector/document search and SQLite for SQL queries (toyota_sales.db).
    """
    def __init__(self, db_path: str = "./lancedb", model_name: str = "gpt-4.1-mini", sqlite_path: str = "toyota_sales.db"):
        """Initialize the RAG agent."""
        self.db_path = db_path
        self.model_name = model_name
        self.db = None
        self.table = None
        self.llm = None
        self.graph = None
        self.sqlite_path = sqlite_path
        self.sqlite_conn = None
        # Initialize components
        self._connect_to_database()
        self._connect_to_sqlite()
        self._setup_llm()
        self._setup_tools()
        self._build_graph()

    def _connect_to_database(self):
        """Connect to LanceDB database."""
        try:
            if not Path(self.db_path).exists():
                raise FileNotFoundError(f"Database not found at {self.db_path}")
            
            self.db = lancedb.connect(self.db_path)
            self.table = self.db.open_table("documents")
            logger.info(f"✅ Connected to LanceDB at {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise

    def _connect_to_sqlite(self):
        """Connect to the SQLite database for SQL queries."""
        try:
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            logger.info(f"✅ Connected to SQLite at {self.sqlite_path}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to SQLite: {e}")
            raise

    def _setup_llm(self):
        """Setup the language model."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(model=self.model_name, temperature=0)
        logger.info(f"✅ Initialized LLM: {self.model_name}")
    
    def _setup_tools(self):
        """Setup tools for the agent using the centralized tools module."""
        from agents.tools import create_toyota_tools
        self.tools = create_toyota_tools(self.db_path, self.sqlite_path)

    def _build_graph(self):
        """Build the LangGraph workflow using a ReAct agent as the entry point."""
        # Add system message to reinforce citation requirements
        from langchain_core.messages import SystemMessage
        system_message = SystemMessage(content="""You are a helpful assistant for Toyota/Lexus vehicle information and sales data.

CRITICAL CITATION REQUIREMENTS:
- When you use document search tools (search_documents, search_in_document) to answer questions, you MUST always cite your sources at the end of your response.
- Use this exact format for citations:

**Sources:**
- [Document Name, Page X]
- [Document Name, Page Y]

- For single document searches, use "**Source:**" (singular)
- Always include the document filename and page number
- Place citations at the very end of your response

Example:
"According to the warranty policy, coverage extends for 3 years or 36,000 miles, whichever comes first.

**Sources:**
- [Warranty_Policy_Appendix.pdf, Page 5]
- [Contract_Toyota_2023.pdf, Page 12]"

NEVER provide document-based answers without proper citations.""")
        
        # Create the ReAct agent node with system message
        self.react_agent = create_react_agent(
            model=self.llm, 
            tools=self.tools, 
            messages_modifier=system_message.content
        )
        # Build the graph
        workflow = StateGraph(AgentState)
        workflow.add_node("react_agent", self.react_agent)
        workflow.set_entry_point("react_agent")
        workflow.add_edge("react_agent", "__end__")
        self.graph = workflow.compile()
        logger.info("✅ LangGraph ReAct workflow compiled successfully")
        
    def _get_schema_info(self) -> str:
        """Get the schema information from the SQLite database."""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            schema_str = ""
            for table in tables:
                schema_str += f"\nTable: {table}\n"
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                for col in columns:
                    schema_str += f"  - {col[1]} ({col[2]})\n"
            return schema_str.strip()
        except Exception as e:
            return f"Error retrieving schema: {e}"

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Invoke the agent with a user query. The ReAct agent will decide which tool(s) to use.
        """
        try:
            # Always get the schema and prepend to the query
            schema = self._get_schema_info()
            schema_context = f"Database schema:\n{schema}\n\n"
            combined_query = schema_context + query

            initial_state = {
                "messages": [HumanMessage(content=combined_query)],
                "query": query
            }
            result = self.graph.invoke(initial_state)
            final_message = result["messages"][-1]
            # If tool trace is available in result, include it
            tool_trace = result.get("tool_trace", [])
            return {
                "response": final_message.content,
                "messages": result["messages"],
                "query": query,
                "success": True,
                "tool_trace": tool_trace
            }
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "messages": [],
                "query": query,
                "success": False,
                "tool_trace": []
            }

    def stream(self, query: str):
        """
        Stream the agent's response for real-time updates. The ReAct agent will decide which tool(s) to use.
        """
        try:
            # Always get the schema and prepend to the query
            schema = self._get_schema_info()
            schema_context = f"Database schema:\n{schema}\n\n"
            combined_query = schema_context + query

            initial_state = {
                "messages": [HumanMessage(content=combined_query)],
                "query": query
            }
            for chunk in self.graph.stream(initial_state):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming agent response: {e}")
            yield {"error": str(e)}


# --- Expose a compiled graph for LangGraph Studio (like ferranti_assistant.py) ---
from langgraph.graph import StateGraph

def build_rag_agent_graph():
    """Build a RAG agent graph for LangGraph Studio using centralized tools."""
    # Define the workflow for the RAG agent
    workflow = StateGraph(AgentState)
    
    # Default paths for LangGraph Studio
    db_path = "./lancedb"
    sqlite_path = "toyota_sales.db"
    model_name = "gpt-4.1-mini"
    
    # Initialize LLM
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    # Use centralized tools
    from agents.tools import create_toyota_tools
    tools = create_toyota_tools(db_path, sqlite_path)
    
    # Create ReAct agent
    from langgraph.prebuilt import create_react_agent
    react_agent = create_react_agent(model=llm, tools=tools)
    
    # Build workflow
    workflow.add_node("react_agent", react_agent)
    workflow.set_entry_point("react_agent")
    workflow.add_edge("react_agent", "__end__")
    
    return workflow.compile()

# Expose the compiled graph for LangGraph Studio
rag_agent = build_rag_agent_graph()


def main():
    """Main function for testing the agent independently."""
    print("🚀 Toyota/Lexus RAG Agent")
    print("=" * 50)
    
    try:
        # Initialize agent
        agent = RAGAgent()
        print("✅ Agent initialized successfully")
        
        # Expose a compiled graph instance for LangGraph Studio
        rag_agent_instance = RAGAgent()
        rag_agent = rag_agent_instance.graph
        
        # Interactive testing loop
        print("\nType 'quit' to exit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\n🔍 Ask a question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- Ask any question about Toyota/Lexus vehicles")
                    print("- 'list documents' - See available documents")
                    print("- 'search [document] [query]' - Search in specific document")
                    print("- 'quit' - Exit the program")
                    continue
                
                if not user_input:
                    continue
                
                # Invoke agent
                print("\n🤖 Processing...")
                result = agent.invoke(user_input)
                
                if result["success"]:
                    print(f"\n✅ Response:\n{result['response']}")
                else:
                    print(f"\n❌ Error: {result['response']}")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")


if __name__ == "__main__":
    main()
