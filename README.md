# Toyota/Lexus RAG Assistant

A simple RAG assistant that combines vehicle sales data and documents to answer automotive questions. Built with LangGraph, LanceDB, and SQLite.

## What it does

This assistant can:

- **Context-Aware Responses**: Handles Toyota-specific questions only.
- **Query Sales Data**: Answer questions about vehicle sales based on SQL database.
- **Search Documents**: Capable of finding relevant information in manuals, contracts, and warranties documents.
- **Tool Orchestration**: Capable of answer complex questions combining different tools.

## How it works

The assistant uses a multi-step LangGraph workflow with intelligent routing.
Given the user question and chat history as input this is how the agent works:

**Workflow Steps:**

1. **Safety Check**: OpenAI Moderation API filters harmful content
2. **Query Analysis**: LLM classifies the question type and intent
3. **Context Aware Routing**: Routes to appropriate response path:
   - **Toyota-specific**: Uses tools (SQL/documents) to answer
   - **Needs clarification**: Asks for more specific information
   - **Off-topic**: Politely redirects to Toyota/Lexus topics
4. **Tool Loop**: For Toyota questions, iterates between model and tools until complete

![Agent Architecture](media/agent_architecture.png)


## Query Routing & Safety

### Intelligent Query Classification

The assistant uses a multi-step routing system to handle different types of queries:

**Toyota-Specific Queries**: Questions about Toyota/Lexus vehicles, sales data, manuals, warranties
- Routes to tool selection (SQL database or document search)
- Iterates between model and tools until complete answer is generated

**Need More Information**: Vague or unclear questions that need clarification
- Generates specific follow-up questions to guide the user
- Helps narrow down to actionable Toyota-related queries

**Off-Topic Queries**: General questions unrelated to Toyota/Lexus
- Provides polite redirection back to Toyota topics
- Maintains focus on the assistant's domain expertise

### Safety & Security

**Content Moderation**: Uses OpenAI's Moderation API to filter harmful content including sexual content, harassment, hate speech, violence, self-harm, and illicit activities. Flagged content is blocked before processing.

**SQL Injection Protection**:
- Read-only database connections prevent destructive operations
- Only SELECT queries allowed - blocks DROP, DELETE, UPDATE, INSERT, CREATE
- Database rejects any unauthorized statement types
- Note: For production, consider predefined query methods for maximum security

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Setup

1. **Install dependencies**
```bash
uv sync
```

2. **Set environment variable**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

3. **Setup databases**
```bash
make setup-db
```

### Run the app

```bash
# Web interface
make run

# Development mode
make dev
```

Visit `http://localhost:8000` to chat with the assistant.

![Toyota RAG Assistant UI](media/run_ui.png)

## What's included

### Sample Data
- **Sales Data** (SQLite): Vehicle sales by model, country, and date
- **Documents** (LanceDB): Toyota manuals, contracts, and warranty policies

## Configuration

Only one environment variable needed:
```bash
OPENAI_API_KEY=your-api-key

# optional
DEFAULT_MODEL="openai/gpt-4.1-mini"
MAX_SEARCH_RESULTS=10
```

## Project Structure

```
src/
├── assistant/            # Core agent
│   ├── graph.py         # LangGraph workflow
│   ├── tools.py         # SQL and document search tools
│   ├── state.py         # Agent state management
│   ├── context.py       # Configuration
│   ├── prompts.py       # System prompts
│   └── guardrails.py    # Safety features
└── chainlit_app.py      # Web interface

scripts/                 # Database setup and ingestion scripts
├── structured_data_ingestion_pipeline.py       # Ingest structured data into SQLite
├── unstructured_data_ingestion_pipeline.py     # Ingest pdf documents into ChromaDB
data/                    # Sample CSV data
docs/                    # Sample PDF documents
```

## Try these questions

**Using structured sales data:**
- "What were the RAV4 sales in Germany in 2024?"
- "Show me the top countries by vehicle sales"

**Using unstructured documents:**
- "What is the Toyota warranty coverage?"
- "Where is the tire repair kit in the UX?"

**Hybrid:**
- "Compare RAV4 sales and summarize its warranty"

## Document Ingestion Options

### ChromaDB Pipeline (Enhanced)

For advanced document processing with ChromaDB:

```bash
# Ensure dependencies are synced
uv sync

# Quick start with the enhanced pipeline
cd scripts
./run_chroma_pipeline.sh demo

# Run tests
./run_chroma_pipeline.sh test

# Basic ingestion
./run_chroma_pipeline.sh ingest

# See all options
./run_chroma_pipeline.sh help
```

The enhanced ChromaDB pipeline offers:
- **Semantic-aware chunking** with configurable overlap
- **Rich metadata extraction** for better retrieval
- **Batch processing** with error handling
- **Multiple configuration presets** (dev/prod/test/hp)
- **Comprehensive testing** and verification

See [`scripts/README_ChromaDB.md`](scripts/README_ChromaDB.md) for detailed documentation.

### LanceDB Pipeline (Current)

The current implementation uses LanceDB:

```bash
make setup-db  # Includes LanceDB ingestion
```

## Development

```bash
# Code quality
make lint format

# Run tests
make test

# Debug mode
make dev  # Opens LangGraph Studio
```

![LangGraph Studio Development](media/studio_dev.png)

## Troubleshooting

**Database issues:**
```bash
make setup-db  # Recreate databases
```
---

**Built with**: LangGraph, LanceDB, OpenAI, Chainlit
