# Toyota/Lexus RAG Assistant

A production-ready agentic system that intelligently combines structured sales data and unstructured documents to answer automotive industry questions. Built with LangGraph, LanceDB, and advanced routing capabilities.

## 🎯 Overview

This intelligent assistant demonstrates a sophisticated RAG (Retrieval-Augmented Generation) system that can:

- **Query Sales Data**: Execute SQL queries on structured vehicle sales, models, countries, and order types
- **Search Documents**: Extract information from contracts, warranty policies, and owner's manuals using hybrid vector/full-text search
- **Intelligent Routing**: Automatically classify queries and route to appropriate tools or provide clarification
- **Safety Integration**: Content moderation and guardrails for production use
- **Hybrid Analysis**: Combine multiple data sources for comprehensive insights

## 🏗️ Architecture

### Core Components

- **LangGraph Workflow**: Intelligent query routing and tool orchestration
- **Hybrid Search**: OpenAI embeddings + BM25 full-text search with LanceDB
- **SQL Engine**: SQLite with structured sales data analysis
- **Safety Layer**: OpenAI Moderation API integration
- **Async Processing**: Non-blocking SQLite operations for production deployment

### System Flow

```
 Safety Check →  Query Analysis →  Route Decision
                                    ├─ Toyota → Hybrid Search + SQL Tools
                                    ├─ More Info → Clarification Request
                                    └─ Off-topic → Polite Redirection
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-toyota
```

2. **Set up Python environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
# Or using uv (recommended):
uv sync
```

4. **Set up environment variables**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

5. **Initialize databases**
```bash
# Ingest PDF documents to LanceDB
python scripts/simple_pdf_to_lancedb.py

# Ingest CSV sales data to SQLite
python scripts/ingest_to_sqlite.py
```

### Running the Application

#### Option 1: Chainlit Web Interface
```bash
chainlit run src/chainlit_app.py
```
Access the web interface at `http://localhost:8000`

#### Option 2: LangGraph Studio (Development)
```bash
langgraph dev
```
Access LangGraph Studio at `http://localhost:8123`

#### Option 3: Direct Python Testing
```bash
python test_agent.py
```

## 📊 Data Sources

### Structured Data (SQLite)
- **Sales Data**: Vehicle sales by model, country, month, year
- **Dimension Tables**: Countries, models, order types
- **Schema**: Star schema optimized for analytical queries

### Unstructured Data (LanceDB)
- **Vehicle Manuals**: Owner's manuals for various Toyota/Lexus models
- **Contracts**: Toyota and Lexus dealer contracts
- **Warranties**: Warranty policy documents and appendices

## 🛠️ Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your-api-key          # Required for LLM and embeddings
OPENAI_MODEL=gpt-4o-mini            # Default model
LANCEDB_PATH=./lancedb              # Vector database path
SQLITE_PATH=toyota_sales.db         # Sales data path
```

### Context Configuration
Key configurable parameters in `src/react_agent/context.py`:
- `model`: LLM model for reasoning
- `system_prompt`: Main assistant behavior
- `router_system_prompt`: Query classification prompts
- `max_search_results`: Number of document results

## 🔧 Development

### Project Structure
```
src/
├── react_agent/           # Core agent implementation
│   ├── graph.py          # LangGraph workflow definition
│   ├── tools.py          # RAG and SQL tools
│   ├── state.py          # State management
│   ├── context.py        # Configuration
│   ├── prompts.py        # System prompts
│   └── guardrails.py     # Safety integration
├── chainlit_app.py       # Web interface
└── ...

scripts/                  # Data ingestion and utilities
data/                     # Raw CSV sales data
docs/                     # PDF documents for RAG
lancedb/                  # Vector database files
```

### Adding New Tools
1. Define tool function with `@tool` decorator in `tools.py`
2. Add to `TOOLS` list

### Customizing Prompts
Edit prompts in `src/react_agent/prompts.py`:
- `SYSTEM_PROMPT`: Main assistant behavior
- `ROUTER_SYSTEM_PROMPT`: Query classification
- `MORE_INFO_SYSTEM_PROMPT`: Clarification requests
- `GENERAL_SYSTEM_PROMPT`: Off-topic responses

## 🧪 Example Queries

### Sales Data Analysis
```
"What were the monthly RAV4 HEV sales in Germany during 2024?"
"Compare Toyota vs Lexus SUV sales in Western Europe"
"Show me the top 5 countries by total vehicle sales"
```

### Document Search
```
"What is the standard Toyota warranty coverage in Europe?"
"Where is the tire repair kit located in the UX model?"
"What are the maintenance requirements for hybrid vehicles?"
```

### Hybrid Analysis
```
"Compare RAV4 sales performance and summarize its warranty coverage"
"Analyze Lexus market share and explain their customer care policies"
```

## 🛡️ Safety Features

- **Input Moderation**: OpenAI Moderation API for content safety
- **Query Routing**: Intelligent classification and appropriate responses
- **Error Handling**: Graceful degradation and fallback mechanisms
- **Rate Limiting**: Built-in protections for API usage

## 🔍 Monitoring & Debugging

### State Tracking
The system tracks:
- `executed_sql_queries`: All SQL queries executed
- `retrieved_documents`: Documents used for responses
- `router`: Query classification decisions
- `safety`: Safety assessment results

### Logging
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/chainlit_app.py
```

### LangGraph Studio
Use LangGraph Studio for visual debugging:
- Step-by-step execution
- State inspection
- Message flow analysis

## 📈 Performance Considerations

### Database Optimization
- **LanceDB**: Automatic FTS indexing for hybrid search
- **SQLite**: Optimized schema with proper indexing
- **Async Operations**: Non-blocking database operations

### Scaling
- **Thread Pool**: SQLite operations run in separate threads
- **Connection Pooling**: Efficient database connection management
- **Caching**: Vector embeddings cached in LanceDB

## 🔧 Troubleshooting

### Common Issues

**1. Missing OpenAI API Key**
```bash
export OPENAI_API_KEY="your-key-here"
```

**2. Database Not Found**
```bash
# Re-run ingestion scripts
python scripts/simple_pdf_to_lancedb.py
python scripts/ingest_to_sqlite.py
```

### Debug Mode
```bash
# Run with detailed logging
LOG_LEVEL=DEBUG chainlit run src/chainlit_app.py
```

### Development Setup

**Quick Setup:**
```bash
# Complete development environment setup
make dev-setup
```

**Manual Setup:**
```bash
# Install development dependencies
uv sync

# Install pre-commit hooks
make setup-pre-commit

# Initialize databases
make setup-databases
```

**Code Quality:**
```bash
# Run linting
make lint

# Format code
make format

# Type checking
make type-check

# Run all checks
make check-all

# Run tests
make test
```

**Pre-commit Hooks:**
The project uses pre-commit hooks to ensure code quality:
- **Ruff**: Fast Python linter and formatter
- **Type checking**: MyPy static type checking
- **Security**: Bandit security linting
- **Import sorting**: isort for clean imports
- **Basic checks**: Trailing whitespace, file endings, etc.

Hooks run automatically on `git commit` and can be run manually:
```bash
# Run on all files
make pre-commit-run

# Run on staged files only
pre-commit run
```
---

**Built with**: LangGraph, LanceDB, OpenAI, Chainlit, SQLite, Python 3.9+
