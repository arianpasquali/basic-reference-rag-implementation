# Toyota/Lexus RAG Assistant

A simple RAG assistant that combines vehicle sales data and documents to answer automotive questions. Built with LangGraph, LanceDB, and SQLite.

## 🎯 What it does

This assistant can:

- **Query Sales Data**: Answer questions about vehicle sales using SQL
- **Search Documents**: Find information in manuals, contracts, and warranties
- **Smart Routing**: Automatically direct questions to the right tools
- **Safety**: Basic content filtering for safe interactions

## 🏗️ How it works

```
User Question → Safety Check → Route to Tools → Generate Answer
                              ├─ SQL Database (sales data)
                              ├─ Document Search (PDFs)
                              └─ Clarification/Redirect
```

## 🚀 Quick Start

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

## 📊 What's included

### Sample Data
- **Sales Data** (SQLite): Vehicle sales by model, country, and date
- **Documents** (LanceDB): Toyota manuals, contracts, and warranty policies

## 🛠️ Configuration

Only one environment variable needed:
```bash
OPENAI_API_KEY=your-api-key

# optional
DEFAULT_MODEL="openai/gpt-4.1-mini"
```

Other settings in `src/assistant/context.py`:
- `model`: LLM model (default: gpt-4o-mini)
- `max_search_results`: Number of documents to retrieve

## 🔧 Project Structure

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

scripts/                 # Database setup scripts
data/                    # Sample CSV data
docs/                    # Sample PDF documents
```

## 🧪 Try these questions

**Sales data:**
- "What were the RAV4 sales in Germany in 2024?"
- "Show me the top countries by vehicle sales"

**Documents:**
- "What is the Toyota warranty coverage?"
- "Where is the tire repair kit in the UX?"

**Combined:**
- "Compare RAV4 sales and summarize its warranty"

## 🛠️ Development

```bash
# Code quality
make lint format

# Run tests
make test

# Debug mode
make dev  # Opens LangGraph Studio
```

## 🔧 Troubleshooting

**Missing API key:**
```bash
export OPENAI_API_KEY="your-key-here"
```

**Database issues:**
```bash
make setup-db  # Recreate databases
```

---

**Built with**: LangGraph, LanceDB, OpenAI, Chainlit
