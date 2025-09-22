#!/bin/bash
set -e

echo "🚀 Starting Toyota RAG Assistant..."

# Function to check if SQLite database exists and has tables
check_sqlite_db() {
    local db_path="$1"
    if [ ! -f "$db_path" ]; then
        echo "❌ SQLite database not found: $db_path"
        return 1
    fi

    # Check if database has tables
    table_count=$(sqlite3 "$db_path" "SELECT count(*) FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "0")
    if [ "$table_count" -eq 0 ]; then
        echo "❌ SQLite database exists but has no tables: $db_path"
        return 1
    fi

    echo "✅ SQLite database OK: $db_path ($table_count tables)"
    return 0
}

# Function to check if vector database exists
check_vector_db() {
    local db_type="$1"
    local db_path="$2"

    if [ ! -d "$db_path" ]; then
        echo "❌ $db_type database not found: $db_path"
        return 1
    fi

    # Check if directory has any files (basic check)
    if [ -z "$(ls -A "$db_path" 2>/dev/null)" ]; then
        echo "❌ $db_type database directory is empty: $db_path"
        return 1
    fi

    echo "✅ $db_type database OK: $db_path"
    return 0
}

# Check required environment variables
echo "🔑 Checking environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY environment variable is required"
    exit 1
fi
echo "✅ Environment variables OK"

# Database paths
SQLITE_DB="./toyota_sales.db"
CHROMA_DB="./chroma_db"

# Check databases
echo "🔍 Checking databases..."

databases_need_setup=false

# Check SQLite database
if ! check_sqlite_db "$SQLITE_DB"; then
    databases_need_setup=true
fi

# Check ChromaDB vector database
if ! check_vector_db "ChromaDB" "$CHROMA_DB"; then
    databases_need_setup=true
fi

# Setup databases if needed
if [ "$databases_need_setup" = true ]; then
    echo "🔧 Setting up databases..."

    # Run database setup
    if ! make setup-db; then
        echo "❌ ERROR: Database setup failed"
        exit 1
    fi

    echo "✅ Database setup completed"
else
    echo "✅ All databases are available"
fi

# Start the application
echo "🌟 Starting Chainlit application..."
exec uv run chainlit run src/chainlit_app.py --host 0.0.0.0 --port 8000
