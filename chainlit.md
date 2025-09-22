## Welcome to Motor Industry Assistant

Welcome to the Motor Industry Assistant. A proof-of-concept agentic system that helps answer automotive industry questions by combining structured sales data and unstructured documents like owner manuals.

### What This Assistant Can Do

This POC demonstrates an agent that can:

- **Query Sales Data**: Answer questions about vehicle sales, models, countries, and order types using SQL
- **Search Documents**: Extract information from contracts, warranty policies, and owner's manuals using RAG
- **Hybrid Analysis**: Combine both data sources to answer complex questions and get insights

### Example Questions You Can Ask

- **SQL Queries**: "Monthly RAV4 HEV sales in Germany in 2024"
- **Document Search**: "What is the standard Toyota warranty for Europe?"
- **Owner's Manual**: "Where is the tire repair kit located for the UX?"
- **Hybrid Analysis**: "Compare Toyota vs Lexus SUV sales in Western Europe and summarize warranty differences"

### Architecture Overview

This assistant uses:
- **LangGraph Agent** with conditional routing and safety guardrails
- **Question Analysis and Routing** to classify questions as Toyota-related, off-topic, or need clarification
- **OpenAI Moderation** for content safety filtering
- **SQL Tools** for structured sales data queries
- **RAG Tools** for unstructured document search

![Agent Architecture](media/agent_architecture.png)
