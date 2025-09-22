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

![Agent Architecture](https://rag-reference-demo.onrender.com/public/agent_architecture.png)

#### Security Trade-offs

- **Input Guardrails**: OpenAI Moderation API filters all incoming messages before processing. It is a simple solution given the time constraints. It is the very minimum we need to have for the majority of the applications.
- **Read-Only Database Access**: SQLite connections use `mode=ro` to prevent data modification. Ideally LLM should not have direct access to database because we risk SQL injection and data leak. Given the time constraint this is an initial implementation that keeps the agent flexible but do not allow any destructive SQL command to be executed. It is important to mention that ata leakage is not protected enough with the current version.
- **Grounded Responses**: Prompt engineering to mitigate hallucination by requiring source attribution. Further reasoning can be implemented but latency would be penalized. In order to keep improving that we need a proper evaluation pipeline in place to iterate properly.
- **Smart routing with Off-topic detection**: Off-topic conversations are a risky vector for Prompt or SQL injection and jailbreaking. We block that with a router node.
