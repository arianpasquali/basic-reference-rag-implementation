## Welcome to Motor Industry Assistant

Welcome to the Motor Industry Assistant - a proof-of-concept agentic system that helps answer automotive industry questions by intelligently combining structured sales data and unstructured documents.

### What This Assistant Can Do 

This POC demonstrates an intelligent agent that can:

- **Query Sales Data**: Answer questions about vehicle sales, models, countries, and order types using SQL
- **Search Documents**: Extract information from contracts, warranty policies, and owner's manuals using RAG
- **Hybrid Analysis**: Combine both data sources for comprehensive insights

### Example Questions You Can Ask

- **SQL Queries**: "Monthly RAV4 HEV sales in Germany in 2024"
- **Document Search**: "What is the standard Toyota warranty for Europe?"
- **Owner's Manual**: "Where is the tire repair kit located for the UX?"
- **Hybrid Analysis**: "Compare Toyota vs Lexus SUV sales in Western Europe and summarize warranty differences"

### Architecture Overview

This assistant uses:
- **SQL Tool** for structured sales data queries
- **RAG Tool** for unstructured document search
- **Intelligent Router** to select appropriate tools based on question type