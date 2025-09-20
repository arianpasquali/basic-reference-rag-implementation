#!/usr/bin/env python3
"""
Chainlit UI for Toyota/Lexus Assistant.
"""

import logging

import chainlit as cl

# Add src directory to path to import the agent
# sys.path.append(str(Path(__file__).parent))
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from assistant import graph
from assistant.context import Context

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Toyota-specific system prompt
TOYOTA_SYSTEM_PROMPT = """You are a helpful assistant for Toyota/Lexus vehicle information and sales data.

CRITICAL CITATION REQUIREMENTS:
- When you use document search tools (search_documents, search_in_document) to answer questions, you MUST always cite your sources at the end of your response.
- Use this exact format for citations:

**Sources:**
- [Document Name, Page X]
- [Document Name, Page Y]

- For single document searches, use "Source:" (singular)
- Always include the document filename and page number
- Place citations at the very end of your response

Example:
"According to the warranty policy, coverage extends for 3 years or 36,000 miles, whichever comes first.

**Sources:**
- [Warranty_Policy_Appendix.pdf, Page 5]
- [Contract_Toyota_2023.pdf, Page 12]"

NEVER provide document-based answers without proper citations."""


@cl.set_starters
async def set_starters():
    """Set starter messages for common queries."""
    return [
        cl.Starter(
            label="Monthly RAV4 sales in Germany in 2024",
            message="Can you help me get the monthly sales of the RAV4 in Germany in 2024?",
        ),
        cl.Starter(
            label="Toyota warranty for Europe",
            message="What is the standard Toyota warranty for Europe?",
        ),
        cl.Starter(
            label="Toyota sales in Belgium in 2025",
            message="Can you help me get the monthly sales of the Toyota in Belgium in 2025?",
        ),
        cl.Starter(
            label="RAV4 maintenance schedule",
            message="What is the maintenance schedule for my RAV4?",
        ),
        cl.Starter(
            label="Engine oil check procedure",
            message="How do I check the engine oil level in my Toyota vehicle?",
        ),
        cl.Starter(
            label="Tire repair kit location",
            message="Where is the tire repair kit located in my vehicle?",
        ),
        cl.Starter(
            label="Compare Toyota vs Lexus warranty",
            message="What are the key warranty differences between Toyota and Lexus vehicles?",
        ),
    ]


@cl.on_chat_start
async def start():
    """Initialize the Toyota/Lexus RAG agent when a new chat session starts."""
    try:
        logger.info("Starting new chat session")

        # Initialize context for the agent
        context = Context(system_prompt=TOYOTA_SYSTEM_PROMPT, model="openai/gpt-4.1-mini")

        logger.info(f"Context initialized with model: {context.model}")

        # Initialize the agent (graph is already compiled)
        agent = graph

        # Store in session for use in message handler
        cl.user_session.set("agent", agent)
        cl.user_session.set("context", context)

        logger.info("Agent and context stored in session successfully")
        logger.info("Chat session initialization complete")

    except Exception as e:
        logger.error(f"Error during chat initialization: {e}")
        await cl.Message(
            content=f"**Error:** Failed to initialize agent: {e!s}\n\n"
            f"Please refresh the page and try again."
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with LangGraph streaming."""
    agent = cl.user_session.get("agent")
    context = cl.user_session.get("context")

    if not agent or not context:
        await cl.Message(content="Agent not initialized. Please refresh the page.").send()
        return

    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    # Stream messages from the graph, filtering for final responses
    async for msg, metadata in agent.astream(
        {"messages": [HumanMessage(content=message.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
        context=context,
    ):
        # Stream tokens from final response nodes
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata.get("langgraph_node")
            in ["call_model", "ask_for_more_info", "respond_to_offtopic_question"]
        ):
            await final_answer.stream_token(msg.content)

    await final_answer.send()


@cl.on_stop
async def stop():
    """Clean up when chat stops."""
    logger.info("Chat session ended")


if __name__ == "__main__":
    logger.info("Starting Chainlit app...")
    cl.run()
