#!/usr/bin/env python3
"""
Chainlit UI for Toyota/Lexus Assistant.

This module provides a Chainlit-based web interface for the Toyota/Lexus RAG assistant.
It maintains conversation history to enable multi-turn conversations with the LangGraph agent.

Key features:
- Maintains conversation history in user session
- Passes complete message history to LangGraph agent for context-aware responses
- Supports streaming responses from the agent
- Tracks conversation state across multiple user interactions
"""

import logging

import chainlit as cl

# Add src directory to path to import the agent
# sys.path.append(str(Path(__file__).parent))
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from assistant import graph
from assistant.context import Context
from assistant.utils import load_starters_from_csv
from core.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress OpenAI HTTP request logs for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

# System prompts are now centralized in prompts.py


@cl.set_starters
async def set_starters():
    """Set sample conversation starter messages to demonstrate what we can do."""
    try:
        # Load conversation starters from file
        conversation_starters = load_starters_from_csv(settings.STARTERS_CSV_PATH)
        logger.info(
            f"Loaded {len(conversation_starters)} conversation starter messages from csv file"
        )

        # Create Chainlit Starters
        cl_starters = [
            cl.Starter(label=item["label"], message=item["message"])
            for item in conversation_starters
        ]

        return cl_starters

    except Exception as e:
        logger.error(f"Error loading starters from csv file: {e}")
        # Fallback to a basic starter if CSV loading fails
        return [
            cl.Starter(
                label="Toyota vs Lexus warranty",
                message="Compare Toyota vs Lexus warranty",
            ),
            cl.Starter(
                label="RAV4 sales in Germany in 2024",
                message="What was the monthly RAV4 sales in Germany in 2024?",
            ),
        ]


@cl.on_chat_start
async def start():
    """Initialize the Toyota/Lexus RAG agent when a new chat session starts."""
    try:
        logger.info("Starting new chat session")

        # Initialize context for the agent (uses default system prompt from prompts.py)
        context = Context(model=settings.DEFAULT_MODEL)

        logger.info(f"Context initialized with model: {context.model}")

        # Initialize the agent (graph is already compiled)
        agent = graph

        # Initialize conversation history storage
        conversation_history = []

        # Store in session for use in message handler
        cl.user_session.set("agent", agent)
        cl.user_session.set("context", context)
        cl.user_session.set("conversation_history", conversation_history)

        logger.info("Agent, context, and conversation history stored in session successfully")
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
    conversation_history = cl.user_session.get("conversation_history", [])

    if not agent or not context:
        await cl.Message(content="Agent not initialized. Please refresh the page.").send()
        return

    # Add the new user message to conversation history
    user_message = HumanMessage(content=message.content)
    conversation_history.append(user_message)

    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    # Keep track of the assistant's response to add to history later
    assistant_response_content = ""

    # Stream messages from the graph, passing the entire conversation history
    async for msg, metadata in agent.astream(
        {"messages": conversation_history},
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
            assistant_response_content += msg.content

    # Add the assistant's response to conversation history
    if assistant_response_content:
        assistant_message = AIMessage(content=assistant_response_content)
        conversation_history.append(assistant_message)

        # Update the conversation history in the session
        cl.user_session.set("conversation_history", conversation_history)

        logger.info(f"Updated conversation history. Total messages: {len(conversation_history)}")

    await final_answer.send()


@cl.on_stop
async def stop():
    """Clean up when chat stops."""
    logger.info("Chat session ended")


if __name__ == "__main__":
    logger.info("Starting Chainlit app...")
    cl.run()
