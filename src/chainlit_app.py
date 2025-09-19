#!/usr/bin/env python3
"""
Chainlit UI for Toyota/Lexus RAG Agent.
Decoupled from the agent logic for better separation of concerns.
"""

import chainlit as cl
import logging
import sys
from pathlib import Path

# Add src directory to path to import the agent
sys.path.append(str(Path(__file__).parent))

from react_agent import graph
from react_agent.context import Context
from react_agent.state import InputState
from langchain_core.messages import HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Toyota-specific system prompt
TOYOTA_SYSTEM_PROMPT = """You are a helpful assistant for Toyota/Lexus vehicle information and sales data.

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

NEVER provide document-based answers without proper citations.

System time: {system_time}"""

# Global agent instance
agent = None

@cl.set_starters
async def set_starters():
    """Set starter messages for common queries."""
    return [
        cl.Starter(
            label="Monthly RAV4 sales in Germany in 2024",
            message="Can you help me get the monthly sales of the RAV4 HEV in Germany in 2024?",
            icon="/public/logo_dark.png"
        ),
        cl.Starter(
            label="Toyota warranty for Europe",
            message="What is the standard Toyota warranty for Europe?",
            icon="/public/logo_dark.png"
        ),
        cl.Starter(
            label="RAV4 maintenance schedule",
            message="What is the maintenance schedule for my RAV4?",
            icon="/public/logo_dark.png"
        ),
        cl.Starter(
            label="Engine oil check procedure",
            message="How do I check the engine oil level in my Toyota vehicle?",
            icon="/public/logo_dark.png"
        ),
        cl.Starter(
            label="Tire repair kit location",
            message="Where is the tire repair kit located in my vehicle?",
            icon="/public/logo_dark.png"
        ),
        cl.Starter(
            label="Compare Toyota vs Lexus warranty",
            message="What are the key warranty differences between Toyota and Lexus vehicles?",
            icon="/public/logo_dark.png"
        )
    ]

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    global agent

    # Initialize the agent with Toyota-specific context
    context = Context(
        system_prompt=TOYOTA_SYSTEM_PROMPT,
        model="openai/gpt-4.1-mini",
        max_search_results=5
    )
    
    # For now, use the graph directly and pass context in config
    # The Runtime class may have different initialization requirements
    agent = graph
    
    cl.user_session.set("agent", agent)
    cl.user_session.set("context", context)
    
    # Send welcome message
    # await cl.Message(
    #     content="**Welcome to Motors Industry Assistant!**\n\n"
    #            "I can help you with:\n"
    #            "• Vehicle warranty information\n"
    #            "• Maintenance schedules and procedures\n"
    #            "• Sales data and analytics\n"
    #            "• Owner's manual content\n\n"
    #            "Ask me anything about Toyota or Lexus vehicles!"
    # ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with streaming support."""
    # Get agent and context from session
    agent = cl.user_session.get("agent")
    context = cl.user_session.get("context")
    
    if not agent or not context:
        await cl.Message(
            content="**Error:** Agent not initialized. Please refresh the page."
        ).send()
        return
    
    try:
        logger.info(f"Processing message: '{message.content}'")
        logger.info(f"Agent type: {type(agent)}")
        logger.info(f"Context type: {type(context)}")
        
        # Create initial state with the user's message
        initial_state = InputState(
            messages=[HumanMessage(content=message.content)]
        )
        logger.info(f"Initial state created with {len(initial_state.messages)} messages")
        
        # Create config - context is passed separately as per LangGraph docs
        logger.info(f"Context details: {context}")
        logger.info(f"Context model: {context.model}")
        logger.info(f"Context system_prompt length: {len(context.system_prompt)}")
        
        config = {
            "configurable": {
                "thread_id": cl.context.session.id
            }
        }
        logger.info(f"Config created: {config}")
        logger.info(f"Context to be passed: {context}")
        
        # Initialize response message
        final_answer = cl.Message(content="")
        
        # Show thinking indicator
        async with cl.Step(name="Processing your question...") as step:
            try:
                logger.info("Starting streaming attempt...")
                logger.info(f"About to call agent.astream with:")
                logger.info(f"  - initial_state: {initial_state}")
                logger.info(f"  - config: {config}")
                logger.info(f"  - context: {context}")
                
                # Pass context as separate parameter (LangGraph context approach)
                stream = agent.astream(initial_state, config=config, context=context)
                logger.info(f"Stream object created: {stream}")
                
                async for chunk in stream:
                    logger.debug(f"Received chunk: {chunk}")
                    step.output = "Thinking and searching..."
                    
                    # Handle different types of chunks from LangGraph
                    for node_name, node_data in chunk.items():
                        logger.debug(f"Processing node: {node_name}")
                        if node_name == "call_model" and "messages" in node_data:
                            messages = node_data["messages"]
                            if messages:
                                last_message = messages[-1]
                                if hasattr(last_message, 'content') and last_message.content:
                                    # Clear previous content and set new content
                                    final_answer.content = format_response(last_message.content)
                        
                        elif node_name == "tools":
                            step.output = "Found relevant information..."
            
            except Exception as streaming_error:
                logger.error(f"Streaming failed: {type(streaming_error).__name__}: {streaming_error}")
                logger.error(f"Error details: context={context}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
                # Fallback to synchronous processing
                try:
                    logger.info("Attempting synchronous fallback...")
                    result = await agent.ainvoke(initial_state, config=config, context=context)
                    
                    if result and "messages" in result:
                        last_message = result["messages"][-1]
                        if hasattr(last_message, 'content'):
                            final_answer.content = format_response(last_message.content)
                        else:
                            final_answer.content = "Sorry, I couldn't process your request."
                    else:
                        final_answer.content = "Sorry, I couldn't generate a response."
                        
                except Exception as sync_error:
                    logger.error(f"Sync fallback also failed: {type(sync_error).__name__}: {sync_error}")
                    import traceback
                    logger.error(f"Sync fallback traceback: {traceback.format_exc()}")
                    final_answer.content = f"**Error:** Both streaming and sync failed. Last error: {str(sync_error)}"
        
        # Send the final response
        await final_answer.send()
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await cl.Message(
            content=f"**Sorry, I encountered an error:**\n\n{str(e)}\n\n"
                   f"Please try rephrasing your question or contact support."
        ).send()

def format_response(response: str) -> str:
    """Format the agent's response for better display in Chainlit."""
    
    
    # Format source citations if present
    if "source:" in response.lower() or "page" in response.lower():
        response += "\n\n---\n *Information retrieved from official Toyota/Lexus knowledge base*"
    
    return response

@cl.on_stop
async def stop():
    """Clean up when chat stops."""
    logger.info("Chat session ended")

# Custom actions for the UI
@cl.action_callback("list_documents")
async def list_documents_action(action: cl.Action):
    """Action to list available documents."""
    agent = cl.user_session.get("agent")
    context = cl.user_session.get("context")
    
    if not agent or not context:
        await cl.Message(content="Agent not available").send()
        return
    
    try:
        # Import the tools directly
        from react_agent.tools import list_available_documents
        
        # Call the tool to get documents
        documents = list_available_documents()
        
        if not documents:
            await cl.Message(content="No documents found in the database.").send()
            return
        
        content = "**Available Documents:**\n\n"
        
        for doc in documents:
            content += f"• **{doc['filename']}**\n"
            content += f"  Pages: {doc['pages']}\n"
            content += f"  Chunks: {doc['chunks']}\n"
            content += f"  Characters: {doc['total_characters']:,}\n\n"
        
        await cl.Message(content=content).send()
        
    except Exception as e:
        await cl.Message(content=f"Error listing documents: {e}").send()


if __name__ == "__main__":
    # This allows running the Chainlit app directly
    import uvicorn
    
    print("Starting Chainlit RAG UI...")
    print("Open http://localhost:8000 in your browser")
    
    # chainlit run src/chainlit_rag_ui.py
    pass
