"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
try:
    from datetime import UTC
except ImportError:
    # Python 3.9 compatibility
    UTC = timezone.utc
from typing import Dict, List, Literal, cast, Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, State, Router
from react_agent.tools import TOOLS, SearchResult
from react_agent.utils import load_chat_model
from react_agent.guardrails import OpenAIModerator, GuardrailsOutput, SafetyAssessment

from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)

# Define the function that calls the model


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


def format_safety_message(safety: GuardrailsOutput) -> AIMessage:
    """Format a safety message when content is flagged as unsafe."""
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def guard_input(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Check input safety using OpenAI moderation."""
    moderator = OpenAIModerator()
    
    # Get the last user message
    user_messages = [msg.content for msg in state.messages if hasattr(msg, 'content')]
    input_text = " ".join(user_messages) if user_messages else ""
    
    safety_output = await moderator.ainvoke(input_text)
    return {"safety": safety_output}


async def block_unsafe_content(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Block unsafe content and return a safety message."""
    safety: GuardrailsOutput = state.safety
    return {"messages": [format_safety_message(safety)]}


async def analyze_and_route_query(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (State): The current state of the conversation.
        runtime (Runtime[Context]): Runtime with context containing model configuration.

    Returns:
        dict: A dictionary containing the 'router' key with the classification result.
    """
    model = load_chat_model(runtime.context.model)
    
    # Convert messages to the proper format, filtering out ToolMessages for routing
    converted_messages = [{"role": "system", "content": runtime.context.router_system_prompt}]
    
    for msg in state.messages:
        # Skip ToolMessages as they're not relevant for routing decisions
        if msg.type not in ["tool", "tool_message"]:
            converted_messages.append({"role": msg.type, "content": msg.content})
    
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(converted_messages)
    )
    
    # Convert Router model to dict for state storage
    router_dict = {"type": response.type, "logic": response.logic}
    
    logger.info(f"🔀 Query routed as: {response.type} - {response.logic}")
    return {"router": router_dict}


async def ask_for_more_info(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (State): The current state of the conversation.
        runtime (Runtime[Context]): Runtime with context containing model configuration.

    Returns:
        dict: A dictionary with a 'messages' key containing the generated response.
    """
    model = load_chat_model(runtime.context.model)
    system_prompt = runtime.context.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    
    # Convert messages to the proper format, filtering out ToolMessages
    converted_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in state.messages:
        # Skip ToolMessages as they're not relevant for clarification requests
        if msg.type not in ["tool", "tool_message"]:
            converted_messages.append({"role": msg.type, "content": msg.content})
    
    response = await model.ainvoke(converted_messages)
    logger.info("❓ Asked user for more information")
    return {"messages": [response]}


async def respond_to_offtopic_question(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate a response to a general query not related to Toyota/Lexus.

    This node is called when the router classifies the query as a general question.

    Args:
        state (State): The current state of the conversation.
        runtime (Runtime[Context]): Runtime with context containing model configuration.

    Returns:
        dict: A dictionary with a 'messages' key containing the generated response.
    """
    model = load_chat_model(runtime.context.model)
    system_prompt = runtime.context.general_system_prompt.format(
        logic=state.router["logic"]
    )
    
    # Convert messages to the proper format, filtering out ToolMessages
    converted_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in state.messages:
        # Skip ToolMessages as they're not relevant for off-topic responses
        if msg.type not in ["tool", "tool_message"]:
            converted_messages.append({"role": msg.type, "content": msg.content})
    
    response = await model.ainvoke(converted_messages)
    logger.info("💬 Responded to off-topic question")
    return {"messages": [response]}


def route_query(state: State) -> Literal["ask_for_more_info", "respond_to_offtopic_question", "call_model"]:
    """Determine the next step based on the query classification.

    Args:
        state (State): The current state with router classification.

    Returns:
        str: The next step to take based on the router type.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    if not state.router:
        # If no router classification, default to normal tool flow
        return "call_model"
    
    router_type = state.router["type"]
    
    if router_type == "toyota":
        logger.info("🚗 Routing to Toyota tool processing")
        return "call_model"
    elif router_type == "more-info":
        logger.info("❓ Routing to ask for more info")
        return "ask_for_more_info"
    elif router_type == "general":
        logger.info("💬 Routing to off-topic question response")
        return "respond_to_offtopic_question"
    else:
        logger.warning(f"⚠️ Unknown router type {router_type}, defaulting to tool processing")
        return "call_model"


async def call_tools(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Call tools and track executed queries and retrieved documents."""
    from react_agent.tools import search_documents, search_in_document, execute_sql, _sync_execute_sql, _sync_get_schema
    
    # Track executed SQL queries and retrieved documents
    executed_queries = list(state.executed_sql_queries) if state.executed_sql_queries else []
    retrieved_docs = list(state.retrieved_documents) if state.retrieved_documents else []
    
    # Get the last AI message with tool calls
    last_ai_message = None
    for msg in reversed(state.messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            last_ai_message = msg
            break
    
    if not last_ai_message or not last_ai_message.tool_calls:
        # Use standard ToolNode if no tool calls to track
        tool_node = ToolNode(TOOLS)
        return await tool_node.ainvoke(state)
    
    # Execute tools and track results
    tool_messages = []
    
    for tool_call in last_ai_message.tool_calls:
        tool_name = tool_call.get('name', '')
        tool_args = tool_call.get('args', {})
        tool_id = tool_call.get('id', '')
        
        try:
            # Execute the tool directly and capture the result
            if tool_name == "search_documents":
                tool_result = search_documents.invoke(tool_args)
                
                # Convert SearchResult objects to Document objects for state tracking
                for search_result in tool_result:
                    if isinstance(search_result, SearchResult):
                        doc = Document(
                            page_content=search_result.content,
                            metadata={
                                'filename': search_result.filename,
                                'page': search_result.page,
                                'chunk_index': search_result.chunk_index,
                                'relevance_score': search_result.relevance_score,
                                'tool_used': tool_name
                            }
                        )
                        retrieved_docs.append(doc)
                        logger.info(f"📄 Tracked document: {search_result.filename} (page {search_result.page})")
                
                # Create tool message with the result
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_message)
                
            elif tool_name == "search_in_document":
                tool_result = search_in_document.invoke(tool_args)
                
                # Convert SearchResult objects to Document objects for state tracking
                for search_result in tool_result:
                    if isinstance(search_result, SearchResult):
                        doc = Document(
                            page_content=search_result.content,
                            metadata={
                                'filename': search_result.filename,
                                'page': search_result.page,
                                'chunk_index': search_result.chunk_index,
                                'relevance_score': search_result.relevance_score,
                                'tool_used': tool_name
                            }
                        )
                        retrieved_docs.append(doc)
                        logger.info(f"📄 Tracked document: {search_result.filename} (page {search_result.page})")
                
                # Create tool message with the result
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_message)
                
            elif tool_name == "execute_sql":
                query = tool_args.get('query', '')
                
                # Use asyncio.to_thread to avoid blocking the event loop
                tool_result = await asyncio.to_thread(_sync_execute_sql, query)
                
                # Track the SQL query
                if query and query not in executed_queries:
                    executed_queries.append(query)
                    logger.info(f"📝 Tracked SQL query: {query[:100]}...")
                
                # Create tool message with the result
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_message)
                
            elif tool_name == "get_sql_schema":
                # Use asyncio.to_thread to avoid blocking the event loop
                tool_result = await asyncio.to_thread(_sync_get_schema)
                
                # Create tool message with the result
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_message)
                
            else:
                # For other tools, use standard execution
                tool = next((t for t in TOOLS if t.name == tool_name), None)
                if tool:
                    tool_result = tool.invoke(tool_args)
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    tool_messages.append(tool_message)
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            # Create error message
            tool_message = ToolMessage(
                content=f"Error executing {tool_name}: {str(e)}",
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)
    
    # Return the tool results along with updated state
    return {
        "messages": tool_messages,
        "executed_sql_queries": executed_queries,
        "retrieved_documents": retrieved_docs
    }


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define all nodes including safety and routing nodes
builder.add_node("guard_input", guard_input)
builder.add_node("analyze_and_route_query", analyze_and_route_query)
builder.add_node("ask_for_more_info", ask_for_more_info)
builder.add_node("respond_to_offtopic_question", respond_to_offtopic_question)
builder.add_node("call_model", call_model)
builder.add_node("tools", call_tools)
builder.add_node("block_unsafe_content", block_unsafe_content)

# Set the entrypoint as safety guard first
builder.add_edge("__start__", "guard_input")


def check_safety(state: State) -> Literal["unsafe", "safe"]:
    """Check if the input is safe or unsafe."""
    safety: GuardrailsOutput = state.safety
    if safety and safety.safety_assessment == SafetyAssessment.UNSAFE:
        return "unsafe"
    return "safe"


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add safety check after guard_input
builder.add_conditional_edges(
    "guard_input", 
    check_safety, 
    {"unsafe": "block_unsafe_content", "safe": "analyze_and_route_query"}
)

# Add routing after safety check
builder.add_conditional_edges(
    "analyze_and_route_query",
    route_query,
    {
        "ask_for_more_info": "ask_for_more_info",
        "respond_to_offtopic_question": "respond_to_offtopic_question", 
        "call_model": "call_model"
    }
)

# Add end edges for routing nodes
builder.add_edge("ask_for_more_info", "__end__")
builder.add_edge("respond_to_offtopic_question", "__end__")

# Always END after blocking unsafe content
builder.add_edge("block_unsafe_content", "__end__")

# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
