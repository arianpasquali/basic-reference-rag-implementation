"""Define the state structures for the agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from pydantic import BaseModel, Field


class Router(BaseModel):
    """Router model for query classification."""

    type: str = Field(description="The type of query: 'toyota', 'more-info', or 'general'")
    logic: str = Field(description="The reasoning for the classification")


@dataclass
class InputState:
    """Defines the input state for the agent

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    safety: Optional[Any] = field(default=None)
    """
    Safety assessment for the conversation.
    """

    # Additional attributes can be added here as needed.
    retrieved_documents: List[Document] = field(default_factory=list)
    """
    Documents retrieved from the database.
    """

    executed_sql_queries: List[str] = field(default_factory=list)
    """
    SQL queries executed by the agent.
    """

    router: Optional[Dict[str, str]] = field(default=None)
    """
    Router classification for the query.
    Contains 'type' and 'logic' fields from Router model.
    """
