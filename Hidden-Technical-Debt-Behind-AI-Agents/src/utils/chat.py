from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from utils.load_config import LoadConfig
from langgraph.checkpoint.postgres import PostgresSaver
from typing import Dict, Any


CFG = LoadConfig()

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            CFG.chat_llm_system_message,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Ensure that the PostgreSQL tables are initialized on startup
with PostgresSaver.from_conn_string(CFG.db_uri) as setup_checkpointer:
    try:
        setup_checkpointer.setup()
        print("✅ PostgreSQL tables are set up.")
    except Exception as e:
        print(f"⚠️ Setup skipped or failed: {e}")


def call_model(state: MessagesState) -> Dict[str, Any]:
    """
    Given a state (a list of chat messages), this function formats the prompt,
    calls the chat model, and returns the updated state with the model response.

    Args:
        state (MessagesState): The current conversation state/messages.

    Returns:
        Dict[str, Any]: Updated state with the model's response.
    """
    prompt = prompt_template.invoke(state)
    response = CFG.chat_llm.invoke(prompt)
    return {"messages": response}


def build_the_graph() -> StateGraph:
    """
    Builds a LangGraph with a single model node connected to the start.

    Returns:
        StateGraph: The graph definition with nodes and edges.
    """
    graph_builder = StateGraph(state_schema=MessagesState)
    graph_builder.add_edge(START, "model")
    graph_builder.add_node("model", call_model)
    return graph_builder


def run_chat(user_message: str, chat_session_config: Dict[str, Any]) -> str:
    """
    Executes a single turn of chat by invoking the LangGraph with the user message
    and returning the model's response.

    Args:
        user_message (str): The user's input message.
        chat_session_config (Dict[str, Any]): The session-specific configuration, including thread ID.

    Returns:
        str: The model's final response message content.
    """
    with PostgresSaver.from_conn_string(CFG.db_uri) as checkpointer:
        graph_builder = build_the_graph()
        graph = graph_builder.compile(checkpointer=checkpointer)
        input_messages = [HumanMessage(user_message)]
        output = graph.invoke(
            {"messages": input_messages}, chat_session_config)
        return output["messages"][-1].content
