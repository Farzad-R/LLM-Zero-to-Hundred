from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from utils.load_config import LoadConfig
from langgraph.checkpoint.postgres import PostgresSaver

CFG = LoadConfig()

# Define the function for the tools node


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = CFG.stored_vectordb.similarity_search(query, k=CFG.k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Define the function for the query_or_respond node

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = CFG.rag_llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Define the function for the generate node


def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)
              ] + conversation_messages

    # Run
    response = CFG.rag_llm.invoke(prompt)
    return {"messages": [response]}


def compile_the_graph(tools):
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()
    return graph


def run_rag(user_message: str, chat_session_config: str) -> str:
    """
    Processes a user message using a Retrieval-Augmented Generation (RAG) pipeline
    and returns the generated response.

    Args:
        user_message (str): The user's input message.
        chat_session_config (dict): A dictionary containing configuration parameters
            for the chat session (e.g., thread ID, memory options).

    Returns:
        str: The chatbot's final response message, extracted from the graph's output.
    """
    with PostgresSaver.from_conn_string(CFG.db_uri) as checkpointer:
        tools = ToolNode([retrieve])
        graph = compile_the_graph(tools)
        final_state = graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=chat_session_config
        )
        print("final state:", final_state)

        return final_state["messages"][-1].content
