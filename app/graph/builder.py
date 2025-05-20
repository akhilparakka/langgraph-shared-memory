from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import merge_message_runs
from trustcall import create_extractor
from app.graph.spy import Spy
from app.graph.utils import extract_tool_info
from app.graph.graph_state import UpdateMemory, Profile, todo
from datetime import datetime
import uuid


model = ChatOpenAI(model="gpt-4o")


MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. 

You are designed to be a companion to a user, helping them keep track of their todo list.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's todo list
3. General instructions for updating the todo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current todo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the todo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the todo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the todo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""


TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update todo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


def task_master(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    namespace = ("todo", user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile, todo=todo, instructions=instructions
    )

    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    return {"messages": [response]}


def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore):
    message = state["messages"][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call["args"]["update_type"] == "user":
            return "update_profile"
        elif tool_call["args"]["update_type"] == "todo":
            return "update_todo"
        elif tool_call["args"]["update_type"] == "instructions":
            return "update_instructions"
        else:
            raise ValueError


def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    namespace = ("profile", user_id)

    existing_items = store.search(namespace)

    tool_name = "Profile"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )
    result = profile_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated profile",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def update_todo(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""

    user_id = config["configurable"]["user_id"]

    namespace = ("todo", user_id)

    existing_items = store.search(namespace)

    tool_name = "todo"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    spy = Spy()

    todo_extractor = create_extractor(
        model, tools=[todo], tool_choice=tool_name, enable_inserts=True
    ).with_listeners(on_end=spy)

    result = todo_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    tool_calls = state["messages"][-1].tool_calls

    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    print(todo_update_msg)
    return {
        "messages": [
            {
                "role": "tool",
                "content": todo_update_msg,
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")

    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )
    new_memory = model.invoke(
        [SystemMessage(content=system_msg)]
        + state["messages"][:-1]
        + [
            HumanMessage(
                content="Please update the instructions based on the conversation"
            )
        ]
    )

    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated instructions",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


builder = StateGraph(MessagesState)
builder.add_node(task_master)
builder.add_node(update_todo)
builder.add_node(update_profile)
builder.add_node(update_instructions)

builder.add_edge(START, "task_master")
builder.add_conditional_edges("task_master", route_message)
builder.add_edge("update_todo", "task_master")
builder.add_edge("update_profile", "task_master")
builder.add_edge("update_instructions", "task_master")

across_thread_memory = InMemoryStore()

within_thread_memory = MemorySaver()

graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)
