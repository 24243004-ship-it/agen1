
import os
import re
import json
import hashlib
from datetime import date

import streamlit as st
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Daily Task Agent",
    page_icon="✅",
    layout="wide",
)

st.title("✅ Daily Task Planner Agent")
st.write(
    "Create a daily task list using LangGraph ReAct and "
    "remember completed tasks with FAISS."
)

MEMORY_DIR = "completed_tasks_faiss"


# ---------------- LOCAL EMBEDDINGS ----------------
# Local embeddings avoid the need for a separate embedding API.
class LocalEmbeddings(Embeddings):
    def __init__(self, dimension=256):
        self.dimension = dimension

    def embed_query(self, text):
        vector = [0.0] * self.dimension

        for word in re.findall(r"\w+", text.lower()):
            index = int(
                hashlib.md5(word.encode()).hexdigest(), 16
            ) % self.dimension
            vector[index] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]

        return vector

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


embeddings = LocalEmbeddings()


# ---------------- FAISS MEMORY ----------------
def load_memory():
    index_path = os.path.join(MEMORY_DIR, "index.faiss")

    if os.path.exists(index_path):
        return FAISS.load_local(
            MEMORY_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return None


def save_completed_task(user_id, task, completed_date):
    os.makedirs(MEMORY_DIR, exist_ok=True)

    document = Document(
        page_content=(
            f"User ID: {user_id}\n"
            f"Completed task: {task}\n"
            f"Completed date: {completed_date}"
        ),
        metadata={
            "user_id": user_id,
            "completed_date": completed_date,
            "status": "completed",
        },
    )

    store = load_memory()

    if store is None:
        store = FAISS.from_documents(
            [document], embeddings
        )
    else:
        store.add_documents([document])

    store.save_local(MEMORY_DIR)


def search_completed_tasks(user_id, query, k=5):
    store = load_memory()

    if store is None:
        return "No completed-task history has been saved yet."

    results = store.similarity_search(
        query,
        k=k,
        filter={"user_id": user_id},
    )

    if not results:
        return "No matching completed tasks found."

    return "\n\n".join(
        document.page_content for document in results
    )


# ---------------- GEMINI CONFIGURATION ----------------
st.sidebar.header("Gemini API Settings")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
)

model_name = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-2.5-flash", "gemini-2.0-flash"],
)

st.sidebar.markdown(
    "[Get API key](https://aistudio.google.com/apikey)"
)


# ---------------- LANGGRAPH STATE ----------------
class TaskState(TypedDict, total=False):
    user_id: str
    requirements: str
    available_hours: float
    task_history: str
    result: str


# ---------------- AGENT BUILDER ----------------
def build_agent(user_id, api_key, model_name):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
    )

    @tool
    def search_task_history(query: str) -> str:
        """Search the current user's previously completed tasks.
        Use this tool when checking past tasks, repeated activities,
        or earlier progress that may help with today's task list."""
        return search_completed_tasks(user_id, query)

    tools = [search_task_history]

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are a helpful daily task planning agent. "
            "Use search_task_history when past completed tasks are "
            "relevant to the user's requirements. Prepare a practical "
            "daily task list using the requested priorities and "
            "available time. Avoid repeating tasks already completed "
            "unless the user asks for them. Do not invent completed "
            "tasks or claim tasks are complete without confirmation. "
            "Return a clear numbered list with priority, estimated "
            "duration, and a suggested order. Check whether the total "
            "estimated time fits the available hours. If it does not, "
            "identify which tasks should be postponed. Provide concise "
            "decision summaries, not private internal reasoning."
        ),
    )


# ---------------- LANGGRAPH WORKFLOW ----------------
def retrieve_history_node(state: TaskState):
    history = search_completed_tasks(
        state["user_id"],
        state["requirements"],
    )
    return {"task_history": history}


def planner_node(state: TaskState):
    agent = build_agent(
        state["user_id"],
        api_key,
        model_name,
    )

    user_prompt = f"""
Create a daily task list for {date.today()}.

User requirements:
{state['requirements']}

Available hours today:
{state['available_hours']}

Previously retrieved completed-task history:
{state['task_history']}

Use the search_task_history tool if further history lookup is useful.

Return:
1. A short summary
2. Numbered tasks in suggested order
3. Priority and estimated duration for each task
4. A realistic schedule
5. Tasks to postpone if the workload exceeds the available time

Do not mark any task completed. The user will do that in the dashboard.
"""

    response = agent.invoke({
        "messages": [{"role": "user", "content": user_prompt}]
    })

    last_message = response["messages"][-1]
    return {"result": last_message.content}


def build_graph():
    graph = StateGraph(TaskState)

    graph.add_node("RetrieveHistory", retrieve_history_node)
    graph.add_node("PlannerAgent", planner_node)

    graph.set_entry_point("RetrieveHistory")
    graph.add_edge("RetrieveHistory", "PlannerAgent")
    graph.add_edge("PlannerAgent", END)

    return graph.compile()


# ---------------- DASHBOARD ----------------
st.subheader("📝 Your Daily Requirements")

user_id = st.text_input(
    "User ID",
    value="user001",
)

requirements = st.text_area(
    "What do you need to do today?",
    placeholder=(
        "Example:\n"
        "- Complete Python assignment\n"
        "- Study two chapters\n"
        "- Prepare presentation slides\n"
        "- Attend project meeting"
    ),
    height=140,
)

available_hours = st.slider(
    "Available hours today",
    min_value=1,
    max_value=16,
    value=6,
)

if st.button("🚀 Generate Daily Task List", type="primary"):
    if not api_key.strip():
        st.error("Enter your Gemini API key in the sidebar.")
    elif not user_id.strip() or not requirements.strip():
        st.error("Enter your User ID and daily requirements.")
    else:
        try:
            graph = build_graph()

            with st.spinner(
                "Retrieving previous tasks and preparing your plan..."
            ):
                result = graph.invoke({
                    "user_id": user_id.strip(),
                    "requirements": requirements.strip(),
                    "available_hours": available_hours,
                })

            st.session_state["task_plan"] = result["result"]
            st.session_state["task_user_id"] = user_id.strip()

        except Exception as error:
            st.error(f"Could not generate task list: {error}")


# ---------------- DISPLAY PLAN ----------------
if "task_plan" in st.session_state:
    st.divider()
    st.subheader("📋 Today's Suggested Task List")
    st.markdown(st.session_state["task_plan"])

    st.download_button(
        "⬇️ Download Task List",
        data=st.session_state["task_plan"],
        file_name="daily_task_list.md",
        mime="text/markdown",
    )


# ---------------- COMPLETED TASKS ----------------
st.divider()
st.subheader("☑️ Save Completed Tasks")

st.write(
    "Mark tasks you have finished. Each saved task will be "
    "stored in FAISS memory for future queries."
)

completed_user_id = st.text_input(
    "User ID for completed tasks",
    value="user001",
    key="completed_user_id",
)

completed_task = st.text_input(
    "Completed task",
    placeholder="e.g. Finished Python assignment",
)

if st.button("💾 Save Completed Task"):
    if not completed_user_id.strip() or not completed_task.strip():
        st.error("Enter both User ID and completed task.")
    else:
        try:
            save_completed_task(
                completed_user_id.strip(),
                completed_task.strip(),
                str(date.today()),
            )
            st.success("Completed task saved to FAISS memory.")
        except Exception as error:
            st.error(f"Could not save completed task: {error}")


# ---------------- SEARCH MEMORY ----------------
st.divider()
st.subheader("🔍 Retrieve Previous Tasks")

search_user_id = st.text_input(
    "User ID to search",
    value="user001",
    key="search_user_id",
)

search_query = st.text_input(
    "Search query",
    placeholder="e.g. Which assignments did I finish?",
)

if st.button("Search Completed History"):
    if not search_user_id.strip() or not search_query.strip():
        st.error("Enter a User ID and search query.")
    else:
        try:
            history = search_completed_tasks(
                search_user_id.strip(),
                search_query.strip(),
                k=10,
            )
            st.markdown("### Retrieved History")
            st.write(history)
        except Exception as error:
            st.error(f"Could not retrieve history: {error}")


st.caption(
    "Built with LangGraph • LangChain ReAct • FAISS • Gemini AI"
)
