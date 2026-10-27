import os
import re
import hashlib
import streamlit as st

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Software Development Agent",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Software Development Planning Agent")
st.write(
    "Break development tasks into smaller steps using "
    "LangGraph, Gemini AI, and FAISS memory."
)

MEMORY_DIR = "programming_tools_faiss"


# ---------------- LOCAL EMBEDDINGS ----------------
# Creates local vectors without another API key.
class LocalEmbeddings(Embeddings):
    def __init__(self, dimension=128):
        self.dimension = dimension

    def embed_query(self, text):
        vector = [0.0] * self.dimension

        for word in re.findall(r"\w+", text.lower()):
            index = int(
                hashlib.md5(word.encode()).hexdigest(), 16
            ) % self.dimension
            vector[index] += 1.0

        norm = sum(x * x for x in vector) ** 0.5
        if norm:
            vector = [x / norm for x in vector]

        return vector

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


embeddings = LocalEmbeddings()


# ---------------- GEMINI CONFIGURATION ----------------
st.sidebar.header("Gemini Configuration")

api_key = st.sidebar.text_input(
    "Gemini API key",
    type="password",
    help="Enter your key privately. Do not share it."
)

model_name = st.sidebar.selectbox(
    "Select Gemini model",
    ["gemini-2.5-flash", "gemini-2.0-flash"]
)

st.sidebar.markdown(
    "[Get a Gemini API key](https://aistudio.google.com/apikey)"
)


def get_llm():
    if not api_key.strip():
        return None

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key.strip(),
        temperature=0.2
    )


# ---------------- FAISS MEMORY ----------------
def load_memory():
    index_path = os.path.join(MEMORY_DIR, "index.faiss")

    if os.path.exists(index_path):
        return FAISS.load_local(
            MEMORY_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return None


def retrieve_tools(user_id):
    store = load_memory()

    if store is None:
        return "No saved programming preferences found."

    try:
        docs = store.similarity_search(
            f"Programming tools preferred by user {user_id}",
            k=10,
            filter={"user_id": user_id}
        )

        if not docs:
            return "No saved programming preferences found."

        return "\n".join(doc.page_content for doc in docs)

    except Exception:
        return "No saved programming preferences found."


def save_tools(user_id, preferences):
    os.makedirs(MEMORY_DIR, exist_ok=True)

    document = Document(
        page_content=preferences,
        metadata={"user_id": user_id}
    )

    store = load_memory()

    if store is None:
        store = FAISS.from_documents(
            [document], embeddings
        )
    else:
        store.add_documents([document])

    store.save_local(MEMORY_DIR)


# ---------------- GRAPH STATE ----------------
class DevelopmentState(TypedDict, total=False):
    user_id: str
    task: str
    skill_level: str
    language: str
    tools: str
    previous_preferences: str
    constraints: str
    plan: str


# ---------------- PLANNER AGENT ----------------
def planner_agent(state: DevelopmentState):
    llm = get_llm()

    if llm is None:
        return {
            "plan": (
                "Please enter your Gemini API key in the sidebar "
                "to generate a personalized development plan."
            )
        }

    prompt = f"""
You are a software development planning agent.

Break the user's software-development task into small,
ordered, actionable steps.

USER TASK:
{state['task']}

PROGRAMMING LANGUAGE:
{state['language']}

SKILL LEVEL:
{state['skill_level']}

USER'S CURRENT TOOL PREFERENCES:
{state['tools']}

PREVIOUSLY SAVED PREFERENCES FROM FAISS:
{state['previous_preferences']}

ADDITIONAL CONSTRAINTS:
{state['constraints']}

OUTPUT FORMAT:

1. Task summary
2. Recommended tools and why they fit the preferences
3. Numbered development steps, from setup to testing
4. For every step, include:
   - Goal
   - Action to perform
   - Expected result
5. Testing and debugging checklist
6. Final completion checklist

Make the steps specific to the requested task.
Respect the user's skill level and constraints.
Reuse remembered tools when appropriate, but do not force
an unsuitable tool.
Do not claim to execute code or create files.
Provide a concise explanation of decisions, not private
internal reasoning. Mention assumptions when necessary.
"""

    try:
        response = llm.invoke(prompt)
        return {"plan": str(response.content)}

    except Exception as error:
        return {
            "plan": f"Gemini API error: {error}"
        }


# ---------------- LANGGRAPH WORKFLOW ----------------
def build_graph():
    graph = StateGraph(DevelopmentState)

    graph.add_node("Planner", planner_agent)
    graph.set_entry_point("Planner")
    graph.add_edge("Planner", END)

    return graph.compile()


agent = build_graph()


# ---------------- DASHBOARD INPUTS ----------------
st.subheader("📝 Development Task Details")

user_id = st.text_input(
    "User ID",
    value="user001"
)

task = st.text_area(
    "Describe your software-development task",
    placeholder=(
        "Example: Build a student attendance management "
        "system using Python and Flask."
    ),
    height=110
)

col1, col2 = st.columns(2)

with col1:
    language = st.selectbox(
        "Preferred programming language",
        [
            "Python",
            "Java",
            "JavaScript",
            "C++",
            "C#",
            "Other"
        ]
    )

    skill_level = st.selectbox(
        "Experience level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

with col2:
    tools = st.text_input(
        "Preferred programming tools",
        placeholder="e.g. PyCharm, Flask, SQLite, Git"
    )

    constraints = st.text_area(
        "Additional requirements",
        placeholder=(
            "e.g. beginner-friendly, free tools, "
            "must include a dashboard"
        ),
        height=90
    )


# ---------------- GENERATE PLAN ----------------
if st.button("🚀 Break Task into Steps", type="primary"):
    if not user_id.strip():
        st.error("Please enter a User ID.")
    elif not task.strip():
        st.error("Please describe your development task.")
    elif not api_key.strip():
        st.warning(
            "Enter your Gemini API key in the sidebar first."
        )
    else:
        with st.spinner(
            "Retrieving preferences and creating your plan..."
        ):
            previous = retrieve_tools(user_id.strip())

            state = {
                "user_id": user_id.strip(),
                "task": task.strip(),
                "language": language,
                "skill_level": skill_level,
                "tools": tools or "No new tools specified",
                "previous_preferences": previous,
                "constraints": (
                    constraints or "No additional constraints"
                )
            }

            try:
                result = agent.invoke(state)
                st.session_state["dev_plan"] = result["plan"]
                st.session_state["dev_user_id"] = user_id.strip()
                st.session_state["dev_preferences"] = (
                    f"Preferred language: {language}; "
                    f"Skill level: {skill_level}; "
                    f"Programming tools: {tools or 'Not specified'}"
                )

            except Exception as error:
                st.error(f"Could not generate plan: {error}")


# ---------------- DISPLAY PLAN ----------------
if "dev_plan" in st.session_state:
    st.divider()
    st.subheader("📋 Generated Development Plan")
    st.markdown(st.session_state["dev_plan"])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Tool Preferences"):
            try:
                save_tools(
                    st.session_state["dev_user_id"],
                    st.session_state["dev_preferences"]
                )
                st.success(
                    "Preferences saved to FAISS memory. "
                    "Use the same User ID next time."
                )
            except Exception as error:
                st.error(f"Could not save preferences: {error}")

    with col2:
        st.download_button(
            "⬇️ Download Development Plan",
            data=st.session_state["dev_plan"],
            file_name="development_plan.md",
            mime="text/markdown"
        )

st.divider()
st.caption("Powered by LangGraph • Gemini AI • FAISS")
