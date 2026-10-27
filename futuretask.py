
import os
import re
import hashlib
from typing import TypedDict

import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Multi-Agent Task Solver",
    page_icon="🤖",
    layout="wide"
)

DB_DIRECTORY = "task_faiss_memory"


# ---------------- LOCAL EMBEDDINGS ----------------
# Simple local embeddings for storing and retrieving task history.
# A separate embeddings API is not required.

class LocalEmbeddings(Embeddings):
    def __init__(self, dimensions=256):
        self.dimensions = dimensions

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        words = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0

        norm = sum(value * value for value in vector) ** 0.5

        if norm:
            vector = [value / norm for value in vector]

        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


# ---------------- FAISS MEMORY ----------------

def get_vector_store():
    embeddings = LocalEmbeddings()
    index_path = os.path.join(DB_DIRECTORY, "index.faiss")

    if os.path.exists(index_path):
        # Only load a FAISS index created and controlled by this app.
        return FAISS.load_local(
            DB_DIRECTORY,
            embeddings,
            allow_dangerous_deserialization=True
        )

    os.makedirs(DB_DIRECTORY, exist_ok=True)

    initial_document = Document(
        page_content="Initial memory record.",
        metadata={"type": "system"}
    )

    store = FAISS.from_documents(
        [initial_document],
        embeddings
    )

    store.save_local(DB_DIRECTORY)
    return store


def save_task_memory(task, plan, research, final_answer):
    store = get_vector_store()

    content = (
        f"Task: {task}\n\n"
        f"Plan: {plan}\n\n"
        f"Research summary: {research}\n\n"
        f"Final verified response: {final_answer}"
    )

    record_id = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    store.add_documents(
        documents=[
            Document(
                page_content=content,
                metadata={
                    "type": "task_history",
                    "task": task[:500]
                }
            )
        ],
        ids=[record_id]
    )

    store.save_local(DB_DIRECTORY)


def search_task_memory(query, limit=4):
    store = get_vector_store()

    results = store.similarity_search(
        query,
        k=limit
    )

    return [
        document.page_content
        for document in results
        if document.metadata.get("type") == "task_history"
    ]


# ---------------- LANGGRAPH STATE ----------------

class TaskState(TypedDict, total=False):
    task: str
    previous_information: str
    plan: str
    research: str
    final_answer: str


# ---------------- MULTI-AGENT SYSTEM ----------------

def build_task_graph(api_key, model_name):

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )

    # Agent 1: Planner
    def planner_agent(state: TaskState):
        prompt = f"""
You are the Planner Agent in a real-world task-solving system.

User's task:
{state['task']}

Relevant information from previous tasks:
{state['previous_information']}

Break the task into clear, ordered, achievable steps.
Identify the main objective, required resources, constraints,
risks, and expected result.

Return:
1. Objective
2. Ordered action plan
3. Important constraints
4. Expected outcome

Use previous information only when relevant. Do not invent facts.
Show a concise plan, not private internal reasoning.
"""
        response = llm.invoke(prompt)
        return {"plan": response.content}

    # Agent 2: Researcher
    def researcher_agent(state: TaskState):
        prompt = f"""
You are the Researcher Agent.

Original task:
{state['task']}

Plan from the Planner Agent:
{state['plan']}

Previous task information:
{state['previous_information']}

Develop a useful response for each major step in the plan.
Provide relevant facts, practical methods, examples, and
recommendations appropriate to the task.

Rules:
- Do not claim to have searched the live internet.
- Do not invent sources, statistics, or verified facts.
- Clearly flag uncertainty or information that needs checking.
- For calculations, explain the inputs and show the result.
- Keep the response practical and well organized.

Return a structured research draft for the Checker Agent.
"""
        response = llm.invoke(prompt)
        return {"research": response.content}

    # Agent 3: Checker
    def checker_agent(state: TaskState):
        prompt = f"""
You are the Checker Agent. Review the research draft and prepare
the final answer to the user's task.

Original task:
{state['task']}

Plan:
{state['plan']}

Research draft:
{state['research']}

Check for:
1. Relevance to the original task.
2. Coverage of the plan's important steps.
3. Contradictions, arithmetic errors, and unsupported claims.
4. Missing constraints, safety considerations, or practical details.
5. Clarity and usefulness.

Correct issues where possible. If something cannot be verified
from the supplied information, state the limitation rather than
pretending it was verified.

Return the final answer with:
- A short summary
- Actionable steps or recommendations
- Relevant cautions or limitations
- A clear conclusion

Do not expose private internal reasoning.
"""
        response = llm.invoke(prompt)
        return {"final_answer": response.content}

    # Connect the agents into a LangGraph workflow.
    graph = StateGraph(TaskState)

    graph.add_node("Planner", planner_agent)
    graph.add_node("Researcher", researcher_agent)
    graph.add_node("Checker", checker_agent)

    graph.add_edge(START, "Planner")
    graph.add_edge("Planner", "Researcher")
    graph.add_edge("Researcher", "Checker")
    graph.add_edge("Checker", END)

    return graph.compile()


# ---------------- STREAMLIT DASHBOARD ----------------

st.title("🤖 Multi-Agent Real-World Task Solver")

st.write(
    "Enter a task. The Planner breaks it down, the Researcher "
    "develops a solution, and the Checker reviews the final answer."
)

with st.sidebar:
    st.header("⚙️ AI Settings")

    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.0-flash"]
    )

    st.info("Previous tasks are stored locally using FAISS.")


st.subheader("📝 Enter Your Task")

task = st.text_area(
    "What real-world task do you want to solve?",
    value=(
        "Prepare a practical 7-day study schedule for a college student "
        "who has three subjects to revise and four study hours per day."
    ),
    height=130
)

if st.button(
    "🚀 Solve Task",
    type="primary",
    use_container_width=True
):
    if not api_key.strip():
        st.error("Enter your Gemini API key in the sidebar.")
    elif not task.strip():
        st.error("Please enter a task.")
    else:
        try:
            with st.spinner(
                "Planner → Researcher → Checker agents are working..."
            ):
                try:
                    previous_records = search_task_memory(task, limit=4)
                except Exception:
                    previous_records = []

                previous_information = (
                    "\n\n".join(previous_records)
                    if previous_records
                    else "No matching previous task records found."
                )

                app = build_task_graph(
                    api_key.strip(),
                    model_name
                )

                result = app.invoke({
                    "task": task.strip(),
                    "previous_information": previous_information
                })

            st.session_state["task"] = task.strip()
            st.session_state["plan"] = result["plan"]
            st.session_state["research"] = result["research"]
            st.session_state["final_answer"] = result["final_answer"]

            st.success("Task completed and checked!")

        except Exception as error:
            st.error(f"Could not complete the task: {error}")
            st.info(
                "Check your API key, internet connection, model access, "
                "and installed packages."
            )


# Keep results visible after Streamlit reruns.

if st.session_state.get("final_answer"):
    st.divider()

    with st.expander("📋 Planner Agent — Task Breakdown"):
        st.markdown(st.session_state["plan"])

    with st.expander("🔎 Researcher Agent — Research Draft"):
        st.markdown(st.session_state["research"])

    st.subheader("✅ Checker Agent — Final Answer")
    st.markdown(st.session_state["final_answer"])

    st.download_button(
        "⬇️ Download Final Answer",
        data=st.session_state["final_answer"],
        file_name="task_solution.md",
        mime="text/markdown",
        use_container_width=True
    )

    st.button(
        "💾 Save This Task to FAISS Memory",
        key="save_task_button",
        on_click=lambda: save_task_memory(
            st.session_state["task"],
            st.session_state["plan"],
            st.session_state["research"],
            st.session_state["final_answer"]
        )
    )


# ---------------- SEARCH PREVIOUS TASKS ----------------

st.divider()
st.subheader("🧠 Previous Task Memory")

search_query = st.text_input(
    "Search previous tasks",
    placeholder="Example: study timetable or project planning"
)

if st.button("🔍 Search Task Memory", use_container_width=True):
    if not search_query.strip():
        st.warning("Enter a search query.")
    else:
        try:
            records = search_task_memory(search_query, limit=5)

            if records:
                for number, record in enumerate(records, start=1):
                    with st.expander(f"Previous task {number}"):
                        st.markdown(record)
            else:
                st.info("No matching saved tasks found.")

        except Exception as error:
            st.error(f"Memory search failed: {error}")


st.caption(
    "FAISS memory is local. Search uses lightweight local hash embeddings, "
    "so similarity results are approximate. The agents use the AI model's "
    "knowledge and do not independently verify live web information."
)
