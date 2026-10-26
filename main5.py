import os
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

load_dotenv()

API_KEY = os.getenv("", "").strip()
DB_PATH = "assignment_memory_db"

os.makedirs(DB_PATH, exist_ok=True)

st.set_page_config(
    page_title="AI Assignment Planner",
    page_icon="📝",
    layout="wide"
)


# --------------------------------------------------
# 1. EMBEDDINGS AND CHROMADB MEMORY
# --------------------------------------------------

class SimpleEmbeddings(Embeddings):
    """Offline embeddings for basic local memory retrieval."""

    def __init__(self, dimension=256):
        self.dimension = dimension

    def _embed(self, text):
        import hashlib
        import re
        import numpy as np

        vector = np.zeros(self.dimension, dtype=np.float32)

        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            vector[index] += 1.0

        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm

        return vector.tolist()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings():
    if API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=API_KEY
            )
        except Exception:
            pass

    return SimpleEmbeddings()


@st.cache_resource
def get_memory():
    return Chroma(
        collection_name="assignment_preferences",
        embedding_function=get_embeddings(),
        persist_directory=DB_PATH
    )


def save_preferences(student_id, preferences):
    memory = get_memory()

    doc = Document(
        page_content=(
            f"Student ID: {student_id}. "
            f"Preferred task size: {preferences['task_size']}. "
            f"Preferred work time: {preferences['work_time']}. "
            f"Assignment type: {preferences['assignment_type']}. "
            f"Daily available hours: {preferences['daily_hours']}."
        ),
        metadata={
            "student_id": student_id,
            "task_size": preferences["task_size"],
            "work_time": preferences["work_time"],
            "assignment_type": preferences["assignment_type"],
            "daily_hours": int(preferences["daily_hours"]),
            "saved_at": datetime.now().isoformat(timespec="seconds")
        }
    )

    memory.add_documents([doc])


def retrieve_preferences(student_id):
    memory = get_memory()

    result = memory.get(
        where={"student_id": student_id},
        include=["metadatas", "documents"]
    )

    records = result.get("metadatas") or []

    if not records:
        return None

    latest = max(
        records,
        key=lambda item: item.get("saved_at", "")
    )

    return {
        "task_size": latest.get("task_size", "Medium"),
        "work_time": latest.get("work_time", "Evening"),
        "assignment_type": latest.get("assignment_type", "Project"),
        "daily_hours": int(latest.get("daily_hours", 2)),
        "saved_at": latest.get("saved_at", "")
    }


# --------------------------------------------------
# 2. RULE-BASED ASSIGNMENT PLANNER
# --------------------------------------------------

def create_basic_tasks(title, assignment_type, task_size):
    """Fallback task breakdown when no AI API is configured."""

    templates = {
        "Essay": [
            ("Understand the assignment requirements", "High"),
            ("Research the topic and collect references", "High"),
            ("Prepare an outline and main arguments", "High"),
            ("Write the first draft", "High"),
            ("Edit, proofread, and format citations", "Medium"),
            ("Submit the final assignment", "High")
        ],
        "Programming": [
            ("Understand requirements and define features", "High"),
            ("Design the solution and algorithm", "High"),
            ("Set up the project and implement the core code", "High"),
            ("Test features and fix errors", "High"),
            ("Write documentation and examples", "Medium"),
            ("Review and submit the final project", "High")
        ],
        "Research": [
            ("Define the research question and objectives", "High"),
            ("Find credible sources and related studies", "High"),
            ("Organize notes and identify key findings", "High"),
            ("Draft the research report", "High"),
            ("Check evidence, references, and formatting", "Medium"),
            ("Prepare the final submission", "High")
        ],
        "Project": [
            ("Clarify requirements and expected deliverables", "High"),
            ("Research the topic and plan the approach", "High"),
            ("Break the project into modules or sections", "High"),
            ("Complete the main implementation or draft", "High"),
            ("Test, review, and improve the result", "Medium"),
            ("Prepare documentation and final submission", "High")
        ]
    }

    tasks = templates.get(assignment_type, templates["Project"])

    size_multiplier = {
        "Small": 0.7,
        "Medium": 1.0,
        "Large": 1.5
    }.get(task_size, 1.0)

    result = []

    for index, (task, priority) in enumerate(tasks, start=1):
        result.append({
            "id": index,
            "task": task,
            "duration_hours": round(
                max(0.5, (1.0 if index in [1, 6] else 1.5)
                    * size_multiplier),
                1
            ),
            "priority": priority,
            "status": "Pending"
        })

    # Include the actual assignment title in the plan.
    result[0]["task"] += f": {title}"

    return result


# --------------------------------------------------
# 3. LANGCHAIN ASSIGNMENT AGENT
# --------------------------------------------------

def generate_tasks(title, description, assignment_type, task_size):
    """Use a LangChain model when configured; otherwise use fallback."""

    fallback = create_basic_tasks(
        title, assignment_type, task_size
    )

    if not API_KEY:
        return fallback, (
            "Used the built-in planning method because no API key "
            "was configured."
        )

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=API_KEY
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an assignment planning assistant.
Break an assignment into small, actionable tasks.
Plan step by step internally, but return only the final structured result.
Do not reveal private internal chain-of-thought.
Return valid JSON only, with this schema:
{
  "tasks": [
    {
      "task": "short actionable task",
      "duration_hours": 1.0,
      "priority": "High"
    }
  ],
  "rationale": "A short explanation of how the tasks are organized."
}
Create 4 to 8 tasks. Use positive durations.
Priority must be High, Medium, or Low."""
            ),
            (
                "human",
                """Assignment title: {title}
Description: {description}
Type: {assignment_type}
Task size: {task_size}

Make the tasks practical, ordered, and specific to this assignment."""
            )
        ])

        chain = prompt | llm

        response = chain.invoke({
            "title": title,
            "description": description,
            "assignment_type": assignment_type,
            "task_size": task_size
        })

        raw = response.content

        # Handle models that wrap JSON in Markdown fences.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()

        data = json.loads(raw)
        tasks = data["tasks"]

        if not isinstance(tasks, list) or not tasks:
            raise ValueError("The model returned no tasks.")

        clean_tasks = []

        for index, task in enumerate(tasks[:8], start=1):
            duration = float(task["duration_hours"])

            if duration <= 0:
                raise ValueError("Task duration must be positive.")

            priority = task.get("priority", "Medium")
            if priority not in ["High", "Medium", "Low"]:
                priority = "Medium"

            clean_tasks.append({
                "id": index,
                "task": str(task["task"]),
                "duration_hours": duration,
                "priority": priority,
                "status": "Pending"
            })

        return clean_tasks, data.get(
            "rationale",
            "Tasks are organized in a practical completion order."
        )

    except Exception:
        return fallback, (
            "AI generation was unavailable or returned an invalid result. "
            "The built-in planner generated the task list instead."
        )


# --------------------------------------------------
# 4. DASHBOARD
# --------------------------------------------------

st.title("📝 AI Assignment Planner")
st.caption("LangChain • Step-by-Step Task Planning • ChromaDB Memory")

student_id = st.text_input(
    "Student ID",
    value="student01"
).strip()

if not student_id:
    st.warning("Please enter a Student ID.")
    st.stop()

try:
    previous = retrieve_preferences(student_id)
except Exception as error:
    previous = None
    st.warning(f"Could not retrieve memory: {error}")

if previous:
    st.success("Previous assignment preferences retrieved.")
else:
    st.info("New profile. Enter your assignment preferences below.")

default_size = previous["task_size"] if previous else "Medium"
default_time = previous["work_time"] if previous else "Evening"
default_type = previous["assignment_type"] if previous else "Project"
default_hours = previous["daily_hours"] if previous else 2

st.divider()
st.subheader("⚙️ Assignment Details")

with st.form("assignment_form"):
    title = st.text_input(
        "Assignment title",
        placeholder="Example: Student Management System"
    )

    description = st.text_area(
        "Assignment description",
        placeholder="Explain what needs to be completed..."
    )

    col1, col2 = st.columns(2)

    with col1:
        assignment_type = st.selectbox(
            "Assignment type",
            ["Project", "Essay", "Programming", "Research"],
            index=[
                "Project", "Essay", "Programming", "Research"
            ].index(default_type)
        )

        task_size = st.selectbox(
            "Preferred task size",
            ["Small", "Medium", "Large"],
            index=["Small", "Medium", "Large"].index(default_size)
        )

    with col2:
        work_time = st.selectbox(
            "Preferred work time",
            ["Morning", "Afternoon", "Evening", "Night"],
            index=[
                "Morning", "Afternoon", "Evening", "Night"
            ].index(default_time)
        )

        daily_hours = st.slider(
            "Available hours per day",
            min_value=1,
            max_value=12,
            value=max(1, min(12, int(default_hours)))
        )

    generate = st.form_submit_button(
        "Break Assignment into Tasks",
        type="primary",
        use_container_width=True
    )

if generate:
    if not title.strip():
        st.error("Please enter an assignment title.")
    else:
        preferences = {
            "task_size": task_size,
            "work_time": work_time,
            "assignment_type": assignment_type,
            "daily_hours": daily_hours
        }

        try:
            save_preferences(student_id, preferences)

            tasks, rationale = generate_tasks(
                title.strip(),
                description.strip(),
                assignment_type,
                task_size
            )

            st.session_state["assignment_tasks"] = tasks
            st.session_state["assignment_rationale"] = rationale
            st.session_state["assignment_title"] = title.strip()
            st.session_state["assignment_student"] = student_id
            st.session_state["assignment_hours"] = daily_hours

            st.success("Assignment broken into manageable tasks!")

        except Exception as error:
            st.error(f"Could not create the task plan: {error}")


# --------------------------------------------------
# 5. TASK DASHBOARD AND CHECKLIST
# --------------------------------------------------

if (
    st.session_state.get("assignment_tasks")
    and st.session_state.get("assignment_student") == student_id
):
    st.divider()
    st.subheader(
        f"📋 Task Plan: {st.session_state['assignment_title']}"
    )

    tasks = st.session_state["assignment_tasks"]

    total_hours = sum(
        task["duration_hours"] for task in tasks
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tasks", len(tasks))
    c2.metric("Estimated Hours", f"{total_hours:.1f}")
    c3.metric(
        "Available Hours / Day",
        st.session_state["assignment_hours"]
    )

    st.write("**Planning rationale:**", st.session_state["assignment_rationale"])

    st.subheader("Task Checklist")

    for task in tasks:
        key = f"task_{student_id}_{task['id']}"

        checked = st.checkbox(
            f"{task['task']} — {task['duration_hours']} hours "
            f"({task['priority']} priority)",
            key=key
        )

        task["status"] = "Completed" if checked else "Pending"

    completed = sum(
        task["status"] == "Completed" for task in tasks
    )

    st.progress(completed / len(tasks))
    st.write(f"Completed {completed} of {len(tasks)} tasks.")

    st.download_button(
        "Download Assignment Plan",
        data=json.dumps(tasks, indent=2),
        file_name="assignment_plan.json",
        mime="application/json"
    )

st.divider()
st.subheader("🧠 Saved Assignment Preferences")

if previous:
    st.write("**Preferred task size:**", previous["task_size"])
    st.write("**Preferred work time:**", previous["work_time"])
    st.write("**Assignment type:**", previous["assignment_type"])
    st.write("**Available daily hours:**", previous["daily_hours"])
    st.caption(f"Last saved: {previous['saved_at']}")
else:
    st.write("Your preferences will appear here after you create a plan.")

st.caption(
    "ChromaDB persistent memory folder: assignment_memory_db"
)