import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import TypedDict, List, Dict, Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

load_dotenv()

MEMORY_DIR = "study_subject_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

st.set_page_config(
    page_title="Multi-Agent Study Planner",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# 1. FAISS MEMORY
# --------------------------------------------------

class LocalEmbeddings(Embeddings):
    """Simple offline embeddings for a demonstration project."""

    def __init__(self, dimension=256):
        self.dimension = dimension

    def _embed(self, text):
        vector = np.zeros(self.dimension, dtype=np.float32)

        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            vector[index] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector.tolist()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings():
    api_key = os.getenv("", "").strip()

    if api_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )
        except Exception:
            pass

    return LocalEmbeddings()


@st.cache_resource
def get_memory():
    embeddings = get_embeddings()
    index_file = Path(MEMORY_DIR) / "index.faiss"
    pickle_file = Path(MEMORY_DIR) / "index.pkl"

    if index_file.exists() and pickle_file.exists():
        try:
            return FAISS.load_local(
                MEMORY_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            return None

    return None


def save_subjects(student_id, subjects):
    """Save a student's selected subjects to persistent FAISS memory."""

    embeddings = get_embeddings()

    text = (
        f"Student ID: {student_id}. "
        f"Previously selected study subjects: {', '.join(subjects)}."
    )

    document = Document(
        page_content=text,
        metadata={
            "student_id": student_id,
            "subjects": json.dumps(subjects),
            "saved_at": datetime.now().isoformat(timespec="seconds")
        }
    )

    store = get_memory()

    if store is None:
        store = FAISS.from_documents([document], embeddings)
    else:
        store.add_documents([document])

    store.save_local(MEMORY_DIR)

    # Reload the index next time to avoid stale cached state.
    get_memory.clear()


def get_previous_subjects(student_id):
    """Retrieve the latest saved subjects for this student."""

    store = get_memory()

    if store is None:
        return None

    try:
        docs = list(store.docstore._dict.values())

        matches = [
            doc for doc in docs
            if doc.metadata.get("student_id") == student_id
        ]

        if not matches:
            return None

        latest = max(
            matches,
            key=lambda doc: doc.metadata.get("saved_at", "")
        )

        return json.loads(latest.metadata["subjects"])

    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


# --------------------------------------------------
# 2. OPTIONAL LANGUAGE MODEL
# --------------------------------------------------

def get_llm():
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return None

    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
    except Exception:
        return None


def ask_llm(instructions, fallback):
    """Use the configured model when available; otherwise use fallback."""

    llm = get_llm()

    if llm is None:
        return fallback

    try:
        response = llm.invoke(instructions)
        return str(response.content)
    except Exception:
        return fallback


# --------------------------------------------------
# 3. LANGGRAPH AGENT STATE
# --------------------------------------------------

class StudyState(TypedDict, total=False):
    student_id: str
    subjects: List[str]
    study_minutes: int
    break_minutes: int
    plan: List[Dict[str, Any]]
    research: Dict[str, str]
    checker_report: str
    approved: bool


# --------------------------------------------------
# 4. PLANNER AGENT
# --------------------------------------------------

def planner_agent(state: StudyState):
    """Divide available study time between selected subjects."""

    subjects = state["subjects"]
    total_minutes = state["study_minutes"]
    break_minutes = state["break_minutes"]

    count = len(subjects)
    base = total_minutes // count
    remainder = total_minutes % count

    plan = []
    elapsed = 0

    for index, subject in enumerate(subjects):
        duration = base + (1 if index < remainder else 0)

        plan.append({
            "Subject": subject,
            "Study time (min)": duration,
            "Break after (min)": (
                break_minutes if index < count - 1 else 0
            ),
            "Task": "Review concepts and practice questions"
        })

        elapsed += duration

    fallback = (
        f"Planner allocated {total_minutes} minutes across "
        f"{count} subjects, with breaks between sessions."
    )

    summary = ask_llm(
        "You are the Planner Agent. Summarize this timetable briefly "
        "without changing the allocation:\n"
        + json.dumps(plan),
        fallback
    )

    return {"plan": plan, "planner_summary": summary}


# --------------------------------------------------
# 5. RESEARCHER AGENT
# --------------------------------------------------

def researcher_agent(state: StudyState):
    """Suggest useful study topics and revision activities."""

    research = {}

    for subject in state["subjects"]:
        fallback = (
            f"For {subject}: review the core concepts, "
            "practice a few questions, and finish with a short self-test."
        )

        result = ask_llm(
            f"You are the Researcher Agent. Suggest three beginner-friendly "
            f"study activities for the subject '{subject}'. "
            "Do not claim to have searched the internet. "
            "Keep the response short and practical.",
            fallback
        )

        research[subject] = result

    return {"research": research}


# --------------------------------------------------
# 6. CHECKER AGENT
# --------------------------------------------------

def checker_agent(state: StudyState):
    """Verify that all subjects and time allocations are valid."""

    plan = state["plan"]
    expected_subjects = state["subjects"]

    planned_subjects = [item["Subject"] for item in plan]

    total = sum(item["Study time (min)"] for item in plan)
    valid_durations = all(
        item["Study time (min)"] > 0 for item in plan
    )

    approved = (
        planned_subjects == expected_subjects
        and total == state["study_minutes"]
        and valid_durations
    )

    if approved:
        report = (
            f"Plan checked successfully. All {len(plan)} subjects "
            f"are included and the total study time is {total} minutes."
        )
    else:
        report = (
            "Plan validation failed. Check the subjects and time allocations."
        )

    return {
        "approved": approved,
        "checker_report": report
    }


# --------------------------------------------------
# 7. CONNECT THE THREE AGENTS WITH LANGGRAPH
# --------------------------------------------------

def build_study_graph():
    graph = StateGraph(StudyState)

    graph.add_node("Planner", planner_agent)
    graph.add_node("Researcher", researcher_agent)
    graph.add_node("Checker", checker_agent)

    graph.set_entry_point("Planner")
    graph.add_edge("Planner", "Researcher")
    graph.add_edge("Researcher", "Checker")
    graph.add_edge("Checker", END)

    return graph.compile()


# --------------------------------------------------
# 8. STREAMLIT DASHBOARD
# --------------------------------------------------

st.title("📚 Multi-Agent Study Planner")
st.caption("LangGraph • Planner • Researcher • Checker • FAISS")

student_id = st.text_input(
    "Student ID",
    value="student01"
).strip()

if not student_id:
    st.warning("Please enter a Student ID.")
    st.stop()

previous_subjects = get_previous_subjects(student_id)

if previous_subjects:
    st.success("Previous subjects retrieved from FAISS memory.")
    default_subjects = ", ".join(previous_subjects)
else:
    st.info("No saved subjects found. Create your first study plan.")
    default_subjects = "Python, Mathematics, English"

st.divider()
st.subheader("⚙️ Study Preferences")

with st.form("study_form"):
    subjects_text = st.text_input(
        "Subjects (comma-separated)",
        value=default_subjects
    )

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider(
            "Available study hours",
            min_value=1,
            max_value=8,
            value=2
        )

    with col2:
        break_minutes = st.selectbox(
            "Break between subjects",
            [5, 10, 15],
            index=1
        )

    submit = st.form_submit_button(
        "Create Study Plan",
        type="primary",
        use_container_width=True
    )

if submit:
    subjects = list(dict.fromkeys(
        subject.strip()
        for subject in subjects_text.split(",")
        if subject.strip()
    ))

    if not subjects:
        st.error("Enter at least one subject.")
    else:
        try:
            save_subjects(student_id, subjects)

            app = build_study_graph()

            result = app.invoke({
                "student_id": student_id,
                "subjects": subjects,
                "study_minutes": study_hours * 60,
                "break_minutes": break_minutes
            })

            st.session_state["study_result"] = result
            st.session_state["result_student"] = student_id

            st.success("Study plan generated!")

        except Exception as error:
            st.error(f"Could not create the study plan: {error}")


# --------------------------------------------------
# 9. DISPLAY AGENT RESULTS
# --------------------------------------------------

result = st.session_state.get("study_result")

if result and st.session_state.get("result_student") == student_id:
    st.divider()
    st.subheader("🗓️ Your Short Study Plan")

    st.dataframe(
        result["plan"],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🧠 Agent Workflow")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📅 Planner")
        st.write(result.get("planner_summary", "Plan created."))

    with col2:
        st.markdown("### 🔎 Researcher")

        for subject, suggestions in result.get("research", {}).items():
            with st.expander(subject):
                st.write(suggestions)

    with col3:
        st.markdown("### ✅ Checker")

        if result.get("approved"):
            st.success(result["checker_report"])
        else:
            st.error(result["checker_report"])

    st.download_button(
        "Download Study Plan",
        data=json.dumps(result["plan"], indent=2),
        file_name="study_plan.json",
        mime="application/json"
    )

st.divider()
st.subheader("💾 Previous Subject Memory")

if previous_subjects:
    st.write(", ".join(previous_subjects))
    st.caption("Stored locally in the FAISS index.")
else:
    st.write("Your selected subjects will be saved after plan creation.")

st.caption(
    "FAISS index folder: study_subject_memory"
)