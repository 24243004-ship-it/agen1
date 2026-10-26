import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any

import numpy as np
import streamlit as st

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

load_dotenv()

MEMORY_DIR = "faiss_timetable_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

st.set_page_config(
    page_title="AI Timetable Agent",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# 1. OFFLINE-FRIENDLY EMBEDDINGS FOR FAISS
# --------------------------------------------------

class SimpleEmbeddings(Embeddings):
    """Lightweight deterministic embeddings for local demo retrieval."""

    def __init__(self, dimension=256):
        self.dimension = dimension

    def _embed(self, text):
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())

        for token in tokens:
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

    return SimpleEmbeddings()


@st.cache_resource
def get_faiss_memory():
    embeddings = get_embeddings()
    index_path = Path(MEMORY_DIR)

    if (index_path / "index.faiss").exists() and (
        index_path / "index.pkl"
    ).exists():
        try:
            return FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            pass

    return None


def save_preference(student_id, subjects, timing, hours, breaks):
    """Persist one preference record in the FAISS index."""

    embeddings = get_embeddings()

    text = (
        f"Student ID {student_id}. Preferred subjects: "
        f"{', '.join(subjects)}. Preferred study timing: {timing}. "
        f"Daily study hours: {hours}. Break duration: {breaks} minutes."
    )

    document = Document(
        page_content=text,
        metadata={
            "student_id": student_id,
            "subjects": json.dumps(subjects),
            "timing": timing,
            "hours": int(hours),
            "breaks": int(breaks),
            "saved_at": datetime.now().isoformat(timespec="seconds")
        }
    )

    store = get_faiss_memory()

    if store is None:
        store = FAISS.from_documents([document], embeddings)
        st.cache_resource.clear()
    else:
        store.add_documents([document])

    store.save_local(MEMORY_DIR)


def retrieve_preferences(student_id):
    """Retrieve this student's most recent stored preference."""

    store = get_faiss_memory()

    if store is None:
        return None

    try:
        records = store.docstore._dict.values()

        matches = [
            doc for doc in records
            if doc.metadata.get("student_id") == student_id
        ]

        if not matches:
            return None

        latest = max(
            matches,
            key=lambda doc: doc.metadata.get("saved_at", "")
        )

        metadata = latest.metadata

        return {
            "subjects": json.loads(metadata["subjects"]),
            "timing": metadata["timing"],
            "hours": int(metadata["hours"]),
            "breaks": int(metadata["breaks"]),
            "saved_at": metadata["saved_at"]
        }

    except Exception:
        return None


# --------------------------------------------------
# 2. TREE-OF-THOUGHT: THREE CANDIDATE TIMETABLES
# --------------------------------------------------

def create_candidate(subjects, timing, hours, breaks, strategy):
    """Create a timetable using one of three planning strategies."""

    strategies = {
        "Balanced": {"duration": 45, "break": breaks},
        "Deep Focus": {"duration": 60, "break": max(5, breaks)},
        "Short Sessions": {"duration": 30, "break": max(5, breaks)}
    }

    config = strategies[strategy]
    session_length = config["duration"]
    break_length = config["break"]

    total_minutes = int(hours * 60)
    sessions = []

    start_hours = {
        "Morning": 6,
        "Afternoon": 13,
        "Evening": 17,
        "Night": 20
    }

    current = datetime.now().replace(
        hour=start_hours[timing],
        minute=0,
        second=0,
        microsecond=0
    )

    subject_index = 0
    elapsed = 0

    while elapsed < total_minutes:
        duration = min(session_length, total_minutes - elapsed)

        # Stop if there isn't enough time for a useful session.
        if duration < 15:
            break

        subject = subjects[subject_index % len(subjects)]
        end = current + timedelta(minutes=duration)

        sessions.append({
            "Time": (
                current.strftime("%I:%M %p")
                + " - "
                + end.strftime("%I:%M %p")
            ),
            "Subject": subject,
            "Study (min)": duration,
            "Break after (min)": (
                break_length
                if elapsed + duration < total_minutes
                else 0
            )
        })

        current = end + timedelta(
            minutes=break_length
            if elapsed + duration < total_minutes
            else 0
        )

        elapsed += duration + (
            break_length
            if elapsed + duration < total_minutes
            else 0
        )
        subject_index += 1

    return {
        "name": strategy,
        "sessions": sessions,
        "study_minutes": sum(
            item["Study (min)"] for item in sessions
        ),
        "break_minutes": sum(
            item["Break after (min)"] for item in sessions
        )
    }


# --------------------------------------------------
# 3. LANGGRAPH WORKFLOW
# --------------------------------------------------

class PlannerState(TypedDict, total=False):
    student_id: str
    subjects: List[str]
    timing: str
    hours: int
    breaks: int
    candidates: List[Dict[str, Any]]
    evaluated: List[Dict[str, Any]]
    selected: Dict[str, Any]


def generate_options(state: PlannerState):
    """Branch into three possible timetable solutions."""

    options = [
        create_candidate(
            state["subjects"], state["timing"],
            state["hours"], state["breaks"], strategy
        )
        for strategy in ["Balanced", "Deep Focus", "Short Sessions"]
    ]

    return {"candidates": options}


def evaluate_options(state: PlannerState):
    """Score each candidate against simple study-planning criteria."""

    evaluated = []

    for candidate in state["candidates"]:
        session_count = len(candidate["sessions"])

        # A transparent heuristic, not a hidden reasoning trace.
        score = 100

        if session_count < 2:
            score -= 10

        if candidate["name"] == "Deep Focus":
            score -= 5  # Longer sessions may cause fatigue.

        if candidate["name"] == "Short Sessions":
            score -= 2  # More transitions can reduce continuity.

        candidate = dict(candidate)
        candidate["score"] = score
        candidate["evaluation"] = (
            f"{session_count} study sessions; "
            f"{candidate['study_minutes']} study minutes; "
            f"{candidate['break_minutes']} break minutes."
        )
        evaluated.append(candidate)

    return {"evaluated": evaluated}


def select_best_option(state: PlannerState):
    """Select the highest-scoring option."""

    best = max(
        state["evaluated"],
        key=lambda item: item["score"]
    )

    return {"selected": best}


def build_graph():
    graph = StateGraph(PlannerState)

    graph.add_node("generate_options", generate_options)
    graph.add_node("evaluate_options", evaluate_options)
    graph.add_node("select_best", select_best_option)

    graph.set_entry_point("generate_options")
    graph.add_edge("generate_options", "evaluate_options")
    graph.add_edge("evaluate_options", "select_best")
    graph.add_edge("select_best", END)

    return graph.compile()


# --------------------------------------------------
# 4. STREAMLIT DASHBOARD
# --------------------------------------------------

st.title("📚 AI Timetable Selection Agent")
st.caption("LangGraph • Tree-of-Thought • FAISS Memory")

student_id = st.text_input(
    "Student ID",
    value="student01"
).strip()

if not student_id:
    st.warning("Please enter a Student ID.")
    st.stop()

previous = retrieve_preferences(student_id)

if previous:
    st.success("Previous preferences retrieved from FAISS.")
else:
    st.info("No saved preferences found. Create your first timetable.")

default_subjects = (
    ", ".join(previous["subjects"])
    if previous else "Python, Mathematics, English"
)
default_timing = previous["timing"] if previous else "Morning"
default_hours = previous["hours"] if previous else 3
default_breaks = previous["breaks"] if previous else 10

st.subheader("⚙️ Study Preferences")

with st.form("planner_form"):
    subject_text = st.text_input(
        "Subjects (comma-separated)",
        value=default_subjects
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        timing = st.selectbox(
            "Preferred timing",
            ["Morning", "Afternoon", "Evening", "Night"],
            index=["Morning", "Afternoon", "Evening", "Night"].index(
                default_timing
            )
        )

    with c2:
        hours = st.slider(
            "Study hours per day",
            1, 12,
            max(1, min(12, int(default_hours)))
        )

    with c3:
        breaks = st.selectbox(
            "Break duration (minutes)",
            [5, 10, 15, 20],
            index=[5, 10, 15, 20].index(
                default_breaks
                if default_breaks in [5, 10, 15, 20]
                else 10
            )
        )

    submit = st.form_submit_button(
        "Generate Three Timetables",
        type="primary",
        use_container_width=True
    )

if submit:
    subjects = list(dict.fromkeys(
        item.strip()
        for item in subject_text.split(",")
        if item.strip()
    ))

    if not subjects:
        st.error("Please enter at least one subject.")
    else:
        try:
            save_preference(
                student_id, subjects, timing, hours, breaks
            )

            graph = build_graph()

            result = graph.invoke({
                "student_id": student_id,
                "subjects": subjects,
                "timing": timing,
                "hours": hours,
                "breaks": breaks
            })

            st.session_state["candidates"] = result["evaluated"]
            st.session_state["selected"] = result["selected"]
            st.session_state["active_student"] = student_id

            st.success("Three timetable options generated!")

        except Exception as error:
            st.error(f"Unable to generate timetables: {error}")


# --------------------------------------------------
# 5. COMPARE AND SELECT
# --------------------------------------------------

if (
    st.session_state.get("candidates")
    and st.session_state.get("active_student") == student_id
):
    st.divider()
    st.subheader("🌳 Three Possible Timetables")

    candidates = st.session_state["candidates"]

    columns = st.columns(3)

    for index, candidate in enumerate(candidates):
        with columns[index]:
            st.markdown(f"### {candidate['name']}")

            st.metric("Evaluation score", candidate["score"])
            st.write(candidate["evaluation"])

            st.dataframe(
                candidate["sessions"],
                use_container_width=True,
                hide_index=True
            )

    best = st.session_state["selected"]

    st.divider()
    st.subheader("🏆 Recommended Timetable")

    st.success(
        f"Recommended: {best['name']} "
        f"(score: {best['score']})"
    )

    selected_name = st.radio(
        "Choose your preferred timetable",
        [candidate["name"] for candidate in candidates],
        index=[
            candidate["name"] for candidate in candidates
        ].index(best["name"])
    )

    selected = next(
        candidate for candidate in candidates
        if candidate["name"] == selected_name
    )

    st.dataframe(
        selected["sessions"],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Selected Timetable",
        data=json.dumps(selected, indent=2),
        file_name="selected_timetable.json",
        mime="application/json"
    )

st.divider()
st.subheader("🧠 Previous Preferences")

if previous:
    st.write("**Subjects:**", ", ".join(previous["subjects"]))
    st.write("**Preferred timing:**", previous["timing"])
    st.write("**Daily study hours:**", previous["hours"])
    st.write("**Break duration:**", f"{previous['breaks']} minutes")
    st.caption(f"Last saved: {previous['saved_at']}")
else:
    st.write("Preferences will appear after you generate your first timetable.")

st.caption(
    "FAISS stores previous preferences locally in the "
    "'faiss_timetable_memory' folder."
)