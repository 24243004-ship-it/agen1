import os
import json
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings.fake import FakeEmbeddings

load_dotenv()

API_KEY = os.getenv("", "").strip()
DB_PATH = "study_memory_db"

st.set_page_config(
    page_title="AI Study Planner",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def get_memory():
    if API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=API_KEY
            )
        except Exception:
            embeddings = FakeEmbeddings(size=64)
    else:
        embeddings = FakeEmbeddings(size=64)

    return Chroma(
        collection_name="student_preferences",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )


def save_preferences(student_id, subjects, study_time, hours, break_minutes):
    memory = get_memory()

    preference = {
        "student_id": student_id,
        "subjects": subjects,
        "study_time": study_time,
        "hours": hours,
        "break_minutes": break_minutes,
        "saved_at": datetime.now().isoformat(timespec="seconds")
    }

    document = Document(
        page_content=(
            f"Student {student_id} prefers subjects "
            f"{', '.join(subjects)}, study time {study_time}, "
            f"daily study hours {hours}, "
            f"break duration {break_minutes} minutes."
        ),
        metadata={
            "student_id": student_id,
            "subjects": json.dumps(subjects),
            "study_time": study_time,
            "hours": int(hours),
            "break_minutes": int(break_minutes),
            "saved_at": preference["saved_at"]
        }
    )

    memory.add_documents(
        [document],
        ids=[f"{student_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"]
    )

    return preference


def retrieve_preferences(student_id):
    memory = get_memory()

    # Get records belonging to this student only.
    data = memory._collection.get(
        where={"student_id": student_id},
        include=["documents", "metadatas"]
    )

    metadatas = data.get("metadatas") or []

    if not metadatas:
        return None

    # Use the most recently saved preference.
    latest = max(
        metadatas,
        key=lambda item: item.get("saved_at", "")
    )

    try:
        subjects = json.loads(latest.get("subjects", "[]"))
    except (TypeError, json.JSONDecodeError):
        subjects = []

    return {
        "subjects": subjects,
        "study_time": latest.get("study_time", "Morning"),
        "hours": int(latest.get("hours", 3)),
        "break_minutes": int(latest.get("break_minutes", 10)),
        "saved_at": latest.get("saved_at", "")
    }


# --------------------------------------------------
# 2. STUDY PLANNING AGENT
# --------------------------------------------------

class StudyPlannerAgent:

    def __init__(self):
        self.llm = None

        if API_KEY:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    api_key=API_KEY
                )
            except Exception:
                self.llm = None

    def plan(self, subjects, study_time, hours, break_minutes):
        # Step 1: Validate the subjects.
        subjects = [
            subject.strip()
            for subject in subjects
            if subject.strip()
        ]

        if not subjects:
            raise ValueError("Please enter at least one subject.")

        # Step 2: Allocate study sessions.
        total_minutes = int(hours * 60)
        session_minutes = 45

        number_of_sessions = max(
            1, (total_minutes + break_minutes) //
            (session_minutes + break_minutes)
        )

        # Don't create more sessions than the time available.
        number_of_sessions = min(
            number_of_sessions,
            max(1, total_minutes // 25)
        )

        sessions = []
        remaining = total_minutes
        start_hour = {
            "Morning": 6,
            "Afternoon": 13,
            "Evening": 17,
            "Night": 20
        }.get(study_time, 6)

        current_time = datetime.now().replace(
            hour=start_hour,
            minute=0,
            second=0,
            microsecond=0
        )

        # Step 3: Distribute subjects evenly.
        for index in range(number_of_sessions):
            subject = subjects[index % len(subjects)]

            slots_left = number_of_sessions - index
            duration = min(
                session_minutes,
                max(25, remaining // slots_left)
            )

            end_time = current_time + timedelta(minutes=duration)

            sessions.append({
                "Time": (
                    current_time.strftime("%I:%M %p")
                    + " - "
                    + end_time.strftime("%I:%M %p")
                ),
                "Subject": subject,
                "Duration": f"{duration} min",
                "Activity": (
                    "Review concepts and make notes"
                    if index % 2 == 0
                    else "Practice questions and revise"
                )
            })

            remaining -= duration
            current_time = end_time

            if index < number_of_sessions - 1:
                current_time += timedelta(minutes=break_minutes)
                remaining -= break_minutes

        # Step 4: Ask the optional AI model for study advice.
        tips = (
            "Revise your notes after each session and "
            "finish with a short self-test."
        )

        if self.llm:
            try:
                prompt = f"""
You are a study-planning assistant.

Student subjects: {", ".join(subjects)}
Preferred time: {study_time}
Daily study hours: {hours}
Break duration: {break_minutes} minutes.

Give three short, practical study tips for this student.
Do not change the supplied timetable.
Do not reveal private internal reasoning.
"""
                response = self.llm.invoke(prompt)
                tips = response.content
            except Exception:
                pass

        return sessions, tips


# --------------------------------------------------
# 3. DASHBOARD
# --------------------------------------------------

st.title("📚 AI Daily Study Planner")
st.caption(
    "LangChain • Step-by-Step Planning • ChromaDB Memory"
)

# Student identity allows preferences to be retrieved later.
student_id = st.text_input(
    "Student ID",
    value="student01",
    help="Use the same ID next time to retrieve saved preferences."
).strip()

if not student_id:
    st.warning("Enter a Student ID to continue.")
    st.stop()

# Load previous preferences.
try:
    previous = retrieve_preferences(student_id)
except Exception as error:
    previous = None
    st.warning(f"Could not load saved preferences: {error}")

if previous:
    st.success("Previous study preferences found in ChromaDB.")
else:
    st.info("New student profile. Enter your study preferences.")

st.divider()

st.subheader("⚙️ Customize Your Study Plan")

default_subjects = (
    ", ".join(previous["subjects"])
    if previous else "Python, Mathematics, English"
)

default_time = previous["study_time"] if previous else "Morning"
default_hours = previous["hours"] if previous else 3
default_break = previous["break_minutes"] if previous else 10

with st.form("study_preferences"):
    subjects_text = st.text_input(
        "Subjects (separate with commas)",
        value=default_subjects
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        study_time = st.selectbox(
            "Preferred study time",
            ["Morning", "Afternoon", "Evening", "Night"],
            index=[
                "Morning", "Afternoon", "Evening", "Night"
            ].index(default_time)
        )

    with col2:
        hours = st.slider(
            "Daily study hours",
            min_value=1,
            max_value=12,
            value=max(1, min(12, int(default_hours)))
        )

    with col3:
        break_minutes = st.selectbox(
            "Break between sessions",
            [5, 10, 15, 20],
            index=[5, 10, 15, 20].index(
                default_break if default_break in [5, 10, 15, 20]
                else 10
            )
        )

    generate = st.form_submit_button(
        "Generate My Study Schedule",
        type="primary",
        use_container_width=True
    )

if generate:
    subjects = [
        item.strip()
        for item in subjects_text.split(",")
        if item.strip()
    ]

    if not subjects:
        st.error("Enter at least one subject.")
    else:
        try:
            # Save the current preferences to persistent memory.
            save_preferences(
                student_id,
                subjects,
                study_time,
                hours,
                break_minutes
            )

            # Run the planning agent.
            agent = StudyPlannerAgent()
            sessions, tips = agent.plan(
                subjects,
                study_time,
                hours,
                break_minutes
            )

            st.session_state["study_schedule"] = sessions
            st.session_state["study_tips"] = tips
            st.session_state["study_student"] = student_id

            st.success("Schedule generated and preferences saved!")

        except Exception as error:
            st.error(f"Could not generate schedule: {error}")

# Display the latest generated schedule.
if (
    st.session_state.get("study_schedule")
    and st.session_state.get("study_student") == student_id
):
    st.divider()
    st.subheader("🗓️ Your Daily Study Schedule")

    sessions = st.session_state["study_schedule"]

    st.dataframe(
        sessions,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("💡 Study Tips")
    st.write(st.session_state.get("study_tips", ""))

    st.download_button(
        "Download Schedule as JSON",
        data=json.dumps(sessions, indent=2),
        file_name="daily_study_schedule.json",
        mime="application/json"
    )

st.divider()
st.subheader("🧠 Saved Preferences")

if previous:
    st.write("**Previously saved subjects:**", ", ".join(previous["subjects"]))
    st.write("**Preferred time:**", previous["study_time"])
    st.write("**Daily study hours:**", previous["hours"])
    st.write("**Break duration:**", f'{previous["break_minutes"]} minutes')
    st.caption(f'Last saved: {previous["saved_at"]}')
else:
    st.write("Your saved preferences will appear here after your first plan.")

st.caption(
    "ChromaDB data persists in the study_memory_db folder. "
    "Use the same Student ID to retrieve your preferences in a later session."
)