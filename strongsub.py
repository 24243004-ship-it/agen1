
import os
import hashlib
from datetime import date

import streamlit as st
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Exam Revision Planner",
    page_icon="📚",
    layout="wide"
)

DB_PATH = "student_faiss_memory"


# ---------------- LOCAL EMBEDDINGS ----------------
# This demo uses lightweight local embeddings.
# No separate embedding API key is required.

class LocalEmbeddings(Embeddings):
    def __init__(self, dimensions=256):
        self.dimensions = dimensions

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        words = text.lower().split()

        for word in words:
            digest = hashlib.sha256(word.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0

        norm = sum(x * x for x in vector) ** 0.5
        if norm:
            vector = [x / norm for x in vector]

        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


# ---------------- FAISS MEMORY ----------------

def get_memory():
    embeddings = LocalEmbeddings()

    if os.path.exists(os.path.join(DB_PATH, "index.faiss")):
        # Only load a FAISS folder created by this application.
        return FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    initial_doc = Document(
        page_content="Initial memory document for the revision planner.",
        metadata={"student_id": "__system__", "type": "initial"}
    )

    store = FAISS.from_documents([initial_doc], embeddings)
    store.save_local(DB_PATH)
    return store


def save_student_record(student_id, subject, level, marks, notes):
    store = get_memory()

    content = (
        f"Student: {student_id}\n"
        f"Subject: {subject}\n"
        f"Performance: {level}\n"
        f"Marks: {marks if marks.strip() else 'Not provided'}\n"
        f"Notes: {notes if notes.strip() else 'None'}"
    )

    record_id = hashlib.sha256(
        f"{student_id}|{subject}|{level}|{marks}|{notes}".encode()
    ).hexdigest()

    store.add_documents(
        documents=[
            Document(
                page_content=content,
                metadata={
                    "student_id": student_id,
                    "subject": subject,
                    "level": level,
                    "type": "student_record"
                }
            )
        ],
        ids=[record_id]
    )
    store.save_local(DB_PATH)


def retrieve_student_memory(student_id, query, k=8):
    store = get_memory()

    results = store.similarity_search(
        query,
        k=k,
        filter={"student_id": student_id}
    )

    return [
        doc.page_content
        for doc in results
        if doc.metadata.get("type") == "student_record"
    ]


# ---------------- REVISION PLANNER AGENT ----------------

def create_revision_plan(
    api_key,
    model_name,
    student_id,
    exam_date,
    subjects,
    daily_hours,
    memory_records
):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )

    memory_text = "\n\n".join(memory_records) if memory_records else (
        "No previous performance records found for this student."
    )

    prompt = f"""
You are an AI Exam Revision Planner.

Prepare a practical revision plan for the student.

STUDENT ID: {student_id}
TODAY: {date.today()}
EXAM DATE: {exam_date}
AVAILABLE STUDY HOURS PER DAY: {daily_hours}

SUBJECTS AND CURRENT SELF-ASSESSMENT:
{subjects}

RETRIEVED PREVIOUS PERFORMANCE FROM FAISS MEMORY:
{memory_text}

Follow this transparent planning procedure:
1. Identify the subjects and the topics that need revision.
2. Prioritize weak subjects and topics that need improvement.
3. Include strong subjects in shorter revision and practice sessions.
4. Divide the available time into realistic study blocks with breaks.
5. Include active recall, practice questions, and a final mock test
   or self-test where time permits.
6. Adapt the schedule to the number of days remaining before the exam.

OUTPUT FORMAT:
- Brief planning summary
- Priority list with reasons
- Day-by-day timetable in a Markdown table
- Suggested revision method for each subject
- Final revision checklist

RULES:
- Never invent previous marks or performance history.
- If a subject's strength is unknown, label it as unassessed.
- Keep total scheduled study time within the daily hour limit.
- Account for the actual number of days remaining.
- If the exam is today or has passed, clearly adjust the plan.
- Make the plan simple and suitable for a student.
- Show the plan and key scheduling reasons, not private internal
  chain-of-thought.
"""

    response = llm.invoke(prompt)
    return response.content


# ---------------- STREAMLIT DASHBOARD ----------------

st.title("📚 AI Exam Revision Planner")
st.write(
    "Create a personalized revision timetable and use FAISS memory "
    "to remember your strong and weak subjects."
)

with st.sidebar:
    st.header("⚙️ AI Settings")
    api_key = st.text_input("Gemini API Key", type="password")

    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.0-flash"]
    )

st.subheader("👤 Student Details")

student_id = st.text_input(
    "Student ID / Nickname",
    value="student1",
    help="Use the same ID in future sessions to retrieve your records."
)

st.subheader("📝 Subject Performance")

st.caption(
    "Enter one subject per line. Example: Mathematics - Weak; "
    "English - Strong; Physics - Average"
)

subjects = st.text_area(
    "Subjects and strengths / weaknesses",
    value=(
        "Mathematics - Weak\n"
        "Physics - Average\n"
        "Computer Science - Strong\n"
        "English - Strong"
    ),
    height=130
)

col1, col2 = st.columns(2)

with col1:
    exam_date = st.date_input(
        "Upcoming exam date",
        value=date.today()
    )

with col2:
    daily_hours = st.slider(
        "Available study hours per day",
        min_value=1,
        max_value=12,
        value=4
    )

if st.button(
    "🚀 Generate Revision Plan",
    type="primary",
    use_container_width=True
):
    if not api_key.strip():
        st.error("Enter your Gemini API key in the sidebar.")
    elif not student_id.strip():
        st.error("Enter a student ID or nickname.")
    elif not subjects.strip():
        st.error("Enter at least one subject.")
    else:
        try:
            with st.spinner("Retrieving memory and preparing your plan..."):
                history = retrieve_student_memory(
                    student_id.strip(),
                    subjects,
                    k=8
                )

                plan = create_revision_plan(
                    api_key=api_key.strip(),
                    model_name=model_name,
                    student_id=student_id.strip(),
                    exam_date=exam_date,
                    subjects=subjects,
                    daily_hours=daily_hours,
                    memory_records=history
                )

            st.session_state["revision_plan"] = plan
            st.session_state["student_id"] = student_id.strip()
            st.session_state["subjects"] = subjects

            st.success("Your revision plan is ready!")

        except Exception as error:
            st.error(f"Could not create plan: {error}")
            st.info(
                "Check your API key, internet connection, model access, "
                "and installed packages."
            )


# Display the plan outside the button block so it remains visible
# after Streamlit reruns.

if st.session_state.get("revision_plan"):
    st.subheader("🗓️ Your Personalized Revision Plan")
    st.markdown(st.session_state["revision_plan"])

    st.download_button(
        "⬇️ Download Revision Plan",
        data=st.session_state["revision_plan"],
        file_name="revision_plan.md",
        mime="text/markdown",
        use_container_width=True
    )


# ---------------- SAVE PERFORMANCE MEMORY ----------------

st.divider()
st.subheader("🧠 Save Subject Performance to FAISS")

st.write(
    "After a test or revision session, save your performance here. "
    "The next plan can use these records."
)

with st.form("save_performance_form"):
    record_student_id = st.text_input(
        "Student ID / Nickname for this record",
        value=st.session_state.get("student_id", "student1")
    )

    record_subject = st.text_input(
        "Subject name",
        placeholder="Example: Mathematics"
    )

    record_level = st.selectbox(
        "Performance level",
        ["Weak", "Average", "Strong"]
    )

    record_marks = st.text_input(
        "Marks (optional)",
        placeholder="Example: 35/50"
    )

    record_notes = st.text_area(
        "Topics to improve / strengths (optional)",
        placeholder="Example: Good at algebra, needs help with calculus"
    )

    save_clicked = st.form_submit_button("💾 Save Performance")

if save_clicked:
    if not record_student_id.strip() or not record_subject.strip():
        st.error("Enter both student ID and subject name.")
    else:
        try:
            save_student_record(
                record_student_id.strip(),
                record_subject.strip(),
                record_level,
                record_marks,
                record_notes
            )
            st.success(
                f"Saved {record_subject} performance to FAISS memory."
            )
        except Exception as error:
            st.error(f"Could not save the record: {error}")


# ---------------- RETRIEVE PAST RECORDS ----------------

st.divider()
st.subheader("🔎 Retrieve Previous Performance")

search_student_id = st.text_input(
    "Student ID to search",
    value=st.session_state.get("student_id", "student1"),
    key="memory_search_student"
)

memory_query = st.text_input(
    "Search topic",
    value="weak subjects and exam performance"
)

if st.button("Search FAISS Memory"):
    if not search_student_id.strip():
        st.warning("Enter a student ID.")
    else:
        try:
            records = retrieve_student_memory(
                search_student_id.strip(),
                memory_query,
                k=10
            )

            if records:
                for index, record in enumerate(records, start=1):
                    with st.expander(f"Previous record {index}"):
                        st.write(record)
            else:
                st.info(
                    "No matching records found. Save a performance record first."
                )
        except Exception as error:
            st.error(f"Memory search failed: {error}")

st.caption(
    "Memory is stored locally in the chroma-independent FAISS folder. "
    "Use the same student ID to retrieve your saved performance."
)
