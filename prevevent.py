import os
import re
import json
import hashlib
import numpy as np
import streamlit as st

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS


# -------------------- CONFIGURATION --------------------

st.set_page_config(
    page_title="College Event Planner AI",
    page_icon="🎓",
    layout="wide"
)

MEMORY_DIR = "college_event_faiss"
VECTOR_SIZE = 256


# -------------------- LOCAL EMBEDDINGS --------------------
# No separate embedding API key required for local FAISS memory.

class LocalEmbeddings(Embeddings):

    def embed_query(self, text: str) -> list[float]:
        vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
        words = re.findall(r"[a-zA-Z0-9+#]+", text.lower())

        for word in words:
            digest = hashlib.sha256(word.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % VECTOR_SIZE
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


embeddings = LocalEmbeddings()


# -------------------- FAISS MEMORY --------------------

def load_memory():
    if os.path.isdir(MEMORY_DIR):
        try:
            return FAISS.load_local(
                MEMORY_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            return None

    return None


vector_store = load_memory()


def save_event_requirements(user_id, requirements):
    global vector_store

    document_text = json.dumps(
        requirements,
        ensure_ascii=False
    )

    document = Document(
        page_content=document_text,
        metadata={
            "user_id": user_id,
            "event_name": requirements["event_name"]
        }
    )

    if vector_store is None:
        vector_store = FAISS.from_documents(
            [document],
            embeddings
        )
    else:
        vector_store.add_documents([document])

    vector_store.save_local(MEMORY_DIR)


def retrieve_previous_events(user_id, query):
    if vector_store is None:
        return []

    try:
        docs = vector_store.similarity_search(
            query,
            k=5,
            filter={"user_id": user_id}
        )

        return [doc.page_content for doc in docs]

    except Exception:
        return []


# -------------------- LANGGRAPH STATE --------------------

class EventState(TypedDict, total=False):
    user_id: str
    event_name: str
    event_type: str
    event_date: str
    participants: int
    budget: str
    venue: str
    requirements: str
    previous_events: list[str]
    plan: str
    api_key: str


# -------------------- STEP 1: RETRIEVE MEMORY --------------------

def memory_agent(state: EventState):
    query = (
        f"Event: {state['event_name']}. "
        f"Type: {state['event_type']}. "
        f"Requirements: {state['requirements']}"
    )

    previous = retrieve_previous_events(
        state["user_id"],
        query
    )

    return {"previous_events": previous}


# -------------------- STEP 2: CREATE EVENT PLAN --------------------

def planning_agent(state: EventState):
    previous = state.get("previous_events", [])

    prompt = f"""
You are a college event planning assistant.

Create a practical step-by-step checklist for this event.

Event name: {state['event_name']}
Event type: {state['event_type']}
Date: {state['event_date']}
Expected participants: {state['participants']}
Budget: {state['budget']}
Venue: {state['venue']}
Additional requirements: {state['requirements']}

Previous event requirements retrieved from memory:
{previous}

Organize the plan in chronological order.
Include task, priority, suggested deadline and responsible team.
Cover approvals, venue, budget, speakers or performers if needed,
equipment, publicity, registration, refreshments, safety,
event-day execution and post-event feedback.

Adapt the tasks to the actual event and budget.
Use previous requirements as suggestions, not mandatory rules.
Do not invent confirmed bookings, permissions or actual deadlines.
Use the event date to suggest relative deadlines when possible.

Give a concise explanation for the plan's ordering, followed by
a clear checklist. Do not reveal private internal reasoning.
"""

    api_key = state.get("api_key", "").strip()

    if api_key:
        try:
            model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                api_key=api_key
            )

            response = model.invoke(prompt)
            return {"plan": response.content}

        except Exception as error:
            st.session_state["agent_error"] = str(error)

    # Offline fallback if no API key is provided.
    plan = f"""
# Event Plan: {state['event_name']}

## Step 1 — Define the event
- Confirm the purpose, theme and expected outcomes.
- Finalize the event format and participant capacity.
- Assign an event coordinator.

## Step 2 — Budget and approvals
- Prepare an estimated budget for the event.
- Request college management permission.
- Confirm funding and sponsorship options.

## Step 3 — Venue and logistics
- Shortlist a venue based on participant count.
- Check seating, electricity, internet and accessibility.
- Arrange microphones, projector and other required equipment.

## Step 4 — Teams and program
- Assign volunteers for registration, logistics and hospitality.
- Prepare the program schedule.
- Contact speakers, judges or performers if applicable.

## Step 5 — Publicity and registration
- Prepare posters and announcements.
- Open participant registration.
- Track registrations and communicate event details.

## Step 6 — Event-day preparation
- Verify venue, equipment and attendance lists.
- Brief volunteers and confirm assigned responsibilities.
- Prepare emergency contacts and basic safety arrangements.

## Step 7 — Conduct the event
- Welcome participants and follow the program schedule.
- Monitor time, equipment and participant support.
- Resolve issues and record attendance.

## Step 8 — Follow-up
- Collect feedback from participants.
- Settle expenses and acknowledge volunteers.
- Prepare a short event report.

## Planning notes
- Event date: {state['event_date']}
- Participants: {state['participants']}
- Budget: {state['budget']}
- Venue: {state['venue']}

The checklist is a general template. Confirm actual permissions,
availability, costs and deadlines with the college.

## Planning approach
Tasks are ordered from approvals and preparation to execution
and follow-up so that dependent activities are addressed early.
"""

    return {"plan": plan}


# -------------------- BUILD LANGGRAPH WORKFLOW --------------------

def build_graph():
    graph = StateGraph(EventState)

    graph.add_node("memory_agent", memory_agent)
    graph.add_node("planning_agent", planning_agent)

    graph.add_edge(START, "memory_agent")
    graph.add_edge("memory_agent", "planning_agent")
    graph.add_edge("planning_agent", END)

    return graph.compile()


# -------------------- STREAMLIT DASHBOARD --------------------

st.title("🎓 College Event Planner AI")

st.write(
    "Plan a college event step by step with LangGraph "
    "and remember previous event requirements using FAISS."
)

with st.sidebar:
    st.header("Student Settings")

    user_id = st.text_input(
        "Student / Organizer ID",
        value="student01"
    ).strip()

    api_key = st.text_input(
        "OpenAI API Key (optional)",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password"
    ).strip()

    st.caption(
        "The offline planner works without an API key. "
        "An API key enables AI-generated, customized plans."
    )

st.subheader("1. Enter Event Details")

left, right = st.columns(2)

with left:
    event_name = st.text_input(
        "Event Name",
        value="College Symposium"
    )

    event_type = st.selectbox(
        "Event Type",
        [
            "Technical Symposium",
            "Workshop",
            "Cultural Fest",
            "Sports Event",
            "Seminar",
            "College Function",
            "Other"
        ]
    )

    event_date = st.date_input(
        "Planned Event Date"
    )

    participants = st.number_input(
        "Expected Participants",
        min_value=1,
        max_value=10000,
        value=200
    )

with right:
    budget = st.selectbox(
        "Estimated Budget",
        [
            "Below ₹5,000",
            "₹5,000–₹15,000",
            "₹15,000–₹50,000",
            "Above ₹50,000",
            "Not decided"
        ]
    )

    venue = st.selectbox(
        "Venue",
        [
            "College Auditorium",
            "Seminar Hall",
            "Classroom",
            "Outdoor Campus",
            "Venue not decided"
        ]
    )

    requirements = st.text_area(
        "Additional Requirements",
        placeholder=(
            "Example: Need guest speakers, certificates, "
            "refreshments and project demonstrations."
        ),
        height=120
    )

st.divider()

generate = st.button(
    "🚀 Create Event Plan",
    type="primary",
    use_container_width=True
)

if generate:
    if not user_id:
        st.error("Please enter a Student / Organizer ID.")
    elif not event_name.strip():
        st.error("Please enter an event name.")
    else:
        initial_state: EventState = {
            "user_id": user_id,
            "event_name": event_name.strip(),
            "event_type": event_type,
            "event_date": str(event_date),
            "participants": int(participants),
            "budget": budget,
            "venue": venue,
            "requirements": requirements,
            "api_key": api_key
        }

        try:
            with st.spinner(
                "Retrieving memory and preparing the event plan..."
            ):
                workflow = build_graph()
                result = workflow.invoke(initial_state)

            st.session_state["event_result"] = result

            st.session_state["current_event"] = {
                "user_id": user_id,
                "event_name": event_name.strip(),
                "event_type": event_type,
                "event_date": str(event_date),
                "participants": int(participants),
                "budget": budget,
                "venue": venue,
                "requirements": requirements
            }

            st.success("Event plan created successfully!")

        except Exception as error:
            st.error(f"Unable to create the plan: {error}")

# -------------------- DISPLAY PLAN --------------------

if "event_result" in st.session_state:
    result = st.session_state["event_result"]
    current = st.session_state["current_event"]

    st.subheader("2. Your Step-by-Step Event Plan")
    st.markdown(result.get("plan", ""))

    previous = result.get("previous_events", [])

    with st.expander("View Previous Requirements Retrieved"):
        if previous:
            for index, item in enumerate(previous, start=1):
                st.markdown(f"**Previous event {index}**")
                st.code(item, language="json")
        else:
            st.info(
                "No previous event requirements found for this ID."
            )

    st.subheader("3. Save Requirements to FAISS Memory")

    st.write(
        "Save the event requirements so they can be retrieved "
        "when you plan another event using the same ID."
    )

    if st.button("💾 Save Event Requirements"):
        try:
            save_event_requirements(
                current["user_id"],
                current
            )

            st.success(
                "Event requirements saved in persistent FAISS memory."
            )

        except Exception as error:
            st.error(f"Could not save requirements: {error}")

    st.download_button(
        "📥 Download Event Plan",
        data=result.get("plan", ""),
        file_name="college_event_plan.md",
        mime="text/markdown"
    )

st.divider()

st.caption(
    "College Event Planner AI | LangGraph workflow | FAISS memory"
)