
import os
import json
import uuid
from datetime import date

import streamlit as st
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Assignment Time Optimizer",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Assignment Time Optimizer")
st.write(
    "Generate, compare, and select assignment plans using "
    "Gemini AI, a Tree-of-Thought-inspired workflow, and ChromaDB."
)

DB_DIR = "./assignment_memory_db"


# ---------------- SIDEBAR ----------------
st.sidebar.header("Gemini API Settings")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
)

model_name = st.sidebar.selectbox(
    "Gemini model",
    ["gemini-2.5-flash", "gemini-2.0-flash"],
)

st.sidebar.markdown(
    "[Get an API key](https://aistudio.google.com/apikey)"
)


# ---------------- CHROMADB MEMORY ----------------
@st.cache_resource
def get_vectorstore(key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=key,
    )

    return Chroma(
        collection_name="assignment_history",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )


def save_assignment_history(store, student_id, assignments, chosen_plan):
    record = {
        "saved_on": str(date.today()),
        "assignments": assignments,
        "selected_plan": chosen_plan,
    }

    document = Document(
        page_content=json.dumps(record, ensure_ascii=False),
        metadata={
            "student_id": student_id,
            "record_id": str(uuid.uuid4()),
        },
    )

    store.add_documents([document])


def retrieve_assignment_history(store, student_id, query):
    try:
        docs = store.similarity_search(
            query,
            k=5,
            filter={"student_id": student_id},
        )

        if not docs:
            return "No previous assignment history found."

        return "\n\n".join(
            doc.page_content for doc in docs
        )

    except Exception as error:
        return f"Previous history could not be retrieved: {error}"


# ---------------- GEMINI ----------------
def get_llm(key, model):
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0.3,
    )


# ---------------- ASSIGNMENT PARSER ----------------
def parse_assignments(text):
    assignments = []

    for line_number, line in enumerate(text.strip().splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        parts = [part.strip() for part in line.split("|")]

        if len(parts) != 4:
            raise ValueError(
                f"Line {line_number}: use "
                "'Assignment | Hours | Days left | Difficulty'."
            )

        title, hours_text, days_text, difficulty_text = parts

        try:
            hours = float(hours_text)
            days_left = int(days_text)
            difficulty = int(difficulty_text)
        except ValueError:
            raise ValueError(
                f"Line {line_number}: hours, days left, "
                "and difficulty must be numbers."
            )

        if not title:
            raise ValueError(f"Line {line_number}: title is required.")

        if hours <= 0 or days_left < 0 or not 1 <= difficulty <= 5:
            raise ValueError(
                f"Line {line_number}: hours must be positive, "
                "days left cannot be negative, and difficulty "
                "must be between 1 and 5."
            )

        assignments.append({
            "title": title,
            "hours": hours,
            "days_left": days_left,
            "difficulty": difficulty,
        })

    if not assignments:
        raise ValueError("Please enter at least one assignment.")

    return assignments


# ---------------- TREE-OF-THOUGHT-INSPIRED PLANNER ----------------
STRATEGIES = {
    "Deadline First": (
        "Prioritize assignments with the nearest deadlines. "
        "Include time for testing and final submission."
    ),
    "Difficulty First": (
        "Start difficult or high-effort assignments early, "
        "while ensuring no deadline is missed."
    ),
    "Time Blocking": (
        "Group similar work into focused sessions, use breaks, "
        "and reserve short review sessions."
    ),
}


def generate_candidate(llm, strategy, description, assignments, history):
    prompt = f"""
You are a college assignment planning assistant.

Generate one practical plan using this strategy:
{strategy}

Strategy guidance:
{description}

Current date: {date.today()}

Current assignments:
{json.dumps(assignments, indent=2)}

Previous assignment history and preferences:
{history}

Requirements:
1. Create an actionable schedule for the current assignments.
2. Respect every deadline. Days left means calendar days remaining.
3. Estimate daily work blocks and breaks.
4. Include a buffer for checking and submission.
5. Never claim that work can be completed in fewer hours than
   the given estimates unless you explicitly explain the assumption.
6. Do not assume the student can work unlimited hours each day.
   If daily availability is unknown, state a reasonable assumption.
7. Use previous preferences only when relevant.
8. Give a concise explanation of why this plan could save time.
9. Identify risks and possible deadline conflicts.
10. Return valid JSON only, with these keys:
{{
  "plan_name": "name",
  "schedule": [
    {{
      "day": "Day 1",
      "tasks": ["task and estimated duration"]
    }}
  ],
  "estimated_total_hours": 10,
  "time_saving_tips": ["tip 1", "tip 2"],
  "reason": "short explanation",
  "risks": ["risk 1"]
}}

Do not claim time savings that cannot be justified by the schedule.
"""

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict) else str(item)
            for item in content
        )

    content = str(content).strip()
    content = content.replace("```json", "").replace("```", "").strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Gemini did not return valid JSON.")

    plan = json.loads(content[start:end + 1])
    plan["strategy"] = strategy

    # Compare plans using the same assignment workload baseline.
    plan["baseline_hours"] = round(
        sum(item["hours"] for item in assignments), 1
    )

    estimated = float(plan.get("estimated_total_hours", 0))
    plan["estimated_total_hours"] = estimated

    if estimated <= 0:
        raise ValueError("A plan must have a positive estimated time.")

    # Estimated savings are an estimate, not a guaranteed result.
    plan["estimated_savings_hours"] = round(
        max(0, plan["baseline_hours"] - estimated), 1
    )

    return plan


def calculate_score(plan, assignments):
    """Transparent heuristic score for comparing candidate plans."""
    total_hours = sum(a["hours"] for a in assignments)
    estimated = plan["estimated_total_hours"]

    score = 100.0

    # Penalize plans that claim a longer completion time.
    if total_hours > 0:
        score -= min(35, max(0, estimated - total_hours) / total_hours * 35)

    # Penalize missing schedule entries or explanations.
    if not plan.get("schedule"):
        score -= 25
    if not plan.get("reason"):
        score -= 10
    if not plan.get("risks"):
        score -= 5

    # Penalize plans with no buffer or submission checks mentioned.
    schedule_text = json.dumps(plan.get("schedule", [])).lower()
    if not any(word in schedule_text for word in ["review", "check", "submit", "buffer"]):
        score -= 10

    return round(max(0, min(100, score)), 1)


# ---------------- DASHBOARD INPUTS ----------------
st.subheader("1. Student and Assignment Details")

student_id = st.text_input("Student ID", value="STU101")

daily_hours = st.slider(
    "Available study hours per day",
    min_value=1,
    max_value=12,
    value=4,
)

preferences = st.text_input(
    "Planning preferences",
    placeholder="e.g. short study sessions, Python first, evening study",
)

assignments_text = st.text_area(
    "Enter assignments (one per line)",
    value=(
        "Python Project | 5 | 3 | 4\n"
        "Database Assignment | 3 | 2 | 3\n"
        "AI Report | 4 | 5 | 5"
    ),
    height=150,
    help=(
        "Format: Assignment Name | Estimated Hours | "
        "Days Left | Difficulty (1-5)"
    ),
)

st.caption(
    "Example: Python Project | 5 | 3 | 4"
)


# ---------------- GENERATE AND COMPARE ----------------
if st.button("🚀 Generate and Compare 3 Plans", type="primary"):
    if not api_key.strip():
        st.error("Enter your Gemini API key in the sidebar.")
    elif not student_id.strip():
        st.error("Enter a Student ID.")
    else:
        try:
            assignments = parse_assignments(assignments_text)

            total_hours = sum(item["hours"] for item in assignments)
            available_hours = daily_hours * max(
                item["days_left"] for item in assignments
            )

            store = get_vectorstore(api_key.strip())

            history = retrieve_assignment_history(
                store,
                student_id.strip(),
                "Previous assignment details, completion time, "
                "planning preferences, and selected plans",
            )

            llm = get_llm(api_key.strip(), model_name)

            plans = []

            progress = st.progress(0)
            status = st.empty()

            for index, (strategy, description) in enumerate(
                STRATEGIES.items()
            ):
                status.write(f"Generating candidate: {strategy}...")
                plan = generate_candidate(
                    llm,
                    strategy,
                    description,
                    assignments,
                    history,
                )

                plan["score"] = calculate_score(plan, assignments)
                plan["available_hours_per_day"] = daily_hours
                plans.append(plan)

                progress.progress((index + 1) / len(STRATEGIES))

            status.empty()
            progress.empty()

            # Highest heuristic score wins; ties favor lower estimated time.
            plans.sort(
                key=lambda p: (
                    p["score"],
                    -p["estimated_total_hours"],
                ),
                reverse=True,
            )

            st.session_state["plans"] = plans
            st.session_state["assignments"] = assignments
            st.session_state["student_id"] = student_id.strip()
            st.session_state["preferences"] = preferences
            st.session_state["daily_hours"] = daily_hours

        except Exception as error:
            st.error(f"Could not generate plans: {error}")


# ---------------- RESULTS ----------------
if "plans" in st.session_state:
    plans = st.session_state["plans"]
    assignments = st.session_state["assignments"]

    st.divider()
    st.subheader("2. Assignment Workload")

    col1, col2, col3 = st.columns(3)
    col1.metric("Assignments", len(assignments))
    col2.metric(
        "Estimated workload",
        f"{sum(a['hours'] for a in assignments):.1f} hours",
    )
    col3.metric(
        "Study hours/day",
        st.session_state["daily_hours"],
    )

    st.subheader("3. Compare the Three Candidate Plans")

    comparison = []
    for plan in plans:
        comparison.append({
            "Strategy": plan["strategy"],
            "Estimated Hours": plan["estimated_total_hours"],
            "Estimated Savings": plan["estimated_savings_hours"],
            "Plan Score": plan["score"],
        })

    st.dataframe(comparison, use_container_width=True)

    best = plans[0]

    st.subheader("🏆 Recommended Plan")
    st.markdown(f"**Strategy:** {best['strategy']}")
    st.markdown(f"**Why selected:** {best.get('reason', '')}")
    st.write(f"Plan score: {best['score']}/100")
    st.write(
        f"Estimated completion time: "
        f"{best['estimated_total_hours']} hours"
    )
    st.write(
        f"Estimated time difference vs. baseline: "
        f"{best['estimated_savings_hours']} hours"
    )

    st.caption(
        "Plan scores are heuristic comparisons, not proof of optimality. "
        "Estimated savings are not guaranteed; validate the estimates "
        "against the actual assignment workload."
    )

    for index, plan in enumerate(plans, start=1):
        with st.expander(
            f"{index}. {plan['strategy']} — Score {plan['score']}/100",
            expanded=(index == 1),
        ):
            st.markdown("**Schedule**")

            for day in plan.get("schedule", []):
                st.markdown(f"**{day.get('day', 'Study session')}**")
                tasks = day.get("tasks", [])
                if tasks:
                    for task in tasks:
                        st.markdown(f"- {task}")
                else:
                    st.write("No tasks specified.")

            st.markdown("**Time-saving tips**")
            for tip in plan.get("time_saving_tips", []):
                st.markdown(f"- {tip}")

            st.markdown("**Risks**")
            for risk in plan.get("risks", []):
                st.markdown(f"- {risk}")

    st.subheader("4. Save Assignment History")

    if st.button("💾 Save Assignments and Selected Plan"):
        try:
            store = get_vectorstore(api_key.strip())

            saved_record = {
                "assignments": assignments,
                "daily_hours": st.session_state["daily_hours"],
                "preferences": st.session_state["preferences"],
                "selected_plan": best,
            }

            save_assignment_history(
                store,
                st.session_state["student_id"],
                saved_record,
                best["strategy"],
            )

            st.success(
                "Assignment history saved to ChromaDB. "
                "Use the same Student ID for future planning."
            )

        except Exception as error:
            st.error(f"Could not save assignment history: {error}")

    download_data = {
        "assignments": assignments,
        "candidate_plans": plans,
        "recommended_plan": best,
    }

    st.download_button(
        "⬇️ Download Comparison (JSON)",
        data=json.dumps(download_data, indent=2, ensure_ascii=False),
        file_name="assignment_plan_comparison.json",
        mime="application/json",
    )

st.divider()
st.caption("Built with LangChain • Gemini AI • ChromaDB • Streamlit")
