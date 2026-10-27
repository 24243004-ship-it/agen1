
import os
import re
import json
import hashlib
from datetime import date

import numpy as np
import streamlit as st
import chromadb

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# -------------------- CONFIGURATION --------------------

st.set_page_config(
    page_title="Study Plan Optimizer AI",
    page_icon="📚",
    layout="wide"
)

DB_PATH = "./study_preferences_db"
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="study_preferences"
)

VECTOR_SIZE = 128


# -------------------- LOCAL EMBEDDINGS --------------------
# Lightweight embeddings for local ChromaDB retrieval.
# No external embedding API is required.

def make_embedding(text):
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


# -------------------- CHROMADB MEMORY --------------------

def save_preferences(student_id, preferences):
    doc_text = json.dumps(preferences, ensure_ascii=False)

    record_id = (
        f"{student_id}_{date.today().isoformat()}_"
        f"{hashlib.sha256(doc_text.encode()).hexdigest()[:10]}"
    )

    # Upsert prevents duplicate IDs when saving the same preferences.
    collection.upsert(
        ids=[record_id],
        documents=[doc_text],
        metadatas=[{"student_id": student_id}],
        embeddings=[make_embedding(doc_text)]
    )


def retrieve_preferences(student_id, query="study preferences"):
    try:
        result = collection.query(
            query_embeddings=[make_embedding(query)],
            n_results=5,
            where={"student_id": student_id}
        )

        documents = result.get("documents", [[]])[0]
        return [json.loads(doc) for doc in documents]

    except Exception:
        return []


# -------------------- TREE-OF-THOUGHT PLANNER --------------------

def generate_plan(subjects, difficult_subjects, daily_hours,
                  exam_days, revision_percent, strategy):

    # More study time for difficult subjects in strategy 2.
    if strategy == "Difficult Subjects First":
        weights = [
            2.0 if subject in difficult_subjects else 1.0
            for subject in subjects
        ]
        revision_fraction = revision_percent / 100

    elif strategy == "Revision Focused":
        weights = [
            1.5 if subject in difficult_subjects else 1.0
            for subject in subjects
        ]
        revision_fraction = max(revision_percent / 100, 0.30)

    else:
        weights = [1.0] * len(subjects)
        revision_fraction = revision_percent / 100

    study_fraction = 1 - revision_fraction
    total_weight = sum(weights)

    available_hours = daily_hours * exam_days
    revision_hours = available_hours * revision_fraction
    subject_hours = available_hours * study_fraction

    allocations = {}

    for subject, weight in zip(subjects, weights):
        allocations[subject] = round(
            subject_hours * weight / total_weight, 2
        )

    # Round-off correction to keep totals consistent.
    used_hours = sum(allocations.values())
    if subjects:
        allocations[subjects[0]] = round(
            allocations[subjects[0]] + subject_hours - used_hours, 2
        )

    return {
        "strategy": strategy,
        "allocations": allocations,
        "revision_hours": round(revision_hours, 2),
        "total_hours": round(available_hours, 2),
        "daily_hours": daily_hours,
        "exam_days": exam_days,
        "revision_percent": round(revision_fraction * 100, 1)
    }


def evaluate_plan(plan, difficult_subjects, preferred_strategy,
                  required_revision_percent):

    allocations = plan["allocations"]
    total_hours = plan["total_hours"]

    # Reward difficult-subject coverage.
    if allocations:
        difficult_hours = sum(
            allocations.get(subject, 0)
            for subject in difficult_subjects
        )
        difficult_ratio = difficult_hours / max(
            sum(allocations.values()), 0.01
        )
    else:
        difficult_ratio = 0

    # Reward revision time close to the student's desired amount.
    revision_gap = abs(
        plan["revision_percent"] - required_revision_percent
    )
    revision_score = max(0, 100 - revision_gap * 2)

    # Reward subject coverage and time distribution.
    positive_subjects = sum(
        1 for hours in allocations.values() if hours > 0
    )
    coverage_score = (
        100 * positive_subjects / max(len(allocations), 1)
    )

    if difficult_subjects:
        difficulty_score = min(100, difficult_ratio * 150)
    else:
        difficulty_score = 100

    preference_score = (
        100 if plan["strategy"] == preferred_strategy else 70
    )

    score = (
        coverage_score * 0.35
        + difficulty_score * 0.25
        + revision_score * 0.25
        + preference_score * 0.15
    )

    # All plans share the same available-hour limit.
    if sum(allocations.values()) + plan["revision_hours"] > total_hours + 0.1:
        score -= 30

    return {
        **plan,
        "score": round(max(0, min(100, score)), 1),
        "coverage_score": round(coverage_score, 1),
        "difficulty_score": round(difficulty_score, 1),
        "revision_score": round(revision_score, 1),
        "preference_score": preference_score
    }


def tree_of_thought(subjects, difficult_subjects, daily_hours,
                    exam_days, revision_percent, preferred_strategy):

    strategies = [
        "Balanced Plan",
        "Difficult Subjects First",
        "Revision Focused"
    ]

    candidates = []

    # Generate three branches, then compare their scores.
    for strategy in strategies:
        plan = generate_plan(
            subjects,
            difficult_subjects,
            daily_hours,
            exam_days,
            revision_percent,
            strategy
        )

        evaluated = evaluate_plan(
            plan,
            difficult_subjects,
            preferred_strategy,
            revision_percent
        )

        candidates.append(evaluated)

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return candidates


# -------------------- LANGCHAIN EXPLANATION --------------------

def explain_recommendation(best_plan, api_key):
    if not api_key:
        return (
            f"The {best_plan['strategy']} was selected because it achieved "
            f"the highest heuristic score of {best_plan['score']}/100. "
            "The score considers subject coverage, difficult-subject "
            "allocation, revision time and your preferred study strategy."
        )

    try:
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=api_key
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a study-planning assistant. Explain the selected "
                "plan clearly using the provided scores. Do not reveal "
                "private internal chain-of-thought. Mention that the score "
                "is a heuristic, not a guarantee of exam performance."
            ),
            (
                "human",
                "Selected plan: {plan}\n"
                "Score: {score}\n"
                "Subject allocations: {allocations}\n"
                "Revision hours: {revision_hours}\n"
                "Explain why this plan fits the constraints in 3-5 sentences."
            )
        ])

        chain = prompt | model
        response = chain.invoke({
            "plan": best_plan["strategy"],
            "score": best_plan["score"],
            "allocations": json.dumps(best_plan["allocations"]),
            "revision_hours": best_plan["revision_hours"]
        })

        return response.content

    except Exception as error:
        return (
            "LLM explanation was unavailable, so a local explanation "
            f"is shown instead. Details: {error}\n\n"
            f"The {best_plan['strategy']} scored "
            f"{best_plan['score']}/100."
        )


# -------------------- DASHBOARD --------------------

st.title("📚 Study Plan Optimizer AI")

st.write(
    "Compare three possible study plans using a Tree-of-Thought-inspired "
    "search, then save your preferences in ChromaDB."
)

with st.sidebar:
    st.header("Student Profile")

    student_id = st.text_input(
        "Student ID",
        value="student01"
    ).strip()

    api_key = st.text_input(
        "OpenAI API key (optional)",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password"
    ).strip()

    st.caption(
        "Without an API key, plan generation and comparison work locally. "
        "The optional LangChain model improves the recommendation explanation."
    )

left, right = st.columns([3, 2])

with left:
    st.subheader("1. Study Requirements")

    subjects_text = st.text_input(
        "Subjects (comma-separated)",
        value="Mathematics, Python, DBMS, Networks"
    )

    difficult_text = st.text_input(
        "Difficult subjects",
        value="Mathematics, Networks"
    )

    daily_hours = st.slider(
        "Available study hours per day",
        min_value=1,
        max_value=12,
        value=4
    )

    exam_days = st.number_input(
        "Days remaining until exam",
        min_value=1,
        max_value=180,
        value=14
    )

    revision_percent = st.slider(
        "Desired revision time (%)",
        min_value=10,
        max_value=50,
        value=25
    )

    preferred_strategy = st.selectbox(
        "Preferred strategy",
        [
            "Balanced Plan",
            "Difficult Subjects First",
            "Revision Focused"
        ]
    )

    if st.button("Load Previous Preferences"):
        previous = retrieve_preferences(
            student_id,
            "study strategy revision daily hours subjects"
        )

        if previous:
            st.session_state["previous_preferences"] = previous
            st.success("Previous preferences retrieved from ChromaDB.")
            st.json(previous)
        else:
            st.info("No saved preferences found for this student ID.")

with right:
    st.subheader("2. How the Agent Works")

    st.markdown("""
    **Step 1 — Retrieve**

    Search ChromaDB for this student's earlier preferences.

    **Step 2 — Generate**

    Create three candidate plans using different study strategies.

    **Step 3 — Evaluate**

    Score coverage, difficult-subject time, revision and strategy fit.

    **Step 4 — Select**

    Recommend the candidate with the highest score.

    **Step 5 — Remember**

    Save the selected strategy and constraints for future sessions.
    """)

if st.button("🌳 Compare Three Study Plans", type="primary",
             use_container_width=True):

    subjects = [
        item.strip()
        for item in subjects_text.split(",")
        if item.strip()
    ]

    difficult_subjects = [
        item.strip()
        for item in difficult_text.split(",")
        if item.strip()
    ]

    if not student_id:
        st.error("Please enter a Student ID.")
    elif not subjects:
        st.error("Please enter at least one subject.")
    else:
        previous = retrieve_preferences(
            student_id,
            "study preferences " + ", ".join(subjects)
        )

        # Use saved preferences if the current input is the same as
        # a previously stored subject/strategy preference.
        if previous:
            st.session_state["previous_preferences"] = previous

        candidates = tree_of_thought(
            subjects,
            difficult_subjects,
            daily_hours,
            int(exam_days),
            revision_percent,
            preferred_strategy
        )

        st.session_state["candidates"] = candidates
        st.session_state["study_input"] = {
            "student_id": student_id,
            "subjects": subjects,
            "difficult_subjects": difficult_subjects,
            "daily_hours": daily_hours,
            "exam_days": int(exam_days),
            "revision_percent": revision_percent,
            "preferred_strategy": preferred_strategy
        }

        st.session_state["explanation"] = explain_recommendation(
            candidates[0], api_key
        )

if "candidates" in st.session_state:
    candidates = st.session_state["candidates"]
    study_input = st.session_state["study_input"]

    st.divider()
    st.subheader("3. Compare the Three Plans")

    columns = st.columns(3)

    for index, candidate in enumerate(candidates):
        with columns[index]:
            st.metric(
                candidate["strategy"],
                f"{candidate['score']}/100"
            )

            st.write(
                f"Coverage: {candidate['coverage_score']}/100"
            )
            st.write(
                f"Difficult subjects: "
                f"{candidate['difficulty_score']}/100"
            )
            st.write(
                f"Revision fit: {candidate['revision_score']}/100"
            )

    best_plan = candidates[0]

    st.success(f"Recommended plan: {best_plan['strategy']}")

    st.subheader("4. Recommended Study Allocation")

    st.write(st.session_state["explanation"])

    st.dataframe(
        [
            {"Subject": subject, "Study Hours": hours}
            for subject, hours in best_plan["allocations"].items()
        ],
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "Revision Hours",
        best_plan["revision_hours"]
    )

    st.metric(
        "Total Available Hours",
        best_plan["total_hours"]
    )

    st.caption(
        "Scores are heuristic comparisons, not predictions of exam marks. "
        "Breaks, sleep and personal commitments should be planned separately."
    )

    st.subheader("5. Save Preferences to ChromaDB")

    if st.button("💾 Save Selected Study Preference"):
        preference = {
            **study_input,
            "selected_strategy": best_plan["strategy"],
            "selected_score": best_plan["score"]
        }

        try:
            save_preferences(student_id, preference)
            st.success(
                "Study preferences saved. Reuse the same Student ID "
                "to retrieve them in a future session."
            )
        except Exception as error:
            st.error(f"Could not save preferences: {error}")

    st.download_button(
        "Download Study Plan JSON",
        data=json.dumps(
            {
                "recommended_plan": best_plan,
                "all_candidates": candidates,
                "student_constraints": study_input
            },
            indent=2
        ),
        file_name="study_plan.json",
        mime="application/json"
    )

st.divider()

st.caption(
    "Study Plan Optimizer AI | LangChain | Tree-of-Thought-inspired "
    "candidate comparison | ChromaDB persistent memory"
)

