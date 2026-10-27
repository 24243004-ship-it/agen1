import streamlit as st
import chromadb
import hashlib
import json
import random
from collections import Counter

# -------------------- CONFIGURATION --------------------

st.set_page_config(
    page_title="Smart Timetable AI",
    page_icon="📚",
    layout="wide"
)

DB_PATH = "./timetable_chroma_db"

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="timetable_preferences"
)


# -------------------- LOCAL EMBEDDING --------------------
# Simple deterministic embeddings: no API key required.

def make_embedding(text, dimensions=128):
    vector = [0.0] * dimensions
    words = text.lower().split()

    for word in words:
        digest = hashlib.sha256(word.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = sum(value * value for value in vector) ** 0.5

    if norm:
        vector = [value / norm for value in vector]

    return vector


# -------------------- CHROMADB MEMORY --------------------

def save_constraints(user_id, constraints):
    document = json.dumps(constraints, ensure_ascii=False)

    record_id = f"{user_id}_{random.randint(100000, 999999)}"

    collection.add(
        ids=[record_id],
        documents=[document],
        metadatas=[{"user_id": user_id}],
        embeddings=[make_embedding(document)]
    )


def retrieve_constraints(user_id):
    try:
        result = collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )

        documents = result.get("documents") or []

        if documents:
            return json.loads(documents[-1])

    except Exception:
        pass

    return None


# -------------------- TREE-OF-THOUGHT ENGINE --------------------

def generate_candidate(subjects, days, periods, strategy):
    slots = len(days) * periods

    # Repeat subjects when there are more timetable slots than subjects.
    subject_list = [
        subjects[i % len(subjects)]
        for i in range(slots)
    ]

    if strategy == "Balanced":
        # Rotate subjects each day to spread them across the week.
        schedule = []

        for day_index in range(len(days)):
            start = day_index % len(subjects)
            daily_subjects = [
                subjects[(start + i) % len(subjects)]
                for i in range(periods)
            ]
            schedule.append(daily_subjects)

        return schedule

    if strategy == "Rotated":
        # Change the ordering of subjects for a different candidate.
        offset = max(1, len(subjects) // 2)
        rotated = subjects[offset:] + subjects[:offset]
        sequence = [
            rotated[i % len(rotated)]
            for i in range(slots)
        ]

    else:
        # Sequential arrangement.
        sequence = subject_list

    return [
        sequence[i * periods:(i + 1) * periods]
        for i in range(len(days))
    ]


def evaluate_timetable(schedule, subjects):
    adjacent_penalty = 0

    # Penalize the same subject appearing in adjacent periods.
    for daily_schedule in schedule:
        for i in range(1, len(daily_schedule)):
            if daily_schedule[i] == daily_schedule[i - 1]:
                adjacent_penalty += 1

    all_subjects = [
        subject
        for daily_schedule in schedule
        for subject in daily_schedule
    ]

    counts = Counter(all_subjects)

    # Penalize uneven subject distribution.
    expected = len(all_subjects) / max(1, len(subjects))
    balance_penalty = sum(
        abs(counts.get(subject, 0) - expected)
        for subject in subjects
    )

    score = max(
        0,
        100 - (adjacent_penalty * 10) - (balance_penalty * 2)
    )

    return {
        "score": round(score, 1),
        "adjacent_repeats": adjacent_penalty,
        "balance_penalty": round(balance_penalty, 1)
    }


def tree_of_thought(subjects, days, periods):
    strategies = ["Sequential", "Balanced", "Rotated"]
    candidates = []

    # Three candidate branches are explored and evaluated.
    for index, strategy in enumerate(strategies, start=1):
        schedule = generate_candidate(
            subjects, days, periods, strategy
        )

        metrics = evaluate_timetable(schedule, subjects)

        candidates.append({
            "name": f"Solution {index}",
            "strategy": strategy,
            "schedule": schedule,
            "score": metrics["score"],
            "adjacent_repeats": metrics["adjacent_repeats"],
            "balance_penalty": metrics["balance_penalty"]
        })

    # Highest-scoring solution is recommended.
    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True
    )

    return candidates


# -------------------- DASHBOARD --------------------

st.title("📚 Smart Timetable AI")
st.write(
    "Tree-of-Thought approach to generate, compare, "
    "and select college timetable solutions."
)

st.info(
    "The agent explores three timetable strategies, evaluates "
    "their subject distribution and consecutive repeats, "
    "and recommends the highest-scoring candidate."
)

# -------------------- USER INPUT --------------------

with st.sidebar:
    st.header("Student Profile")

    user_id = st.text_input(
        "Student ID",
        value="student01"
    ).strip()

    st.caption("Use the same ID to retrieve saved constraints.")

st.subheader("1. Enter Timetable Requirements")

subjects_text = st.text_input(
    "Subjects (comma-separated)",
    value="Python, DBMS, Mathematics, English, Networks"
)

days_text = st.text_input(
    "College days (comma-separated)",
    value="Monday, Tuesday, Wednesday, Thursday, Friday"
)

periods = st.number_input(
    "Periods per day",
    min_value=1,
    max_value=10,
    value=5
)

break_period = st.checkbox(
    "Include a lunch break after period 3",
    value=False
)

extra_constraints = st.text_area(
    "Additional preferences",
    placeholder=(
        "Example: Mathematics in the morning, "
        "avoid repeated subjects, Friday should be lighter."
    )
)

# -------------------- LOAD PREVIOUS MEMORY --------------------

if st.button("Load Previous Preferences"):
    previous = retrieve_constraints(user_id)

    if previous:
        st.session_state["previous_constraints"] = previous
        st.success("Previous constraints retrieved from ChromaDB.")
        st.json(previous)
    else:
        st.warning("No saved preferences found for this Student ID.")

# -------------------- GENERATE SOLUTIONS --------------------

if st.button("🌳 Generate and Compare 3 Timetables",
             type="primary"):

    subjects = [
        item.strip()
        for item in subjects_text.split(",")
        if item.strip()
    ]

    days = [
        item.strip()
        for item in days_text.split(",")
        if item.strip()
    ]

    if not user_id:
        st.error("Please enter a Student ID.")
    elif len(subjects) < 2:
        st.error("Enter at least two subjects.")
    elif not days:
        st.error("Enter at least one college day.")
    else:
        constraints = {
            "subjects": subjects,
            "days": days,
            "periods_per_day": int(periods),
            "lunch_break_after_period_3": break_period,
            "additional_preferences": extra_constraints
        }

        previous = retrieve_constraints(user_id)

        # Use the saved constraints as a reference.
        if previous:
            st.session_state["previous_constraints"] = previous

        candidates = tree_of_thought(
            subjects, days, int(periods)
        )

        st.session_state["candidates"] = candidates
        st.session_state["current_constraints"] = constraints
        st.session_state["current_user_id"] = user_id

        st.success("Three timetable candidates generated!")

# -------------------- DISPLAY COMPARISON --------------------

if "candidates" in st.session_state:
    candidates = st.session_state["candidates"]
    constraints = st.session_state["current_constraints"]

    st.subheader("2. Compare the Three Solutions")

    comparison_columns = st.columns(3)

    for index, candidate in enumerate(candidates):
        with comparison_columns[index]:
            st.metric(
                candidate["name"],
                f'{candidate["score"]}/100'
            )
            st.write(f'**Strategy:** {candidate["strategy"]}')
            st.write(
                f'Adjacent repeats: {candidate["adjacent_repeats"]}'
            )
            st.write(
                f'Balance penalty: {candidate["balance_penalty"]}'
            )

    st.caption(
        "Scores are heuristic estimates, not a guarantee of "
        "teacher availability or room-conflict resolution."
    )

    st.subheader("3. View Timetable")

    candidate_names = [item["name"] for item in candidates]

    selected_name = st.selectbox(
        "Choose a timetable to inspect",
        candidate_names
    )

    selected = next(
        item for item in candidates
        if item["name"] == selected_name
    )

    schedule = selected["schedule"]
    days = constraints["days"]
    periods_per_day = constraints["periods_per_day"]

    table_data = {"Day": days}

    for period_index in range(periods_per_day):
        table_data[f"Period {period_index + 1}"] = [
            schedule[day_index][period_index]
            for day_index in range(len(days))
        ]

    if constraints["lunch_break_after_period_3"]:
        if periods_per_day >= 3:
            st.caption(
                "Lunch break requested after period 3. "
                "This simple version records the preference but "
                "does not create a separate break column."
            )

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("4. Save Your Selected Constraints")

    st.write(
        "Select the solution you prefer, then save the current "
        "constraints to ChromaDB for future sessions."
    )

    if st.button("💾 Save Constraints to ChromaDB"):
        try:
            save_constraints(
                st.session_state["current_user_id"],
                constraints
            )
            st.success(
                "Constraints saved successfully! "
                "They can be retrieved using the same Student ID."
            )
        except Exception as error:
            st.error(f"Unable to save memory: {error}")

    st.download_button(
        "Download Selected Timetable JSON",
        data=json.dumps(
            {
                "selected_solution": selected,
                "constraints": constraints
            },
            indent=2
        ),
        file_name="college_timetable.json",
        mime="application/json"
    )

st.divider()

st.caption(
    "Smart Timetable AI | Tree-of-Thought candidate search | "
    "ChromaDB persistent memory"
)