
import os
import re
import hashlib
import streamlit as st

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Healthy Meal Planner AI",
    page_icon="🥗",
    layout="wide"
)

st.title("🥗 Healthy Meal Planner AI")
st.write(
    "Multi-agent meal planning using Gemini AI, "
    "LangGraph, and FAISS memory."
)

MEMORY_DIR = "food_preferences_faiss"


# ---------------- LOCAL EMBEDDINGS ----------------
# FAISS memory embeddings work locally without another API.
class LocalEmbeddings(Embeddings):
    def __init__(self, dimension=128):
        self.dimension = dimension

    def embed_query(self, text):
        vector = [0.0] * self.dimension

        for word in re.findall(r"\w+", text.lower()):
            index = int(
                hashlib.md5(word.encode()).hexdigest(), 16
            ) % self.dimension
            vector[index] += 1.0

        norm = sum(x * x for x in vector) ** 0.5
        if norm:
            vector = [x / norm for x in vector]

        return vector

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


embeddings = LocalEmbeddings()


# ---------------- GEMINI MODEL ----------------
api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help="Get your key from Google AI Studio."
)

model_name = st.sidebar.selectbox(
    "Gemini model",
    ["gemini-2.5-flash", "gemini-2.0-flash"]
)

st.sidebar.caption(
    "Get an API key: "
)


def get_llm():
    if not api_key.strip():
        return None

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key.strip(),
        temperature=0.3
    )


# ---------------- FAISS MEMORY ----------------
def load_vectorstore():
    index_file = os.path.join(MEMORY_DIR, "index.faiss")

    if os.path.exists(index_file):
        return FAISS.load_local(
            MEMORY_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return None


def retrieve_preferences(user_id):
    store = load_vectorstore()

    if store is None:
        return "No previous preferences saved."

    try:
        results = store.similarity_search(
            f"Food preferences for user {user_id}",
            k=8,
            filter={"user_id": user_id}
        )

        if not results:
            return "No previous preferences saved."

        return "\n".join(doc.page_content for doc in results)

    except Exception:
        return "No previous preferences found."


def save_preferences(user_id, preferences):
    os.makedirs(MEMORY_DIR, exist_ok=True)

    document = Document(
        page_content=preferences,
        metadata={"user_id": user_id}
    )

    store = load_vectorstore()

    if store is None:
        store = FAISS.from_documents([document], embeddings)
    else:
        store.add_documents([document])

    store.save_local(MEMORY_DIR)


# ---------------- GRAPH STATE ----------------
class MealState(TypedDict, total=False):
    user_id: str
    preferences: str
    previous_preferences: str
    diet: str
    allergies: str
    budget: str
    goal: str
    request: str
    planner_result: str
    researcher_result: str
    checker_result: str


# ---------------- AI HELPER ----------------
def ask_gemini(prompt):
    llm = get_llm()

    if llm is None:
        return (
            "Gemini API key missing. Enter your API key "
            "in the sidebar to generate AI meal suggestions."
        )

    try:
        response = llm.invoke(prompt)
        return str(response.content)

    except Exception as error:
        return f"Gemini API error: {error}"


# ---------------- AGENT 1: PLANNER ----------------
def planner_agent(state: MealState):
    prompt = f"""
You are the Planner Agent for a simple healthy one-day meal plan.

User ID: {state['user_id']}
Diet type: {state['diet']}
Allergies or foods to avoid: {state['allergies']}
Budget: {state['budget']}
Health goal: {state['goal']}
New preferences: {state['preferences']}
Previously remembered preferences: {state['previous_preferences']}
Special request: {state['request']}

Create a practical one-day plan with:
- Breakfast
- Morning snack
- Lunch
- Evening snack
- Dinner

For every meal, list foods, main ingredients, and a short reason
why it fits the user's preferences. Include approximate calories
only when reasonable. Respect allergies and dietary restrictions.
Do not promise medical outcomes. Avoid ingredients listed as allergens.
"""
    result = ask_gemini(prompt)

    if result.startswith("Gemini API error") or result.startswith(
        "Gemini API key missing"
    ):
        result = f"""
Sample one-day plan (fallback; not personalized by Gemini)

Breakfast: Oats with fruit and unsweetened milk or suitable alternative.
Morning snack: A seasonal fruit.
Lunch: Rice or whole grains, dal, and cooked vegetables.
Evening snack: Roasted chickpeas, if suitable.
Dinner: Vegetable soup with whole grains and a suitable protein.

Diet: {state['diet']}
Foods to avoid: {state['allergies']}
Budget: {state['budget']}

Please review ingredients and allergens before eating.
"""
    return {"planner_result": result}


# ---------------- AGENT 2: RESEARCHER ----------------
def researcher_agent(state: MealState):
    prompt = f"""
You are the Researcher Agent. Review the proposed meal plan.

User diet: {state['diet']}
Allergies / foods to avoid: {state['allergies']}
Budget: {state['budget']}
Goal: {state['goal']}
Remembered preferences: {state['previous_preferences']}

Draft plan:
{state['planner_result']}

Improve the plan using general nutrition knowledge and practical,
affordable ingredients. Consider variety, vegetables, fruit, whole
grains, and suitable protein sources. Respect the stated diet and
allergies. Do not claim to have browsed the internet or verified
current prices. Mention substitutions where useful.
Return the revised meal plan and a short explanation.
"""
    result = ask_gemini(prompt)

    if result.startswith("Gemini API error") or result.startswith(
        "Gemini API key missing"
    ):
        result = (
            state["planner_result"]
            + "\n\nResearcher note: Include varied vegetables, fruit, "
              "whole grains, and an appropriate protein source. "
              "Check ingredient labels for allergens."
        )

    return {"researcher_result": result}


# ---------------- AGENT 3: CHECKER ----------------
def checker_agent(state: MealState):
    prompt = f"""
You are the Checker Agent. Audit this proposed one-day meal plan.

Diet: {state['diet']}
Allergies / foods to avoid: {state['allergies']}
Budget: {state['budget']}
Goal: {state['goal']}
Previous food preferences: {state['previous_preferences']}

Plan:
{state['researcher_result']}

Check:
1. Does it follow the diet type?
2. Are listed allergens or avoided foods included?
3. Is there reasonable variety across meals?
4. Is the plan practical for the budget?
5. What should be changed or verified?

Return:
- Status: PASS or NEEDS CHANGES
- Issues found
- Corrected final meal plan
- Short safety note

If an allergy conflict appears, clearly flag it and do not recommend
the conflicting meal. Do not present this as medical advice.
"""
    result = ask_gemini(prompt)

    if result.startswith("Gemini API error") or result.startswith(
        "Gemini API key missing"
    ):
        result = (
            "Status: REVIEW NEEDED\n\n"
            "Gemini was unavailable, so an AI safety check could not "
            "be completed.\n\n"
            + state["researcher_result"]
            + "\n\nSafety note: Verify every ingredient and allergen "
              "before preparing the meals."
        )

    return {"checker_result": result}


# ---------------- LANGGRAPH WORKFLOW ----------------
def build_graph():
    graph = StateGraph(MealState)

    graph.add_node("Planner", planner_agent)
    graph.add_node("Researcher", researcher_agent)
    graph.add_node("Checker", checker_agent)

    graph.set_entry_point("Planner")
    graph.add_edge("Planner", "Researcher")
    graph.add_edge("Researcher", "Checker")
    graph.add_edge("Checker", END)

    return graph.compile()


meal_graph = build_graph()


# ---------------- DASHBOARD INPUTS ----------------
st.subheader("👤 Student / User Details")

user_id = st.text_input("User ID", value="user001")

col1, col2 = st.columns(2)

with col1:
    diet = st.selectbox(
        "Diet type",
        [
            "Vegetarian",
            "Vegan",
            "Non-vegetarian",
            "Eggetarian"
        ]
    )

    goal = st.selectbox(
        "Meal goal",
        [
            "Balanced everyday eating",
            "More vegetables and fibre",
            "Higher-protein meals",
            "Budget-friendly healthy meals"
        ]
    )

    budget = st.selectbox(
        "Daily budget",
        ["Low budget", "Moderate budget", "Flexible budget"]
    )

with col2:
    allergies = st.text_input(
        "Allergies / foods to avoid",
        placeholder="e.g. peanuts, dairy, shellfish; None"
    )

    preferences = st.text_area(
        "Food preferences",
        placeholder="e.g. South Indian food, less oil, likes lentils"
    )

    request = st.text_input(
        "Extra request",
        placeholder="e.g. quick meals, no oven, office lunch"
    )


# ---------------- RUN AGENTS ----------------
if st.button("🍽️ Generate Healthy Meal Plan", type="primary"):
    if not user_id.strip():
        st.error("Please enter a User ID.")
    elif not api_key.strip():
        st.warning(
            "Enter your Gemini API key in the sidebar to run the "
            "Planner, Researcher, and Checker AI agents."
        )
    else:
        with st.spinner("Agents are preparing and checking your plan..."):
            previous = retrieve_preferences(user_id.strip())

            initial_state = {
                "user_id": user_id.strip(),
                "preferences": preferences or "No new preferences",
                "previous_preferences": previous,
                "diet": diet,
                "allergies": allergies or "None specified",
                "budget": budget,
                "goal": goal,
                "request": request or "No extra request"
            }

            try:
                result = meal_graph.invoke(initial_state)

                st.session_state["meal_result"] = result
                st.session_state["meal_user_id"] = user_id.strip()
                st.session_state["meal_preferences"] = (
                    f"Diet: {diet}; Goal: {goal}; Budget: {budget}; "
                    f"Allergies/avoid: {allergies or 'None specified'}; "
                    f"Preferences: {preferences or 'Not specified'}; "
                    f"Extra request: {request or 'None'}"
                )
                st.success("All three agents completed their workflow.")

            except Exception as error:
                st.error(f"Could not generate the meal plan: {error}")


# ---------------- DISPLAY RESULTS ----------------
if "meal_result" in st.session_state:
    result = st.session_state["meal_result"]

    st.divider()
    st.header("📋 Agent Results")

    with st.expander("1️⃣ Planner Agent — Initial Plan", expanded=True):
        st.markdown(result.get("planner_result", "No result"))

    with st.expander("2️⃣ Researcher Agent — Improved Plan", expanded=True):
        st.markdown(result.get("researcher_result", "No result"))

    st.subheader("3️⃣ Checker Agent — Final Review")
    st.markdown(result.get("checker_result", "No result"))

    st.caption(
        "AI-generated suggestions may contain errors. Check allergens, "
        "ingredient labels, and portion sizes. Consult a qualified "
        "health professional for medical dietary needs."
    )

    if st.button("💾 Save My Food Preferences to FAISS"):
        try:
            save_preferences(
                st.session_state["meal_user_id"],
                st.session_state["meal_preferences"]
            )
            st.success(
                "Preferences saved! Use the same User ID next time."
            )
        except Exception as error:
            st.error(f"Could not save preferences: {error}")

    st.download_button(
        "⬇️ Download Meal Plan",
        data=(
            "PLANNER AGENT\n"
            + result.get("planner_result", "")
            + "\n\nRESEARCHER AGENT\n"
            + result.get("researcher_result", "")
            + "\n\nCHECKER AGENT\n"
            + result.get("checker_result", "")
        ),
        file_name="healthy_meal_plan.txt",
        mime="text/plain"
    )

st.divider()
st.caption("Powered by LangGraph • Gemini AI • FAISS")
