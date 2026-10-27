
import os
import re
import json
import hashlib

import streamlit as st
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.agents import AgentExecutor, create_react_agent


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Student Monthly Budget Planner",
    page_icon="💰",
    layout="wide"
)

DB_DIRECTORY = "student_expense_memory"
COLLECTION_NAME = "expense_categories"


# ---------------- LOCAL EMBEDDINGS ----------------
# Simple local embeddings mean no separate embedding API is required.

class LocalEmbeddings(Embeddings):
    def __init__(self, dimensions=256):
        self.dimensions = dimensions

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        words = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]

        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


# ---------------- CHROMADB MEMORY ----------------

@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIRECTORY,
        embedding_function=LocalEmbeddings()
    )


def save_expense_categories(student_id, categories, monthly_budget, notes):
    store = get_vector_store()

    content = (
        f"Student ID: {student_id}\n"
        f"Expense categories: {categories}\n"
        f"Monthly budget: Rs. {monthly_budget}\n"
        f"Notes: {notes}"
    )

    record_id = hashlib.sha256(
        f"{student_id}|{content}".encode("utf-8")
    ).hexdigest()

    store.add_documents(
        documents=[
            Document(
                page_content=content,
                metadata={
                    "student_id": student_id,
                    "type": "expense_record",
                    "categories": categories
                }
            )
        ],
        ids=[record_id]
    )


def search_expense_memory(student_id, query, limit=5):
    store = get_vector_store()

    results = store.similarity_search(
        query,
        k=limit,
        filter={"student_id": student_id}
    )

    return "\n\n".join(
        document.page_content
        for document in results
        if document.metadata.get("type") == "expense_record"
    ) or "No previous expense records found."


# ---------------- REACT AGENT ----------------

def create_budget_agent(api_key, model_name, student_id):

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )

    @tool
    def calculate_budget(budget_details: str) -> str:
        """Calculate available spending money and total expenses.
        Input must be JSON with monthly_budget, savings_goal,
        and fixed_expenses. fixed_expenses must be a list of numbers.
        Example: {"monthly_budget":10000,"savings_goal":1000,
        "fixed_expenses":[3000,800]}"""

        try:
            data = json.loads(budget_details)

            budget = float(data["monthly_budget"])
            savings = float(data["savings_goal"])
            fixed = [float(value) for value in data["fixed_expenses"]]

            if budget < 0 or savings < 0 or any(value < 0 for value in fixed):
                return "Error: amounts cannot be negative."

            total_fixed = sum(fixed)
            available = budget - savings - total_fixed

            return (
                f"Monthly budget: Rs. {budget:.2f}\n"
                f"Savings goal: Rs. {savings:.2f}\n"
                f"Total fixed expenses: Rs. {total_fixed:.2f}\n"
                f"Remaining for variable expenses: Rs. {available:.2f}\n"
                f"Budget status: "
                f"{'Within budget' if available >= 0 else 'Over budget'}"
            )

        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return (
                "Invalid input. Provide JSON with monthly_budget, "
                "savings_goal, and fixed_expenses as a list of numbers."
            )

    @tool
    def search_previous_expenses(query: str) -> str:
        """Search ChromaDB for this student's previously saved expense
        categories and budget records. Give a short search query."""

        return search_expense_memory(student_id, query)

    tools = [calculate_budget, search_previous_expenses]

    prompt = PromptTemplate.from_template(
        """You are a helpful ReAct budget-planning agent for college students.

Use the tools when needed:
- calculate_budget: calculate savings, fixed costs, and available money.
- search_previous_expenses: retrieve this student's saved expense categories.

Student ID: {student_id}

Create a realistic monthly budget based on the student's request.
Use the supplied amounts; never invent actual expenses or past records.
If a required amount is missing, clearly state your assumption or ask
for clarification. Keep the budget within the available money.
Include categories such as food, travel, accommodation, study materials,
mobile recharge, personal expenses, and entertainment when relevant.
Previous records are hints, not guaranteed current expenses.

Follow the ReAct format below. Use observations from tools to inform
your answer. Do not expose private internal reasoning; provide a concise
summary of the calculations and recommendations in the final answer.

Question: {input}

{agent_scratchpad}"""
    )

    # Supply the student ID to the prompt at runtime.
    prompt = prompt.partial(student_id=student_id)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6
    )


# ---------------- STREAMLIT DASHBOARD ----------------

st.title("💰 AI Monthly Budget Planner")
st.write(
    "Plan your monthly college expenses using a ReAct agent "
    "and remember expense categories with ChromaDB."
)

with st.sidebar:
    st.header("⚙️ AI Settings")
    api_key = st.text_input("Gemini API Key", type="password")

    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.0-flash"]
    )

st.subheader("👨‍🎓 Student Details")

student_id = st.text_input(
    "Student ID / Nickname",
    value="student1",
    help="Use the same ID in future sessions to retrieve saved expenses."
)

col1, col2 = st.columns(2)

with col1:
    monthly_budget = st.number_input(
        "Monthly budget (₹)",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    accommodation = st.number_input(
        "Accommodation / Hostel (₹)",
        min_value=0.0,
        value=3000.0,
        step=100.0
    )

    food = st.number_input(
        "Food expenses (₹)",
        min_value=0.0,
        value=2000.0,
        step=100.0
    )

with col2:
    travel = st.number_input(
        "Travel expenses (₹)",
        min_value=0.0,
        value=800.0,
        step=100.0
    )

    study = st.number_input(
        "Study materials (₹)",
        min_value=0.0,
        value=500.0,
        step=100.0
    )

    savings_goal = st.number_input(
        "Monthly savings goal (₹)",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

mobile = st.number_input(
    "Mobile / Internet (₹)",
    min_value=0.0,
    value=300.0,
    step=50.0
)

other_expenses = st.text_input(
    "Other expenses (optional)",
    value="Entertainment, personal care"
)

notes = st.text_area(
    "Additional requirements",
    placeholder=(
        "Example: I travel by bus, want to save more, "
        "and need money for project materials."
    )
)

if st.button(
    "🚀 Create Monthly Budget",
    type="primary",
    use_container_width=True
):
    if not api_key.strip():
        st.error("Enter your Gemini API key in the sidebar.")
    elif not student_id.strip():
        st.error("Enter a student ID or nickname.")
    else:
        try:
            fixed_expenses = [
                accommodation,
                food,
                travel,
                study,
                mobile
            ]

            fixed_total = sum(fixed_expenses)
            available = monthly_budget - savings_goal - fixed_total

            request_text = f"""
Create a monthly budget for a college student.

Monthly budget: Rs. {monthly_budget:.2f}
Savings goal: Rs. {savings_goal:.2f}

User-provided expenses:
Accommodation: Rs. {accommodation:.2f}
Food: Rs. {food:.2f}
Travel: Rs. {travel:.2f}
Study materials: Rs. {study:.2f}
Mobile/internet: Rs. {mobile:.2f}
Other interests/categories: {other_expenses}
Additional requirements: {notes}

The known expenses total Rs. {fixed_total:.2f}.
After savings and known expenses, the remaining amount is Rs. {available:.2f}.

Use the calculation tool to verify the figures and search previous
expense records for useful category ideas.

Provide:
1. A category-wise monthly budget table.
2. Estimated amount for each category.
3. Total planned expenses and savings.
4. Remaining balance or budget shortfall.
5. Two practical money-saving tips.

Do not exceed the budget silently. If the requested savings and costs
do not fit, identify the shortfall and suggest adjustments.
Clearly distinguish user-provided figures from suggested allocations.
"""

            with st.spinner("ReAct agent is planning your budget..."):
                agent = create_budget_agent(
                    api_key.strip(),
                    model_name,
                    student_id.strip()
                )

                result = agent.invoke({"input": request_text})

            st.session_state["budget_result"] = result["output"]
            st.session_state["budget_student_id"] = student_id.strip()
            st.session_state["budget_categories"] = (
                f"Accommodation, Food, Travel, Study materials, "
                f"Mobile/Internet, {other_expenses}"
            )
            st.session_state["budget_amount"] = monthly_budget
            st.session_state["budget_notes"] = notes

            st.success("Monthly budget created!")

        except Exception as error:
            st.error(f"Could not create the budget: {error}")
            st.info(
                "Check your API key, internet connection, model access, "
                "and installed package versions."
            )


# ---------------- DISPLAY BUDGET ----------------

if st.session_state.get("budget_result"):
    st.subheader("📊 Your Monthly Budget Plan")
    st.markdown(st.session_state["budget_result"])

    st.download_button(
        "⬇️ Download Budget Plan",
        data=st.session_state["budget_result"],
        file_name="student_monthly_budget.md",
        mime="text/markdown",
        use_container_width=True
    )


# ---------------- SAVE CATEGORY MEMORY ----------------

st.divider()
st.subheader("🧠 Save Expense Categories to ChromaDB")

st.write(
    "Save the categories you used so future budgets can retrieve them."
)

with st.form("save_expenses_form"):
    save_student_id = st.text_input(
        "Student ID for memory",
        value=st.session_state.get("budget_student_id", "student1")
    )

    saved_categories = st.text_area(
        "Expense categories",
        value=st.session_state.get(
            "budget_categories",
            "Food, Travel, Accommodation, Study materials, Mobile"
        )
    )

    saved_budget = st.number_input(
        "Monthly budget to remember (₹)",
        min_value=0.0,
        value=float(st.session_state.get("budget_amount", 10000.0)),
        step=500.0
    )

    saved_notes = st.text_area(
        "Notes about this budget",
        value=st.session_state.get("budget_notes", "")
    )

    save_clicked = st.form_submit_button("💾 Save to ChromaDB")

if save_clicked:
    if not save_student_id.strip() or not saved_categories.strip():
        st.warning("Enter a student ID and expense categories.")
    else:
        try:
            save_expense_categories(
                save_student_id.strip(),
                saved_categories.strip(),
                saved_budget,
                saved_notes.strip()
            )
            st.success("Expense categories saved successfully!")
        except Exception as error:
            st.error(f"Could not save memory: {error}")


# ---------------- SEARCH OLD EXPENSE CATEGORIES ----------------

st.divider()
st.subheader("🔎 Search Previous Expense Categories")

search_student_id = st.text_input(
    "Student ID to search",
    value=st.session_state.get("budget_student_id", "student1"),
    key="memory_student_id"
)

search_query = st.text_input(
    "Search query",
    value="food, travel and study expenses"
)

if st.button("Search ChromaDB Memory", use_container_width=True):
    if not search_student_id.strip() or not search_query.strip():
        st.warning("Enter a student ID and search query.")
    else:
        try:
            matches = search_expense_memory(
                search_student_id.strip(),
                search_query.strip(),
                limit=5
            )

            if matches == "No previous expense records found.":
                st.info(matches)
            else:
                st.markdown(matches)

        except Exception as error:
            st.error(f"Memory search failed: {error}")

st.caption(
    "ChromaDB memory is stored locally. Local hash embeddings provide "
    "approximate similarity search, not full semantic understanding."
)
