import os
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fake import FakeEmbeddings

load_dotenv()

# Optional AI and embedding support
API_KEY = os.getenv("", "").strip()

DATA_DIR = Path("purchase_data")
DATA_DIR.mkdir(exist_ok=True)

PURCHASE_FILE = DATA_DIR / "purchases.json"
FAISS_DIR = DATA_DIR / "faiss_index"

st.set_page_config(
    page_title="Smart Purchase Agent",
    page_icon="🛒",
    layout="wide"
)

# ------------------ STORAGE ------------------

def load_purchases():
    if PURCHASE_FILE.exists():
        try:
            with open(PURCHASE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_purchases(purchases):
    with open(PURCHASE_FILE, "w", encoding="utf-8") as f:
        json.dump(purchases, f, indent=2, ensure_ascii=False)


def get_embeddings():
    if API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=API_KEY
            )
        except Exception:
            pass

    # Offline fallback: FAISS still works, but retrieval quality
    # is lower than with a semantic embedding model.
    return FakeEmbeddings(size=64)


@st.cache_resource
def get_vector_store():
    embeddings = get_embeddings()

    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        try:
            return FAISS.load_local(
                str(FAISS_DIR),
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            pass

    return FAISS.from_documents(
        [Document(page_content="Purchase memory initialized.")],
        embeddings
    )


def save_to_memory(purchase):
    store = get_vector_store()

    text = (
        f"Purchase ID: {purchase['id']}. "
        f"Date: {purchase['date']}. "
        f"Items: {purchase['items']}. "
        f"Total cost: Rs. {purchase['total']:.2f}."
    )

    store.add_documents([
        Document(
            page_content=text,
            metadata={"purchase_id": purchase["id"]}
        )
    ])

    store.save_local(str(FAISS_DIR))


def search_memory(query):
    store = get_vector_store()
    results = store.similarity_search(query, k=5)

    return [
        doc.page_content
        for doc in results
        if doc.metadata.get("purchase_id")
    ]


# ------------------ PURCHASE TOOLS ------------------

@tool
def calculate_total(items_json: str) -> str:
    """Calculate the total purchase cost from a JSON list.
    Each item must contain name, quantity and price.
    Example: [{"name":"Pen","quantity":2,"price":10}]
    """
    try:
        items = json.loads(items_json)
        total = 0.0

        for item in items:
            quantity = float(item["quantity"])
            price = float(item["price"])

            if quantity < 0 or price < 0:
                return "Error: quantity and price cannot be negative."

            total += quantity * price

        return f"Total purchase cost: Rs. {total:.2f}"

    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return "Invalid item data. Provide valid JSON with name, quantity and price."


@tool
def retrieve_previous_purchases(question: str) -> str:
    """Search FAISS memory for relevant previous purchases.
    Use this when the user asks about earlier purchases or past spending.
    """
    try:
        results = search_memory(question)

        if not results:
            return "No matching previous purchases found."

        return "\n".join(results)

    except Exception as exc:
        return f"Could not retrieve purchase memory: {exc}"


# ------------------ REACT AGENT ------------------

def create_purchase_agent():
    if not API_KEY:
        return None

    from langchain_openai import ChatOpenAI
    from langchain_classic.agents import (
        create_react_agent,
        AgentExecutor
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=API_KEY
    )

    tools = [calculate_total, retrieve_previous_purchases]

    prompt = PromptTemplate.from_template("""
You are a helpful shopping purchase assistant.

Answer the user's question accurately.
Use calculate_total for arithmetic.
Use retrieve_previous_purchases for questions about purchase history.
Do not invent previous purchases.
Use this format for your reasoning and tool actions:

Question: {input}
Thought: Decide which tool is useful.
{agent_scratchpad}
""".replace(
        "Thought: Decide which tool is useful.",
        "Answer the question using the available tools when needed."
    ) + """

You have access to these tools:
{tools}

Tool names: {tool_names}

Follow the ReAct format:
Thought: brief decision
Action: one of [{tool_names}]
Action Input: input for the tool
Observation: tool result
... repeat if needed ...
Thought: I have enough information
Final: response to the user

Question: {input}
{agent_scratchpad}
""")

    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5
    )


# ------------------ DASHBOARD ------------------

st.title("🛒 Smart Purchase Agent")
st.caption("LangChain ReAct • FAISS Purchase Memory • Python Dashboard")

purchases = load_purchases()

total_spent = sum(p["total"] for p in purchases)
total_items = sum(
    len(p["items"]) for p in purchases
)

c1, c2, c3 = st.columns(3)

c1.metric("Total Spent", f"₹{total_spent:,.2f}")
c2.metric("Purchase Records", len(purchases))
c3.metric("Item Types / Entries", total_items)

st.divider()

# ------------------ ADD PURCHASE ------------------

st.subheader("➕ Add a Purchase")

with st.form("purchase_form", clear_on_submit=True):
    item_name = st.text_input("Item name", placeholder="Example: Notebook")
    col1, col2 = st.columns(2)

    with col1:
        quantity = st.number_input(
            "Quantity", min_value=1, value=1, step=1
        )

    with col2:
        price = st.number_input(
            "Price per item (₹)",
            min_value=0.0,
            value=50.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "Save Purchase",
        type="primary"
    )

    if submitted:
        if not item_name.strip():
            st.error("Please enter an item name.")
        else:
            item = {
                "name": item_name.strip(),
                "quantity": int(quantity),
                "price": float(price)
            }

            total = item["quantity"] * item["price"]

            purchase = {
                "id": str(int(datetime.now().timestamp() * 1000000)),
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "items": [item],
                "total": total
            }

            purchases.append(purchase)
            save_purchases(purchases)

            try:
                save_to_memory(purchase)
                st.success(
                    f"Purchase saved! Total: ₹{total:,.2f}. "
                    "FAISS memory updated."
                )
            except Exception as exc:
                st.warning(
                    f"Purchase saved, but FAISS update failed: {exc}"
                )

            st.rerun()

# ------------------ PURCHASE HISTORY ------------------

st.subheader("📋 Purchase History")

if purchases:
    rows = []

    for purchase in reversed(purchases):
        names = ", ".join(
            f"{i['name']} × {i['quantity']}"
            for i in purchase["items"]
        )

        rows.append({
            "Date": purchase["date"],
            "Items": names,
            "Total (₹)": round(purchase["total"], 2)
        })

    st.dataframe(rows, use_container_width=True)
else:
    st.info("No purchases yet. Add your first purchase above.")

# ------------------ AI CHAT ------------------

st.divider()
st.subheader("🤖 Ask Your Purchase Agent")

st.write(
    "Examples: 'Calculate 3 pens at ₹10 each' or "
    "'What did I purchase earlier?'"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about costs or previous purchases...")

if question:
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer = ""

        try:
            agent = create_purchase_agent()

            if agent:
                result = agent.invoke({"input": question})
                answer = result.get(
                    "output", "I could not generate an answer."
                )
            else:
                q = question.lower()

                if any(word in q for word in [
                    "previous", "earlier", "history", "past",
                    "before", "last purchase"
                ]):
                    matches = search_memory(question)

                    if matches:
                        answer = "\n\n".join(matches)
                    else:
                        answer = "No matching previous purchases found."

                elif any(word in q for word in [
                    "total", "spent", "spending", "cost",
                    "purchase amount"
                ]):
                    answer = (
                        f"Your recorded total spending is "
                        f"₹{sum(p['total'] for p in load_purchases()):,.2f}."
                    )
                else:
                    answer = (
                        "Add purchases using the form. I can calculate "
                        "totals and search purchase history. For natural "
                        "language ReAct AI, configure OPENAI_API_KEY in .env."
                    )

        except Exception as exc:
            answer = (
                f"Agent error: {exc}\n\n"
                "Check your API key, internet connection, and installed "
                "LangChain packages."
            )

        st.markdown(answer)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

# ------------------ MEMORY STATUS ------------------

st.divider()
st.subheader("🧠 FAISS Memory Status")

st.write(
    "Memory folder:",
    str(FAISS_DIR.resolve())
)

st.write(
    "Embedding mode:",
    "OpenAI embeddings" if API_KEY else "Offline demo embeddings"
)

st.caption(
    "Purchase records are saved in purchase_data/purchases.json. "
    "FAISS index is saved in purchase_data/faiss_index."
)