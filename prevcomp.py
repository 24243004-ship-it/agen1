
import hashlib
import re
from typing import TypedDict

import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma


# -------------------- CONFIGURATION --------------------

st.set_page_config(
    page_title="Programming Language Comparator",
    page_icon="💻",
    layout="wide"
)

DB_DIRECTORY = "chroma_comparison_memory"
COLLECTION_NAME = "language_comparisons"


# -------------------- LOCAL EMBEDDINGS --------------------
# Local embeddings avoid needing a separate embedding API.
# ChromaDB stores and retrieves comparison documents.

class LocalEmbeddings(Embeddings):
    def __init__(self, dimensions=256):
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude:
            vector = [value / magnitude for value in vector]

        return vector


@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIRECTORY,
        embedding_function=LocalEmbeddings()
    )


# -------------------- LANGGRAPH STATE --------------------

class ComparisonState(TypedDict, total=False):
    language1: str
    language2: str
    criteria: str
    plan: str
    research: str
    final_result: str


# -------------------- AGENT WORKFLOW --------------------

def build_comparison_graph(api_key: str, model_name: str):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )

    def planner_agent(state: ComparisonState):
        prompt = f"""
You are the Planner Agent.

Prepare a simple comparison plan for these programming languages:
Language 1: {state['language1']}
Language 2: {state['language2']}

Compare them using these criteria:
{state['criteria']}

Give a clear plan with the comparison criteria and the points
the Researcher Agent should investigate. Do not write the final
comparison yet.
"""
        response = llm.invoke(prompt)
        return {"plan": response.content}

    def researcher_agent(state: ComparisonState):
        prompt = f"""
You are the Researcher Agent.

Prepare a factual, beginner-friendly comparison of:
{state['language1']} and {state['language2']}.

Follow this plan:
{state['plan']}

Cover these criteria:
{state['criteria']}

Include:
- A Markdown comparison table
- Strengths and limitations of each language
- Common use cases
- Which language may suit a beginner, with context
- A short conclusion

Avoid unsupported exact performance claims. Explain that the best
choice depends on the project. If a fact varies by version or
environment, say so. Do not invent benchmark results.
"""
        response = llm.invoke(prompt)
        return {"research": response.content}

    def checker_agent(state: ComparisonState):
        prompt = f"""
You are the Checker Agent. Review and improve this comparison.

Languages: {state['language1']} and {state['language2']}
Required criteria: {state['criteria']}

Research draft:
{state['research']}

Check that:
1. Both languages are compared fairly.
2. The required criteria are covered.
3. The result is clear and suitable for a student.
4. There are no obvious contradictions or unsupported precise claims.

Correct issues and return ONLY the final comparison in Markdown.
Keep the table and a concise conclusion. Do not claim that you
verified facts using live sources.
"""
        response = llm.invoke(prompt)
        return {"final_result": response.content}

    graph = StateGraph(ComparisonState)
    graph.add_node("Planner", planner_agent)
    graph.add_node("Researcher", researcher_agent)
    graph.add_node("Checker", checker_agent)

    graph.add_edge(START, "Planner")
    graph.add_edge("Planner", "Researcher")
    graph.add_edge("Researcher", "Checker")
    graph.add_edge("Checker", END)

    return graph.compile()


# -------------------- STREAMLIT DASHBOARD --------------------

st.title("💻 Programming Language Comparison")
st.write(
    "Compare two programming languages using three AI agents "
    "and save previous results in ChromaDB."
)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your API key here. Do not share it."
    )

    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.0-flash"]
    )

    st.info("Previous comparisons are stored locally in ChromaDB.")

col1, col2 = st.columns(2)

with col1:
    language1 = st.text_input(
        "First programming language",
        value="Python"
    )

with col2:
    language2 = st.text_input(
        "Second programming language",
        value="Java"
    )

criteria = st.multiselect(
    "Comparison criteria",
    [
        "Syntax and readability",
        "Ease of learning",
        "Performance",
        "Applications and use cases",
        "Libraries and ecosystem",
        "Career opportunities"
    ],
    default=[
        "Syntax and readability",
        "Ease of learning",
        "Performance",
        "Applications and use cases"
    ]
)

compare_clicked = st.button(
    "🚀 Compare Languages",
    type="primary",
    use_container_width=True
)

if compare_clicked:
    if not api_key.strip():
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not language1.strip() or not language2.strip():
        st.error("Please enter both programming language names.")
    elif language1.strip().lower() == language2.strip().lower():
        st.error("Please enter two different programming languages.")
    elif not criteria:
        st.error("Please select at least one comparison criterion.")
    else:
        try:
            with st.spinner("The Planner, Researcher and Checker agents are working..."):
                app = build_comparison_graph(api_key.strip(), model_name)

                result = app.invoke({
                    "language1": language1.strip(),
                    "language2": language2.strip(),
                    "criteria": ", ".join(criteria)
                })

                final_result = result["final_result"]

            st.session_state["last_result"] = final_result
            st.session_state["last_languages"] = (
                language1.strip(),
                language2.strip()
            )

            st.success("Comparison completed!")
            st.subheader("📊 Final Comparison")
            st.markdown(final_result)

            if st.button("💾 Save Comparison to ChromaDB"):
                store = get_vector_store()
                document = Document(
                    page_content=(
                        f"Programming language comparison: "
                        f"{language1.strip()} vs {language2.strip()}\n\n"
                        f"Criteria: {', '.join(criteria)}\n\n"
                        f"{final_result}"
                    ),
                    metadata={
                        "language1": language1.strip(),
                        "language2": language2.strip(),
                        "criteria": ", ".join(criteria)
                    }
                )

                store.add_documents(
                    [document],
                    ids=[hashlib.sha256(
                        (
                            language1.strip().lower()
                            + "|"
                            + language2.strip().lower()
                            + "|"
                            + final_result
                        ).encode("utf-8")
                    ).hexdigest()]
                )
                st.success("Comparison saved in ChromaDB memory!")

        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.info(
                "Check your internet connection, API key, model access, "
                "and installed package versions."
            )


# Show the latest result again after Streamlit reruns
if "last_result" in st.session_state and not compare_clicked:
    st.subheader("📊 Latest Comparison")
    st.markdown(st.session_state["last_result"])


# -------------------- SEARCH PREVIOUS COMPARISONS --------------------

st.divider()
st.header("🧠 Previous Comparison Memory")

search_query = st.text_input(
    "Search saved comparisons",
    placeholder="Example: Python vs Java"
)

if st.button("🔍 Search ChromaDB", use_container_width=True):
    if not search_query.strip():
        st.warning("Enter a language or topic to search.")
    else:
        try:
            store = get_vector_store()
            documents = store.similarity_search(search_query, k=5)

            if not documents:
                st.info("No saved comparisons found.")
            else:
                st.subheader("Retrieved Results")
                for index, document in enumerate(documents, start=1):
                    with st.expander(f"Saved comparison {index}"):
                        st.caption(
                            f"{document.metadata.get('language1', '?')} "
                            f"vs {document.metadata.get('language2', '?')}"
                        )
                        st.markdown(document.page_content)

        except Exception as error:
            st.error(f"Memory search failed: {error}")

st.caption(
    "Note: ChromaDB uses local, lightweight hash embeddings in this demo. "
    "Search is approximate and may not understand meaning as well as "
    "a dedicated semantic embedding model."
)
