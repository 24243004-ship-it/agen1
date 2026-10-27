
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
    page_title="College Project File Organizer",
    page_icon="📁",
    layout="wide"
)

DB_DIRECTORY = "project_category_memory"
COLLECTION_NAME = "project_categories"


# -------------------- LOCAL EMBEDDINGS --------------------
# Simple local embeddings for ChromaDB.
# No separate embedding API is needed.

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

        norm = sum(x * x for x in vector) ** 0.5
        if norm:
            vector = [x / norm for x in vector]

        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


# -------------------- CHROMADB MEMORY --------------------

@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIRECTORY,
        embedding_function=LocalEmbeddings()
    )


def save_categories(project_name, categories, solution_name):
    store = get_vector_store()

    content = (
        f"Project: {project_name}\n"
        f"Chosen organization method: {solution_name}\n"
        f"Categories: {categories}"
    )

    record_id = hashlib.sha256(content.encode("utf-8")).hexdigest()

    store.add_documents(
        documents=[
            Document(
                page_content=content,
                metadata={
                    "project_name": project_name,
                    "solution": solution_name,
                    "categories": categories
                }
            )
        ],
        ids=[record_id]
    )


def retrieve_categories(query, limit=5):
    store = get_vector_store()
    results = store.similarity_search(query, k=limit)

    return [doc.page_content for doc in results]


# -------------------- LANGGRAPH STATE --------------------

class OrganizerState(TypedDict, total=False):
    project_name: str
    project_description: str
    file_list: str
    previous_categories: str
    solutions: str
    final_result: str


# -------------------- TREE-OF-THOUGHT AGENT --------------------

def build_graph(api_key, model_name):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.4
    )

    def generate_solutions(state: OrganizerState):
        prompt = f"""
You are a college project file organization assistant.

Project name: {state['project_name']}
Project description: {state['project_description']}
Existing files or file types:
{state['file_list']}

Previously used project categories from memory:
{state['previous_categories']}

Use a Tree-of-Thought-inspired approach. Generate exactly THREE
different possible solutions for organizing this project.

SOLUTION 1: SUBJECT / COMPONENT BASED
Group files by application components, subjects, or modules.

SOLUTION 2: PROJECT PHASE BASED
Group files by planning, design, development, testing, and submission.

SOLUTION 3: FILE TYPE BASED
Group files by source code, documents, images, datasets, and other formats.

For EACH solution provide:
1. Folder structure using a readable tree.
2. Which files belong in each folder.
3. Two advantages.
4. One possible disadvantage.
5. Explain when this approach is useful.

Adapt the folder names to the actual project. If filenames were not
provided, give suitable example filenames and label them as examples.

Do not claim that files have actually been moved. These are proposed
folder structures only.
"""

        response = llm.invoke(prompt)
        return {"solutions": response.content}

    def evaluate_solutions(state: OrganizerState):
        prompt = f"""
You are the Solution Evaluation Agent.

Evaluate the following THREE proposed folder organization solutions
for a college project.

Project: {state['project_name']}
Description: {state['project_description']}
Files: {state['file_list']}

Solutions:
{state['solutions']}

Compare the options on:
- Ease of use
- Maintainability
- Suitability for college submission
- Ease of finding files
- Suitability for this specific project

Return:
1. A comparison table with all three options.
2. A recommendation with clear reasons.
3. A final suggested folder tree.
4. A short checklist for organizing the files safely.

Do not claim that you inspected or moved real files. Prefer a clear,
practical structure. If two solutions are equally suitable, say so.
Keep the original three options visible in your evaluation.
"""

        response = llm.invoke(prompt)
        return {"final_result": response.content}

    graph = StateGraph(OrganizerState)

    graph.add_node("GenerateThreeSolutions", generate_solutions)
    graph.add_node("EvaluateSolutions", evaluate_solutions)

    graph.add_edge(START, "GenerateThreeSolutions")
    graph.add_edge("GenerateThreeSolutions", "EvaluateSolutions")
    graph.add_edge("EvaluateSolutions", END)

    return graph.compile()


# -------------------- STREAMLIT DASHBOARD --------------------

st.title("📁 AI College Project File Organizer")

st.write(
    "Generate three alternative folder structures using a "
    "Tree-of-Thought-inspired workflow and remember previous categories "
    "with ChromaDB."
)

with st.sidebar:
    st.header("⚙️ AI Settings")

    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.0-flash"]
    )

st.subheader("📝 Project Details")

project_name = st.text_input(
    "Project name",
    value="College Management System"
)

project_description = st.text_area(
    "Project description",
    value=(
        "A college project containing Python code, a database, "
        "project documentation, diagrams, and testing files."
    )
)

file_list = st.text_area(
    "Existing filenames or file types (optional)",
    value=(
        "app.py\n"
        "database.py\n"
        "requirements.txt\n"
        "project_report.pdf\n"
        "system_design.png\n"
        "test_app.py"
    ),
    height=140
)

if st.button(
    "🌳 Generate Three Solutions",
    type="primary",
    use_container_width=True
):
    if not api_key.strip():
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not project_name.strip() or not project_description.strip():
        st.error("Please enter the project name and description.")
    else:
        try:
            with st.spinner("Generating and evaluating three solutions..."):
                try:
                    old_records = retrieve_categories(
                        project_name + " " + project_description,
                        limit=5
                    )
                except Exception:
                    old_records = []

                previous_categories = (
                    "\n\n".join(old_records)
                    if old_records
                    else "No previous project categories found."
                )

                app = build_graph(api_key.strip(), model_name)

                result = app.invoke({
                    "project_name": project_name.strip(),
                    "project_description": project_description.strip(),
                    "file_list": file_list.strip() or "No file list supplied.",
                    "previous_categories": previous_categories
                })

            st.session_state["solutions"] = result["solutions"]
            st.session_state["final_result"] = result["final_result"]
            st.session_state["project_name"] = project_name.strip()
            st.session_state["project_description"] = (
                project_description.strip()
            )
            st.session_state["file_list"] = file_list.strip()

            st.success("Three solutions generated and evaluated!")

        except Exception as error:
            st.error(f"An error occurred: {error}")
            st.info(
                "Check your API key, internet connection, model access, "
                "and installed packages."
            )


# Keep the generated output visible after Streamlit reruns.

if st.session_state.get("solutions"):
    st.subheader("🌳 Three Possible Solutions")
    st.markdown(st.session_state["solutions"])

    st.divider()
    st.subheader("🏆 Evaluation and Recommendation")
    st.markdown(st.session_state["final_result"])

    st.divider()
    st.subheader("💾 Save a Used Project Category")

    st.write(
        "After reviewing the suggestions, save the organization method "
        "you decide to use. It will be available in future searches."
    )

    with st.form("save_category_form"):
        chosen_solution = st.selectbox(
            "Chosen organization method",
            [
                "Subject / Component Based",
                "Project Phase Based",
                "File Type Based",
                "Custom Method"
            ]
        )

        categories = st.text_area(
            "Categories and folder names to remember",
            value=(
                "src, docs, diagrams, tests, data, "
                "requirements, final_submission"
            )
        )

        save_clicked = st.form_submit_button("Save to ChromaDB")

    if save_clicked:
        if not categories.strip():
            st.warning("Please enter at least one category.")
        else:
            try:
                save_categories(
                    st.session_state["project_name"],
                    categories.strip(),
                    chosen_solution
                )
                st.success("Project categories saved in ChromaDB memory!")
            except Exception as error:
                st.error(f"Could not save categories: {error}")


# -------------------- SEARCH PAST CATEGORIES --------------------

st.divider()
st.subheader("🧠 Previous Project Category Memory")

search_query = st.text_input(
    "Search previously used categories",
    placeholder="Example: Python project with report and testing files"
)

if st.button("🔍 Search Memory", use_container_width=True):
    if not search_query.strip():
        st.warning("Please enter a search query.")
    else:
        try:
            records = retrieve_categories(search_query.strip(), limit=5)

            if records:
                for number, record in enumerate(records, start=1):
                    with st.expander(f"Previous category record {number}"):
                        st.markdown(record)
            else:
                st.info("No previous categories found.")
        except Exception as error:
            st.error(f"Memory search failed: {error}")

st.caption(
    "ChromaDB stores category records locally. Local hash embeddings "
    "provide approximate similarity search; they do not fully understand "
    "the meaning of a query."
)
