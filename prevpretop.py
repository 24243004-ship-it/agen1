
import os
import re
import json
import hashlib
from typing import TypedDict

import numpy as np
import streamlit as st

from langgraph.graph import StateGraph, START, END
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Presentation Multi-Agent",
    page_icon="🤖",
    layout="wide"
)

MEMORY_DIR = "presentation_faiss_memory"
EMBEDDING_SIZE = 256


# ---------------- LOCAL EMBEDDINGS ----------------
# Local embeddings allow FAISS memory to work without an API key.

class LocalEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        vector = np.zeros(EMBEDDING_SIZE, dtype=np.float32)

        words = re.findall(r"[a-zA-Z0-9+#]+", text.lower())

        for word in words:
            digest = hashlib.sha256(word.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_SIZE
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)

        if norm:
            vector /= norm

        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


embeddings = LocalEmbeddings()


# ---------------- FAISS MEMORY ----------------

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


def retrieve_previous_topics(topic: str, user_id: str) -> list[str]:
    if vector_store is None:
        return []

    try:
        docs = vector_store.similarity_search(
            f"User ID: {user_id}. Presentation topic: {topic}",
            k=10,
            filter={"user_id": user_id}
        )

        return [doc.page_content for doc in docs]

    except Exception:
        return []


def save_topic(topic: str, user_id: str):
    global vector_store

    document = Document(
        page_content=f"Presentation topic: {topic}",
        metadata={"user_id": user_id, "topic": topic}
    )

    if vector_store is None:
        vector_store = FAISS.from_documents(
            [document], embeddings
        )
    else:
        vector_store.add_documents([document])

    vector_store.save_local(MEMORY_DIR)


# ---------------- GRAPH STATE ----------------

class PresentationState(TypedDict, total=False):
    topic: str
    audience: str
    slide_count: int
    previous_topics: list[str]
    plan: str
    research: str
    final_presentation: str
    checker_notes: str
    use_ai: bool


# ---------------- LLM HELPER ----------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    api_key = st.session_state.get("api_key", "").strip()

    if not api_key:
        return ""

    try:
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=api_key
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        response = (prompt | model).invoke({"input": user_prompt})

        return response.content

    except Exception as error:
        st.session_state["agent_error"] = str(error)
        return ""


# ---------------- AGENT 1: PLANNER ----------------

def planner_agent(state: PresentationState):
    topic = state["topic"]
    audience = state["audience"]
    slide_count = state["slide_count"]

    previous = state.get("previous_topics", [])

    prompt = f"""
    Create a concise presentation outline.

    Topic: {topic}
    Audience: {audience}
    Number of slides: {slide_count}

    Previously used topics for this user:
    {previous}

    Make a logical slide-by-slide outline.
    Include an introduction and a conclusion.
    Do not repeat the same content across slides.
    """

    result = call_llm(
        "You are the Planner Agent. Create clear presentation structures.",
        prompt
    )

    if not result:
        standard_topics = [
            "Introduction",
            "Definition and Overview",
            "Types and Key Concepts",
            "Applications and Examples",
            "Benefits and Limitations",
            "Future Scope",
            "Conclusion"
        ]

        headings = standard_topics[:slide_count]

        while len(headings) < slide_count:
            headings.insert(-1, "Key Examples and Discussion")

        result = "\n".join(
            f"Slide {i + 1}: {heading}"
            for i, heading in enumerate(headings)
        )

        result = f"Presentation topic: {topic}\n" + result

    return {"plan": result}


# ---------------- AGENT 2: RESEARCHER ----------------

def researcher_agent(state: PresentationState):
    topic = state["topic"]
    plan = state["plan"]
    audience = state["audience"]

    prompt = f"""
    Prepare concise slide content using this outline.

    Main topic: {topic}
    Audience: {audience}
    Outline:
    {plan}

    For each slide, provide:
    - Slide heading
    - Two or three short bullet points
    - A simple example where useful

    Keep the language understandable.
    Do not invent statistics or citations.
    """

    result = call_llm(
        "You are the Researcher Agent. Explain concepts accurately and clearly.",
        prompt
    )

    if not result:
        result = f"""
Slide 1: Introduction to {topic}
- {topic} is the main subject of this presentation.
- It is important in modern education, research and technology.
- This presentation introduces its key ideas.

Slide 2: Definition and Overview
- Understand the meaning and purpose of {topic}.
- Learn its fundamental concepts.
- Example: Study a simple real-world use case.

Slide 3: Types and Key Concepts
- Explore the major concepts related to {topic}.
- Understand how the concepts differ.
- Example: Compare two common approaches.

Slide 4: Applications
- {topic} can be explored through practical examples.
- Applications depend on the field and requirements.
- Example: Examine its use in education or business.

Slide 5: Benefits and Limitations
- Benefits may include improved efficiency and new capabilities.
- Limitations depend on cost, quality and implementation.
- Responsible use is important.

Slide 6: Future Scope
- Research and development may introduce new possibilities.
- Skills and knowledge will continue to matter.
- Real-world evaluation remains essential.

Slide 7: Conclusion
- Review the important points discussed.
- Remember the practical uses and limitations.
- Thank you.
"""

    return {"research": result}


# ---------------- AGENT 3: CHECKER ----------------

def checker_agent(state: PresentationState):
    topic = state["topic"]
    research = state["research"]

    prompt = f"""
    Review this short presentation about {topic}.

    Content:
    {research}

    Check:
    1. Repeated points
    2. Clarity and organization
    3. Unsupported factual claims
    4. Whether the content fits a student presentation

    Return a polished final presentation with slide headings
    and concise bullet points. Correct obvious writing issues.
    """

    result = call_llm(
        "You are the Checker Agent. Review and improve presentation quality.",
        prompt
    )

    if not result:
        result = research.strip()

    notes = (
        "Checker completed a basic review. "
        "The offline fallback uses a predefined template; "
        "it does not independently verify facts."
    )

    if st.session_state.get("api_key", "").strip() and result:
        notes = (
            "Checker Agent reviewed clarity, repetition, structure "
            "and unsupported claims. Verify important facts before presenting."
        )

    return {
        "final_presentation": result,
        "checker_notes": notes
    }


# ---------------- LANGGRAPH WORKFLOW ----------------

def build_presentation_graph():
    graph = StateGraph(PresentationState)

    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("checker", checker_agent)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "checker")
    graph.add_edge("checker", END)

    return graph.compile()


# ---------------- DASHBOARD ----------------

st.title("🤖 AI Presentation Multi-Agent System")

st.write(
    "Prepare a short presentation with Planner, Researcher "
    "and Checker agents. FAISS remembers previously used topics."
)

with st.sidebar:
    st.header("Settings")

    user_id = st.text_input(
        "Student ID",
        value="student01"
    ).strip()

    st.session_state["api_key"] = st.text_input(
        "OpenAI API key (optional)",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password"
    )

    st.caption(
        "Without an API key, the app uses a basic offline template. "
        "LLM-powered agent outputs require a valid API key."
    )

left, right = st.columns([3, 2])

with left:
    st.subheader("Presentation Details")

    topic = st.text_input(
        "Presentation topic",
        value="Artificial Intelligence"
    ).strip()

    audience = st.selectbox(
        "Audience",
        ["College Students", "School Students", "General Audience"]
    )

    slide_count = st.slider(
        "Number of slides",
        min_value=3,
        max_value=7,
        value=5
    )

    generate = st.button(
        "🚀 Generate Presentation",
        type="primary",
        use_container_width=True
    )

with right:
    st.subheader("Multi-Agent Workflow")

    st.markdown("""
    **1. Planner Agent**  
    Creates a slide outline.

    **2. Researcher Agent**  
    Writes slide content and examples.

    **3. Checker Agent**  
    Reviews and polishes the content.

    **4. FAISS Memory**  
    Retrieves and stores previous topics.
    """)

if generate:
    if not user_id:
        st.error("Please enter a Student ID.")
    elif not topic:
        st.warning("Please enter a presentation topic.")
    else:
        previous_topics = retrieve_previous_topics(topic, user_id)

        initial_state: PresentationState = {
            "topic": topic,
            "audience": audience,
            "slide_count": slide_count,
            "previous_topics": previous_topics
        }

        try:
            with st.spinner("Agents are preparing your presentation..."):
                workflow = build_presentation_graph()
                output = workflow.invoke(initial_state)

            st.session_state["output"] = output
            st.session_state["current_topic"] = topic
            st.session_state["current_user"] = user_id

            st.success("Presentation generated successfully!")

        except Exception as error:
            st.error(f"Workflow error: {error}")

if "output" in st.session_state:
    output = st.session_state["output"]

    st.divider()
    st.subheader("📋 Agent 1: Planner Output")
    st.markdown(output.get("plan", ""))

    st.subheader("🔎 Agent 2: Researcher Output")
    st.markdown(output.get("research", ""))

    st.subheader("✅ Agent 3: Checker Output")
    st.markdown(output.get("final_presentation", ""))

    st.info(output.get("checker_notes", ""))

    st.download_button(
        "📥 Download Presentation Content",
        data=output.get("final_presentation", ""),
        file_name="AI_Presentation.txt",
        mime="text/plain"
    )

    st.divider()
    st.subheader("💾 Save Topic to FAISS Memory")

    st.write(
        f"Current topic: **{st.session_state['current_topic']}**"
    )

    if st.button("Save This Topic"):
        try:
            save_topic(
                st.session_state["current_topic"],
                st.session_state["current_user"]
            )
            st.success(
                "Topic saved! Use the same Student ID in a future session."
            )
        except Exception as error:
            st.error(f"Memory save failed: {error}")

    if st.button("View Previously Used Topics"):
        topics = retrieve_previous_topics(
            st.session_state["current_topic"],
            st.session_state["current_user"]
        )

        if topics:
            st.write(topics)
        else:
            st.info("No saved topics found for this user.")

st.divider()
st.caption(
    "Built with Python, LangGraph, FAISS and Streamlit."
)
