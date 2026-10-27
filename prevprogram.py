import os
import re
import hashlib
import numpy as np
import streamlit as st

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

st.set_page_config(
    page_title="CodeCompass AI",
    page_icon="💻",
    layout="wide"
)

MEMORY_FOLDER = "programming_faiss_memory"
EMBEDDING_SIZE = 256


# -------------------- LOCAL EMBEDDINGS --------------------
# API key illamalum FAISS memory work aagum.
# These embeddings are for local preference retrieval,
# not a replacement for a production semantic embedding model.

class SimpleEmbeddings(Embeddings):

    def _embed(self, text):
        vector = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
        words = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_SIZE
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


embeddings = SimpleEmbeddings()


# -------------------- FAISS MEMORY --------------------

def load_memory():
    if os.path.isdir(MEMORY_FOLDER):
        try:
            return FAISS.load_local(
                MEMORY_FOLDER,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            return None
    return None


def save_preference(student_id, language, requirements):
    global vector_store

    memory_text = (
        f"User ID: {student_id}. "
        f"Previously preferred programming language: {language}. "
        f"Requirements: {requirements}"
    )

    document = Document(
        page_content=memory_text,
        metadata={
            "student_id": student_id,
            "language": language
        }
    )

    if vector_store is None:
        vector_store = FAISS.from_documents(
            [document], embeddings
        )
    else:
        vector_store.add_documents([document])

    vector_store.save_local(MEMORY_FOLDER)


def retrieve_preferences(student_id, requirements):
    if vector_store is None:
        return []

    try:
        results = vector_store.similarity_search(
            f"User ID: {student_id}. {requirements}",
            k=8,
            filter={"student_id": student_id}
        )
        return [doc.page_content for doc in results]
    except Exception:
        return []


vector_store = load_memory()


# -------------------- RECOMMENDATION TOOL --------------------

@tool
def compare_programming_languages(requirements: str) -> str:
    """Compare programming languages and recommend suitable options
    for the user's stated project, skill level, and requirements."""

    text = requirements.lower()

    choices = [
        (
            "Python",
            ["ai", "machine learning", "data science", "automation",
             "beginner", "easy", "backend", "chatbot", "deep learning"]
        ),
        (
            "JavaScript",
            ["website", "web development", "frontend", "browser",
             "interactive website", "full stack"]
        ),
        (
            "Java",
            ["android", "enterprise", "large application",
             "banking", "spring boot"]
        ),
        (
            "C++",
            ["game engine", "competitive programming", "embedded",
             "high performance", "system programming"]
        ),
        (
            "C#",
            ["unity", "unity game", "windows application", ".net"]
        ),
        (
            "SQL",
            ["database", "query", "reporting", "relational data"]
        ),
        (
            "Swift",
            ["ios", "iphone", "ipad", "apple app"]
        ),
        (
            "Kotlin",
            ["android app", "modern android"]
        ),
        (
            "R",
            ["statistical analysis", "statistics", "research analysis"]
        ),
        (
            "Go",
            ["cloud services", "microservices", "concurrent server"]
        )
    ]

    scores = []
    for language, keywords in choices:
        score = sum(1 for keyword in keywords if keyword in text)
        scores.append((score, language))

    scores.sort(reverse=True)
    best_score, best_language = scores[0]

    if best_score == 0:
        best_language = "Python"
        reason = (
            "Python is a beginner-friendly general-purpose option. "
            "More details about the project would improve this recommendation."
        )
    else:
        reason = (
            f"{best_language} matches the requirements because "
            "the described tasks align with its common use cases."
        )

    alternatives = [
        language for score, language in scores
        if language != best_language and score > 0
    ][:3]

    return (
        f"Top recommendation: {best_language}\n"
        f"Reason: {reason}\n"
        f"Other possible choices: {', '.join(alternatives) if alternatives else 'Need more details to compare.'}"
    )


# -------------------- REACT AGENT --------------------

def run_agent(requirements, previous_preferences, api_key):
    memory_context = (
        "\n".join(previous_preferences)
        if previous_preferences
        else "No previous preferences found for this user."
    )

    prompt = f"""
You are CodeCompass, a programming language recommendation assistant.

Use the compare_programming_languages tool to compare suitable languages.
This is a ReAct workflow: reason about the requirements, use the tool,
then give a concise final recommendation. Do not reveal private
internal chain-of-thought; provide a short explanation instead.

User requirements:
{requirements}

Relevant previous preferences retrieved from FAISS:
{memory_context}

Your final answer should include:
1. Recommended programming language
2. Why it fits the requirements
3. Two advantages
4. One limitation
5. One alternative language

Treat previous preferences as context, not as a rule.
If the current requirements differ, prioritize the current requirements.
"""

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

    agent = create_react_agent(
        model=model,
        tools=[compare_programming_languages]
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })

    return result["messages"][-1].content


# -------------------- FALLBACK WITHOUT API KEY --------------------

def fallback_recommendation(requirements, previous_preferences):
    tool_result = compare_programming_languages.invoke({
        "requirements": requirements
    })

    previous_text = (
        "\n".join(previous_preferences)
        if previous_preferences
        else "No previous preferences saved yet."
    )

    return (
        "### Recommendation\n\n"
        f"{tool_result}\n\n"
        "### Previous preferences\n\n"
        f"{previous_text}\n\n"
        "_Offline mode is active. Add an OpenAI API key to enable "
        "the LangGraph ReAct LLM agent._"
    )


# -------------------- DASHBOARD UI --------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #176b55;
    }
    .subtitle {
        font-size: 16px;
        color: #777777;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">💻 CodeCompass AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Smart programming language advisor '
    'powered by LangGraph and FAISS memory</div>',
    unsafe_allow_html=True
)

st.divider()

with st.sidebar:
    st.header("⚙️ Settings")

    student_id = st.text_input(
        "User ID",
        value="student01",
        help="Use the same ID to retrieve your saved preferences."
    ).strip()

    api_key = st.text_input(
        "OpenAI API key (optional)",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Required for the LLM-powered ReAct agent."
    ).strip()

    st.info(
        "FAISS stores previous language preferences locally "
        "in the programming_faiss_memory folder."
    )

left, right = st.columns([3, 2])

with left:
    st.subheader("🎯 Your requirements")

    requirements = st.text_area(
        "What do you want to build?",
        placeholder=(
            "Example: I am a beginner. I want to build an AI chatbot "
            "and analyze data using machine learning."
        ),
        height=150
    )

    preferred_language = st.selectbox(
        "Your current preferred language",
        [
            "Python", "JavaScript", "Java", "C++", "C#",
            "SQL", "Swift", "Kotlin", "R", "Go", "Not sure"
        ]
    )

    skill_level = st.selectbox(
        "Your programming experience",
        ["Beginner", "Intermediate", "Advanced"]
    )

    project_type = st.selectbox(
        "Project type",
        [
            "AI / Machine Learning",
            "Website",
            "Mobile App",
            "Game Development",
            "Data Analysis",
            "Automation",
            "General Programming"
        ]
    )

    recommend_clicked = st.button(
        "🚀 Recommend Language",
        type="primary",
        use_container_width=True
    )

with right:
    st.subheader("🧠 How it works")

    st.markdown(
        """
        **1. Understand**

        Identify the project, skill level and goals.

        **2. Retrieve**

        Search previous preferences from FAISS.

        **3. Reason + Act**

        The LangGraph ReAct agent can use a comparison tool.

        **4. Recommend**

        Present a suitable language and explanation.

        **5. Remember**

        Save the user's chosen preference for future sessions.
        """
    )

    st.caption(
        "Without an API key, the local recommendation fallback "
        "works, but the LLM-based ReAct agent is not activated."
    )

if recommend_clicked:
    if not student_id:
        st.error("Please enter a User ID.")
    elif not requirements.strip():
        st.warning("Please describe your programming requirements.")
    else:
        full_requirements = (
            f"{requirements}. "
            f"Skill level: {skill_level}. "
            f"Project type: {project_type}. "
            f"Current preference: {preferred_language}."
        )

        with st.spinner("Analyzing requirements and searching memory..."):
            previous = retrieve_preferences(
                student_id, full_requirements
            )

            try:
                if api_key:
                    answer = run_agent(
                        full_requirements,
                        previous,
                        api_key
                    )
                else:
                    answer = fallback_recommendation(
                        full_requirements,
                        previous
                    )

                st.session_state["recommendation"] = answer
                st.session_state["last_student_id"] = student_id
                st.session_state["last_requirements"] = full_requirements

            except Exception as error:
                st.error(
                    "The AI agent could not run. Check your API key, "
                    "internet connection, and installed package versions."
                )
                st.caption(f"Technical details: {error}")

if "recommendation" in st.session_state:
    st.divider()
    st.subheader("✨ Language recommendation")
    st.markdown(st.session_state["recommendation"])

    st.subheader("💾 Save your preference")

    st.write(
        "After reviewing the recommendation, choose which language "
        "you want the app to remember."
    )

    chosen_language = st.selectbox(
        "Language to remember",
        [
            "Python", "JavaScript", "Java", "C++", "C#",
            "SQL", "Swift", "Kotlin", "R", "Go"
        ],
        key="language_to_save"
    )

    if st.button("Save preference to FAISS"):
        try:
            save_preference(
                st.session_state["last_student_id"],
                chosen_language,
                st.session_state["last_requirements"]
            )
            st.success(
                f"Saved {chosen_language} preference for "
                f"{st.session_state['last_student_id']}."
            )
        except Exception as error:
            st.error(f"Could not save preference: {error}")

st.divider()
st.caption(
    "CodeCompass AI | LangGraph ReAct + FAISS | "
    "Local memory is stored on this computer."
)