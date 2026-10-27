
import os
import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Academic History Agent",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Academic History Agent")
st.write(
    "Ask questions about previous academic activities "
    "using a LangChain ReAct agent and ChromaDB memory."
)

DB_DIR = "./student_academic_chroma"


# ---------------- SIDEBAR ----------------
st.sidebar.header("Gemini API Settings")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
)

model_name = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-2.5-flash", "gemini-2.0-flash"],
)

st.sidebar.markdown(
    "[Get an API key from Google AI Studio]"
    ""
)


# ---------------- CHROMADB ----------------
@st.cache_resource
def get_vectorstore(key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=key,
    )

    return Chroma(
        collection_name="student_academic_records",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )


# ---------------- SAVE STUDENT RECORD ----------------
def save_record(vectorstore, student_id, activity, details):
    text = (
        f"Student ID: {student_id}\n"
        f"Academic activity: {activity}\n"
        f"Details: {details}"
    )

    document = Document(
        page_content=text,
        metadata={
            "student_id": student_id,
            "activity": activity,
        },
    )

    vectorstore.add_documents([document])


# ---------------- REACT AGENT ----------------
def create_academic_agent(vectorstore, student_id, api_key, model_name):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
    )

    @tool
    def search_academic_history(question: str) -> str:
        """Search the current student's stored academic records.
        Input should be a natural-language question about activities,
        assignments, exams, projects, attendance, or achievements."""

        results = vectorstore.similarity_search(
            question,
            k=6,
            filter={"student_id": student_id},
        )

        if not results:
            return "No matching academic records were found."

        return "\n\n".join(
            doc.page_content for doc in results
        )

    tools = [search_academic_history]

    prompt = PromptTemplate.from_template(
        """You are a helpful student academic history assistant.

Answer the user's question using the student's stored records.
You must use the search_academic_history tool before answering
questions about previous activities.

Only report facts supported by the retrieved records.
If information is missing, clearly say that it is not available.
Do not invent grades, attendance, dates, or achievements.
Do not reveal records belonging to other students.

You have access to these tools:

{tools}

Use this format:

Question: the question to answer
Thought: briefly identify what information is needed
Action: one of [{tool_names}]
Action Input: the search query
Observation: the tool result
... (repeat if necessary)
Thought: I have enough information to answer
Final Answer: the answer for the student

Question: {input}
Thought:{agent_scratchpad}"""
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
    )


# ---------------- DASHBOARD ----------------
tab1, tab2 = st.tabs(
    ["📝 Add Academic Record", "💬 Ask About Previous Activities"]
)


with tab1:
    st.subheader("Store Student Information")

    student_id_input = st.text_input(
        "Student ID",
        key="save_student_id",
        placeholder="e.g. STU101",
    )

    activity = st.selectbox(
        "Academic Activity",
        [
            "Assignment",
            "Examination",
            "Project",
            "Attendance",
            "Workshop",
            "Achievement",
            "Internship",
            "Other",
        ],
    )

    details = st.text_area(
        "Activity Details",
        placeholder=(
            "e.g. Python assignment submitted on 12 August; "
            "scored 85/100."
        ),
        height=120,
    )

    if st.button("💾 Save Academic Record", type="primary"):
        if not api_key.strip():
            st.error("Enter your Gemini API key in the sidebar.")
        elif not student_id_input.strip() or not details.strip():
            st.error("Enter both Student ID and activity details.")
        else:
            try:
                with st.spinner("Saving record to ChromaDB..."):
                    store = get_vectorstore(api_key.strip())
                    save_record(
                        store,
                        student_id_input.strip(),
                        activity,
                        details.strip(),
                    )
                st.success("Academic record saved successfully!")

            except Exception as error:
                st.error(f"Could not save record: {error}")


with tab2:
    st.subheader("Ask a Question")

    query_student_id = st.text_input(
        "Student ID",
        key="query_student_id",
        placeholder="e.g. STU101",
    )

    question = st.text_area(
        "Your question",
        placeholder=(
            "e.g. What projects have I completed? "
            "What was my last assignment score?"
        ),
        height=90,
    )

    if st.button("🔎 Find Answer", type="primary"):
        if not api_key.strip():
            st.error("Enter your Gemini API key in the sidebar.")
        elif not query_student_id.strip() or not question.strip():
            st.error("Enter Student ID and your question.")
        else:
            try:
                with st.spinner(
                    "Searching records and preparing an answer..."
                ):
                    store = get_vectorstore(api_key.strip())

                    executor = create_academic_agent(
                        store,
                        query_student_id.strip(),
                        api_key.strip(),
                        model_name,
                    )

                    result = executor.invoke(
                        {"input": question.strip()}
                    )

                st.subheader("📚 Agent Answer")
                st.markdown(result["output"])

            except Exception as error:
                st.error(f"Could not answer the question: {error}")


st.divider()
st.caption(
    "Built with LangChain • ReAct Agent • ChromaDB • Gemini AI"
)
st.caption(
    "Use fictional student data for demos. Protect real academic "
    "records and restrict access to authorized users."
)
