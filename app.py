import streamlit as st
import fitz
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.agent import create_tool_calling_agent

def extract_pages(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return [{"page_num": i+1, "text": page.get_text()} for i, page in enumerate(doc)]

def build_agent(pages):
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(
        "\n\n".join(p["text"] for p in pages)
    )
    vs = Chroma.from_texts(chunks, HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={"device": "cpu"}
    ), persist_directory="medical_db")
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF for relevant information. Always try this first."""
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs) if docs else "NO_CONTENT_FOUND"

    @tool
    def summarize_pages(page_numbers: str) -> str:
        """Summarize specific pages or all pages from the research paper. 
        Pass 'all' to summarize all pages, or comma-separated page numbers like '1,2,3'."""
        if page_numbers.strip().lower() == "all":
            selected = pages
        else:
            nums = [int(n.strip()) for n in page_numbers.split(",") if n.strip().isdigit()]
            selected = [p for p in pages if p["page_num"] in nums]
        if not selected:
            return "No matching pages found."
        return "\n\n".join(
            f"**Page {p['page_num']}:**\n{p['text'][:2000]}" for p in selected
        )

    @tool
    def web_search(query: str) -> str:
        """Search the web for latest medical guidelines. Use only if PDF has no answer."""
        r = requests.get("https://api.search.tinyfish.ai", headers={"X-API-Key": "TINYFISH_API_KEY"}, params={"query": query}).json()
        return "\n\n".join(f"{x['title']}: {x.get('snippet','')}" for x in r.get("results", [])) or "No results."

    tools = [search_pdf, summarize_pages, web_search]
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key="GROQ_API_KEY", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant. "
         "If the user asks to summarize pages, use the summarize_pages tool. "
         "For all other questions, use search_pdf first; fall back to web_search if NO_CONTENT_FOUND. "
         "Format answers with bullet points and tables. Current year: 2026."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=True), len(chunks)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Research Assistant")
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
if uploaded_file and st.session_state.get("filename") != uploaded_file.name:
    st.session_state.clear()
    st.session_state["filename"] = uploaded_file.name
    with st.spinner("Indexing PDF..."):
        pages = extract_pages(uploaded_file)
        st.session_state["agent"], n = build_agent(pages)
    st.success(f"Ready! Indexed {n} chunks from {len(pages)} pages.")
question = st.text_input("Ask a question about the paper")
if question:
    st.write(st.session_state["agent"].invoke({"input": question})["output"] if "agent" in st.session_state else "Please upload a PDF first.")
