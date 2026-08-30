from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
import fitz
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st


def extract_pages(uploaded_file):
  doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
  return [
      {"page_num": i + 1, "text": page.get_text()}
      for i, page in enumerate(doc)
  ]


def summarize_page(llm, text, page_num):
  if not text.strip():
    return "No readable text."

  return llm.invoke(
      f"You are a medical research assistant. Summarize page {page_num} "
      f"of this research paper in 3-5 bullet points:\n\n{text[:3000]}"
  ).content


def build_agent(pages):
  chunks = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=50
  ).split_text("\n\n".join(p["text"] for p in pages))
  vs = Chroma.from_texts(
      chunks,
      HuggingFaceEmbeddings(
          model_name="sentence-transformers/all-MiniLM-L12-v2",
          model_kwargs={"device": "cpu"},
      ),
      persist_directory="medical_db",
  )
  retriever = vs.as_retriever(search_kwargs={"k": 5})

  @tool
  def search_pdf(query: str) -> str:
    """Search the uploaded PDF first."""
    docs = retriever.invoke(query)
    return (
        "\n\n".join(d.page_content for d in docs) if docs else "NO_CONTENT_FOUND"
    )

  @tool
  def web_search(query: str) -> str:
    """Search the web if PDF has no answer."""
    r = requests.get(
        "https://api.search.tinyfish.ai",
        headers={"X-API-Key": st.secrets["TINYFISH_API_KEY"]},
        params={"query": query},
    ).json()

    return (
        "\n\n".join(
            f"{x['title']}: {x.get('snippet','')}" for x in r.get("results", [])
        )
        or "No results."
    )

  llm = ChatGroq(
      model="llama-3.3-70b-versatile",
      groq_api_key=st.secrets["GROQ_API_KEY"],
      temperature=0,
  )
  prompt = ChatPromptTemplate.from_messages([
      (
          "system",
          (
              "You are a medical research assistant. Search PDF first; fall back"
              " to web if NO_CONTENT_FOUND. Use bullet points and tables."
              " Current year: 2026."
          ),
      ),
      ("human", "{input}"),
      ("placeholder", "{agent_scratchpad}"),
  ])
  return (
      AgentExecutor(
          agent=create_tool_calling_agent(
              llm, [search_pdf, web_search], prompt
          ),
          tools=[search_pdf, web_search],
          verbose=True,
      ),
      len(chunks),
  )


# ── UI ─────────────────────────────────────────────────────────────
st.title("Medical Research Assistant")
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
if uploaded_file and st.session_state.get("filename") != uploaded_file.name:
  st.session_state.clear()
  st.session_state["filename"] = uploaded_file.name
  pages = extract_pages(uploaded_file)

  llm = ChatGroq(
      model="openai/gpt-oss-120b",
      groq_api_key=st.secrets["GROQ_API_KEY"],
      temperature=0,
  )

  st.subheader("Page Summaries")
  for p in pages:
    with st.expander(f"Page {p['page_num']}"):
      st.write(summarize_page(llm, p["text"], p["page_num"]))
  st.session_state["agent"], n = build_agent(pages)
  st.success(f"Ready! Indexed {n} chunks.")
question = st.text_input("Ask a question about the paper")
if question:
  st.write(
      st.session_state["agent"].invoke({"input": question})["output"]
      if "agent" in st.session_state
      else "Please upload a PDF first."
  )
