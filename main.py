import os
import math
import hashlib
import requests
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SCALEDOWN_API_KEY = os.getenv("SCALEDOWN_API_KEY")


def _resolve_model_client_config() -> tuple[str, str | None]:
    """Resolve API key/base for OpenAI-compatible clients.

    If OPENAI_API_KEY is actually an OpenRouter key (sk-or-v1...), route requests
    to OpenRouter automatically unless OPENAI_API_BASE is explicitly provided.
    """
    api_key = (OPENAI_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please add it to your .env file.")

    api_base = (os.getenv("OPENAI_API_BASE") or "").strip()
    if not api_base and api_key.startswith("sk-or-v1"):
        api_base = "https://openrouter.ai/api/v1"

    return api_key, (api_base or None)


def _build_local_embeddings():
    """Build a local embedding model as a fallback when API embeddings fail."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = (os.getenv("LOCAL_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2").strip()
    return HuggingFaceEmbeddings(model_name=model_name)


class _HashEmbeddings(Embeddings):
    """Small dependency-free embedding fallback.

    This keeps the app functional when external embedding APIs fail and
    sentence-transformers is unavailable.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % self.dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

class DashboardResponse(BaseModel):
    summary: str = Field(
        description=(
            "Short 1–2 sentence summary of the answer, grounded only in the document. "
            "If unknown, say exactly: The document does not provide information on this."
        )
    )
    paragraph: str = Field(
        description=(
            "A richer plain-language explanation (2–4 sentences) giving more detail and nuance, "
            "still strictly grounded in the retrieved context."
        )
    )
    impact_points: List[str] = Field(
        description="2–4 short impact points. Each must be concise and directly grounded in context."
    )
    key_clauses: List[str] = Field(
        description="1–4 exact section/clause references from context. Empty list if unavailable."
    )
    acts_referenced: List[str] = Field(
        default_factory=list,
        description="Acts or laws explicitly referenced in the retrieved context. Empty list if none."
    )

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: str
    compressed_context: str

class CitizenDashboardApp:
    def __init__(self, pdf_path: str):
        self.compressor_url = "https://api.scaledown.xyz/compress/raw/"
        self.pdf_path = pdf_path
        self.load_pdf(pdf_path)

    def load_pdf(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.setup_rag()
        self.setup_graph()

    def setup_rag(self):
        print("[1/4] Loading and parsing document")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        if not documents:
            raise RuntimeError(
                "No pages were extracted from the PDF. "
                "Please upload a text-based PDF (not image-only) or a different file."
            )

        # Keep only pages that contain extractable text.
        documents = [d for d in documents if (d.page_content or "").strip()]
        if not documents:
            raise RuntimeError(
                "PDF text extraction returned empty content for all pages. "
                "This file may be scanned/image-only. Please upload a text-based PDF."
            )

        print("[2/4] Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)
        docs = [d for d in docs if (d.page_content or "").strip()]
        if not docs:
            raise RuntimeError(
                "No text chunks were generated from this PDF. "
                "Try a different PDF or one with selectable text."
            )
        self._all_chunks = docs

        print("[3/4] Embedding chunks into Chroma")
        api_key, api_base = _resolve_model_client_config()
        requested_embedding_model = (os.getenv("EMBEDDING_MODEL") or "").strip()
        if requested_embedding_model:
            model_candidates = [requested_embedding_model]
        elif api_base and "openrouter.ai" in api_base:
            # OpenRouter commonly expects provider-prefixed model names for embeddings.
            model_candidates = [
                "openai/text-embedding-3-small",
                "text-embedding-3-small",
            ]
        else:
            model_candidates = ["text-embedding-3-small"]

        last_exc = None
        for model_name in model_candidates:
            try:
                embed_kwargs = {
                    "model": model_name,
                    "openai_api_key": api_key,
                }
                if api_base:
                    embed_kwargs["openai_api_base"] = api_base

                self.embeddings = OpenAIEmbeddings(**embed_kwargs)
                self.vectorstore = Chroma.from_documents(
                    docs,
                    self.embeddings,
                    collection_name="citizen_dashboard",
                )
                print(f"Embedding model selected: {model_name}")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            print("Remote embedding failed, switching to local embedding model")
            try:
                self.embeddings = _build_local_embeddings()
                self.vectorstore = Chroma.from_documents(
                    docs,
                    self.embeddings,
                    collection_name="citizen_dashboard",
                )
                print("Embedding model selected: local sentence-transformers")
            except Exception as local_exc:
                print("Local sentence-transformers unavailable, switching to hash embeddings")
                try:
                    self.embeddings = _HashEmbeddings()
                    self.vectorstore = Chroma.from_documents(
                        docs,
                        self.embeddings,
                        collection_name="citizen_dashboard",
                    )
                    print("Embedding model selected: hash fallback")
                except Exception as hash_exc:
                    base_hint = api_base or "https://api.openai.com/v1"
                    raise RuntimeError(
                        "Failed to build embeddings/vector store. "
                        f"Tried API models: {', '.join(model_candidates)}. "
                        f"Base URL: {base_hint}. "
                        f"API last error: {last_exc}. "
                        f"Local model error: {local_exc}. "
                        f"Hash fallback error: {hash_exc}"
                    ) from hash_exc
        top_k = int((os.getenv("RETRIEVER_TOP_K") or "1").strip())
        self.retriever_k = max(1, top_k)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.retriever_k})

        print("[4/4] Initializing model with structured output")
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        self.default_max_tokens = int((os.getenv("LLM_MAX_TOKENS") or "700").strip())
        self._build_structured_llm(self.default_max_tokens)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are an expert civic education assistant specializing in Indian law.\n"
            "You MUST return a rich but concise structured answer ONLY, following the schema fields exactly.\n\n"
            "CRITICAL INSTRUCTIONS TO PREVENT HALLUCINATION:\n"
            "1. ONLY rely on the provided retrieved context below. Do not use outside knowledge.\n"
            "2. If the user's question cannot be answered, use exactly: The document does not provide information on this.\n"
            "3. DO NOT invent false section numbers or clauses.\n\n"
            "OUTPUT RULES:\n"
            "- summary: 1–2 sentences that give a clear top-level answer.\n"
            "- paragraph: 2–4 sentences of plain-language detail (think 2–3 lines of text), expanding on the summary.\n"
            "- impact_points: 2–4 bullets, each a full phrase that clearly explains who is affected or what changes.\n"
            "- key_clauses: only exact section/ clause references that appear in the context.\n"
            "- acts_referenced: only Acts or laws that are explicitly named in the context.\n\n"
            "Make sure paragraph and impact_points actually add NEW detail beyond the summary, not just restate it.\n\n"
            "Context:\n{context}"),
            MessagesPlaceholder(variable_name="messages")
        ])


    def _build_structured_llm(self, max_tokens: int):
        llm_kwargs = {
            "model_name": self.model_name,
            "openai_api_key": self.api_key,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if self.api_base:
            llm_kwargs["openai_api_base"] = self.api_base

        self.llm = ChatOpenAI(**llm_kwargs)
        self.structured_llm = self.llm.with_structured_output(DashboardResponse)

    def retrieve_node(self, state: AgentState):
        """Retrieve relevant context from Chroma DB based on current user message."""
        query = state["messages"][-1].content
        print("\nRetrieving relevant context")
        raw_context = self.retrieve_context(query)
        return {"context": raw_context}

    def retrieve_context(self, query: str) -> str:
        primary_docs = self.retriever.invoke(query)
        merged = []
        seen = set()

        def _add_docs(items):
            for d in items:
                text = (d.page_content or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    merged.append(d)

        _add_docs(primary_docs)

        # Adaptive recall: start cheap (k=1), then expand only if context looks too thin.
        raw_context = "\n\n".join(d.page_content for d in merged)
        if len(raw_context) < 700:
            extra_k = max(4, self.retriever_k + 3)
            try:
                extra_docs = self.vectorstore.similarity_search(query, k=extra_k)
                _add_docs(extra_docs)
            except Exception:
                pass

            # Lexical fallback for low-quality embeddings or noisy provider responses.
            q_tokens = {t for t in query.lower().split() if len(t) > 2}
            if q_tokens and self._all_chunks:
                scored = []
                for d in self._all_chunks:
                    text = (d.page_content or "")
                    low = text.lower()
                    score = sum(1 for t in q_tokens if t in low)
                    if score > 0:
                        scored.append((score, d))
                scored.sort(key=lambda x: x[0], reverse=True)
                lexical_docs = [d for _, d in scored[:3]]
                _add_docs(lexical_docs)

        return "\n\n".join(d.page_content for d in merged)

    def compress_node(self, state: AgentState):
        """Compress the retrieved context using ScaleDown API."""
        raw_context = state["context"]
        query = state["messages"][-1].content
        return {"compressed_context": self.compress_context(query, raw_context)}

    def compress_context(self, query: str, raw_context: str) -> str:
        if not SCALEDOWN_API_KEY:
            return raw_context

        # For shorter contexts, skip compression to preserve details.
        if len(raw_context) < 1200:
            return raw_context

        print("Compressing token payload via ScaleDown")
        payload = {
            "prompt": query,
            "context": raw_context
        }
        headers = {
            "x-api-key": SCALEDOWN_API_KEY
        }

        try:
            response = requests.post(self.compressor_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data.get("successful") and "results" in data:
                res = data["results"]
                orig_tokens = res.get('original_prompt_tokens')
                comp_tokens = res.get('compressed_prompt_tokens')
                savings = 0 if not orig_tokens else round((1 - comp_tokens/orig_tokens)*100)
                print(f"Tokens compressed: {orig_tokens} -> {comp_tokens} ({savings}% saved)")
                compressed = res.get("compressed_prompt", raw_context)
                if isinstance(compressed, str) and len(compressed.strip()) >= 300:
                    return compressed
                return raw_context
        except Exception:
            pass

        return raw_context

    def generate_node(self, state: AgentState):
        """Generate structured response using Pydantic, LLM, and explicitly prevent hallucination."""
        print("Generating structured answer")
        compressed = state["compressed_context"]
        
        chain = self.prompt | self.structured_llm
        try:
            result = chain.invoke({"context": compressed, "messages": state["messages"]})
        except Exception as exc:
            err_text = str(exc)
            # OpenRouter 402: insufficient credits for requested max_tokens.
            if "402" in err_text or "requires more credits" in err_text.lower():
                fallback_tokens = max(200, min(512, self.default_max_tokens // 2))
                print(f"Credit limit hit, retrying with max_tokens={fallback_tokens}")
                self._build_structured_llm(fallback_tokens)
                chain = self.prompt | self.structured_llm
                result = chain.invoke({"context": compressed, "messages": state["messages"]})
            else:
                raise

        return {"messages": [AIMessage(content=result.model_dump_json())]}

    def setup_graph(self):
        """Construct the LangGraph state machine with Chat Memory."""
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("compress", self.compress_node)
        workflow.add_node("generate", self.generate_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "compress")
        workflow.add_edge("compress", "generate")
        workflow.add_edge("generate", END)

        # Chat Memory checkpointer
        memory = InMemorySaver()
        self.app = workflow.compile(checkpointer=memory)

    def ask_structured(self, query: str, thread_id: str = "citizens-dashboard") -> DashboardResponse:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke({"messages": [HumanMessage(content=query)]}, config=config)

        raw = result["messages"][-1].content
        return DashboardResponse.model_validate_json(raw)

    def run(self, query: str, thread_id: str = "citizens-dashboard"):
        result = self.ask_structured(query, thread_id)
        lines = [
            f"Summary: {result.summary}",
            "Impact Points:",
            *[f"- {point}" for point in result.impact_points],
            "Key Clauses:",
            *([f"- {clause}" for clause in result.key_clauses] if result.key_clauses else ["- No specific clauses found in context."]),
        ]
        return "\n".join(lines)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    pdf_file = "English_FinanceBill21202610322PM.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"File not found: {pdf_file}")
    else:
        app = CitizenDashboardApp(pdf_file)
        
        # Original Query
        app.run("Summarize the key tax changes for salaried individuals.", thread_id="user1")
        
        # Follow-up memory query
        print("\n\n-- Running follow up question to test chat memory --")
        app.run("Wait, does this apply to businesses too?", thread_id="user1")
