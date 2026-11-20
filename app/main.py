import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

load_dotenv()

from app.schemas import QueryRequest
from app.vectorstore import build_final_multimodel_document, index_documents_to_pinecone
from app.retriever import create_hybrid_retriever, create_reranked_retriever
from app.generator import multimodal_rag_answer

final_retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global final_retriever
    pdf_path = r"data/NovaCore_Multimodal_Company_Report_2026.pdf"

    try:
        print("Building document corpus...")
        final_docs = build_final_multimodel_document(pdf_path)

        print("Connecting to Pinecone...")
        vectorstore = index_documents_to_pinecone(final_docs)

        print("Initializing Hybrid + Reranked Retriever...")
        hybrid_ret = create_hybrid_retriever(final_docs, vectorstore)
        final_retriever = create_reranked_retriever(hybrid_ret)
        print("Pipeline fully loaded and ready!")
    except Exception:
        print("STARTUP FAILED:")
        traceback.print_exc()
        final_retriever = None

    yield

    final_retriever = None

app = FastAPI(
    title="Multimodal Corporate RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "retriever_ready": final_retriever is not None}

@app.post("/query")
def query_endpoint(request: QueryRequest):
    if not final_retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized.")
    try:
        return multimodal_rag_answer(request.query, final_retriever)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")