from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

def create_hybrid_retriever(docs_corpus, vstore):
    bm25retriever = BM25Retriever.from_documents(docs_corpus)
    bm25retriever.k = 8

    vector_retriever = vstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    ensemble = EnsembleRetriever(
        retrievers=[bm25retriever, vector_retriever],
        weights=[0.5, 0.5]   
    )
    return ensemble


def create_reranked_retriever(base_retriever):
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=6)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return compression_retriever