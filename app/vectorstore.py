import os
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from app.ingestion import extract_pdf_content, generate_image_summaries

def build_final_multimodel_document(pdf_path, img_dir="extracted_images", cache_file="image_summaries_cache.json"):
    raw_extracted_docs = extract_pdf_content(pdf_path, output_img_dir=img_dir)
    summaries = generate_image_summaries(img_dir=img_dir, cache_file=cache_file)

    text_docs = [doc for doc in raw_extracted_docs if doc.metadata["source_type"] == "text"]
    table_docs = [doc for doc in raw_extracted_docs if doc.metadata["source_type"] == "table"]

    image_summary_docs = []
    for item in summaries:
        image_summary_docs.append(Document(
            page_content=item["summary"],
            metadata={
                "page_number": int(item["page_number"]),
                "source_type": "image_summary",
                "image_path": item["image_path"]
            }
        ))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked_text_docs = text_splitter.split_documents(text_docs)
    final_document = chunked_text_docs + table_docs + image_summary_docs
    return final_document


def index_documents_to_pinecone(docs_corpus, index_name="multimodel-rag-project"):
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    doc_ids = []
    for doc in docs_corpus:
        unique_string = f"{doc.metadata.get('source_type')}_{doc.metadata.get('page_number')}_{doc.page_content}"
        doc_id = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
        doc_ids.append(doc_id)
        
    vectorstore.add_documents(documents=docs_corpus, ids=doc_ids)
    return vectorstore