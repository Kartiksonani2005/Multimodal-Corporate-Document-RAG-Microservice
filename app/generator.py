import os
import time
import base64
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def multimodal_rag_answer(query, retriever):
    start_time = time.time()
    retrieved_docs = retriever.invoke(query)
    
    context_texts = []
    image_payloads = []
    sources = []
    images_sent = []
    
    for doc in retrieved_docs:
        source_type = doc.metadata.get("source_type")
        page_num = doc.metadata.get("page_number")
        sources.append(f"Page {page_num} ({source_type})")
        
        if source_type in ["text", "table"]:
            context_texts.append(f"[Page {page_num} - {source_type.upper()}]:\n{doc.page_content}")
        elif source_type == "image_summary":
            context_texts.append(f"[Page {page_num} - IMAGE SUMMARY]:\n{doc.page_content}")
            img_path = doc.metadata.get("image_path")
            
            if img_path and os.path.exists(img_path) and len(images_sent) < 2 and img_path not in images_sent:
                b64_img = encode_image(img_path)
                mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
                image_payloads.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_img}"}
                })
                images_sent.append(img_path)

    combined_context = "\n\n---\n\n".join(context_texts)
    
    system_prompt = (
        "You are an expert enterprise AI assistant. Answer the user's question accurately and concisely "
        "using ONLY the provided context and embedded visual data. Always cite the specific page numbers."
    )
    
    user_prompt = f"Context:\n{combined_context}\n\nUser Question: {query}\nAnswer:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}] + image_payloads}
    ]

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=messages,
        temperature=0.1,
        max_tokens=800
    )
    
    latency = time.time() - start_time
    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(sources)),
        "images_sent": images_sent,
        "latency": round(latency, 2)
    }