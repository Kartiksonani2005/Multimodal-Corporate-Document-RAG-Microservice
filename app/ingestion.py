import os
import hashlib
import json
import base64
import pymupdf
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

def extract_pdf_content(pdf_path, output_img_dir="extracted_images"):
    os.makedirs(output_img_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    structured_documents = []
    seen_image_hashes = {}  

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        page_text = page.get_text("text")
        lines = page_text.split("\n")
        cleaned_lines = [
            line for line in lines 
            if not line.startswith("NovaCore Systems Ltd. - Fictional company report") 
            and not line.strip().startswith("Page ")
        ]
        page_text = "\n".join(cleaned_lines).strip()

        if page_text:
            structured_documents.append(Document(
                page_content=page_text,
                metadata={"page_number": int(page_number), "source_type": "text"}
            ))

        tabs = page.find_tables()
        for tab in tabs:
            table_df = tab.extract()
            if table_df and len(table_df) > 0:
                header = table_df[0]
                separator = ["---"] * len(header)
                rows = table_df[1:]
                
                md_lines = [
                    " | ".join(str(cell or "").strip() for cell in header),
                    " | ".join(separator)
                ]
                for row in rows:
                    md_lines.append(" | ".join(str(cell or "").strip() for cell in row))
                markdown_table = "\n".join(md_lines)
                
                structured_documents.append(Document(
                    page_content=f"### Table on Page {page_number}\n{markdown_table}",
                    metadata={"page_number": int(page_number), "source_type": "table"}
                ))

        image_list = page.get_images(full=True)
        for image_index, image in enumerate(image_list):
            xref = image[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            if len(image_bytes) < 5000:
                continue

            img_hash = hashlib.md5(image_bytes).hexdigest()

            if img_hash not in seen_image_hashes:
                image_filename = f"page_{page_number}_img_{image_index}.{image_ext}"
                image_filepath = os.path.join(output_img_dir, image_filename).replace("\\", "/")

                with open(image_filepath, "wb") as f:
                    f.write(image_bytes)
                
                seen_image_hashes[img_hash] = image_filepath

    return structured_documents


def generate_image_summaries(img_dir="extracted_images", cache_file="image_summaries_cache.json"):

    load_dotenv()
    
    vision_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.0, max_tokens=1024)

    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding="utf-8") as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            cache = {}

    image_summaries = []
    if not os.path.exists(img_dir):
        return image_summaries

    prompt_version_tag = "v2_transcription_strict"

    for filename in sorted(os.listdir(img_dir)):
        if filename.lower().endswith((".png", ".jpeg", ".jpg")):
            image_path = os.path.join(img_dir, filename).replace("\\", "/")
            
            with open(image_path, "rb") as image_file:
                img_bytes = image_file.read()
                img_hash = hashlib.md5(img_bytes).hexdigest()
                cache_key = f"{img_hash}_{prompt_version_tag}"

            if cache_key in cache:
                image_summaries.append(cache[cache_key])
                continue

            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            mime_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Transcribe and describe this visual element, chart, table, or diagram in complete detail. Extract all numbers, labels, metrics, and text explicitly visible."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}
                    }
                ]
            )

            response = vision_llm.invoke([message])
            summary_text = response.content
            if "<think>" in summary_text and "</think>" in summary_text:
                summary_text = summary_text.split("</think>")[-1].strip()

            page_num = 1
            if "page_" in filename:
                try:
                    page_num = int(filename.split("_")[1])
                except ValueError:
                    pass

            summary_data = {
                "filename": filename,
                "page_number": page_num,
                "image_path": image_path,
                "summary": summary_text
            }

            cache[cache_key] = summary_data
            image_summaries.append(summary_data)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=4, ensure_ascii=False)

    return image_summaries