import os
import hashlib
import json
import base64
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from app.config import VISION_MODEL, CACHE_FILE, OUTPUT_IMG_DIR

def generate_image_summaries(img_dir=OUTPUT_IMG_DIR, cache_file=CACHE_FILE):
    vision_llm = ChatGroq(model=VISION_MODEL, temperature=0.0, max_tokens=1024)

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
                        "text": "Transcribe and describe this visual element, chart, table, or diagram in complete detail. Extract all numbers, labels, metrics, and text explicitly visible. Do not interpret or extrapolate beyond what is shown. Mark estimated values clearly."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}
                    }
                ]
            )

            response = vision_llm.invoke([message])
            summary_text = response.content
            
            # Clean reasoning tags if model outputs them
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