import os
import re
import json
import cv2
import numpy as np
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

load_dotenv()

# ---------------------------------------------------------
# PDF TEXT DETECTION
# ---------------------------------------------------------
def pdf_has_text(pdf_path: str, min_chars: int = 200) -> bool:
    reader = PdfReader(pdf_path)
    total = 0
    for page in reader.pages:
        text = page.extract_text()
        if text:
            total += len(text.strip())
    return total >= min_chars


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def ocr_pdf(pdf_path: str) -> str:
    images = convert_from_path(pdf_path, dpi=300)
    texts = []

    for img in images:
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        text = pytesseract.image_to_string(thresh, config="--psm 6")
        texts.append(text)

    return "\n".join(texts)


def extract_text_with_fallback(pdf_path: str) -> str:
    if pdf_has_text(pdf_path):
        text = extract_pdf_text(pdf_path)
        if len(text.strip()) >= 200:
            return text
    return ocr_pdf(pdf_path)


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------
def normalize_soil_text(raw_text: str) -> str:
    text = raw_text.lower()
    text = re.sub(r"ds/m|dsm", "ds m-1", text)
    text = re.sub(r"ppm|mg/kg", "ppm", text)
    text = re.sub(r"%", " percent", text)

    aliases = {
        "oc": "organic carbon",
        "org carbon": "organic carbon",
        "potash": "available potassium",
        "k2o": "available potassium",
        "p2o5": "available phosphorus",
    }

    for k, v in aliases.items():
        text = text.replace(k, v)

    return re.sub(r"\n{2,}", "\n", text).strip()


# ---------------------------------------------------------
# LLM JSON EXTRACTION
# ---------------------------------------------------------
def extract_json_from_llm(text: str) -> dict:
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError("LLM did not return JSON")
    return json.loads(match.group(1))

def call_gemini_soil_analysis(cleaned_text: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=2048,
        timeout=10,          # ✅ HARD STOP
        max_retries=0        # ✅ NO RETRIES
    )

    
    prompt = f"""
    You are an API that extracts soil analysis data.
    
    STRICT RULES:
    - VALID JSON ONLY
    - NO markdown
    - DO NOT rename keys
    - DO NOT add keys
    
    RETURN EXACT SCHEMA:
    
    {{
      "soil_parameters": [
        {{
          "name": string,
          "value": number | null,
          "unit": string | null,
          "status": string,
          "notes": string
        }}
      ],
      "advisory": {{
        "summary": string,
        "recommendations": [
          {{
            "issue": string,
            "action": string,
            "priority": "LOW" | "MEDIUM" | "HIGH"
          }}
        ]
      }},
      "confidence": "LOW" | "MEDIUM" | "HIGH"
    }}
    
    Soil report:
    {cleaned_text}
    """

    try:
        response = llm.invoke(prompt)
        return extract_json_from_llm(response.content)

    except ChatGoogleGenerativeAIError as e:
        return {
            "error": "LLM_UNAVAILABLE",
            "message": "Soil advisory service temporarily unavailable",
            "confidence": "LOW"
        }
