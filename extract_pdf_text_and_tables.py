import json
import pdfplumber
from pathlib import Path

# Initialize structured data containers
extracted_text = {}
extracted_tables = {}
plain_text_pages = []

with pdfplumber.open("manuals/hl_series_iom.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        page_num = i + 1
        
        # 1. Extract regular text
        text = page.extract_text()
        if text:
            extracted_text[f"page_{page_num}"] = text
            plain_text_pages.append(f"--- PAGE {page_num} ---\n{text}")
            
        # 2. Extract structured tables
        tables = page.extract_tables()
        if tables:
            extracted_tables[f"page_{page_num}"] = tables

# Save text content to text.json
with open("text.json", "w", encoding="utf-8") as text_file:
    json.dump(extracted_text, text_file, ensure_ascii=False, indent=4)

# Save table matrices to tables.json
with open("tables.json", "w", encoding="utf-8") as tables_file:
    json.dump(extracted_tables, tables_file, ensure_ascii=False, indent=4)

Path("output.txt").write_text("\n\n".join(plain_text_pages), encoding="utf-8")
print("Extraction complete! Saved output.txt, text.json, and tables.json.")