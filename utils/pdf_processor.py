import os
import requests
import fitz # PyMuPDF
from tqdm.auto import tqdm

def download_pdf(pdf_path: str, url: str):
    """
    Tải file PDF từ URL nếu nó chưa tồn tại.
    """
    if not os.path.exists(pdf_path):
        print(f"File '{pdf_path}' không tồn tại, đang tải xuống...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                file.write(response.content)
            print(f"Đã tải và lưu file dưới dạng {pdf_path}")
        else:
            print(f"Không thể tải file. Mã trạng thái: {response.status_code}")
    else:
        print(f"File '{pdf_path}' đã tồn tại.")

def text_formatter(text: str) -> str:
    """
    Thực hiện định dạng văn bản cơ bản (thay thế dấu xuống dòng bằng khoảng trắng).
    """
    cleaned_text = text.replace("\n", " ").strip()
    # Các chức năng định dạng văn bản tiềm năng khác có thể đặt ở đây
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Mở một file PDF, đọc nội dung văn bản từng trang, và thu thập thống kê.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number, # Giữ nguyên số trang để dễ debug, sau này có thể điều chỉnh
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4, # 1 token = ~4 ký tự
            "text": text
        })
    doc.close()
    return pages_and_texts