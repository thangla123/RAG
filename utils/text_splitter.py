import re
from spacy.lang.en import English
from tqdm.auto import tqdm
import pandas as pd

# Khởi tạo SpaCy NLP pipeline (chỉ 1 lần)
try:
    nlp = English()
    nlp.add_pipe("sentencizer")
except Exception as e:
    print(f"Lỗi khi tải SpaCy model: {e}. Vui lòng chạy 'python -m spacy download en_core_web_sm' trong môi trường ảo của bạn.")
    raise

def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Chia input_list thành các sublist có kích thước slice_size (hoặc gần nhất có thể).
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def process_text_into_chunks(pages_and_texts: list[dict], num_sentence_chunk_size: int, min_token_length: int) -> list[dict]:
    """
    Xử lý danh sách các trang và văn bản để tạo ra các chunk có kích thước phù hợp.
    """
    # Bước 1: Chia văn bản thành các câu bằng SpaCy
    for item in tqdm(pages_and_texts, desc="Chia trang thành câu"):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])

    # Bước 2: Chia các câu thành các chunk
    for item in tqdm(pages_and_texts, desc="Chia câu thành chunk"):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                             slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    # Bước 3: Tạo các item riêng cho mỗi chunk và tính toán thống kê
    pages_and_chunks = []
    for item in tqdm(pages_and_texts, desc="Tạo chunk riêng biệt"):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Nối các câu thành một đoạn văn bản (chunk)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            # Thêm khoảng trắng sau dấu chấm nếu không có (ví dụ: ".A" -> ". A")
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Lấy thống kê về chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # Ước tính 1 token ~ 4 ký tự

            pages_and_chunks.append(chunk_dict)

    # Bước 4: Lọc các chunk quá ngắn
    # Convert sang DataFrame để dễ lọc, sau đó chuyển lại list of dicts
    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_filtered = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

    print(f"Tổng số chunk sau lọc: {len(pages_and_chunks_filtered)}")
    return pages_and_chunks_filtered