import os
import sys
from dotenv import load_dotenv
import pandas as pd
import torch
from huggingface_hub import login, HfFolder
import numpy as np

# Thêm đường dẫn thư mục gốc project vào sys.path để import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config
from utils.pdf_processor import download_pdf, open_and_read_pdf
from utils.text_splitter import process_text_into_chunks
from utils import embedding_utils # Import module đầy đủ để gọi load_embedding_model bên trong ask()
from utils.llm_utils import load_llm_and_tokenizer, ask

def setup_huggingface_login():
    """Đăng nhập Hugging Face nếu token chưa có trong cache hoặc chưa được thiết lập."""
    token = HfFolder.get_token()
    if token is None:
        print("Vui lòng dán Hugging Face token của bạn vào đây (từ hf.co/settings/tokens):")
        login()
        token = HfFolder.get_token()
    if token:
        print("Đã đăng nhập Hugging Face.")
    else:
        print("Không thể đăng nhập Hugging Face. Vui lòng kiểm tra token của bạn.")

def main():
    # Load biến môi trường từ .env (nếu có)
    load_dotenv()
    setup_huggingface_login()

    # Thiết bị (CPU hoặc CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    # 1. Tải và xử lý PDF
    print("--- Bước 1: Tải và xử lý PDF ---")
    pdf_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    download_pdf(config.PDF_PATH, pdf_url)

    pages_and_texts = open_and_read_pdf(config.PDF_PATH)
    pages_and_chunks = process_text_into_chunks(pages_and_texts, config.NUM_SENTENCE_CHUNK_SIZE, config.MIN_TOKEN_LENGTH)

    # 2. Tạo hoặc tải Embeddings
    print("--- Bước 2: Tải/Tạo Embeddings ---")
    text_chunks_and_embedding_df = pd.DataFrame(pages_and_chunks) # Khởi tạo DataFrame từ các chunk đã xử lý

    if os.path.exists(config.EMBEDDINGS_SAVE_PATH):
        print(f"File embeddings '{config.EMBEDDINGS_SAVE_PATH}' đã tồn tại, đang tải...")
        loaded_df = pd.read_csv(config.EMBEDDINGS_SAVE_PATH)
        # Chuyển đổi cột embedding từ chuỗi sang numpy array hoặc list (tùy thuộc vào cách bạn lưu)
        # Đây là một điểm có thể cần điều chỉnh tùy thuộc vào định dạng lưu
        loaded_df["embedding"] = loaded_df["embedding"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x) # Sử dụng eval() cẩn thận nếu nguồn không đáng tin
        # Đảm bảo giữ đúng thứ tự và số lượng chunk sau khi lọc và xử lý
        text_chunks_and_embedding_df = loaded_df # Giả sử file đã lưu là file cuối cùng sau xử lý và lọc

        # Convert embeddings sang torch tensor và đưa lên device
        embeddings = torch.tensor(text_chunks_and_embedding_df["embedding"].tolist(), dtype=torch.float32).to(device)
        print("Đã tải embeddings từ file.")
    else:
        print("File embeddings không tồn tại, đang tạo embeddings mới...")
        embedding_model = embedding_utils.load_embedding_model(config.EMBEDDING_MODEL_NAME, device)
        
        # Lấy danh sách các chunk_text để tạo embeddings
        text_chunks_for_embedding = [item["sentence_chunk"] for item in pages_and_chunks] # Lấy từ pages_and_chunks đã lọc

        # Tạo embeddings cho tất cả các chunk
        print("Đang tạo embeddings (quá trình này có thể mất vài phút)...")
        embeddings = embedding_model.encode(text_chunks_for_embedding, batch_size=32, convert_to_tensor=True).to(device)
        
        # Thêm cột embeddings vào DataFrame của các chunk đã xử lý
        text_chunks_and_embedding_df["embedding"] = embeddings.tolist()
        text_chunks_and_embedding_df.to_csv(config.EMBEDDINGS_SAVE_PATH, index=False)
        print(f"Đã tạo và lưu embeddings vào '{config.EMBEDDINGS_SAVE_PATH}'.")

    # 3. Tải LLM và Tokenizer
    print("--- Bước 3: Tải LLM và Tokenizer ---")
    tokenizer, llm_model = load_llm_and_tokenizer(
        config.LLM_MODEL_NAME,
        config.USE_QUANTIZATION_CONFIG,
        config.ATTN_IMPLEMENTATION
    )
    print("Đã tải LLM và Tokenizer.")

    # 4. Chạy RAG Pipeline
    print("\n--- Bước 4: Khởi động NutriChat RAG Pipeline ---")
    print("Chào mừng đến với NutriChat!")
    print("Gõ 'thoat' để kết thúc chương trình.")

    # Đảm bảo pages_and_chunks luôn ở dạng list[dict] cho hàm ask()
    # Nếu bạn đã tải từ CSV, text_chunks_and_embedding_df là DataFrame chứa đủ thông tin
    # Bạn có thể cần chuyển nó về list[dict] nếu các hàm utils mong đợi như vậy
    pages_and_chunks_for_rag = text_chunks_and_embedding_df.to_dict(orient="records")

    while True:
        query = input("\nBạn có câu hỏi gì về dinh dưỡng? ")
        if query.lower() == 'thoat':
            print("Cảm ơn bạn đã sử dụng NutriChat. Tạm biệt!")
            break
        
        if not query.strip(): # Xử lý câu hỏi rỗng
            print("Vui lòng nhập câu hỏi.")
            continue

        print("\nĐang tìm kiếm câu trả lời...")
        try:
            answer = ask(
                query=query,
                tokenizer=tokenizer,
                llm_model=llm_model,
                embeddings=embeddings,
                pages_and_chunks=pages_and_chunks_for_rag, # Truyền data đã định dạng cho hàm ask
                temperature=config.DEFAULT_TEMPERATURE,
                max_new_tokens=config.DEFAULT_MAX_NEW_TOKENS,
                format_answer_text=True,
                return_answer_only=True
            )
            print(f"\nCâu trả lời của NutriChat:\n{answer}")
        except Exception as e:
            print(f"Đã xảy ra lỗi khi tạo câu trả lời: {e}")
            print("Vui lòng thử lại hoặc kiểm tra kết nối/cấu hình model.")

if __name__ == "__main__":
    main()