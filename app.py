# # app.py

# import os
# import sys
# import json
# from dotenv import load_dotenv
# import pandas as pd
# import torch
# from flask import Flask, request, jsonify
# from flask_cors import CORS # Dùng để cho phép frontend gọi API
# import numpy as np # Import numpy để sử dụng eval()

# # --- QUAN TRỌNG: Thiết lập biến môi trường HF_HOME trước khi import transformers ---
# # Điều này đảm bảo cache của Hugging Face được lưu vào thư mục `models` của project.
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "models")
# # --- ---

# # Thêm đường dẫn thư mục gốc của project vào sys.path để import các module
# sys.path.append(PROJECT_ROOT)

# # Import các file module của bạn
# import config
# from utils.pdf_processor import download_pdf, open_and_read_pdf
# from utils.text_splitter import process_text_into_chunks
# from utils import embedding_utils 
# from utils.llm_utils import load_llm_and_tokenizer, ask

# # Khởi tạo Flask App
# app = Flask(__name__)
# # Cho phép tất cả các tên miền truy cập (để chạy local frontend)
# CORS(app)

# # Khai báo các biến toàn cục để lưu model, tokenizer và embeddings
# tokenizer = None
# llm_model = None
# embedding_model = None # Embedding model không cần giữ lại sau khi tạo embeddings
# embeddings = None
# pages_and_chunks = None

# def initialize_models():
#     """
#     Tải tất cả các model và dữ liệu cần thiết.
#     Hàm này chỉ chạy một lần khi server khởi động.
#     """
#     global tokenizer, llm_model, embeddings, pages_and_chunks
    
#     print("--- Khởi tạo Server ---")
    
#     # 0. Đăng nhập Hugging Face (nếu token chưa có trong cache)
#     load_dotenv()
    
#     # 1. Tải và xử lý PDF
#     print("\n1. Tải và xử lý PDF...")
#     pdf_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
#     download_pdf(config.PDF_PATH, pdf_url)

#     pages_and_texts = open_and_read_pdf(config.PDF_PATH)
    
#     # 2. Tạo hoặc tải Embeddings
#     print("\n2. Tải/Tạo Embeddings...")
#     if os.path.exists(config.EMBEDDINGS_SAVE_PATH):
#         print(f"File embeddings '{config.EMBEDDINGS_SAVE_PATH}' đã tồn tại, đang tải...")
#         loaded_df = pd.read_csv(config.EMBEDDINGS_SAVE_PATH)
        
#         # Chuyển đổi cột embedding từ chuỗi sang numpy array
#         # Sử dụng eval() cẩn thận nếu nguồn không đáng tin
#         loaded_df["embedding"] = loaded_df["embedding"].apply(lambda x: np.array(eval(x)))

#         pages_and_chunks = loaded_df.to_dict(orient="records")
        
#         # Chuyển đổi embeddings sang torch tensor và đưa lên device
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         embeddings = torch.tensor(loaded_df["embedding"].tolist(), dtype=torch.float32).to(device)
#         print("Đã tải embeddings từ file.")
#     else:
#         print("File embeddings không tồn tại, đang tạo embeddings mới...")
#         pages_and_chunks_processed = process_text_into_chunks(pages_and_texts, config.NUM_SENTENCE_CHUNK_SIZE, config.MIN_TOKEN_LENGTH)
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         embedding_model = embedding_utils.load_embedding_model(config.EMBEDDING_MODEL_NAME, device)
        
#         text_chunks_for_embedding = [item["sentence_chunk"] for item in pages_and_chunks_processed]

#         print("Đang tạo embeddings (quá trình này có thể mất vài phút)...")
#         embeddings = embedding_model.encode(text_chunks_for_embedding, batch_size=32, convert_to_tensor=True).to(device)
        
#         df_chunks = pd.DataFrame(pages_and_chunks_processed)
#         df_chunks["embedding"] = embeddings.tolist()
#         df_chunks.to_csv(config.EMBEDDINGS_SAVE_PATH, index=False)
#         pages_and_chunks = df_chunks.to_dict(orient="records")
#         print(f"Đã tạo và lưu embeddings vào '{config.EMBEDDINGS_SAVE_PATH}'.")

#     # 3. Tải LLM và Tokenizer
#     print("\n3. Tải LLM và Tokenizer...")
#     tokenizer, llm_model = load_llm_and_tokenizer(
#         config.LLM_MODEL_NAME,
#         config.USE_QUANTIZATION_CONFIG,
#         config.ATTN_IMPLEMENTATION
#     )
#     print("Đã tải LLM và Tokenizer.")
#     print("\n--- Khởi tạo Server Hoàn tất. Sẵn sàng phục vụ! ---")

# @app.route('/ask', methods=['POST'])
# def handle_ask():
#     """Endpoint để xử lý câu hỏi RAG từ frontend."""
#     data = request.json
#     query = data.get('query')

#     if not query:
#         return jsonify({"error": "Missing 'query' parameter"}), 400

#     print(f"\nNhận được truy vấn: {query}")
    
#     try:
#         # Gọi hàm ask từ utils.llm_utils
#         answer, context_items = ask(
#             query=query,
#             tokenizer=tokenizer,
#             llm_model=llm_model,
#             embeddings=embeddings,
#             pages_and_chunks=pages_and_chunks,
#             temperature=config.DEFAULT_TEMPERATURE,
#             max_new_tokens=config.DEFAULT_MAX_NEW_TOKENS,
#             return_answer_only=False
#         )

#         # Trả về câu trả lời và ngữ cảnh
#         response_data = {
#             "answer": answer,
#             "context_items": [
#                 {
#                     "page_number": item["page_number"],
#                     "sentence_chunk": item["sentence_chunk"],
#                     "score": item["score"]
#                 } for item in context_items
#             ]
#         }
#         return jsonify(response_data)

#     except Exception as e:
#         print(f"Lỗi khi xử lý truy vấn: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     initialize_models()
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


# # app.py
# # Server backend với các tính năng LLM được thêm vào.

# import os
# import sys
# import json
# from dotenv import load_dotenv
# import pandas as pd
# import torch
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import requests # Thêm thư viện requests cho API call

# # --- QUAN TRỌNG: Thiết lập biến môi trường HF_HOME trước khi import transformers ---
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "models")
# # --- ---

# sys.path.append(PROJECT_ROOT)

# import config
# from utils.pdf_processor import download_pdf, open_and_read_pdf
# from utils.text_splitter import process_text_into_chunks
# from utils import embedding_utils 
# from utils.llm_utils import load_llm_and_tokenizer, ask

# app = Flask(__name__)
# CORS(app)

# tokenizer = None
# llm_model = None
# embeddings = None
# pages_and_chunks = None

# def initialize_models():
#     global tokenizer, llm_model, embeddings, pages_and_chunks

#     print("--- Khởi tạo Server ---")
    
#     load_dotenv()

#     print("\n1. Tải và xử lý PDF...")
#     pdf_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
#     download_pdf(config.PDF_PATH, pdf_url)

#     pages_and_texts = open_and_read_pdf(config.PDF_PATH)
    
#     print("\n2. Tải/Tạo Embeddings...")
#     if os.path.exists(config.EMBEDDINGS_SAVE_PATH):
#         print(f"File embeddings '{config.EMBEDDINGS_SAVE_PATH}' đã tồn tại, đang tải...")
#         loaded_df = pd.read_csv(config.EMBEDDINGS_SAVE_PATH)
#         loaded_df["embedding"] = loaded_df["embedding"].apply(lambda x: np.array(eval(x)))
#         pages_and_chunks = loaded_df.to_dict(orient="records")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         embeddings = torch.tensor(loaded_df["embedding"].tolist(), dtype=torch.float32).to(device)
#         print("Đã tải embeddings từ file.")
#     else:
#         print("File embeddings không tồn tại, đang tạo embeddings mới...")
#         pages_and_chunks_processed = process_text_into_chunks(pages_and_texts, config.NUM_SENTENCE_CHUNK_SIZE, config.MIN_TOKEN_LENGTH)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         embedding_model = embedding_utils.load_embedding_model(config.EMBEDDING_MODEL_NAME, device)
#         text_chunks_for_embedding = [item["sentence_chunk"] for item in pages_and_chunks_processed]
#         print("Đang tạo embeddings (quá trình này có thể mất vài phút)...")
#         embeddings = embedding_model.encode(text_chunks_for_embedding, batch_size=32, convert_to_tensor=True).to(device)
#         df_chunks = pd.DataFrame(pages_and_chunks_processed)
#         df_chunks["embedding"] = embeddings.tolist()
#         df_chunks.to_csv(config.EMBEDDINGS_SAVE_PATH, index=False)
#         pages_and_chunks = df_chunks.to_dict(orient="records")
#         print(f"Đã tạo và lưu embeddings vào '{config.EMBEDDINGS_SAVE_PATH}'.")

#     print("\n3. Tải LLM và Tokenizer...")
#     tokenizer, llm_model = load_llm_and_tokenizer(
#         config.LLM_MODEL_NAME,
#         config.USE_QUANTIZATION_CONFIG,
#         config.ATTN_IMPLEMENTATION
#     )
#     print("Đã tải LLM và Tokenizer.")
#     print("\n--- Khởi tạo Server Hoàn tất. Sẵn sàng phục vụ! ---")

# @app.route('/ask', methods=['POST'])
# def handle_ask():
#     """Endpoint để xử lý câu hỏi RAG từ frontend."""
#     # (Đoạn code này giữ nguyên từ trước)
#     data = request.json
#     query = data.get('query')
#     if not query:
#         return jsonify({"error": "Missing 'query' parameter"}), 400
#     print(f"\nNhận được truy vấn: {query}")
#     try:
#         answer, context_items = ask(
#             query=query,
#             tokenizer=tokenizer,
#             llm_model=llm_model,
#             embeddings=embeddings,
#             pages_and_chunks=pages_and_chunks,
#             temperature=config.DEFAULT_TEMPERATURE,
#             max_new_tokens=config.DEFAULT_MAX_NEW_TOKENS,
#             return_answer_only=False
#         )
#         response_data = {
#             "answer": answer,
#             "context_items": [
#                 {
#                     "page_number": item["page_number"],
#                     "sentence_chunk": item["sentence_chunk"],
#                     "score": item["score"]
#                 } for item in context_items
#             ]
#         }
#         return jsonify(response_data)
#     except Exception as e:
#         print(f"Lỗi khi xử lý truy vấn: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/generate_prompt', methods=['POST'])
# def handle_generate_prompt():
#     """
#     Endpoint mới để tạo prompt bằng Gemini API.
#     """
#     data = request.json
#     keywords = data.get('keywords')
#     if not keywords:
#         return jsonify({"error": "Missing 'keywords' parameter"}), 400
    
#     print(f"\nNhận được từ khóa để tạo prompt: {keywords}")

#     # Chuẩn bị payload cho Gemini API
#     # Sử dụng gemini-2.5-flash-preview-05-20 cho tốc độ
#     prompt = f"Từ các từ khóa sau, hãy tạo một câu hỏi chi tiết, chuyên sâu và hay về chủ đề dinh dưỡng, như một người dùng đang nghiên cứu. Các từ khóa: {keywords}"
    
#     chat_history = []
#     chat_history.append({"role": "user", "parts": [{"text": prompt}]})
#     payload = {"contents": chat_history}
#     api_key = os.getenv("GEMINI_API_KEY", "") # Lấy API key từ biến môi trường
#     apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

#     try:
#         response = requests.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})
#         response.raise_for_status() # Báo lỗi nếu status code không phải 2xx
#         result = response.json()
        
#         # Xử lý kết quả từ Gemini API
#         if result.get("candidates") and len(result["candidates"]) > 0:
#             generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
#             # Đảm bảo câu hỏi có dấu chấm hỏi
#             if not generated_text.strip().endswith('?'):
#                 generated_text = generated_text.strip() + '?'
#             return jsonify({"generated_prompt": generated_text})
#         else:
#             return jsonify({"error": "Không thể tạo prompt từ Gemini API."}), 500
#     except requests.exceptions.RequestException as e:
#         print(f"Lỗi khi gọi Gemini API: {e}")
#         return jsonify({"error": f"Lỗi khi gọi Gemini API: {e}"}), 500

# if __name__ == '__main__':
#     initialize_models()
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


# app.py
# Server backend, chạy RAG pipeline và cung cấp một API cho frontend.
# Đảm bảo đã cài đặt Flask: pip install Flask

# import os
# import sys
# import json
# from dotenv import load_dotenv
# import pandas as pd
# import torch
# from flask import Flask, request, jsonify, render_template, send_from_directory
# from flask_cors import CORS 
# import numpy as np

# # --- QUAN TRỌNG: Thiết lập biến môi trường HF_HOME trước khi import transformers ---
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "models")
# # --- ---

# sys.path.append(PROJECT_ROOT)

# import config
# from utils.pdf_processor import download_pdf, open_and_read_pdf
# from utils.text_splitter import process_text_into_chunks
# from utils import embedding_utils 
# from utils.llm_utils import load_llm_and_tokenizer, ask

# # Khởi tạo Flask App
# # Thêm tham số template_folder và static_folder để Flask biết nơi tìm file
# # Thư mục template_folder mặc định là 'templates', chúng ta đã đổi thành 'frontend'
# # Thư mục static_folder mặc định là 'static', chúng ta đã đổi thành 'frontend'
# app = Flask(__name__,
#             template_folder='frontend',
#             static_folder='frontend')
# CORS(app)

# tokenizer = None
# llm_model = None
# embeddings = None
# pages_and_chunks = None

# def initialize_models():
#     """
#     Tải tất cả các model và dữ liệu cần thiết.
#     Hàm này chỉ chạy một lần khi server khởi động.
#     """
#     global tokenizer, llm_model, embeddings, pages_and_chunks
    
#     print("--- Khởi tạo Server ---")
    
#     load_dotenv()

#     print("\n1. Tải và xử lý PDF...")
#     pdf_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
#     download_pdf(config.PDF_PATH, pdf_url)

#     pages_and_texts = open_and_read_pdf(config.PDF_PATH)
    
#     print("\n2. Tải/Tạo Embeddings...")
#     if os.path.exists(config.EMBEDDINGS_SAVE_PATH):
#         print(f"File embeddings '{config.EMBEDDINGS_SAVE_PATH}' đã tồn tại, đang tải...")
#         loaded_df = pd.read_csv(config.EMBEDDINGS_SAVE_PATH)
#         loaded_df["embedding"] = loaded_df["embedding"].apply(lambda x: np.array(eval(x)))
#         pages_and_chunks = loaded_df.to_dict(orient="records")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         embeddings = torch.tensor(loaded_df["embedding"].tolist(), dtype=torch.float32).to(device)
#         print("Đã tải embeddings từ file.")
#     else:
#         print("File embeddings không tồn tại, đang tạo embeddings mới...")
#         pages_and_chunks_processed = process_text_into_chunks(pages_and_texts, config.NUM_SENTENCE_CHUNK_SIZE, config.MIN_TOKEN_LENGTH)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         embedding_model = embedding_utils.load_embedding_model(config.EMBEDDING_MODEL_NAME, device)
#         text_chunks_for_embedding = [item["sentence_chunk"] for item in pages_and_chunks_processed]
#         print("Đang tạo embeddings (quá trình này có thể mất vài phút)...")
#         embeddings = embedding_model.encode(text_chunks_for_embedding, batch_size=32, convert_to_tensor=True).to(device)
#         df_chunks = pd.DataFrame(pages_and_chunks_processed)
#         df_chunks["embedding"] = embeddings.tolist()
#         df_chunks.to_csv(config.EMBEDDINGS_SAVE_PATH, index=False)
#         pages_and_chunks = df_chunks.to_dict(orient="records")
#         print(f"Đã tạo và lưu embeddings vào '{config.EMBEDDINGS_SAVE_PATH}'.")

#     print("\n3. Tải LLM và Tokenizer...")
#     tokenizer, llm_model = load_llm_and_tokenizer(
#         config.LLM_MODEL_NAME,
#         config.USE_QUANTIZATION_CONFIG,
#         config.ATTN_IMPLEMENTATION
#     )
#     print("Đã tải LLM và Tokenizer.")
#     print("\n--- Khởi tạo Server Hoàn tất. Sẵn sàng phục vụ! ---")

# # Endpoint mới để phục vụ file HTML
# @app.route('/')
# def serve_index():
#     # Sử dụng render_template để phục vụ file index.html từ thư mục 'frontend'
#     return render_template('index.html')

# # Endpoint để phục vụ các file tĩnh (CSS, JS, images, v.v.)
# @app.route('/<path:filename>')
# def serve_static(filename):
#     return send_from_directory(app.static_folder, filename)

# @app.route('/ask', methods=['POST'])
# def handle_ask():
#     """Endpoint để xử lý câu hỏi RAG từ frontend."""
#     data = request.json
#     query = data.get('query')

#     if not query:
#         return jsonify({"error": "Missing 'query' parameter"}), 400

#     print(f"\nNhận được truy vấn: {query}")
    
#     try:
#         answer, context_items = ask(
#             query=query,
#             tokenizer=tokenizer,
#             llm_model=llm_model,
#             embeddings=embeddings,
#             pages_and_chunks=pages_and_chunks,
#             temperature=config.DEFAULT_TEMPERATURE,
#             max_new_tokens=config.DEFAULT_MAX_NEW_TOKENS,
#             return_answer_only=False
#         )
#         response_data = {
#             "answer": answer,
#             "context_items": [
#                 {
#                     "page_number": item["page_number"],
#                     "sentence_chunk": item["sentence_chunk"],
#                     "score": item["score"]
#                 } for item in context_items
#             ]
#         }
#         return jsonify(response_data)
#     except Exception as e:
#         print(f"Lỗi khi xử lý truy vấn: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     initialize_models()
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


import os
import sys
import json
from dotenv import load_dotenv
import pandas as pd
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS 
import numpy as np

# --- QUAN TRỌNG: Thiết lập biến môi trường HF_HOME trước khi import transformers ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "models")
# --- ---

sys.path.append(PROJECT_ROOT)

import config
from utils.pdf_processor import download_pdf, open_and_read_pdf
from utils.text_splitter import process_text_into_chunks
from utils import embedding_utils 
from utils.llm_utils import load_llm_and_tokenizer, ask

# Khởi tạo Flask App
app = Flask(__name__,
            template_folder='frontend',
            static_folder='frontend')
CORS(app)

tokenizer = None
llm_model = None
embeddings = None
pages_and_chunks = None

def initialize_models():
    """
    Tải tất cả các model và dữ liệu cần thiết.
    Hàm này chỉ chạy một lần khi server khởi động.
    """
    global tokenizer, llm_model, embeddings, pages_and_chunks
    
    print("--- Khởi tạo Server ---")
    
    load_dotenv()

    print("\n1. Tải và xử lý PDF...")
    pdf_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    download_pdf(config.PDF_PATH, pdf_url)

    pages_and_texts = open_and_read_pdf(config.PDF_PATH)
    
    print("\n2. Tải/Tạo Embeddings...")
    if os.path.exists(config.EMBEDDINGS_SAVE_PATH):
        print(f"File embeddings '{config.EMBEDDINGS_SAVE_PATH}' đã tồn tại, đang tải...")
        loaded_df = pd.read_csv(config.EMBEDDINGS_SAVE_PATH)
        loaded_df["embedding"] = loaded_df["embedding"].apply(lambda x: np.array(eval(x)))
        pages_and_chunks = loaded_df.to_dict(orient="records")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = torch.tensor(loaded_df["embedding"].tolist(), dtype=torch.float32).to(device)
        print("Đã tải embeddings từ file.")
    else:
        print("File embeddings không tồn tại, đang tạo embeddings mới...")
        pages_and_chunks_processed = process_text_into_chunks(pages_and_texts, config.NUM_SENTENCE_CHUNK_SIZE, config.MIN_TOKEN_LENGTH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model = embedding_utils.load_embedding_model(config.EMBEDDING_MODEL_NAME, device)
        text_chunks_for_embedding = [item["sentence_chunk"] for item in pages_and_chunks_processed]
        print("Đang tạo embeddings (quá trình này có thể mất vài phút)...")
        embeddings = embedding_model.encode(text_chunks_for_embedding, batch_size=32, convert_to_tensor=True).to(device)
        df_chunks = pd.DataFrame(pages_and_chunks_processed)
        df_chunks["embedding"] = embeddings.tolist()
        df_chunks.to_csv(config.EMBEDDINGS_SAVE_PATH, index=False)
        pages_and_chunks = df_chunks.to_dict(orient="records")
        print(f"Đã tạo và lưu embeddings vào '{config.EMBEDDINGS_SAVE_PATH}'.")

    print("\n3. Tải LLM và Tokenizer...")
    tokenizer, llm_model = load_llm_and_tokenizer(
        config.LLM_MODEL_NAME,
        config.USE_QUANTIZATION_CONFIG,
        config.ATTN_IMPLEMENTATION
    )
    print("Đã tải LLM và Tokenizer.")
    print("\n--- Khởi tạo Server Hoàn tất. Sẵn sàng phục vụ! ---")

# Endpoint mới để phục vụ file HTML
@app.route('/')
def serve_index():
    return render_template('index.html')

# Endpoint để phục vụ các file tĩnh (CSS, JS, images, v.v.)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# THÊM ENDPOINT NÀY ĐỂ PHỤC VỤ FILE PDF
@app.route('/data/<path:filename>')
def serve_data(filename):
    """Phục vụ các tệp từ thư mục 'data'."""
    data_folder = os.path.join(PROJECT_ROOT, "data")
    return send_from_directory(data_folder, filename)

@app.route('/ask', methods=['POST'])
def handle_ask():
    """Endpoint để xử lý câu hỏi RAG từ frontend."""
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    print(f"\nNhận được truy vấn: {query}")
    
    try:
        answer, context_items = ask(
            query=query,
            tokenizer=tokenizer,
            llm_model=llm_model,
            embeddings=embeddings,
            pages_and_chunks=pages_and_chunks,
            temperature=config.DEFAULT_TEMPERATURE,
            max_new_tokens=config.DEFAULT_MAX_NEW_TOKENS,
            return_answer_only=False
        )
        response_data = {
            "answer": answer,
            "context_items": [
                {
                    "page_number": item["page_number"],
                    # "sentence_chunk": item["sentence_chunk"],
                    # "score": item["score"]
                } for item in context_items
            ]
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Lỗi khi xử lý truy vấn: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_models()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)