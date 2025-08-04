import os
import torch
from transformers.utils import is_flash_attn_2_available

# --- Cấu hình chung ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Thư mục gốc của project

# --- Cấu hình Data ---
PDF_FILE_NAME = "human-nutrition-text.pdf"
PDF_PATH = os.path.join(PROJECT_ROOT, "data", PDF_FILE_NAME)
EMBEDDINGS_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "text_chunks_and_embeddings_paraphrase-multilingual-mpnet-base-v2.csv")

# --- Cấu hình Embedding Model ---
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# --- Cấu hình LLM ---
# Bạn có thể thay đổi model_id ở đây nếu muốn thử Gemma-7b-it,
# nhưng hãy đảm bảo GPU của bạn có đủ VRAM.
# Tùy thuộc vào thiết bị của bạn, bạn sẽ điều chỉnh LLM_MODEL_NAME và USE_QUANTIZATION_CONFIG
LLM_MODEL_NAME = "google/gemma-2b-it" # Gemma 2B Instruction-Tuned
# USE_QUANTIZATION_CONFIG = True # Sử dụng 4-bit quantization nếu VRAM hạn chế (ví dụ: < 16GB cho 7B)

# Tự động xác định USE_QUANTIZATION_CONFIG và ATTN_IMPLEMENTATION dựa trên GPU
GPU_MEMORY_GB = 0
if torch.cuda.is_available():
    GPU_MEMORY_BYTES = torch.cuda.get_device_properties(0).total_memory
    GPU_MEMORY_GB = round(GPU_MEMORY_BYTES / (2**30))

if GPU_MEMORY_GB < 5.1:
    print(f"LƯU Ý: Bộ nhớ GPU khả dụng của bạn là {GPU_MEMORY_GB}GB, có thể không đủ để chạy Gemma LLM cục bộ mà không lượng tử hóa sâu hơn.")
    USE_QUANTIZATION_CONFIG = True # Cố gắng dùng lượng tử hóa nếu bộ nhớ thấp
    LLM_MODEL_NAME = "google/gemma-2b-it" # Luôn ưu tiên 2B cho bộ nhớ thấp
elif GPU_MEMORY_GB < 8.1:
    print(f"Bộ nhớ GPU: {GPU_MEMORY_GB}GB | Model đề xuất: Gemma 2B ở độ chính xác 4-bit.")
    USE_QUANTIZATION_CONFIG = True
    LLM_MODEL_NAME = "google/gemma-2b-it"
elif GPU_MEMORY_GB < 19.0:
    print(f"Bộ nhớ GPU: {GPU_MEMORY_GB}GB | Model đề xuất: Gemma 2B ở float16 hoặc Gemma 7B ở 4-bit.")
    USE_QUANTIZATION_CONFIG = False # Thử float16 cho 2B hoặc 4-bit cho 7B
    LLM_MODEL_NAME = "google/gemma-7b-it" # Mặc định 2B float16 nếu không đủ 7B, có thể đổi thủ công
elif GPU_MEMORY_GB >= 19.0:
    print(f"Bộ nhớ GPU: {GPU_MEMORY_GB}GB | Model đề xuất: Gemma 7B ở 4-bit hoặc float16.")
    USE_QUANTIZATION_CONFIG = False # Mặc định float16
    LLM_MODEL_NAME = "google/gemma-7b-it"

# Cấu hình Flash Attention 2
if torch.cuda.is_available() and (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    ATTN_IMPLEMENTATION = "flash_attention_2"
else:
    ATTN_IMPLEMENTATION = "sdpa"
print(f"[INFO] Sử dụng cài đặt lượng tử hóa: {USE_QUANTIZATION_CONFIG}")
print(f"[INFO] Model LLM đã chọn: {LLM_MODEL_NAME}")
print(f"[INFO] Phương pháp Attention: {ATTN_IMPLEMENTATION}")


# --- Cấu hình xử lý văn bản ---
NUM_SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LENGTH = 30 # Lọc các chunk có ít hơn 30 token


# --- Cấu hình LLM Generation ---
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_NEW_TOKENS = 512

# --- Cấu hình thư mục Models (Hugging Face cache) ---
# Thư mục này dùng để tải và lưu trữ model.
# Hugging Face sẽ tự động tạo cấu trúc con bên trong.
HUGGINGFACE_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.environ["HF_HOME"] = HUGGINGFACE_MODELS_DIR # Đặt biến môi trường cho HF cache