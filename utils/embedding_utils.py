import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from time import perf_counter as timer
from tqdm.auto import tqdm

def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    """
    Tải và khởi tạo mô hình embedding SentenceTransformer.
    """
    print(f"[INFO] Đang tải mô hình embedding: {model_name} về {device}...")
    embedding_model = SentenceTransformer(model_name_or_path=model_name, device=device)
    print("Đã tải mô hình embedding.")
    return embedding_model

def retrieve_relevant_resources(query: str,
                                embeddings: torch.Tensor,
                                model: SentenceTransformer,
                                n_resources_to_return: int = 5,
                                print_time: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tạo embedding cho truy vấn và trả về top k điểm số và chỉ số từ embeddings.
    """
    # Tạo embedding cho truy vấn
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Lấy điểm số dot product trên embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Thời gian lấy điểm số trên {len(embeddings)} embeddings: {end_time-start_time:.5f} giây.")

    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return scores, indices

def print_wrapped(text: str, wrap_length: int = 80):
    """
    In văn bản đã được wrap để dễ đọc.
    """
    import textwrap
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def print_top_results_and_scores(query: str,
                                 embeddings: torch.Tensor,
                                 pages_and_chunks: list[dict],
                                 n_resources_to_return: int = 5):
    """
    Nhận một truy vấn, truy xuất các tài nguyên liên quan nhất và in chúng ra theo thứ tự giảm dần.
    """
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  model=load_embedding_model(config.EMBEDDING_MODEL_NAME, "cuda" if torch.cuda.is_available() else "cpu"), # Tải lại model nếu chưa có
                                                  n_resources_to_return=n_resources_to_return)

    print(f"Truy vấn: '{query}'\n")
    print("Kết quả:")
    for score, index in zip(scores, indices):
        print(f"Điểm: {score:.4f}")
        print("Văn bản:")
        print_wrapped(pages_and_chunks[index.item()]["sentence_chunk"]) # .item() để lấy giá trị số nguyên từ tensor
        print(f"Số trang: {pages_and_chunks[index.item()]['page_number']}")
        print("\n")

# Các hàm dot_product và cosine_similarity (chỉ để minh họa, không dùng trong pipeline chính)
def dot_product(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    return torch.dot(vector1, vector2)

def cosine_similarity(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    dot_product_val = torch.dot(vector1, vector2)
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))
    return dot_product_val / (norm_vector1 * norm_vector2)