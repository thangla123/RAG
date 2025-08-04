import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import config # Import config từ thư mục gốc
from . import embedding_utils # Thêm dòng này
import os

def load_llm_and_tokenizer(model_id: str, use_quantization_config: bool, attn_implementation: str):
    """
    Tải và khởi tạo LLM và tokenizer.
    """
    print(f"[INFO] Đang tải LLM: {model_id} và tokenizer...")

    quantization_config = None
    if use_quantization_config:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id,
        # local_files_only=True # Chỉ load local nếu chắc chắn đã có, nếu không thì để mặc định
    )

    # LLM
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        # low_cpu_mem_usage=False, # Sử dụng toàn bộ bộ nhớ nếu có
        attn_implementation=attn_implementation,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        # local_files_only=True # Chỉ load local nếu chắc chắn đã có, nếu không thì để mặc định
    )

    if not use_quantization_config:
        if torch.cuda.is_available():
            llm_model.to("cuda")
        else:
            print("[CẢNH BÁO] CUDA không khả dụng, model sẽ chạy trên CPU (có thể chậm).")

    print("Đã tải LLM và tokenizer.")
    return tokenizer, llm_model


def prompt_formatter(query: str, context_items: list[dict], tokenizer: AutoTokenizer) -> str:
    """
    Format prompt tối ưu cho Google Gemma-7B-IT, dùng chat template chuẩn của Gemma.
    """
    # Ghép các đoạn ngữ cảnh lại
    context = "- " + "\n- ".join(item["sentence_chunk"] for item in context_items)

    # Prompt chính: rõ ràng, gọn, đúng với cách hướng dẫn của Gemma
    user_prompt = f"""
Bạn là một trợ lý AI hiểu ngôn ngữ tự nhiên, được cung cấp ngữ cảnh và truy vấn để trả lời chi tiết, mạch lạc và chính xác.

Ngữ cảnh:
{context}

Truy vấn:
{query}

Yêu cầu:
- Chỉ dựa trên thông tin trong ngữ cảnh để trả lời.
- Không đoán nếu không đủ thông tin.
- Trình bày rõ ràng, có thể liệt kê nếu cần.
"""

    # Đóng gói theo chuẩn chat template của Gemma
    dialogue = [
        {"role": "user", "content": user_prompt}
    ]

    # Áp dụng template theo tokenizer của Gemma
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

# def prompt_formatter(query: str, context_items: list[dict], tokenizer: AutoTokenizer) -> str:
#     """
#     Tăng cường truy vấn với ngữ cảnh dựa trên văn bản từ context_items.
#     """
#     # Nối các mục ngữ cảnh thành một đoạn văn bản
#     context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

#     # Tạo một base prompt với các ví dụ để giúp model
#     base_prompt = f"""Dựa trên các mục ngữ cảnh sau, vui lòng trả lời truy vấn.
# Hãy tự suy nghĩ bằng cách trích xuất các đoạn liên quan từ ngữ cảnh trước khi trả lời truy vấn.
# Đừng trả lại suy nghĩ, chỉ trả lại câu trả lời.
# Đảm bảo câu trả lời của bạn giải thích chi tiết nhất có thể.
# Sử dụng các ví dụ sau đây để tham khảo kiểu câu trả lời lý tưởng.

# Ví dụ 1:
# Truy vấn: Vitamin tan trong chất béo là gì?
# Trả lời: Các vitamin tan trong chất béo bao gồm Vitamin A, Vitamin D, Vitamin E và Vitamin K. Các vitamin này được hấp thụ cùng với chất béo trong chế độ ăn uống và có thể được lưu trữ trong mô mỡ và gan của cơ thể để sử dụng sau này. Vitamin A quan trọng cho thị lực, chức năng miễn dịch và sức khỏe làn da. Vitamin D đóng vai trò quan trọng trong việc hấp thụ canxi và sức khỏe xương. Vitamin E hoạt động như một chất chống oxy hóa, bảo vệ tế bào khỏi bị hư hại. Vitamin K rất cần thiết cho quá trình đông máu và chuyển hóa xương.

# Ví dụ 2:
# Truy vấn: Nguyên nhân gây bệnh tiểu đường loại 2 là gì?
# Trả lời: Bệnh tiểu đường loại 2 thường liên quan đến tình trạng dinh dưỡng quá mức, đặc biệt là việc tiêu thụ quá nhiều calo dẫn đến béo phì. Các yếu tố bao gồm chế độ ăn nhiều đường tinh luyện và chất béo bão hòa, có thể dẫn đến kháng insulin, một tình trạng mà các tế bào của cơ thể không phản ứng hiệu quả với insulin. Theo thời gian, tuyến tụy không thể sản xuất đủ insulin để kiểm soát lượng đường trong máu, dẫn đến bệnh tiểu đường loại 2. Ngoài ra, việc tiêu thụ calo quá mức mà không có đủ hoạt động thể chất làm tăng nguy cơ bằng cách thúc đẩy tăng cân và tích tụ mỡ, đặc biệt là xung quanh vùng bụng, góp phần làm tăng tình trạng kháng insulin.

# Ví dụ 3:
# Truy vấn: Tầm quan trọng của việc hydrat hóa đối với hiệu suất thể chất là gì?
# Trả lời: Hydrat hóa rất quan trọng đối với hiệu suất thể chất vì nước đóng vai trò then chốt trong việc duy trì thể tích máu, điều hòa nhiệt độ cơ thể và đảm bảo vận chuyển chất dinh dưỡng và oxy đến các tế bào. Hydrat hóa đầy đủ là điều cần thiết cho chức năng cơ bắp, sức bền và phục hồi tối ưu. Mất nước có thể dẫn đến giảm hiệu suất, mệt mỏi và tăng nguy cơ mắc các bệnh liên quan đến nhiệt, chẳng hạn như say nắng. Uống đủ nước trước, trong và sau khi tập thể dục giúp đảm bảo hiệu suất thể chất và phục hồi đỉnh cao.

# Bây giờ sử dụng các mục ngữ cảnh sau để trả lời truy vấn của người dùng:
# {context}

# Các đoạn liên quan: <extract relevant passages from the context here>
# Truy vấn của người dùng: {query}
# Trả lời:"""

#     # Cập nhật base prompt với các mục ngữ cảnh và truy vấn
#     base_prompt = base_prompt.format(context=context, query=query)

#     # Tạo template prompt cho model đã được điều chỉnh hướng dẫn
#     dialogue_template = [
#         {"role": "user",
#          "content": base_prompt}
#     ]

#     # Áp dụng chat template
#     prompt = tokenizer.apply_chat_template(
#         conversation=dialogue_template,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     return prompt

def ask(query: str,
        tokenizer: AutoTokenizer,
        llm_model: AutoModelForCausalLM,
        embeddings: torch.Tensor,
        pages_and_chunks: list[dict], # Giữ nguyên dạng list[dict] cho hàm này
        temperature: float = config.DEFAULT_TEMPERATURE,
        max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
        format_answer_text: bool = True,
        return_answer_only: bool = True) -> (str or tuple[str, list[dict]]):
    """
    Nhận một truy vấn, tìm kiếm tài nguyên/ngữ cảnh liên quan và tạo câu trả lời dựa trên các tài nguyên đó.
    """
    # Lấy điểm số và chỉ số của các kết quả liên quan hàng đầu
    embedding_model = embedding_utils.load_embedding_model(config.EMBEDDING_MODEL_NAME, embeddings.device.type) # Tải embedding model ở đây
    scores, indices = embedding_utils.retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        model=embedding_model,
        n_resources_to_return=5
    )

    # Tạo danh sách các mục ngữ cảnh
    context_items = [pages_and_chunks[i.item()] for i in indices] # .item() để lấy giá trị số nguyên từ tensor

    # Thêm điểm số vào mục ngữ cảnh
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu().item() # Đưa điểm số về CPU và lấy giá trị Python

    # Định dạng prompt với các mục ngữ cảnh
    prompt = prompt_formatter(query=query,
                              context_items=context_items,
                              tokenizer=tokenizer)

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(llm_model.device) # Đảm bảo input_ids trên cùng device với model

    # Tạo đầu ra token
    outputs = llm_model.generate(
        **input_ids,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens
    )

    # Chuyển đầu ra token thành văn bản
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Thay thế các token đặc biệt và thông báo trợ giúp không cần thiết
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "").strip()

    if return_answer_only:
        return output_text

    return output_text, context_items