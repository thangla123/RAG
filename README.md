# RAG-DEMO-CHAT

Ứng dụng **RAG-DEMO-CHAT** là một chatbot được xây dựng dựa trên kiến trúc **Retrieval-Augmented Generation (RAG)**. Mục tiêu của ứng dụng là trả lời các câu hỏi của người dùng bằng cách tìm kiếm và tham khảo thông tin từ một tài liệu cụ thể (trong trường hợp này là `human-nutrition-text.pdf`).

Ứng dụng bao gồm hai phần chính:

* **Backend:** Được phát triển bằng **Python** sử dụng framework **Flask**, có nhiệm vụ xử lý các câu hỏi, tìm kiếm thông tin liên quan từ cơ sở dữ liệu nhúng (embeddings) và tạo ra câu trả lời.

* **Frontend:** Một giao diện web đơn giản được xây dựng bằng **HTML, CSS và JavaScript**, cho phép người dùng tương tác trực tiếp với chatbot.

---

## Cấu trúc dự án

Dưới đây là cấu trúc thư mục của dự án để bạn dễ dàng hình dung:

<pre lang="markdown"> ```text RAG-DEMO-CHAT/ ├── data/ │ └── human-nutrition-text.pdf # Tài liệu nguồn để chatbot tham khảo ├── frontend/ │ ├── assets/ │ │ ├── bot_avatar.png │ │ └── user_avatar.png │ └── index.html # Giao diện chính của ứng dụng cùng logic xử lý tương tác giao diện ├── models/ ├── notebooks/ ├── utils/ │ ├── embedding_utils.py │ ├── llm_utils.py │ ├── pdf_processor.py │ └── text_splitter.py ├── .env # Biến môi trường ├── .gitignore ├── app.py # Điểm khởi đầu của ứng dụng Flask ├── config.py ├── main.py ├── README.md # File mô tả dự án này └── requirements.txt # Danh sách các thư viện Python cần thiết ``` </pre>

## Hướng dẫn cài đặt và sử dụng

Để chạy ứng dụng, bạn cần thực hiện theo các bước sau.

###  Cài đặt môi trường

1.  **Tạo môi trường ảo:**

    ```bash
    python -m venv venv-rag
    ```

2.  **Kích hoạt môi trường ảo:**

Trên WSL/Linux/macOS:

      ```bash
      source venv-rag/bin/activate
      ```

# Hướng dẫn cài đặt PyTorch 2.3.1 với CUDA 12.1

Đây là các bước để cài đặt **PyTorch 2.3.1** cùng với các thư viện cần thiết, bao gồm cả hướng dẫn cài đặt **CUDA Toolkit** cho **WSL** (Windows Subsystem for Linux).

---

##  Cài đặt PyTorch

Chạy lệnh sau để cài đặt **PyTorch 2.3.1** tương thích với **CUDA 12.1**:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

##  Cài đặt các thư viện khác

```bash
pip install --no-cache-dir \
    PyMuPDF==1.23.26 \
    matplotlib==3.8.3 \
    numpy==1.26.4 \
    pandas==2.2.1 \
    requests==2.31.0 \
    sentence-transformers==2.5.1 \
    tqdm==4.66.2 \
    transformers==4.38.2 \
    accelerate \
    bitsandbytes \
    spacy \
    jupyter \
    wheel \
    python-dotenv==1.1.1 \
    Flask==3.1.1 \
    flask-cors==6.0.1
```

## Cài đặt CUDA Toolkit 12.9.1 cho WSL
Thực hiện các lệnh sau một cách tuần tự để cài đặt CUDA Toolkit 12.9.1:

```bash
wget [https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin)
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget [https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb](https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb)
sudo dpkg -i cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
```


## Cập nhật biến môi trường PATH
Chạy các lệnh sau để thêm CUDA vào PATH và LD_LIBRARY_PATH:

```bash
echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Cài đặt Flash-attn 
*Lưu ý*: Chỉ cài đặt thư viện này nếu GPU của bạn hỗ trợ compute capability >= 8.0

```bash
pip install -q flash-attn==2.5.8 --no-build-isolation
```

### Khởi động ứng dụng

1.  **Chạy ứng dụng Flask:**
    Sau khi dữ liệu đã được chuẩn bị, bạn có thể khởi động backend bằng lệnh:

    ```bash
    python app.py
    ```

    Nếu mọi thứ hoạt động bình thường, bạn sẽ thấy thông báo server đã chạy tại một địa chỉ, thường là `http://127.0.0.1:5000`.

### Sử dụng ứng dụng

1.  **Mở trình duyệt:**
    Truy cập vào địa chỉ `http://127.0.0.1:5000` trên trình duyệt của bạn.

2.  **Tương tác với chatbot:**
    Giao diện chatbot sẽ xuất hiện. Bạn có thể bắt đầu nhập câu hỏi của mình liên quan đến chủ đề `human-nutrition` (dinh dưỡng con người) và nhận câu trả lời từ bot.

# Cập nhật mô hình
Bạn có thể dễ dàng thay đổi mô hình Embedding và LLM bằng cách chỉnh sửa file .env hoặc config.py.

## Cập nhật mô hình Embedding
File **.env** chứa huggingface token để login
Nếu bạn muốn sử dụng một mô hình Embedding khác (ví dụ: một mô hình mới hơn hoặc phù hợp hơn), hãy chỉnh sửa tên mô hình trong **config.py**:

```bash
EMBEDDING_MODEL_NAME = "your-new-embedding-model-name"
```

## Cập nhật LLMs
Để thay đổi mô hình LLM, bạn chỉ cần cập nhật tên mô hình trong file config.py:
```bash
LLM_MODEL_NAME = "your-new-llm-model-name"
```
