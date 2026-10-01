# Chain Reaction AI

Dự án huấn luyện AI chơi game **Chain Reaction** trên bàn 5×5 bằng AlphaZero-style: mạng neural PolicyValueNet kết hợp thuật toán tìm kiếm MCTS.

---

## 📁 Cấu Trúc Project

```
chain_ai/
├── engine.py        # Luật chơi Chain Reaction, môi trường Gym
├── brain.py         # Mạng neural (ResNet), MCTS, hàm huấn luyện
├── train.py         # Vòng lặp tự đấu → huấn luyện → đánh giá
├── play.py          # Giao diện pygame để đấu với AI
├── books.py         # Ghi nhận khai cuộc và chiến thuật vào SQLite
├── test_engine.py   # Kiểm thử tự động
└── requirements.txt # Thư viện cần thiết
```

---

## ⚙️ Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu |
|---|---|
| Python | **3.11 hoặc 3.12** (khuyên dùng 3.12) |
| RAM | Tối thiểu 4 GB |
| GPU | Tùy chọn (CUDA) — CPU vẫn chạy được |
| OS | Windows / Linux / macOS |

> ⚠️ **Không dùng Python 3.14** — pygame chưa hỗ trợ. Tải Python 3.12 tại [python.org](https://www.python.org/downloads/release/python-3128/).

---

## 🚀 Cài Đặt

### Windows (VS Code / Command Prompt)

```bash
cd D:\chain_ai
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
cd ~/chain_ai
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Kích Hoạt Môi Trường (mỗi lần mở terminal mới)

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## ✅ Kiểm Tra Sau Cài Đặt

```bash
python -m compileall engine.py brain.py books.py train.py play.py test_engine.py
python -m pytest -q
```

Tất cả test phải **passed** trước khi bắt đầu huấn luyện.

---

## 🏋️ Huấn Luyện

### Chạy Thử (Nhanh — Kiểm Tra Luồng)

```bash
python train.py --out runs/smoke --iterations 1 --channels 16 --blocks 1 --games-per-iteration 2 --simulations 16 --eval-simulations 16 --train-steps 5 --batch-size 16 --arena-games 4 --baseline-games 2
```

> Cấu hình này chỉ kiểm tra luồng vận hành, không đủ để tạo AI mạnh.

### Huấn Luyện Thực Tế

```bash
python train.py --out runs/chain --iterations 100 --channels 64 --blocks 4 --games-per-iteration 32 --simulations 256 --eval-simulations 256 --train-steps 400 --batch-size 128 --arena-games 40 --baseline-games 10
```

### Huấn Luyện Sâu Hơn (1000 mô phỏng)

```bash
python train.py --out runs/chain --iterations 200 --channels 64 --blocks 4 --games-per-iteration 32 --simulations 1000 --eval-simulations 1000 --train-steps 400 --batch-size 128 --arena-games 40 --baseline-games 10
```

> **Tiếp tục huấn luyện**: Chạy lại cùng `--out` để tiếp tục từ checkpoint.  
> **Thay kiến trúc mạng**: Dùng thư mục `--out` khác.

---

## ⚡ Bật Numba (Tăng Tốc Engine)

Chỉ bật sau khi tất cả test đã passed.

```bash
# Windows (PowerShell)
$env:CHAIN_NUMBA="1"

# Linux / macOS
export CHAIN_NUMBA=1
```

---

## 📊 Theo Dõi Training

```bash
tensorboard --logdir runs/chain/tensorboard
```

Mở trình duyệt vào `http://localhost:6006` để xem biểu đồ loss, win rate, v.v.

---

## 🎮 Chơi Game

### Đấu Với AI

```bash
python play.py --checkpoint runs/chain/best.pt --human 1 --simulations 256
```

### Xem AI Tự Đấu

```bash
python play.py --checkpoint runs/chain/best.pt --watch --simulations 256
```

### Dùng Gợi Ý Từ Sách Khai Cuộc

```bash
python play.py --checkpoint runs/chain/best.pt --book runs/chain/books.sqlite --simulations 256
```

---

## 📖 Xuất Sách Khai Cuộc

```bash
python books.py --db runs/chain/books.sqlite --out runs/chain/textbook
```

Tạo ra `textbook.md` và `books.json` trong thư mục `runs/chain/textbook/`.

---

## ☁️ Huấn Luyện Trên Google Colab (GPU)

Tạo notebook `train_colab.ipynb`, chọn runtime GPU, thêm các ô sau:

**Ô 1 — Kết nối Drive**
```python
from google.colab import drive
drive.mount("/content/drive")
```

**Ô 2 — Vào thư mục dự án**
```python
%cd /content/drive/MyDrive/chain_ai
%pip install -r requirements.txt
```

> Sao chép thư mục dự án vào `MyDrive/chain_ai` trước.

**Ô 3 — Kiểm tra GPU và chạy tests**
```python
import subprocess, sys, torch

print("PyTorch:", torch.__version__)
print("CUDA khả dụng:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
```

**Ô 4 — Huấn luyện**
```python
import os, subprocess, sys

os.environ["CHAIN_NUMBA"] = "1"

subprocess.run([
    sys.executable, "train.py",
    "--out", "/content/drive/MyDrive/chain_ai/runs/chain",
    "--device", "auto",
    "--iterations", "100",
    "--channels", "64", "--blocks", "4",
    "--games-per-iteration", "32",
    "--simulations", "256", "--eval-simulations", "256",
    "--train-steps", "400", "--batch-size", "128",
    "--arena-games", "40", "--baseline-games", "10",
], check=True)
```

**Ô 5 — TensorBoard**
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/chain_ai/runs/chain/tensorboard
```

> ⚠️ Colab có thể ngắt phiên bất ngờ. Checkpoint được ghi sau mỗi ván lên Drive giúp phục hồi, nhưng có thể làm chậm tốc độ tự đấu.

---

## 📦 Thư Viện Sử Dụng

| Thư viện | Mục đích |
|---|---|
| `torch` | Mạng neural PolicyValueNet |
| `numpy` | Tính toán mảng, MCTS |
| `gymnasium` | Môi trường game chuẩn |
| `pygame` | Giao diện đồ họa |
| `tensorboard` | Theo dõi quá trình huấn luyện |
| `numba` | Tăng tốc engine (tùy chọn) |
| `pytest` | Kiểm thử tự động |

---

## 🔖 Ghi Chú

- **Checkpoint** được lưu vào `runs/chain/trainer.pt` (đầy đủ) và `runs/chain/best.pt` (chơi game).
- **Sách khai cuộc** (`books.sqlite`) tích lũy qua các lần tự đấu, chỉ là thống kê quan sát, không phải chứng minh chắc thắng.
- **Bàn cờ 5×5**: Mỗi ô nổ khi đạt đủ số hàng xóm (góc=2, cạnh=3, giữa=4). Người thắng khi xóa hết ô của đối thủ.
  




🗂️ Cơ bản
Thông số	Mặc định	Ý nghĩa
--out	runs/chain	Thư mục lưu checkpoint, tensorboard, sách
--device	auto	auto tự chọn GPU/CPU, hoặc đặt cuda/cpu
--seed	42	Seed ngẫu nhiên để tái tạo kết quả
🧠 Kiến trúc Mạng Neural
Thông số	Mặc định	Ý nghĩa
--channels	64	Số kênh trong mạng ResNet. Càng lớn → mạng càng mạnh nhưng chậm hơn
--blocks	4	Số khối ResNet. Càng nhiều → hiểu sâu hơn nhưng tốn RAM/GPU

⚠️ Nếu thay --channels hoặc --blocks, bắt buộc dùng --out mới — không thể tiếp tục checkpoint cũ.

🎮 Tự Đấu (Self-Play)
Thông số	Mặc định	Ý nghĩa
--iterations	100	Tổng số chu kỳ huấn luyện. 1 chu kỳ = tự đấu → train → đánh giá
--games-per-iteration	16	Số ván tự đấu mỗi chu kỳ trước khi train
--simulations	128	Số lần MCTS mô phỏng mỗi nước trong tự đấu. Càng cao → AI đi chuẩn hơn nhưng chậm hơn
--explore-plies	10	Số nước đầu mỗi ván chọn ngẫu nhiên (khám phá). Sau đó AI chọn nước tốt nhất
🏋️ Huấn Luyện
Thông số	Mặc định	Ý nghĩa
--train-steps	200	Số bước gradient mỗi chu kỳ
--batch-size	128	Số mẫu mỗi bước train. Lớn hơn → ổn định hơn, cần RAM nhiều hơn
--replay-size	50000	Kích thước bộ nhớ lưu dữ liệu tự đấu cũ để học lại
--learning-rate	0.002	Tốc độ học. Quá lớn → không hội tụ, quá nhỏ → học chậm
⚔️ Đánh Giá (Arena)
Thông số	Mặc định	Ý nghĩa
--arena-games	40	Số ván AI mới đấu với AI cũ để quyết định có thay thế không. Phải là số chẵn
--eval-simulations	128	Số mô phỏng MCTS khi đánh giá trong arena
--baseline-games	10	Số ván đấu với AI đi ngẫu nhiên để đo tiến bộ. Phải là số chẵn, đặt 0 để bỏ qua

AI mới được chấp nhận khi Wilson lower bound > 0.5 (tức là thắng đủ tin cậy, không phải may mắn).