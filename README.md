# Flappy Bird: Double Dueling DQN

Agent học chơi Flappy Bird từ ảnh (`pixel`, CNN) hoặc vector trạng thái (`state`, MLP).
Kỹ thuật: Double DQN, Dueling, Huber loss, soft target update, AMP, replay buffer đều.

Phần thưởng sau shaping: chết -1, qua ống +1, bay vượt mép trên màn hình -0,5, còn sống +0,1.

## Cấu trúc

```
flappy_rl/
  config.py      AgentConfig, TrainConfig (kiểm tra tham số)
  envs.py        wrapper pixel/state và reward shaping
  models.py      mạng Dueling CNN/MLP
  replay.py      replay buffer
  agent.py       Double DQN, checkpoint
  training.py    vòng huấn luyện, chọn checkpoint tốt nhất
  evaluation.py  đánh giá greedy, quay video
  export.py      xuất replay/weights cho trình xem 3D
  viewer.py      nhúng dữ liệu vào assets/flappy_3d.html
  plotting.py    đồ thị huấn luyện
notebooks/flappy_bird_rl.ipynb   train, đánh giá, xem 3D
tests/                           test CPU nhẹ, dùng môi trường giả
scripts/package_source.py        đóng gói dist/flappy_bird.zip
```

## Sử dụng

```bash
pip install -e .[dev]
ruff check .
python -m pytest
```

CI (GitHub Actions) chạy `ruff` và `pytest` trên CPU cho mỗi lần push/PR.

Mở `notebooks/flappy_bird_rl.ipynb`, chọn `MODE` rồi chạy từng cell.
Train tiếp từ checkpoint: đặt `RESUME` (cấu hình và chế độ đọc từ checkpoint).
Kết quả mỗi lần chạy nằm trong `runs/<mode>_<thời gian>/`; vẽ lại đồ thị bằng `History.load`.

Trình xem 3D đọc replay của cả hai chế độ; chạy mạng trực tiếp trong trình duyệt
chỉ hỗ trợ MLP (`state`).

## Lưu ý

- Checkpoint không chứa replay buffer và RNG nên resume không tái lập từng bước.
- Replay `pixel` 50.000 transition chiếm khoảng 2,6 GiB RAM.
- Điểm chọn checkpoint dùng seed cố định (`seed + 10000`); điểm holdout dùng seed khác.
