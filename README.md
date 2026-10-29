# Flappy Bird: Double Dueling DQN

Agent học chơi Flappy Bird từ ảnh (`pixel`, CNN) hoặc vector trạng thái (`state`, MLP).
Kỹ thuật: Double DQN, Dueling, n-step return (n = 3), Huber loss, soft target update, AMP,
replay buffer đều, khám phá có thiên lệch (hành động ngẫu nhiên chỉ vỗ cánh 8%).

Train theo ngân sách bước môi trường: `pixel` 2 triệu bước, `state` 1 triệu bước.
Chế độ `pixel` tắt ảnh nền của game, cắt phần đất và chỉ giữ mặt nạ chim + ống (84x84, 4 khung).

### Vì sao bản cũ không học được

Bản cũ dừng sau 1500 episode (khoảng 75 nghìn bước), epsilon về 0,01 trước khi agent thấy phần
thưởng qua ống. Hành động ngẫu nhiên vỗ cánh 50% làm chim dính trần và chết ở ống đầu tiên:
đo trên game thật, 0% episode ngẫu nhiên ghi được điểm, nên agent không có tín hiệu để học.
Với xác suất vỗ cánh 8%, khoảng 8% episode ngẫu nhiên qua được ống.

Tham khảo: [uvipen/Flappy-bird-deep-Q-learning-pytorch](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch)
(2 triệu bước, cắt đất, ảnh nhị phân), [johnnycode8/dqn_pytorch](https://github.com/johnnycode8/dqn_pytorch)
(Double + Dueling DQN trên `flappy-bird-gymnasium`), bài viết DDQN của Lamboqin (khám phá không đều).

## Kết quả: `pixel`, 2 triệu bước

Thư mục [flappy_bird_output/pixel_20261029_074054](flappy_bird_output/pixel_20261029_074054)
(train trên GPU Colab, 1 giờ 25 phút, khoảng 390 bước/giây).

| Chỉ số | Giá trị |
|---|---|
| Holdout greedy, 10 episode, seed khác, tối đa 10.000 bước (khoảng 262 ống) | **193,7 ống** |
| Eval greedy tốt nhất (5 episode, tối đa 5.000 bước, trần 132 ống) | 132, lần đầu ở bước 600.000 (`best.pt`) |
| Điểm cao nhất lúc train | 265 ống |
| Điểm trung bình lúc train (epsilon 0,01) sau 250.000 bước | khoảng 30 ống |

![Đồ thị huấn luyện](flappy_bird_output/pixel_20261029_074054/training_results.png)

- Agent bắt đầu học khi epsilon về 0,01 (bước 250.000): điểm trung bình episode tăng từ 1,0
  (250 nghìn bước đầu) lên 28,8 (bước 250 nghìn đến 500 nghìn), rồi giữ quanh 30 đến 35.
- Cả 3 episode trong video (tối đa 1.500 bước) và 3 episode replay 3D (tối đa 3.000 bước) đều
  chạm giới hạn bước mà chim chưa chết (39 và 79 ống).
- Điểm lúc train thấp hơn điểm greedy vì 1% hành động ngẫu nhiên đủ làm chim chết ở khe hẹp.
- Điểm eval dao động mạnh giữa các lần (22 đến 132): chính sách DQN dao động, cộng thêm chỉ 5
  episode mỗi lần eval. Giới hạn bước làm 132 và khoảng 262 là trần đo, không phải giới hạn của agent.
- Mỗi ống tốn khoảng 38 bước (khung hình).

Trong repo có `best.pt` (Git LFS, 52 MB), log, đồ thị, video, replay 3D và notebook đã chạy
(`flappy_bird_rl_executed.ipynb`). `last.pt` (bước 2.000.000) chỉ lưu trên Drive.

Phần thưởng sau shaping: chết -1, qua ống +1, bay vượt mép trên màn hình -0,5, còn sống +0,1.

## Cấu trúc

```
flappy_rl/
  config.py      AgentConfig, TrainConfig (kiểm tra tham số)
  envs.py        wrapper pixel/state và reward shaping
  models.py      mạng Dueling CNN/MLP
  replay.py      replay buffer, n-step collector
  agent.py       Double DQN, checkpoint
  training.py    vòng huấn luyện theo bước, chọn checkpoint tốt nhất
  runtime.py     thiết bị, seed, thư mục output
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
Train tiếp khi Colab ngắt: đặt `RESUME_DIR` là thư mục lần chạy cũ (agent nạp `last.pt`).

Kết quả mỗi lần chạy nằm trong `flappy_bird_output/<mode>_<thời gian>/`. Trên Colab thư mục gốc là
`My Drive/flappy_bird_output/`; đổi bằng biến môi trường `FLAPPY_OUTPUT`.

```
flappy_bird_output/pixel_20261029_071937/
  config.json            cấu hình agent và train
  best.pt, last.pt       checkpoint tốt nhất (điểm greedy) và mới nhất
  progress.csv           log theo bước (điểm, loss, epsilon, tốc độ, điểm eval)
  history.json           từng episode và từng lần eval; đọc lại bằng History.load
  training_results.png   đồ thị
  holdout.json           điểm đánh giá cuối (seed khác)
  gameplay.mp4           video
  replay.json, weights.json, viewer.html   giao diện 3D
```

Trình xem 3D đọc replay của cả hai chế độ; chạy mạng trực tiếp trong trình duyệt
chỉ hỗ trợ MLP (`state`).

## Lưu ý

- Checkpoint không chứa replay buffer và RNG nên resume không tái lập từng bước; khi resume,
  buffer được nạp lại bằng `warmup_steps` bước với epsilon hiện tại.
- Tốc độ môi trường `pixel` khoảng 700 bước/giây (render pygame); `state` nhanh hơn nhiều.
- Video của chế độ `pixel` không có ảnh nền, đúng như ảnh agent nhìn thấy.
- Replay `pixel` 50.000 transition chiếm khoảng 2,6 GiB RAM.
- Điểm chọn checkpoint dùng seed cố định (`seed + 10000`); điểm holdout dùng seed khác.
