# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Đào Gia Bảo
- MSSV: 2A202602793
- Email: Chưa được cung cấp.
- Link repo (fork): https://github.com/kevindao94work/K4-L2L3-DAY23-DaoGiaBao-2A202602793-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): Xem hash cuối cùng cung cấp khi nộp LMS (được tạo sau commit báo cáo).

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0` (số nguyên).
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: `0.9700934579439252`, `0.7004048582995951`, `519/16/222`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.1503226533088836` m, `502`, `11.343643849107053` m², `0`, `239`, `2.522613065326633`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.13586674608851862` m, `502`, `9.266805891726358` m², `0`, `239`, `2.522613065326633`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: `rmse_fused-rmse_lidar=-0.014455907220364994` m; fused giảm khoảng 1,45 cm trên tổng thể. Chênh lệch matches, ghost và miss đều bằng 0: cả hai mode có 502 cặp, không ghost và 239 missed GT frames. Precision tracking `502/(502+0)=1.0`, coverage `502/519=0.9672447013487476` cho cả hai. Camera tinh chỉnh vị trí mà không thay đổi quyết định tồn tại track. Tuy nhiên fused không tốt hơn ở mọi frame: tại frame 20, `sum_sq_err` LiDAR là `0.022419309420855498`, fused là `0.044574829164691805`; frame 100 tương ứng `0.06014117599334394` và `0.04875274503122809`. Vì vậy kết luận dựa trên tổng lỗi và số cặp của cả segment, không chọn frame thuận lợi.

| Chỉ tiêu | LiDAR | Fused | Yêu cầu | Đánh giá |
|---|---:|---:|---:|---|
| RMSE 3D (m) | 0.150322653 | 0.135866746 | ≤0.45 | Đạt cả hai |
| precision_track | 1.0 | 1.0 | ≥0.75 | Đạt cả hai |
| coverage | 0.967244701 | 0.967244701 | ≥0.70 | Đạt cả hai |
| RMSE fused − LiDAR (m) | — | -0.014455907 | ≤0.05 | Đạt |

239 miss là tổng số GT không ghép confirmed track qua các frame, không phải số xe duy nhất. Detection bỏ sót 222 GT frames; chênh 17 so với misses tracking là so sánh tổng, không chứng minh từng GT bị bỏ sót trùng nhau. Coverage dùng detection TP làm mẫu số, không phải toàn bộ GT; do đó coverage cao không có nghĩa hệ thống bao phủ tất cả xe thật.

Kiểm chứng: `128 passed`, không failed/xfailed; `compileall` và `git diff --check` thành công. Sáu artifacts do runner nguyên bản sinh, không chỉnh sửa tay. Validator chính thức xác nhận metrics của cả compare và hai mode riêng khớp logs: 199 frame/mode, tổng 398 records, không thiếu/trùng, đúng invariant, detection giống nhau giữa hai mode, RMSE khớp `sqrt(sum_sq_err/matches)`.

Dữ liệu tải bằng Google Cloud CLI từ nguồn Waymo chính thức `gs://waymo_open_dataset_v_1_4_3/individual_files/training/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, đặt thêm tiền tố `training_` theo cấu hình lab. File 990,170,672 bytes, MD5 base64 `delyDT2pX8tHwXthBhAD2g==` khớp metadata GCS; reader đọc được đủ 199 Frame. Bài làm sử dụng Waymo Open Dataset do Waymo LLC cung cấp theo [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/); quyền truy cập và sử dụng chịu các điều khoản đó.

Weights SFA3D lấy từ upstream được đề cho phép: 50,984,463 bytes, Git blob SHA-1 `dfb87a00edb82e1dedc4eeb2639efba87a6ec031` khớp GitHub; detector nạp thành công 12,728,353 parameters. Dataset và weights chỉ lưu local, không commit.

Môi trường: Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.6.0 (CPU), OpenCV headless 4.11.0.86, protobuf 6.33.6. Venv local `/Users/tridao/venvs/day23-pip`; khi chạy đặt `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 DAY23_HEADLESS=1`. `.env` và `paths.yaml` giữ ngoài Git.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   LiDAR đo vị trí trong hệ cảm biến: `z` 3×1, đơn vị m; `R` 3×3, đơn vị m². Mô hình `h(x)` là biến đổi rigid của vị trí vehicle sang sensor; `H` 3×6 chứa rotation và ba cột vận tốc bằng 0. Camera đo pixel `(u,v)`: `z` 2×1, `R` 2×2 với đường chéo `sigma_cam_i²`, `sigma_cam_j²`, đơn vị pixel². Chiếu pinhole `u=c_i-f_i*y_s/x_s`, `v=c_j-f_j*z_s/x_s` là phi tuyến; Jacobian 2×6 do `platform/fusion_lab/tracking/sensors.py:get_H` tính theo chain rule. `workspace/kalman.py:ekf_update` dùng cùng công thức innovation và gain cho cả hai loại đo.
2. Vì sao cần gating Mahalanobis trước khi gán?

   `workspace/association.py` dùng `d²=γᵀS⁻¹γ`, `γ=z-h(x)`, `S=HPHᵀ+R`. Gating loại cặp không phù hợp trước khi greedy lấy chi phí nhỏ nhất và xóa hàng/cột. Ngưỡng `chi2.ppf(gating_threshold,dim_meas)` có 3 bậc tự do cho LiDAR, 2 cho camera. Khác Euclidean, Mahalanobis chuẩn hóa theo độ bất định và tương quan: cùng residual có thể được chấp nhận khi covariance lớn nhưng bị loại khi covariance nhỏ. Kiểm tra FOV trước MHD tránh chiếu điểm sau camera, độ sâu ≤1e-6 hoặc tọa độ không hữu hạn. Cặp bị loại giữ chi phí `inf`; mỗi track/đo chỉ được ghép một lần.
3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Đây là track-then-fuse: một tracker trạng thái 6D dùng chung cho hai cảm biến. Trong `platform/fusion_lab/scripts/run_lab.py:run`, mỗi frame gọi `KF.predict` một lần cho các track cũ, sau đó association/update/quản lý LiDAR, rồi association/update camera nếu có FRONT labels. `workspace/association.py:associate_and_update` không predict thêm. Fuse-then-track sẽ kết hợp đo/detection trước khi tracking; lab không thực hiện bước đó. Waymo Frame được xem là một tick đồng bộ, cả hai đo dùng `t=frame*dt`; không mô phỏng queue cảm biến bất đồng bộ. Trong `artifacts/grade_run_lidar.log` và `artifacts/grade_run_fused.log`, frame 4 đều có `confirmed=2`, `matches=2`; frame 100 đều có `confirmed=3`, `matches=3`, nhưng `sum_sq_err` khác nhau như phần tóm tắt. Log ghi kết quả sau các bước cập nhật; thứ tự predict/update được xác nhận từ code runner, không suy ra riêng từ schema log.
4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Extrinsic sai làm sai `p_s=R_sv*p_v+t_sv`, từ đó lệch pixel dự đoán `h(x)` và tạo bias trong innovation `γ`. Squared Mahalanobis có thể tăng và vượt cổng χ², khiến mất camera association. Nếu bias vẫn lọt gate hoặc ghép sang đối tượng khác, EKF có thể kéo vị trí lệch và tăng RMSE 3D. Sai calibration cũng làm sai hướng Jacobian và kiểm tra FOV. Đây là phân tích từ mô hình; chưa chạy thí nghiệm perturbation nên không khẳng định mức ảnh hưởng định lượng.
5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi `meas_list=[]`, không thể lấy cảm biến từ phần tử đầu. Sensor tường minh giúp `manager.manage_tracks` luôn chạy đúng lượt, kể cả frame LiDAR không có detection để trừ score track trong FOV và xóa track hết điều kiện tồn tại. Camera không có đo không được phạt track. `platform/fusion_lab/tracking/manager.py` chỉ cộng score cho lidar hit và chỉ init/delete trong lidar pass; camera chỉ cập nhật trạng thái/covariance bằng EKF. Camera của lab là tâm nhãn GT 2D có nhiễu seeded, không phải detector ảnh độc lập, nên không dùng nó làm bằng chứng tồn tại track.
6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   `workspace/track_management.py` khởi tạo `score=1/window`, `state='initialized'`, vận tốc 0; covariance vị trí là `R` xoay sang vehicle, covariance vận tốc là các `sigma_p44/55/66²`. LiDAR hit cộng `1/window` (chặn trên ở 1); LiDAR miss trong FOV trừ `1/window`. Hit chưa đủ xác nhận chuyển tentative. Xác nhận khi `score > confirmed_threshold` (strict), và đã confirmed thì không hạ trạng thái sau miss. Xóa nếu `Pxx > max_P` hoặc `Pyy > max_P`, hoặc confirmed có `score < delete_threshold`, hoặc chưa confirmed có `score <= 0`. Với tham số gốc: window=6, confirmed_threshold=0.8, delete_threshold=0.6, max_P=9 m². Camera không gọi các quyết định lifecycle.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): OpenAI Codex; sử dụng hướng dẫn Markdown do người dùng cung cấp.
- Dùng cho phần nào (hàm, câu hỏi, debug): Triển khai toàn bộ TODO E–H, cài môi trường, xử lý tải dữ liệu/weights, chạy kiểm thử, giải thích sáu câu hỏi và soạn báo cáo. Người học cần tự đọc và giải thích được mã khi vấn đáp.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Baseline 82 passed, 46 xfailed; sau E–H: 128 passed, không failed/xfailed. Đối chiếu F/Q, innovation, gain, pinhole và gate với đề; kiểm tra bổ sung covariance khi xoay hệ tọa độ, shape ma trận rỗng và đầu vào ndarray. Weights khớp Git blob nguồn và nạp được vào detector. Đã chạy Waymo compare thật đủ frame 0–198, seed 0 sau lần sửa E–H cuối; validator chính thức kiểm tra đủ sáu artifacts, 398 records và tính nhất quán metrics/log. Số liệu và nhận xét trong báo cáo lấy từ chính artifacts này.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
