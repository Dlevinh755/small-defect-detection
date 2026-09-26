# Kế hoạch Thiết kế Thực nghiệm & Nghiên cứu
## Đề tài: Phát hiện lỗi bề mặt kích thước nhỏ (Small Surface Defect Detection)

> **Cách dùng tài liệu này**
> - `⏳ TBD` = chưa chốt, **đợi kết quả thực nghiệm** rồi điền / lên kế hoạch tiếp.
> - `❓ HỎI` = cần thêm thông tin (giảng viên, nhóm, doanh nghiệp) trước khi quyết định.
> - `✅` = đã chốt. Khi chốt một quyết định, ghi lại vào **Mục 11 – Nhật ký quyết định**.
> - Mọi con số thời gian GPU trong tài liệu là **ước lượng**, cập nhật lại sau lần chạy thật đầu tiên.

---

## Mục lục
0. Thông tin chung
1. Bài toán, câu hỏi nghiên cứu, giả thuyết
2. Dữ liệu
3. Protocol thực nghiệm chung (áp dụng cho MỌI lần chạy)
4. Giai đoạn 1 – Thường kỳ: Baseline + cải tiến nhỏ
5. Giai đoạn 2 – Giữa kỳ: Cải tiến Backbone / Loss + bảng so sánh
6. Giai đoạn 3 – Cuối kỳ: Transfer Learning sang dữ liệu thực tế
7. Nhánh Publish (song song, không bắt buộc cho học phần)
8. Rủi ro & phương án dự phòng
9. Danh sách câu hỏi mở
10. Cấu trúc repo & quy ước đặt tên
11. Nhật ký quyết định & nhật ký thực nghiệm
12. Tài liệu tham khảo
Phụ lục A – Code snippet

---

## 0. Thông tin chung

| Mục | Nội dung |
|---|---|
| Học phần | Deep Learning ❓ HỎI tên chính xác học phần |
| Thành viên | ❓ HỎI – số người, tên, vai trò (xem Mục 5.3) |
| Mốc Thường kỳ | ❓ HỎI ngày nộp |
| Mốc Giữa kỳ | ❓ HỎI ngày nộp |
| Mốc Cuối kỳ | ❓ HỎI ngày nộp |
| Hình thức báo cáo | Slide (tương tự báo cáo N1) ❓ HỎI có cần kèm report viết / code không |
| Tài nguyên | Kaggle T4 16GB (~30 giờ/tuần, tối đa 12 giờ/phiên – kiểm tra quota thực tế), Colab free T4 làm dự phòng |
| Mục tiêu phụ | Có thể phát triển thành paper hội nghị (Mục 7) |

**Yêu cầu của giảng viên (tóm tắt):**
- **Thường kỳ:** Baseline + 1 cải tiến nhỏ.
- **Giữa kỳ:** Cải tiến sâu hơn trên **Backbone hoặc Loss function** + **bảng so sánh thực nghiệm nhiều mô hình**, thể hiện đóng góp của từng thành viên.
- **Cuối kỳ:** Dùng mô hình chung đã tối ưu, **Transfer Learning** sang tập dữ liệu thực tế (ít dữ liệu, gần bài toán doanh nghiệp).
- Gợi ý kỹ thuật: attention (CAM, SimAM), module trích xuất đặc trưng (C3), backbone DINO/DINOv2 cho DETR, tùy biến IoU loss cho box chồng lấn / đối tượng nhỏ.

---

## 1. Bài toán, câu hỏi nghiên cứu, giả thuyết

### 1.1. Phát biểu bài toán
Phát hiện (định vị bằng bounding box + phân loại) các lỗi bề mặt công nghiệp có **kích thước nhỏ** so với ảnh, trong điều kiện mô hình phải nhẹ (chạy được thời gian thực) và dữ liệu thực tế thường ít, mất cân bằng.

### 1.2. Định nghĩa "lỗi nhỏ"
Dùng **cả hai** định nghĩa, báo cáo song song:

| Loại | Định nghĩa | Ghi chú |
|---|---|---|
| Tuyệt đối (COCO) | small: diện tích < 32², medium: 32²–96², large: > 96² (pixel, theo tọa độ ảnh gốc) | Dễ so sánh với tài liệu; nhưng phụ thuộc độ phân giải ảnh gốc |
| Tương đối | Diện tích box / diện tích ảnh | So sánh công bằng giữa các bộ dữ liệu khác độ phân giải |

- Ngưỡng tương đối cụ thể (vd. <0.5%, 0.5–2%, >2%): ⏳ TBD – chốt **sau EDA** (Mục 2.4), chọn sao cho mỗi nhóm có đủ mẫu (gợi ý ≥ 50 box/nhóm ở tập test).
- Lưu ý NEU-DET (200×200): ngưỡng 32² ≈ 2.5% diện tích ảnh → ghi rõ trong báo cáo.

### 1.3. Câu hỏi nghiên cứu (RQ)
- **RQ1 – Chẩn đoán:** Các detector phổ biến (one-stage CNN, two-stage, Transformer) suy giảm hiệu năng thế nào khi kích thước lỗi giảm? Lỗi chủ yếu là **bỏ sót**, **định vị lệch** hay **nhầm lớp**?
- **RQ2 – Cải tiến:** Các can thiệp có căn cứ (độ phân giải đặc trưng – P2/SPD-Conv; attention ở backbone; loss dành cho box nhỏ – NWD/biến thể IoU) cải thiện **nhóm lỗi nhỏ** bao nhiêu, với chi phí tính toán bao nhiêu?
- **RQ3 – Chuyển giao:** Mô hình chung đã cải tiến có chuyển giao tốt sang dữ liệu thực tế ít mẫu hơn so với khởi tạo từ COCO không, ở các mức dữ liệu 10–100%?

### 1.4. Giả thuyết (kiểm chứng bằng thực nghiệm)
- **H1:** Recall và AP giảm mạnh ở nhóm small, rõ nhất ở bộ dữ liệu ảnh gốc lớn bị resize (GC10-DET, PKU-PCB).
- **H2:** Thêm nhánh P2 cải thiện AP_small nhưng tăng FLOPs đáng kể; hiệu quả phụ thuộc bộ dữ liệu.
- **H3:** Với box nhỏ, IoU nhạy với sai lệch vài pixel → loss dựa trên NWD/biến thể IoU cải thiện AP75_small nhiều hơn AP50.
- **H4:** Transfer từ mô hình chung cho lợi thế lớn nhất khi dữ liệu đích ít (10–25%).

> Kết quả có thể bác bỏ giả thuyết – vẫn là phát hiện có giá trị, ghi nhận trung thực.

---

## 2. Dữ liệu

### 2.1. Vai trò các bộ dữ liệu

| Dataset | Miền | Quy mô / độ phân giải | Nhãn gốc | Vai trò | Trạng thái |
|---|---|---|---|---|---|
| NEU-DET | Thép cán nóng | 1800 ảnh, 200×200, 6 lớp | bbox VOC | Nguồn (chính) | ✅ |
| GC10-DET | Tấm thép | ~2300 ảnh có nhãn, 2048×1000, 10 lớp | bbox VOC | Nguồn | ✅ |
| PKU-Market-PCB | PCB | ~690 ảnh, độ phân giải rất cao, 6 lớp | bbox VOC | Nguồn (lỗi cực nhỏ) | ⏳ TBD – xác nhận tải được & chất lượng nhãn |
| Magnetic Tile | Ngói từ | 1344 ảnh (392 lỗi), 5 lớp | mask → bbox | Nguồn (thay thế cho PKU/KSDD2 nếu cần) | ⏳ TBD |
| **KolektorSDD2** | Linh kiện điện | 3335 ảnh (356 lỗi), ~230×630 | mask → bbox | **Đích (Cuối kỳ)** – giữ riêng, KHÔNG dùng để tinh chỉnh ở GĐ1–2 | ✅ đề xuất |
| Dữ liệu tự thu thập | ❓ | ❓ | ❓ | Đích thay thế / bổ sung | ❓ HỎI – có nguồn doanh nghiệp/xưởng không? |

**Quyết định mở:**
- ❓ HỎI giảng viên: báo cáo Thường kỳ cần đúng 4 dataset? Nếu có → dùng NEU-DET, GC10-DET, PKU-PCB, Magnetic Tile (giữ KSDD2 cho cuối kỳ).
- ⏳ TBD: nếu có dữ liệu thật tự thu thập → KSDD2 chuyển thành dữ liệu nguồn.

### 2.2. Tải dữ liệu
- [ ] NEU-DET – mirror Kaggle / trang NEU
- [ ] GC10-DET – Kaggle (`alex000kim/gc10det`)
- [ ] PKU-Market-PCB – mirror Kaggle
- [ ] Magnetic Tile – GitHub của tác giả (Huang et al.)
- [ ] KolektorSDD2 – vicos.si/resources/kolektorsdd2
- [ ] Ghi lại **nguồn, phiên bản, license, ngày tải** vào `data/README.md`

### 2.3. Chuyển đổi & chia dữ liệu
**Đầu ra cho mỗi dataset:**
1. Định dạng YOLO (`images/`, `labels/`, `data.yaml`) – dùng cho Ultralytics.
2. File COCO JSON cho tập **test** (tọa độ ảnh gốc) – dùng cho pycocotools và torchvision.

**Quy tắc:**
- Chia train/val/test **cố định**, seed chia = `42`, lưu danh sách tên file vào `splits/<dataset>/{train,val,test}.txt`. Không bao giờ chia lại.
- NEU-DET: 1440/180/180 (giống DAC-YOLO, để đối chiếu).
- GC10-DET: 80/10/10 ⏳ TBD – kiểm tra có split chuẩn trong tài liệu không.
- KSDD2: giữ train/test gốc; tách val từ train (stratified theo có/không lỗi).
- Chia **stratified theo lớp** nếu có thể.
- Augmentation chỉ áp dụng **sau khi chia** (tránh rò rỉ).
- `image_id` trong COCO JSON đặt theo **tên file (stem)** để khớp với `predictions.json` của Ultralytics.
- Mask → bbox (KSDD2, Magnetic Tile): tách connected components; bỏ vùng < `min_area` pixel (⏳ TBD, đề xuất 4–9 px) – ghi lại số box bị loại. Ảnh không lỗi → file nhãn rỗng (giữ làm ảnh nền).

### 2.4. Phân tích dữ liệu (EDA) – bắt buộc cho báo cáo Thường kỳ
- [ ] Số ảnh, số box, số lớp, số box/ảnh
- [ ] Phân bố lớp (kiểm tra mất cân bằng)
- [ ] **Histogram diện tích box** (tuyệt đối & tương đối, trục log)
- [ ] Tỉ lệ small/medium/large theo từng lớp
- [ ] **GC10-DET, PKU-PCB:** kích thước box **trước và sau resize về 640** → minh chứng lỗi bị co nhỏ
- [ ] Tỉ lệ khung hình (aspect ratio) của box – lỗi dạng vết dài (scratch, weld line)
- [ ] Ảnh minh họa lỗi nhỏ nhất mỗi lớp
- ⏳ TBD sau EDA: chốt ngưỡng nhóm kích thước tương đối (Mục 1.2)

---

## 3. Protocol thực nghiệm chung (áp dụng cho MỌI lần chạy)

### 3.1. Cấu hình huấn luyện cố định

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Kích thước ảnh | 640 | Giống nhau cho mọi mô hình (torchvision: `min_size=max_size=640`) |
| Khởi tạo | Pretrained COCO | Transfer learning chuẩn |
| Epoch (YOLO) | 100 | ⏳ TBD – giảm nếu thiếu quota |
| Epoch (RT-DETR) | 50–72 | ⏳ TBD |
| Epoch (Faster R-CNN) | ~24 | ⏳ TBD |
| Batch | 16 (YOLO) / ⏳ TBD cho mô hình nặng theo VRAM | Ghi lại batch thực tế |
| Optimizer / LR | Mặc định của từng framework | Không tinh chỉnh riêng cho mô hình nào (công bằng) |
| Augmentation | Mặc định Ultralytics (mosaic, HSV, flip…) | Ghi đầy đủ tham số vào slide |
| Seed | GĐ1: 1 seed (`0`); GĐ2 trở đi: ≥ 3 seed (`0,1,2`); kết quả chốt/paper: 5 seed | |
| Early stopping | Tắt (hoặc patience lớn) | Tránh so sánh không đồng đều |

### 3.2. Chỉ số đánh giá
**Độ chính xác (trên tập test, 1 lần duy nhất sau khi chọn checkpoint bằng val):**
- COCO: AP (0.5:0.95), AP50, AP75, **AP_small, AP_medium, AP_large**, AR tương ứng
- Precision, Recall (tại conf mặc định 0.25) ⏳ TBD – chốt ngưỡng conf dùng chung
- **Recall theo nhóm kích thước tương đối** (Phụ lục A.3) – chỉ số chính cho câu hỏi nghiên cứu
- Per-class AP (bảng phụ)

**Chẩn đoán loại lỗi:**
- Confusion matrix (Ultralytics tự sinh)
- **TIDE** (`pip install tidecv`): tách lỗi thành Cls / Loc / Both / Dupe / Bkg / Missed → dùng để chọn hướng cải tiến (Mục 5.1)

**Hiệu năng tính toán:**
- Params (M), GFLOPs @640
- FPS trên T4, batch 1, sau warmup ≥ 50 ảnh, ghi rõ FP16/FP32

**Thống kê (từ GĐ2):**
- Báo cáo **mean ± std** qua các seed
- Welch's t-test giữa baseline và biến thể; chỉ gọi là "cải thiện" khi p < 0.05 **và** chênh lệch > std

### 3.3. Ghi log & tái lập
- Tên run: `<phase>_<dataset>_<model>_<variant>_s<seed>` – ví dụ `p1_neu_yolo11n_base_s0`
- Mỗi run lưu: config (yaml), `best.pt`, `results.csv`, `predictions.json`, COCO eval, thời gian train
- Tổng hợp tất cả vào `results/master_results.csv` (1 dòng / run)
- Tùy chọn: MLflow file store trong `/kaggle/working/mlruns` → tải về sau mỗi phiên
- Ghi phiên bản: `ultralytics`, `torch`, `torchvision`, CUDA
- Kaggle: chạy bằng **Save Version → Save & Run All**; lưu output sau mỗi run (không đợi hết notebook)

---

## 4. Giai đoạn 1 – Thường kỳ: Baseline + cải tiến nhỏ

### 4.1. Mục tiêu
1. Trả lời sơ bộ **RQ1** (chẩn đoán).
2. Chọn **mô hình chung** cho cả học phần.
3. Có **1 cải tiến nhỏ** đúng yêu cầu.

### 4.2. Mô hình

| # | Mô hình | Họ | Vai trò | Framework |
|---|---|---|---|---|
| M1 | YOLO11n | One-stage CNN | Baseline / ứng viên mô hình chung | Ultralytics |
| M2 | Faster R-CNN R50-FPN (v2) | Two-stage, có P2 trong FPN | Baseline đối chứng | torchvision |
| M3 | RT-DETR-l | Transformer | Baseline đối chứng | Ultralytics |
| M4 | **YOLO11n-P2** | M1 + nhánh detect stride 4 | **Cải tiến nhỏ** | Ultralytics (yaml tùy chỉnh) |

- ❓ HỎI giảng viên: cải tiến ở **neck/head (P2)** có được chấp nhận cho Thường kỳ không, hay bắt buộc backbone/loss?
  - Nếu bắt buộc backbone → thay M4 bằng **YOLO11n + SimAM** (không tham số, chèn sau các khối C3k2 ở backbone).
- Trích dẫn nguồn gốc P2: FPN (Lin 2017), TPH-YOLOv5 (Zhu 2021) + 1–2 bài cùng miền (Mục 12). **Không** trình bày P2 là đóng góp mới.
- Mô hình chung dự kiến: **YOLO11n** ⏳ TBD – xác nhận sau khi có kết quả GĐ1.

### 4.3. Ngân sách GPU (ước lượng – cập nhật sau lần chạy thật)

| Mô hình | NEU | GC10 | PKU/MT | Dataset 4 | Tổng |
|---|---|---|---|---|---|
| YOLO11n | ~0.7h | ~1.2h | ~0.4h | ~1h | ~3.3h |
| YOLO11n-P2 | ~0.9h | ~1.5h | ~0.5h | ~1.3h | ~4.2h |
| Faster R-CNN | ~1h | ~1.5h | ~0.5h | ~1.2h | ~4.2h |
| RT-DETR-l | ~2h | ~3.5h | ~1h | ~3h | ~9.5h |
| **Tổng** | | | | | **~21h** |

Thứ tự ưu tiên chạy: YOLO11n → YOLO11n-P2 → Faster R-CNN → RT-DETR (chạy sau cùng, cắt epoch nếu thiếu quota).

### 4.4. Lịch 3 ngày

**Ngày 1 – Dữ liệu + bắt đầu train**
- [ ] Script convert cả 4 bộ → YOLO + COCO JSON test (Phụ lục A.1, A.2)
- [ ] Kiểm tra trực quan: vẽ box lên ~20 ảnh mỗi bộ (bắt lỗi convert)
- [ ] Chạy nền: YOLO11n + YOLO11n-P2 trên NEU-DET và PKU/MT
- [ ] EDA (Mục 2.4)

**Ngày 2 – Chạy hết baseline**
- [ ] YOLO11n, YOLO11n-P2 trên GC10 và dataset 4
- [ ] Faster R-CNN (pipeline torchvision) trên 4 bộ
- [ ] Bắt đầu RT-DETR

**Ngày 3 – Đánh giá & slide**
- [ ] Hoàn tất RT-DETR (hoặc ghi rõ số epoch bị cắt)
- [ ] COCO eval, recall theo nhóm kích thước, TIDE
- [ ] Ảnh minh họa: lỗi nhỏ bị bỏ sót / định vị lệch / nhận nhầm
- [ ] Làm slide

### 4.5. Bảng kết quả (điền sau thực nghiệm)

| Dataset | Model | AP | AP50 | AP75 | AP_s | AP_m | AP_l | Recall_small (tương đối) | Params | GFLOPs | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEU-DET | YOLO11n | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | | | |
| NEU-DET | YOLO11n-P2 | ⏳ | | | | | | | | | |
| … | … | | | | | | | | | | |

### 4.6. Cấu trúc slide Thường kỳ
1. Bài toán, câu hỏi nghiên cứu, định nghĩa "lỗi nhỏ"
2. Datasets (bảng số ảnh / lớp / split)
3. Data Analysis (phân bố kích thước; GC10/PKU trước–sau resize)
4. Augmentation (tham số)
5. Mô hình + cải tiến nhỏ (sơ đồ P2 / SimAM, có trích dẫn)
6. Kết quả (bảng 4.5 + biểu đồ AP_small theo mô hình)
7. Phân tích lỗi (TIDE + ảnh minh họa)
8. Kế hoạch Giữa kỳ (dựa trên chẩn đoán – Mục 5.1)

### 4.7. Kết luận GĐ1 (điền sau)
- Mô hình chung: ⏳ TBD
- Loại lỗi chiếm ưu thế theo TIDE: ⏳ TBD
- P2 có giúp AP_small không? Ở dataset nào? ⏳ TBD
- → Hướng cải tiến GĐ2: ⏳ TBD (theo bảng quyết định 5.1)

---

## 5. Giai đoạn 2 – Giữa kỳ: Cải tiến Backbone / Loss + bảng so sánh

> ⏳ **Toàn bộ lựa chọn cụ thể ở mục này chốt SAU khi có kết quả GĐ1.** Dưới đây là khung + ứng viên.

### 5.1. Bảng quyết định: chẩn đoán → hướng cải tiến

| Nếu kết quả GĐ1 cho thấy… | Nguyên nhân khả dĩ | Hướng cải tiến ưu tiên |
|---|---|---|
| Recall_small thấp, TIDE: **Missed** cao | Đặc trưng độ phân giải thấp, mất thông tin khi downsample | P2 (giữ/nhẹ hóa), **SPD-Conv** ở backbone, tăng imgsz / SAHI khi suy luận |
| AP50_small ổn nhưng **AP75_small thấp**, TIDE: **Loc** cao | IoU quá nhạy với sai lệch vài pixel ở box nhỏ | **Loss:** NWD (kết hợp CIoU), Wise-IoU, Inner-IoU |
| Nhầm lớp giữa các lỗi giống nhau, TIDE: **Cls** cao | Backbone chưa đủ phân biệt | **Attention ở backbone:** SimAM, CBAM, Coordinate Attention |
| Nhiều FP trên nền (vd. ảnh không lỗi), TIDE: **Bkg** cao | Nhiễu kết cấu nền | Attention lọc nền, hard negative, cân bằng lại ảnh nền |
| P2 giúp nhưng FLOPs tăng quá nhiều | Chi phí nhánh stride 4 | Biến thể **P2–P4 (bỏ P5)** hoặc P2 dùng depthwise conv |

### 5.2. Ứng viên cải tiến (có trích dẫn)

| Nhóm | Kỹ thuật | Ý tưởng | Độ khó cài đặt (Ultralytics) |
|---|---|---|---|
| Backbone | SimAM | Attention 3D không tham số | Thấp–vừa (đăng ký module) |
| Backbone | CBAM / Coordinate Attention | Attention kênh + không gian | Vừa |
| Backbone | SPD-Conv | Space-to-depth thay strided conv | Vừa |
| Loss | NWD | Box ≈ Gaussian, dùng Wasserstein distance | Vừa (sửa loss + assigner) |
| Loss | Wise-IoU / Inner-IoU | Biến thể IoU | Thấp–vừa (sửa hàm IoU) |
| Neck/Head | P2, P2–P4 | Nhánh độ phân giải cao | Thấp (yaml) |

**Ghi chú cài đặt (kiểm tra lại theo phiên bản `ultralytics` đang dùng):**
- Module mới: định nghĩa trong `ultralytics/nn/modules/`, đăng ký trong `parse_model` (`ultralytics/nn/tasks.py`), rồi gọi trong yaml.
- Loss box: `BboxLoss` trong `ultralytics/utils/loss.py`; hàm IoU trong `ultralytics/utils/metrics.py`. Nếu đổi độ đo cho cả **assigner** (TaskAlignedAssigner) → ghi rõ là đổi cả assigner hay chỉ loss.
- Cài Ultralytics ở chế độ editable (`pip install -e .`) từ bản fork của nhóm để quản lý thay đổi bằng git.

### 5.3. Phân công (đáp ứng yêu cầu "đóng góp từng thành viên")

| Thành viên | Phụ trách | Biến thể | Trạng thái |
|---|---|---|---|
| ❓ | Backbone | YOLO11n + ⏳ (SimAM / SPD-Conv / CA) | |
| ❓ | Loss | YOLO11n + ⏳ (NWD / Wise-IoU) | |
| ❓ | Neck/Head | YOLO11n-P2 / P2–P4 | |
| ❓ | Dữ liệu, đánh giá, thống kê, tổng hợp | Kết hợp tốt nhất | |

❓ HỎI: nhóm có bao nhiêu người? Nếu ít hơn 4 → gộp vai trò.

### 5.4. Thiết kế ablation
Chạy trên ⏳ TBD bộ dữ liệu (gợi ý: 2 bộ nguồn đại diện – NEU-DET + GC10-DET), **3 seed**:

| ID | Baseline | +Neck (P2) | +Backbone | +Loss | Ghi chú |
|---|---|---|---|---|---|
| A0 | ✓ | | | | YOLO11n |
| A1 | ✓ | ✓ | | | |
| A2 | ✓ | | ✓ | | |
| A3 | ✓ | | | ✓ | |
| A4 | ✓ | ✓ | ✓ | | |
| A5 | ✓ | ✓ | | ✓ | |
| A6 | ✓ | | ✓ | ✓ | |
| A7 | ✓ | ✓ | ✓ | ✓ | Mô hình đầy đủ |

- Ước lượng: 8 cấu hình × 2 dataset × 3 seed × ~1h ≈ **48 giờ GPU** → vượt 1 tuần quota. ⏳ TBD phương án: (a) chia qua 2 tuần; (b) chỉ chạy A0, A1–A3, A7 đủ 3 seed, còn lại 1 seed; (c) dùng thêm Colab.
- Báo cáo mean ± std, t-test A_i vs A0 (Mục 3.2).

### 5.5. Bảng kết quả Giữa kỳ (điền sau)

| ID | AP | AP_s | AP75_s | Recall_small | Params | GFLOPs | FPS | p-value vs A0 (AP_s) |
|---|---|---|---|---|---|---|---|---|
| A0 | ⏳ | | | | | | | – |
| … | | | | | | | | |

### 5.6. Kết luận GĐ2 (điền sau)
- Cấu hình mô hình chung cuối cùng: ⏳ TBD
- Đóng góp của từng thành phần: ⏳ TBD

---

## 6. Giai đoạn 3 – Cuối kỳ: Transfer Learning sang dữ liệu thực tế

> ⏳ Phụ thuộc: mô hình chung (GĐ2) + quyết định dữ liệu đích (Mục 2.1).

### 6.1. Dữ liệu đích
- Mặc định: **KolektorSDD2** (dữ liệu thật từ doanh nghiệp, 246 ảnh lỗi train, mất cân bằng nặng).
- ❓ HỎI: có thể tự thu thập dữ liệu thật (xưởng / doanh nghiệp quen / tự chụp sản phẩm lỗi) không? Nếu có:
  - Quy mô tối thiểu gợi ý: vài trăm ảnh, ≥ 50–100 mẫu lỗi
  - Công cụ gán nhãn: ⏳ TBD (CVAT / Label Studio / Roboflow)
  - Quy ước gán nhãn: ⏳ TBD – viết hướng dẫn gán nhãn ngắn để nhất quán
  - Quyền sử dụng dữ liệu (đặc biệt nếu định publish): ❓ HỎI doanh nghiệp

### 6.2. Thiết kế thí nghiệm

**Chiến lược khởi tạo:**
| ID | Khởi tạo | Ý nghĩa |
|---|---|---|
| T1 | COCO pretrained (YOLO11n gốc) | Mốc so sánh |
| T2 | Mô hình chung **chưa cải tiến** (YOLO11n train trên dữ liệu nguồn) | Tác dụng của dữ liệu nguồn |
| T3 | Mô hình chung **đã cải tiến** (A7 hoặc cấu hình chốt) | Tác dụng của cải tiến + dữ liệu nguồn |

- Dữ liệu nguồn dùng để tạo T2/T3: ⏳ TBD – train gộp các bộ nguồn hay chỉ bộ gần miền nhất?
- Xử lý khác biệt số lớp: thay head phân loại (Ultralytics tự làm khi số lớp đổi), giữ backbone + neck.

**Lượng dữ liệu đích:** 10%, 25%, 50%, 100% tập train (lấy mẫu stratified, cố định seed).

**Chiến lược fine-tune:** ⏳ TBD thử (a) fine-tune toàn bộ; (b) đóng băng backbone (`freeze=<số layer backbone>`) vài epoch đầu rồi mở.

**Chỉ số bổ sung cho bài toán thực tế:**
- Tỉ lệ phát hiện ở mức ảnh (ảnh lỗi có ≥ 1 box đúng)
- Tỉ lệ báo nhầm trên ảnh không lỗi (false alarm rate)
- Đường cong AP_small theo % dữ liệu cho T1/T2/T3

### 6.3. Kết quả & kết luận (điền sau)
- ⏳ TBD

---

## 7. Nhánh Publish (song song, không bắt buộc)

**Định vị bài:** không phải "X-YOLO thêm module", mà là *cải tiến có căn cứ cho lỗi nhỏ, đánh giá phân tầng theo kích thước, nhiều seed, và chuyển giao sang bài toán công nghiệp ít dữ liệu.*

- [ ] Bảng tổng quan tài liệu 30–50 bài gần nhất: phương pháp, dataset, có nhiều seed không, có đánh giá theo kích thước không, có tách riêng tác động từng module không → chứng minh khoảng trống
- [ ] Mở rộng lên 5 seed cho kết quả chính
- [ ] Công bố code + file split + script đánh giá
- Venue: ❓ HỎI / ⏳ TBD (SOICT, KSE, RIVF, ACIIDS; tạp chí: Pattern Recognition Letters, IEEE Access, J. Imaging…) – kiểm tra deadline
- ❓ HỎI giảng viên: có đồng ý hướng dẫn / đồng tác giả không?

---

## 8. Rủi ro & phương án dự phòng

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Hết quota GPU | Còn < 5h trước deadline | Cắt epoch RT-DETR; bỏ bớt dataset cho mô hình nặng; dùng Colab |
| Phiên Kaggle bị ngắt | Mất output | Lưu output sau mỗi run; chia nhỏ notebook |
| Lỗi convert nhãn | mAP bất thường thấp / box lệch khi vẽ | Luôn vẽ kiểm tra trước khi train |
| `image_id` COCO không khớp | COCO eval ra 0 hoặc -1 | Đặt id theo stem tên file; kiểm tra số ảnh khớp |
| Cải tiến không giúp | Chênh lệch < std | Báo cáo trung thực; phân tích nguyên nhân – vẫn là kết quả |
| Dataset tải lỗi / nhãn kém | | Dùng Magnetic Tile thay thế |
| Không có dữ liệu thật | | Dùng KSDD2 làm đích |

---

## 9. Danh sách câu hỏi mở (❓ HỎI)

**Hỏi giảng viên**
1. Ngày nộp và hình thức (slide / report / code) của từng mốc?
2. Cải tiến nhỏ ở Thường kỳ có thể ở neck/head (P2) không, hay phải backbone/loss?
3. Bắt buộc số lượng mô hình / dataset tối thiểu?
4. Có yêu cầu dùng mô hình cụ thể (DETR + DINO/DINOv2)?
5. Dữ liệu "thực tế" ở Cuối kỳ có chấp nhận KSDD2 không, hay phải tự thu thập?
6. Có hỗ trợ hướng dẫn nếu nhóm muốn viết paper?

**Hỏi nhóm**
7. Số thành viên, phân công (Mục 5.3)?
8. Tổng quota GPU nhóm có thể dùng mỗi tuần (mỗi người dùng tài khoản riêng của mình cho phần việc mình phụ trách)?

**Hỏi bên ngoài**
9. Có nguồn dữ liệu lỗi thật từ doanh nghiệp / xưởng không? Điều kiện sử dụng?

---

## 10. Cấu trúc repo & quy ước

```
small-defect-detection/
├── README.md
├── KE_HOACH_NGHIEN_CUU.md          # tài liệu này
├── data/
│   └── README.md                   # nguồn, license, ngày tải
├── splits/<dataset>/{train,val,test}.txt
├── configs/
│   ├── data/<dataset>.yaml
│   └── models/yolo11n-p2.yaml, ...
├── scripts/
│   ├── convert_voc.py
│   ├── convert_mask.py
│   ├── eda.py
│   ├── eval_coco.py
│   └── recall_by_size.py
├── ultralytics/                    # fork, pip install -e
├── notebooks/                      # notebook Kaggle
└── results/
    ├── master_results.csv
    └── <run_name>/
```

---

## 11. Nhật ký quyết định & nhật ký thực nghiệm

### 11.1. Nhật ký quyết định
| Ngày | Quyết định | Lý do / bằng chứng | Người quyết |
|---|---|---|---|
| | Chọn YOLO11n làm ứng viên mô hình chung | Nhẹ, nhanh trên T4, dễ sửa, đối chiếu được DAC-YOLO | |
| | Giữ KSDD2 làm dữ liệu đích, không dùng ở GĐ1–2 | Đảm bảo transfer learning là dữ liệu mới | |
| | | | |

### 11.2. Nhật ký thực nghiệm
| Ngày | Run name | Mục đích | Kết quả chính | Ghi chú / sự cố |
|---|---|---|---|---|
| | | | | |

---

## 12. Tài liệu tham khảo (bổ sung dần)

**Dữ liệu**
- Song & Yan (2013) – NEU surface defect database (NEU-DET)
- Lv et al. (2020) – GC10-DET
- Huang, Qiu & Yuan (2020) – Surface defect saliency of magnetic tile, *The Visual Computer*
- Božič, Tabernik & Skočaj (2021) – Mixed supervision for surface-defect detection (KolektorSDD2), *Computers in Industry*
- PKU-Market-PCB – ⏳ bổ sung trích dẫn chuẩn

**Mô hình**
- Jocher & Qiu (2024) – Ultralytics YOLO11
- Ren et al. (2017) – Faster R-CNN
- Zhao et al. (2024) – RT-DETR (DETRs Beat YOLOs on Real-time Object Detection)
- Lin et al. (2017) – Feature Pyramid Networks

**Vật thể nhỏ / cải tiến**
- Zhu et al. (2021) – TPH-YOLOv5 (thêm head dự đoán cho vật thể nhỏ)
- Wang et al. (2021) – NWD: Normalized Gaussian Wasserstein Distance for Tiny Object Detection
- Sunkara & Luo (2022) – SPD-Conv
- Yang et al. (2021) – SimAM
- Woo et al. (2018) – CBAM; Hou et al. (2021) – Coordinate Attention
- Tong et al. (2023) – Wise-IoU
- Bolya et al. (2020) – TIDE: A General Toolbox for Identifying Object Detection Errors
- Akyon et al. (2022) – SAHI (Slicing Aided Hyper Inference)
- Li et al. (2027) – DAC-YOLO, *Pattern Recognition* 182 (bài tham chiếu cùng chủ đề)
- ⏳ Bổ sung các bài YOLO11/YOLOv8 + P2 cho lỗi bề mặt (đường ray, PCB)

---

## Phụ lục A – Code snippet

### A.1. VOC XML → YOLO + COCO JSON
```python
import xml.etree.ElementTree as ET, json
from pathlib import Path

def voc_to_yolo_and_coco(xml_dir, img_dir, out_lbl_dir, class_names, coco_out=None, stems=None):
    cls2id = {c: i for i, c in enumerate(class_names)}
    out_lbl_dir = Path(out_lbl_dir); out_lbl_dir.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": c} for i, c in enumerate(class_names)]}
    ann_id = 0
    for xml in sorted(Path(xml_dir).glob("*.xml")):
        if stems is not None and xml.stem not in stems:
            continue
        r = ET.parse(xml).getroot()
        W = int(r.find("size/width").text); H = int(r.find("size/height").text)
        img_id = xml.stem                      # id = stem tên file -> khớp predictions.json
        coco["images"].append({"id": img_id, "file_name": f"{xml.stem}.jpg", "width": W, "height": H})
        lines = []
        for o in r.findall("object"):
            c = cls2id[o.find("name").text.strip()]
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": c,
                                        "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
        (out_lbl_dir / f"{xml.stem}.txt").write_text("\n".join(lines))
    if coco_out:
        Path(coco_out).write_text(json.dumps(coco))
```
> Lưu ý: kiểm tra đuôi ảnh (.jpg/.png/.bmp) và tên lớp chính xác của từng bộ. Nếu pycocotools yêu cầu `image_id` kiểu số, tạo bảng ánh xạ stem → số và áp dụng **giống nhau** cho cả GT lẫn prediction. Kiểm tra `category_id` trong `predictions.json` của Ultralytics có khớp với id lớp trong GT hay không.

### A.2. Mask → bbox (KSDD2, Magnetic Tile)
```python
import cv2, numpy as np

def mask_to_boxes(mask_path, min_area=4):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    m = (m > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes, dropped = [], 0
    for x, y, w, h, area in stats[1:]:           # bỏ nhãn 0 = nền
        if area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h)))
        else:
            dropped += 1
    return boxes, m.shape, dropped               # ghi lại số vùng bị loại
```
> KSDD2: kiểm tra quy ước tên file ảnh/mask trong bản tải về (ảnh và mask GT đi theo cặp). Mask Magnetic Tile có thể là ảnh saliency → nhị phân hóa với ngưỡng (⏳ TBD) và ghi lại.

### A.3. Recall theo nhóm kích thước tương đối
```python
import numpy as np

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)

def recall_by_size(gts, preds, img_areas, bins, iou_thr=0.5, conf_thr=0.25):
    """
    gts[img]   = list of (x1,y1,x2,y2,cls)           -- tọa độ ảnh gốc
    preds[img] = list of (x1,y1,x2,y2,cls,score)
    img_areas[img] = W*H ;  bins = [0, 0.005, 0.02, 1.0] (⏳ chốt sau EDA)
    """
    hit = np.zeros(len(bins) - 1); tot = np.zeros(len(bins) - 1)
    for img, g_list in gts.items():
        p_list = sorted([p for p in preds.get(img, []) if p[5] >= conf_thr], key=lambda p: -p[5])
        used = set()
        for g in g_list:
            rel = (g[2]-g[0]) * (g[3]-g[1]) / img_areas[img]
            k = min(np.searchsorted(bins, rel, side="right") - 1, len(bins) - 2)
            tot[k] += 1
            for j, p in enumerate(p_list):
                if j not in used and p[4] == g[4] and iou(p, g) >= iou_thr:
                    used.add(j); hit[k] += 1; break
    return {f"[{bins[i]},{bins[i+1]})": (hit[i] / tot[i] if tot[i] else float("nan"), int(tot[i]))
            for i in range(len(tot))}
```

### A.4. Train + COCO eval (Ultralytics)
```python
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

run = "p1_neu_yolo11n-p2_s0"
model = YOLO("configs/models/yolo11n-p2.yaml").load("yolo11n.pt")   # chuyển trọng số khớp được
model.train(data="configs/data/neu.yaml", imgsz=640, epochs=100, batch=16,
            seed=0, deterministic=True, name=run, project="results")

best = YOLO(f"results/{run}/weights/best.pt")
best.val(data="configs/data/neu.yaml", split="test", save_json=True, conf=0.001, name=f"{run}_test")

gt = COCO("data/coco/neu_test.json")
dt = gt.loadRes(f"results/{run}_test/predictions.json")
E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
```
> `yolov8-p2.yaml` có sẵn trong Ultralytics. Với YOLO11, nếu phiên bản đang dùng chưa có `yolo11-p2.yaml` thì copy `yolo11.yaml` và thêm nhánh upsample + Detect ở stride 4 (ghi rõ trong báo cáo là cấu hình tự viết).
