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
4. Giai đoạn 1 – Thường kỳ: Baseline + xử lý mất cân bằng (tăng cường) + cải tiến nhỏ
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

**Lưu ý về PKU-Market-PCB:** bộ "PCB Defects" trên Roboflow (dùng trong slide N1 và notebook DETR-SE) có cùng 6 lớp (missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper) → nhiều khả năng **cùng nguồn PKU-Market-PCB**, không tính là một bộ riêng. PKU-Market-PCB được tạo từ khoảng 10 bo mạch gốc có lỗi thêm nhân tạo → khi chia dữ liệu cần kiểm tra **rò rỉ theo bo mạch** (cùng một bo xuất hiện ở cả train và test). ⏳ TBD: nếu tên file cho biết bo gốc thì chia theo bo (group split).

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
- [ ] Phân bố lớp **theo cả số ảnh lẫn số box**, trên từng split (train/val/test) → điền bảng 2.5.2
- [ ] Tỉ lệ ảnh có lỗi / ảnh không lỗi (KSDD2, Magnetic Tile)
- [ ] **Histogram diện tích box** (tuyệt đối & tương đối, trục log)
- [ ] Tỉ lệ small/medium/large theo từng lớp
- [ ] **GC10-DET, PKU-PCB:** kích thước box **trước và sau resize về 640** → minh chứng lỗi bị co nhỏ
- [ ] Tỉ lệ khung hình (aspect ratio) của box – lỗi dạng vết dài (scratch, weld line)
- [ ] Ảnh minh họa lỗi nhỏ nhất mỗi lớp
- ⏳ TBD sau EDA: chốt ngưỡng nhóm kích thước tương đối (Mục 1.2)

### 2.5. Xử lý mất cân bằng dữ liệu

#### 2.5.1. Ba dạng mất cân bằng cần theo dõi
| Dạng | Mô tả | Bộ dữ liệu bị ảnh hưởng | Có cần xử lý riêng? |
|---|---|---|---|
| **Giữa các lớp lỗi** | Có lớp rất ít mẫu (vd. rolled pit, crease ở GC10; fray ở Magnetic Tile) | GC10-DET (nặng), Magnetic Tile (vừa), NEU-DET (nhẹ – cân bằng theo ảnh, nhưng số box/lớp không đều) | Có (Mục 2.5.3) |
| **Ảnh lỗi vs ảnh không lỗi** | Ảnh không lỗi chiếm đa số | KSDD2 (~1:8), Magnetic Tile (392:952) | Có (Mục 2.5.3, 6.2) |
| **Theo kích thước lỗi** | Lỗi nhỏ nhiều/ít tùy bộ | Tất cả | Đây là **trọng tâm đề tài** – đo trong EDA, xử lý bằng cải tiến kiến trúc/loss (Mục 5) |
| Vật thể vs nền (trong ảnh) | Đặc trưng có sẵn của detection | Tất cả | **Không** – detector hiện đại đã xử lý (TAL assigner, focal/BCE, Hungarian matching) |

#### 2.5.2. Bảng thống kê mất cân bằng (✅ đã điền sau EDA – chi tiết: [KET_QUA_GD1.md](KET_QUA_GD1.md) Mục 2.5)
| Dataset | Lớp nhiều nhất (số box) | Lớp ít nhất (số box) | Tỉ lệ max/min | Ảnh lỗi : ảnh không lỗi | Số box test của lớp ít nhất | Mức độ |
|---|---|---|---|---|---|---|
| NEU-DET | inclusion (788) | pitted_surface (348) | 2.26 | chỉ có ảnh lỗi | 44 | nhẹ |
| GC10-DET | silk_spot (704) | crease (61) | 11.54 | ~1 : 0 (2 ảnh sạch) | 8 | nặng – hiếm: crescent_gap, rolled_pit, crease, waist_folding |
| PKU-PCB | spurious_copper (405) | spur (386) | 1.05 | chỉ có ảnh lỗi | 52 | nhẹ |
| Magnetic Tile | break (96) | fray (30) | 3.20 | 1 : 2.47 | 4 | vừa – hiếm: fray |
| KSDD2 | 1 lớp | – | – | 1 : 8.4 (356 / 2979) | – | (lỗi / sạch – GĐ3) |

Quy ước mức độ (đề xuất): max/min < 3 → nhẹ; 3–10 → vừa; > 10 → nặng. Lớp có **< 10 box ở tập test** → đánh dấu "AP không ổn định", không rút kết luận riêng cho lớp đó.

#### 2.5.3. Phương pháp xử lý

**(a) Cấp chia dữ liệu – BẮT BUỘC, áp dụng ngay từ GĐ1**
- Chia **stratified theo lớp** (mỗi ảnh gán theo lớp hiếm nhất có trong ảnh) để lớp hiếm có mặt ở cả train/val/test.
- **Chỉ can thiệp vào tập train. Tuyệt đối KHÔNG cân bằng lại val/test** – hai tập này phải giữ phân bố thật, nếu không kết quả bị thổi phồng.
- Ảnh không lỗi giữ trong val/test đúng tỉ lệ gốc (để đo báo nhầm).

**(b) Cấp dữ liệu train – ÁP DỤNG TỪ GĐ1** (quy trình chi tiết: Mục 2.5.5)
| Kỹ thuật | Cách làm | Áp dụng cho | Ghi chú |
|---|---|---|---|
| **Class-aware augmentation offline** (chính, GĐ1) | Ảnh chứa lớp hiếm được sinh thêm bản sao **đã biến đổi** (lật/xoay + độ sáng, tương phản, gamma, nhiễu nhẹ); số bản sao theo hệ số repeat factor r_c = min(cap, max(1, sqrt(t / f_c))), f_c = tỉ lệ ảnh train chứa lớp c | Bộ có mất cân bằng lớp (GC10, MT; NEU/PKU nếu EDA cho thấy max/min ≥ 3) | Code: Phụ lục A.6. Khác RFS thuần ở chỗ **không nhân bản y hệt** → giảm học thuộc |
| **Copy-paste lỗi hiếm (bbox)** (chính, GĐ1) | Cắt lỗi lớp hiếm kèm viền nền, dán sang ảnh train khác, không chồng box, hòa trộn biên mềm + khớp độ sáng nền; **không thu nhỏ** patch | GC10, MT, PKU (lỗi nhỏ, nền đồng nhất) | Code: Phụ lục A.7. Không dùng cho lớp có box chiếm gần hết ảnh (vd. crazing, rolled-in scale của NEU) |
| Repeat Factor Sampling thuần (Gupta et al. 2019, LVIS) | Nhân bản y hệt ảnh chứa lớp hiếm | Như trên | Phương án dự phòng nếu thiếu thời gian. Code: Phụ lục A.5 |
| Điều chỉnh tỉ lệ ảnh nền | Giữ 1:1, 1:3, toàn bộ ảnh không lỗi trong **train** | KSDD2, Magnetic Tile | Siêu tham số ở GĐ3 (Mục 6.2) |
| Copy-paste theo mask (Ghiasi et al. 2021) | Dán chính xác theo đường viền lỗi | KSDD2, Magnetic Tile (có mask) | GĐ3 / mở rộng. `copy_paste` của Ultralytics cần nhãn segment; bản bbox (A.7) dùng được ngay |
| Mosaic (mặc định YOLO) | Ghép 4 ảnh | Tất cả | Đã có sẵn trong baseline – không tính là cải tiến |

**(c) Cấp hàm loss – ứng viên cho GĐ2 (cùng nhóm "cải tiến Loss")**
| Kỹ thuật | Ý tưởng | Cài đặt |
|---|---|---|
| Class-weighted BCE | Trọng số lớp ∝ 1/tần suất (hoặc 1/sqrt) | Sửa loss phân loại trong `ultralytics/utils/loss.py` (⏳ kiểm tra phiên bản có sẵn tham số class weight chưa) |
| Focal loss (Lin et al. 2017) | Giảm trọng số mẫu dễ, tập trung mẫu khó | Có sẵn lớp `FocalLoss` trong Ultralytics – cần nối vào `v8DetectionLoss` |

**(d) Cấp đánh giá – BẮT BUỘC, áp dụng ngay từ GĐ1**
- Luôn báo **AP theo từng lớp kèm số box test** của lớp đó (bảng phụ).
- mAP là trung bình macro theo lớp → lớp hiếm nặng ngang lớp phổ biến, nhưng dao động mạnh giữa các seed → báo mean ± std từ GĐ2.
- Bộ có ảnh không lỗi (KSDD2, Magnetic Tile): thêm **chỉ số mức ảnh** – tỉ lệ phát hiện ảnh lỗi, tỉ lệ báo nhầm trên ảnh không lỗi. **Không dùng accuracy** (đoán "không lỗi" hết đã đạt ~89% trên KSDD2).

#### 2.5.4. Chính sách theo giai đoạn
| Giai đoạn | Làm gì với mất cân bằng |
|---|---|
| **GĐ1 – Thường kỳ** | (a) + (d) + **(b): class-aware augmentation + copy-paste** (Mục 2.5.5). Tất cả mô hình train trên **cùng một phiên bản dữ liệu đã cân bằng** (`<ds>_bal_v1`) → so sánh giữa mô hình vẫn công bằng. Thêm **run đối chứng** YOLO11n trên dữ liệu gốc để đo tác dụng của việc cân bằng. |
| **GĐ2 – Giữa kỳ** | Dữ liệu `_bal` thành mặc định **nếu GĐ1 cho thấy có lợi**. Nếu lớp hiếm vẫn thấp → thêm biến thể cấp loss (class-weighted / focal) vào ablation (Mục 5.4). ⏳ TBD sau GĐ1. |
| **GĐ3 – Cuối kỳ** | Tỉ lệ ảnh nền trong train là siêu tham số khi fine-tune KSDD2; báo cáo chỉ số mức ảnh. |

#### 2.5.5. Quy trình cân bằng bằng tăng cường dữ liệu (GĐ1)

**Nguyên tắc bắt buộc**
1. Làm **sau khi chia** train/val/test. Chỉ sinh dữ liệu cho **train**; nguồn patch copy-paste cũng chỉ lấy từ train. Val/test giữ nguyên.
2. **Sinh offline một lần** (seed = 42), lưu thành phiên bản dữ liệu cố định `<dataset>_bal_v1` (Kaggle Dataset). Mọi mô hình dùng chung phiên bản này.
3. Chạy bước sinh dữ liệu trong **notebook CPU** của Kaggle → không tốn quota GPU.
4. **Không thu nhỏ ảnh / patch, không random crop** → tránh làm lỗi nhỏ biến mất hoặc bị cắt.
5. Tổng số ảnh train tăng **tối đa +50%** (giữ ngân sách GPU và tránh lệch phân bố quá xa thực tế).
6. Mục tiêu: mỗi lớp hiếm đạt **≥ 1/3 số ảnh của lớp nhiều nhất** trong train (⏳ điều chỉnh sau khi xem bảng 2.5.2).

**Cấu hình theo bộ dữ liệu** (⏳ chốt sau EDA)
| Dataset | Có cân bằng? | Phép hình học (offline) | Copy-paste | Ghi chú |
|---|---|---|---|---|
| NEU-DET | Chỉ khi max/min (theo box) ≥ 3 | hflip, vflip, rot90, rot180 | Không (nhiều lỗi phủ gần hết ảnh) | Ảnh vuông, kết cấu không định hướng |
| GC10-DET | **Có** (mất cân bằng nặng) | hflip, vflip (⏳ kiểm tra trực quan trước khi thêm rot90 – một số lỗi có thể gắn với hướng cán) | **Có** cho lớp hiếm có box nhỏ/vừa | Bộ chính để đánh giá tác dụng |
| PKU-PCB | Chỉ khi max/min ≥ 3 (thường khá cân bằng) | hflip, vflip, rot90, rot180 | Tùy chọn (lỗi nhỏ, nền mạch lặp lại) | Dùng bản **gốc**, không dùng bản Roboflow đã tăng cường 3x |
| Magnetic Tile | **Có** (fray, crack ít) | hflip, vflip, rot180 | **Có** | Có thể nâng cấp lên copy-paste theo mask |

**Phép biến đổi quang học** (mọi bản sinh thêm): tương phản ×[0.8, 1.2], độ sáng ±20, gamma [0.8, 1.25] (50%), nhiễu Gauss σ 2–6 (30%), blur 3×3 (20%).
> Ultralytics vẫn áp augmentation online mặc định (mosaic, fliplr, HSV…) cho **mọi** run → phần offline nên ưu tiên các phép mà mặc định không có (vflip, rot90/180, gamma, nhiễu).

**Kiểm tra chất lượng (QA) – bắt buộc trước khi train**
- [ ] Vẽ box lên **30 ảnh sinh ra** mỗi bộ (15 augment + 15 copy-paste): box phải khớp lỗi, patch dán không lộ đường viền.
- [ ] Bảng **trước / sau** số ảnh và số box theo lớp (đưa vào slide).
- [ ] Ghi `t`, `cap`, số ảnh sinh thêm, danh sách phép biến đổi vào nhật ký thực nghiệm.
- [ ] Kiểm tra số ảnh val/test **không đổi**.

**Đánh giá tác dụng của việc cân bằng**
- Định nghĩa **lớp hiếm** trước khi train (từ train gốc): số box < 1/3 lớp nhiều nhất. ⏳ Ghi danh sách lớp hiếm từng bộ.
- Chỉ số: **AP_rare** (trung bình AP các lớp hiếm), **AP_common** (các lớp còn lại), mAP, Recall_small.
- Tiêu chí "có lợi": AP_rare tăng **và** AP_common không giảm quá ~1 điểm. Vì GĐ1 chỉ 1 seed → ghi rõ là kết quả sơ bộ; nếu còn quota, chạy thêm seed 1 cho cặp đối chứng trên GC10.
- **Công bằng về số bước huấn luyện:** dữ liệu `_bal` có nhiều ảnh hơn → nhiều iteration hơn mỗi epoch. Run đối chứng trên dữ liệu gốc đặt `epochs = 100 × |train_bal| / |train_gốc|` (khớp số iteration), để phần cải thiện không chỉ đến từ việc train lâu hơn.

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
| Augmentation online | Mặc định Ultralytics (mosaic, HSV, flip…) | Ghi đầy đủ tham số vào slide |
| Dữ liệu train | `<dataset>_bal_v1` (cân bằng offline – Mục 2.5.5), trừ run đối chứng | Cùng một phiên bản cho mọi mô hình |
| Seed | GĐ1: 1 seed (`0`); GĐ2 trở đi: ≥ 3 seed (`0,1,2`); kết quả chốt/paper: 5 seed | |
| Early stopping | Tắt (hoặc patience lớn) | Tránh so sánh không đồng đều |

### 3.2. Chỉ số đánh giá
**Độ chính xác (trên tập test, 1 lần duy nhất sau khi chọn checkpoint bằng val):**
- COCO: AP (0.5:0.95), AP50, AP75, **AP_small, AP_medium, AP_large**, AR tương ứng
- Precision, Recall (tại conf mặc định 0.25) ⏳ TBD – chốt ngưỡng conf dùng chung
- **Recall theo nhóm kích thước tương đối** (Phụ lục A.3) – chỉ số chính cho câu hỏi nghiên cứu
- Per-class AP **kèm số box test mỗi lớp** (bảng phụ – xem Mục 2.5.3d)
- Bộ có ảnh không lỗi: tỉ lệ phát hiện ảnh lỗi & tỉ lệ báo nhầm ảnh không lỗi
- ⚠️ Khi chạy COCOeval: **không lọc prediction theo ngưỡng score cao** (dùng conf ≈ 0.001), nếu không AP/AR bị hạ giả tạo

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

## 4. Giai đoạn 1 – Thường kỳ: Baseline + xử lý mất cân bằng (tăng cường) + cải tiến nhỏ

### 4.1. Mục tiêu
1. Trả lời sơ bộ **RQ1** (chẩn đoán).
2. Chọn **mô hình chung** cho cả học phần.
3. **Xử lý mất cân bằng bằng tăng cường dữ liệu** và đo tác dụng (Mục 2.5.5).
4. Có **1 cải tiến nhỏ** ở mô hình (P2 hoặc SimAM).

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

**Ma trận run GĐ1**
| Nhóm | Run | Dữ liệu | Bộ dữ liệu | Mục đích |
|---|---|---|---|---|
| Chính | M1–M4 | `_bal_v1` | Cả 4 bộ | So sánh mô hình (cùng dữ liệu) |
| Đối chứng cân bằng | YOLO11n | **gốc**, epoch khớp iteration | Bộ có cân bằng (GC10, MT; + NEU/PKU nếu có) | Đo tác dụng của cân bằng: so với M1 trên `_bal` |
| Tùy chọn (nếu dư quota) | YOLO11n | gốc + **chỉ class-aware aug** | GC10 | Tách đóng góp: augmentation vs copy-paste |

> Hai "cải tiến" của GĐ1 đo độc lập: **cân bằng dữ liệu** (YOLO11n gốc vs YOLO11n `_bal`) và **P2** (YOLO11n `_bal` vs YOLO11n-P2 `_bal`).

### 4.3. Ngân sách GPU (ước lượng – cập nhật sau lần chạy thật)

Giả định: GC10 `_bal` tăng ~+50% ảnh train, MT ~+40%, NEU/PKU tăng 0–20%.

| Run | NEU | GC10 | PKU | MT | Tổng |
|---|---|---|---|---|---|
| YOLO11n (`_bal`) | ~0.8h | ~1.8h | ~0.5h | ~0.7h | ~3.8h |
| YOLO11n-P2 (`_bal`) | ~1.0h | ~2.3h | ~0.7h | ~0.9h | ~4.9h |
| Faster R-CNN (`_bal`) | ~1.1h | ~2.3h | ~0.7h | ~0.9h | ~5.0h |
| RT-DETR-l (`_bal`, 50 epoch) | ~2h | ~5h | ~1.2h | ~1.8h | ~10h |
| Đối chứng YOLO11n (gốc, epoch khớp iteration) | (~0.8h nếu NEU có cân bằng) | ~1.8h | – | ~0.7h | ~2.5h |
| **Tổng lõi** | | | | | **~26h** |
| Tùy chọn: GC10 chỉ-augment | | ~1.6h | | | +1.6h |

- Sinh dữ liệu `_bal` chạy trên **CPU notebook** → 0 giờ GPU.
- ~26h sát quota ~30h/tuần của **một** tài khoản → **nên chia run theo tài khoản Kaggle của từng thành viên** (mỗi người chạy phần mình phụ trách). ❓ HỎI nhóm.
- Nếu vẫn thiếu: RT-DETR chỉ chạy 2 bộ (NEU + GC10) hoặc giảm còn 36–40 epoch, ghi rõ trong slide.

Thứ tự ưu tiên chạy: YOLO11n `_bal` → đối chứng YOLO11n gốc → YOLO11n-P2 → Faster R-CNN → RT-DETR (sau cùng).

### 4.4. Lịch 3 ngày

**Ngày 1 – Dữ liệu, cân bằng, bắt đầu train**
- [ ] Script convert cả 4 bộ → YOLO + COCO JSON test (Phụ lục A.1, A.2); chia stratified
- [ ] Kiểm tra trực quan: vẽ box lên ~20 ảnh mỗi bộ (bắt lỗi convert)
- [ ] EDA (Mục 2.4) → điền bảng 2.5.2, chốt bộ nào cần cân bằng + danh sách lớp hiếm
- [ ] **CPU notebook:** sinh `_bal_v1` (Phụ lục A.6, A.7) → QA 30 ảnh/bộ → upload thành Kaggle Dataset
- [ ] GPU (song song): bắt đầu YOLO11n trên bộ không cần cân bằng (NEU/PKU) trong lúc chờ `_bal`
- [ ] Chạy nền: YOLO11n `_bal` + đối chứng YOLO11n gốc trên GC10

**Ngày 2 – Chạy hết baseline**
- [ ] YOLO11n-P2 `_bal` trên 4 bộ; đối chứng MT
- [ ] Faster R-CNN (pipeline torchvision) trên 4 bộ `_bal`
- [ ] Bắt đầu RT-DETR

**Ngày 3 – Đánh giá & slide**
- [ ] Hoàn tất RT-DETR (hoặc ghi rõ số epoch bị cắt)
- [ ] COCO eval, recall theo nhóm kích thước, TIDE
- [ ] AP theo lớp → AP_rare / AP_common cho cặp đối chứng (gốc vs `_bal`)
- [ ] Ảnh minh họa: lỗi nhỏ bị bỏ sót / định vị lệch / nhận nhầm
- [ ] Làm slide

### 4.5. Bảng kết quả (điền sau thực nghiệm)

**Bảng A – So sánh mô hình (dữ liệu `_bal`)**

| Dataset | Model | AP | AP50 | AP75 | AP_s | AP_m | AP_l | Recall_small (tương đối) | Params | GFLOPs | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEU-DET | YOLO11n | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | | | |
| NEU-DET | YOLO11n-P2 | ⏳ | | | | | | | | | |
| … | … | | | | | | | | | | |

**Bảng B – Tác dụng của xử lý mất cân bằng (YOLO11n)**

| Dataset | Dữ liệu | Số ảnh train | mAP | AP_rare | AP_common | Recall_small | AP từng lớp hiếm |
|---|---|---|---|---|---|---|---|
| GC10-DET | Gốc (epoch khớp iteration) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| GC10-DET | `_bal` (aug + copy-paste) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| GC10-DET | Chỉ aug (tùy chọn) | ⏳ | | | | | |
| Magnetic Tile | Gốc / `_bal` | ⏳ | | | | | |

### 4.6. Cấu trúc slide Thường kỳ
1. Bài toán, câu hỏi nghiên cứu, định nghĩa "lỗi nhỏ"
2. Datasets (bảng số ảnh / lớp / split)
3. Data Analysis (phân bố kích thước; GC10/PKU trước–sau resize; **phân bố lớp – bảng 2.5.2**)
4. **Xử lý mất cân bằng:** biểu đồ số ảnh/box theo lớp trước–sau; ảnh minh họa bản augment và copy-paste; tham số `t`, `cap`
5. Augmentation online (tham số mặc định Ultralytics)
6. Mô hình + cải tiến nhỏ (sơ đồ P2 / SimAM, có trích dẫn)
7. Kết quả: Bảng A (mô hình) + Bảng B (tác dụng cân bằng) + biểu đồ AP_small theo mô hình
8. Phân tích lỗi (TIDE + ảnh minh họa)
9. Kế hoạch Giữa kỳ (dựa trên chẩn đoán – Mục 5.1)

### 4.7. Kết luận GĐ1 (✅ số liệu đầy đủ: [KET_QUA_GD1.md](KET_QUA_GD1.md), 1 seed)
- Mô hình chung: **YOLO11n** – kém mô hình tốt nhất 1–3 AP (GC10/NEU/MT), 7 AP trên PCB (RT-DETR-l 56.7 so với 49.7), nhưng 6.4 GFLOPs so với 105–451.
- Loại lỗi chiếm ưu thế theo TIDE: ⏳ đọc từ `results/figures/p1/tide_*.png`.
- P2 có giúp AP_small không? **Không**: AP gần như không đổi trên GC10/PCB, giảm 4.6 AP trên NEU; chỉ AP_rel_small GC10 +2.4. Trên PCB (lỗi nhỏ nhất) cũng không giúp → hướng tile / SAHI.
- Cân bằng bằng tăng cường: **không có lợi** trên GC10 (AP_rare 42.8 dữ liệu gốc → 39.8 chỉ tăng cường → 35.6 tăng cường + copy-paste; AP_common gần như không đổi); copy-paste kém hơn chỉ tăng cường. MT không kết luận được (fray 4 box test).
- → Hướng cải tiến GĐ2 (đề xuất, chờ xác nhận): dữ liệu gốc; backbone = Coordinate Attention (+2.6 AP GC10 trong screening); loss = class-weighted BCE (+2.3 / +1.1 AP; +8.7 AP_rel_small GC10), có thể thêm Wise-IoU; SimAM và NWD kém hơn mốc.

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
| **Sau khi đã cân bằng bằng tăng cường (GĐ1)**, AP lớp hiếm vẫn thấp hơn hẳn (và lớp đó có đủ mẫu test để tin được) | Tăng cường chưa đủ / lớp hiếm khó về bản chất | **Class-weighted / focal loss** (Mục 2.5.3c); tăng `t`/`cap`; copy-paste theo mask |

### 5.2. Ứng viên cải tiến (có trích dẫn)

| Nhóm | Kỹ thuật | Ý tưởng | Độ khó cài đặt (Ultralytics) |
|---|---|---|---|
| Backbone | SimAM | Attention 3D không tham số | Thấp–vừa (đăng ký module) |
| Backbone | CBAM / Coordinate Attention | Attention kênh + không gian | Vừa |
| Backbone | SPD-Conv | Space-to-depth thay strided conv | Vừa |
| Loss | NWD | Box ≈ Gaussian, dùng Wasserstein distance | Vừa (sửa loss + assigner) |
| Loss | Wise-IoU / Inner-IoU | Biến thể IoU | Thấp–vừa (sửa hàm IoU) |
| Backbone | SE (Squeeze-and-Excitation) | Channel attention (đã thử trong notebook DETR-SE – cần chạy kèm baseline) | Thấp–vừa |
| Loss | Class-weighted BCE / Focal loss | Xử lý mất cân bằng lớp | Thấp–vừa |
| Dữ liệu | Class-aware aug + copy-paste | **Đã làm ở GĐ1** – GĐ2 chỉ tinh chỉnh `t`, `cap` nếu cần | – |
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
- Dữ liệu cho mọi cấu hình A0–A7: `_bal_v1` nếu GĐ1 cho thấy có lợi, ngược lại dùng dữ liệu gốc. ⏳ TBD sau GĐ1.
- ⏳ TBD: nếu lớp hiếm vẫn thấp sau cân bằng → thêm **A8 = A0 + class-weighted/focal loss**, đánh giá bằng AP_rare + mean ± std trên GC10-DET.

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

**Tỉ lệ ảnh không lỗi trong train đích:** ⏳ TBD thử 1:1, 1:3 và toàn bộ (Mục 2.5.3b). Val/test giữ nguyên tỉ lệ gốc. Khi lấy 10–25% dữ liệu, lấy mẫu **stratified theo có/không lỗi** để không mất hết ảnh lỗi.

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
| Lớp hiếm có quá ít mẫu test | < 10 box test, AP nhảy mạnh giữa seed | Gắn nhãn "không ổn định"; không kết luận riêng cho lớp đó; cân nhắc gộp val+test hoặc cross-validation |
| Cân bằng lại nhầm cả val/test | Kết quả tốt bất thường | Chỉ sinh dữ liệu cho train; kiểm tra số ảnh val/test không đổi |
| Copy-paste tạo "dấu vết" giả (mô hình học đường viền dán thay vì lỗi) | QA thấy viền; AP_rare tăng trên train nhưng không tăng trên test | Tăng `margin` hòa trộn; chỉ dán lên nền cùng loại; so với run "chỉ aug" để tách tác dụng |
| Học thuộc lớp hiếm (quá ít mẫu gốc) | Train loss lớp hiếm rất thấp, test không tăng | Giới hạn `cap` ≤ 4; đa dạng phép biến đổi; không nhân bản y hệt |
| Cân bằng làm giảm lớp phổ biến | AP_common giảm > 1 điểm | Giảm `t`; giảm số ảnh copy-paste |
| Rò rỉ theo bo mạch (PKU-PCB) | Kết quả test gần bằng train | Chia theo bo gốc nếu xác định được |

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
│   ├── balance_aug.py              # A.6 + A.7
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
| 2026-09-27 | **GĐ1 có xử lý mất cân bằng** bằng class-aware augmentation + copy-paste (offline, chỉ train); mọi mô hình dùng chung `_bal_v1`; thêm run đối chứng YOLO11n trên dữ liệu gốc (epoch khớp iteration) | Yêu cầu của nhóm; vẫn đo được tác dụng nhờ đối chứng; so sánh mô hình vẫn công bằng | Vinh |
| | Không dùng DETR gốc làm baseline chính (giữ RT-DETR) | Notebook DETR-SE: chỉ dùng C5 (stride 32), chưa hội tụ sau 30 epoch, ~4 phút/epoch | |
| 2026-09-26 | Nhóm kích thước tương đối `[0, 0.005, 0.02, 1]` | EDA: GC10 có 80 / 51 / 227 box test (đều ≥ 50); không ngưỡng chung nào đạt cho mọi bộ | |
| 2026-09-26 | Cân bằng chế độ auto: NEU, PCB không cân bằng (max/min < 3); GC10, MT cân bằng; GC10 chưa bật xoay 90° | Bảng 2.5.2 | |
| 2026-09-26 | Không đưa NEU / PCB vào run đối chứng cân bằng | Không áp dụng cân bằng cho hai bộ này | |
| 2026-09-26 | PKU-PCB giữ chia ngẫu nhiên (không chia theo bo) | Dễ so sánh với tài liệu; chia theo bo chỉ còn ~1 bo cho test. Ghi là hạn chế | |
| 2026-09-27 | Cố định `splits/` trong repo (commit `2df7f89`) | Mọi phiên / tài khoản dùng chung một cách chia | |
| 2026-09-27 | `cache='ram'` cho PCB, GC10 | Đọc ảnh lớn qua ổ mạng làm GPU chờ; áp dụng như nhau cho mọi mô hình trên một bộ | |
| 2026-09-27 | Bỏ box trùng y hệt trong NEU; bỏ ảnh MT có mask rỗng; bỏ file rác KSDD2; ánh xạ `10_yaozhed`, bỏ nhãn `d` ở GC10 | Kiểm tra dữ liệu thật (KET_QUA_GD1.md Mục 2.2) | |

### 11.2. Nhật ký thực nghiệm
| Ngày | Run name | Mục đích | Kết quả chính | Ghi chú / sự cố |
|---|---|---|---|---|
| 2026-09-26 | `download_data.py`, EDA | Tải, kiểm tra, phân tích dữ liệu | 5 bộ đạt kiểm tra; bảng 2.5.2 | Phát hiện lỗi nhãn GC10 / MT / KSDD2 / NEU → đã xử lý |
| 2026-09-26 | `<ds>_bal_v1`, `gc10_aug_v1` | Cân bằng offline | GC10 +566 ảnh (+30.8%), MT +251 (+23.5%), NEU / PCB không đổi | t = 0.2, cap = 4, seed 42 |
| 2026-09-27 | `p1_*` (19 run), `p2s_*` (22 run) | GĐ1 + screening GĐ2 | Xem KET_QUA_GD1.md Mục 5–6 | 5 phiên song song / 3 tài khoản; lỗi đo GFLOPs sau train → sửa `3d6fab6`, đánh giá lại không train lại |
| 2026-09-28 | Phiên tổng hợp | Đánh giá lại 41 run, báo cáo | GĐ1 ≈ 16.8 giờ GPU, screening ≈ 16.4 giờ GPU | Sửa bảng AP theo lớp và biểu đồ > 8 mô hình (`54763ca`, `a4d8fbb`) |

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

**Mất cân bằng dữ liệu**
- Gupta, Dollár & Girshick (2019) – LVIS: A Dataset for Large Vocabulary Instance Segmentation (Repeat Factor Sampling)
- Lin et al. (2017) – Focal Loss for Dense Object Detection
- Ghiasi et al. (2021) – Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation
- Hu, Shen & Sun (2018) – Squeeze-and-Excitation Networks
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

### A.5. Repeat Factor Sampling – tạo danh sách train có oversample lớp hiếm
```python
import math, os
from collections import Counter
from pathlib import Path

def repeat_factor_list(train_txt, label_dir, out_dir, t=0.2):
    """
    train_txt : file danh sách ảnh train (1 đường dẫn / dòng)
    label_dir : thư mục nhãn YOLO (<stem>.txt)
    out_dir   : thư mục chứa ảnh + nhãn đã nhân bản (symlink)
    t         : ngưỡng tần suất (⏳ thử 0.1–0.3); lớp có f_c < t được lặp nhiều hơn
    """
    imgs = [Path(l.strip()) for l in open(train_txt) if l.strip()]
    cls_per_img = {}
    for im in imgs:
        lb = Path(label_dir) / f"{im.stem}.txt"
        rows = [r.split() for r in lb.read_text().splitlines() if r.strip()] if lb.exists() else []
        cls_per_img[im] = {int(r[0]) for r in rows}
    n = len(imgs)
    freq = Counter(c for cs in cls_per_img.values() for c in cs)          # số ảnh chứa lớp c
    r_c = {c: max(1.0, math.sqrt(t / (k / n))) for c, k in freq.items()}  # hệ số lặp theo lớp

    out_img, out_lbl = Path(out_dir, "images"), Path(out_dir, "labels")
    out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for im in imgs:
        cs = cls_per_img[im]
        r = max((r_c[c] for c in cs), default=1.0)                        # ảnh nền: 1 lần
        reps = math.floor(r) + (1 if (r - math.floor(r)) >= 0.5 else 0)  # làm tròn cố định (tái lập)
        for k in range(max(reps, 1)):
            suffix = "" if k == 0 else f"_rep{k}"
            dst_i = out_img / f"{im.stem}{suffix}{im.suffix}"
            dst_l = out_lbl / f"{im.stem}{suffix}.txt"
            if not dst_i.exists(): os.symlink(im.resolve(), dst_i)
            src_l = Path(label_dir) / f"{im.stem}.txt"
            if src_l.exists() and not dst_l.exists(): os.symlink(src_l.resolve(), dst_l)
            for c in cs: stats[c] += 1
    print("Hệ số lặp theo lớp:", {c: round(v, 2) for c, v in sorted(r_c.items())})
    print("Số ảnh chứa mỗi lớp sau oversample:", dict(sorted(stats.items())))
    return r_c
```
> Chỉ chạy trên **train**. Mỗi bản sao có tên khác (`_rep1`, `_rep2`…) để Ultralytics không gộp trùng; kiểm tra số ảnh train in ra khi bắt đầu train. Nếu Kaggle không cho symlink ra ngoài `/kaggle/working`, dùng `shutil.copy`. Ghi lại `t` và bảng hệ số lặp vào nhật ký thực nghiệm.

### A.6 + A.7. Cân bằng bằng tăng cường: class-aware augmentation + copy-paste (`scripts/balance_aug.py`)
Chỉ dùng OpenCV + NumPy (có sẵn trên Kaggle). Đã chạy kiểm tra tự động trên dữ liệu giả lập: box đúng vị trí sau lật/xoay (cả ảnh không vuông), lỗi dán giữ nguyên độ tương phản và không lộ viền, kết quả tái lập với cùng seed.

```python
import math, random
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

# ---------- I/O nhãn YOLO ----------
def read_yolo(lbl_path):
    if not Path(lbl_path).exists():
        return []
    out = []
    for r in Path(lbl_path).read_text().splitlines():
        p = r.split()
        if len(p) == 5:
            out.append([int(p[0])] + [float(v) for v in p[1:]])
    return out

def yolo_to_xyxy(lbls, W, H):
    return [[c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H] for c, x, y, w, h in lbls]

def xyxy_to_yolo(boxes, W, H):
    out = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = np.clip([x1, x2], 0, W); y1, y2 = np.clip([y1, y2], 0, H)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    return out

# ---------- Hệ số nhân bản theo lớp (Repeat Factor, có giới hạn) ----------
def class_repeat_factors(label_files, t=0.2, cap=4.0):
    n = len(label_files)
    freq = Counter()
    for lf in label_files:
        freq.update({b[0] for b in read_yolo(lf)})
    return {c: min(cap, max(1.0, math.sqrt(t / (k / n)))) for c, k in freq.items()}, freq

# ---------- Phép tăng cường giữ nguyên box (không thu nhỏ ảnh để không làm lỗi nhỏ biến mất) ----------
def geo_transform(img, boxes, op):
    H, W = img.shape[:2]
    out = []
    if op == "hflip":
        img = img[:, ::-1]
        out = [[c, W - x2, y1, W - x1, y2] for c, x1, y1, x2, y2 in boxes]
    elif op == "vflip":
        img = img[::-1, :]
        out = [[c, x1, H - y2, x2, H - y1] for c, x1, y1, x2, y2 in boxes]
    elif op == "rot90":            # xoay 90° ngược chiều kim đồng hồ
        img = np.rot90(img, 1)
        out = [[c, y1, W - x2, y2, W - x1] for c, x1, y1, x2, y2 in boxes]
    elif op == "rot180":
        img = img[::-1, ::-1]
        out = [[c, W - x2, H - y2, W - x1, H - y1] for c, x1, y1, x2, y2 in boxes]
    else:
        out = boxes
    return np.ascontiguousarray(img), out

def photo_transform(img, rng):
    f = img.astype(np.float32)
    alpha = rng.uniform(0.8, 1.2)                  # contrast
    beta = rng.uniform(-20, 20)                    # brightness
    f = f * alpha + beta
    if rng.random() < 0.5:                         # gamma
        g = rng.uniform(0.8, 1.25)
        f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, g)
    if rng.random() < 0.3:                         # nhiễu Gauss nhẹ
        f = f + rng.normal(0, rng.uniform(2, 6), f.shape)
    f = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:                         # blur rất nhẹ (tránh xóa lỗi nhỏ)
        f = cv2.GaussianBlur(f, (3, 3), 0)
    return f

def class_aware_augment(img_paths, label_dir, out_img_dir, out_lbl_dir,
                        geo_ops=("hflip", "vflip", "rot180"), t=0.2, cap=4.0, seed=42):
    """
    Sinh bản tăng cường OFFLINE cho ảnh chứa lớp hiếm (chỉ tập TRAIN).
    Số bản sinh thêm cho mỗi ảnh = round(max_c r_c) - 1, r_c tính theo repeat factor có trần `cap`.
    Mỗi bản sinh thêm dùng một phép biến đổi KHÁC nhau (không nhân bản y hệt).
    """
    rng = np.random.default_rng(seed)
    label_files = [Path(label_dir) / f"{Path(p).stem}.txt" for p in img_paths]
    r_c, freq_before = class_repeat_factors(label_files, t, cap)
    Path(out_img_dir).mkdir(parents=True, exist_ok=True); Path(out_lbl_dir).mkdir(parents=True, exist_ok=True)
    added = Counter(); n_new = 0
    for p, lf in zip(img_paths, label_files):
        lbls = read_yolo(lf)
        if not lbls:
            continue
        r = max(r_c[c] for c, *_ in lbls)
        extra = r - 1.0                                   # làm tròn ngẫu nhiên (có seed) để lớp
        n_extra = int(extra) + int(rng.random() < extra - int(extra))  # mất cân bằng vừa vẫn được tăng
        if n_extra <= 0:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        H, W = img.shape[:2]
        boxes = yolo_to_xyxy(lbls, W, H)
        ops = list(rng.permutation(list(geo_ops)))
        for k in range(n_extra):
            op = ops[k % len(ops)]
            im2, b2 = geo_transform(img, boxes, op)
            im2 = photo_transform(im2, rng)
            H2, W2 = im2.shape[:2]
            stem = f"{Path(p).stem}_aug{k+1}"
            cv2.imwrite(str(Path(out_img_dir) / f"{stem}{Path(p).suffix}"), im2)
            (Path(out_lbl_dir) / f"{stem}.txt").write_text("\n".join(xyxy_to_yolo(b2, W2, H2)))
            added.update(c for c, *_ in lbls); n_new += 1
    return {"repeat_factor": {c: round(v, 2) for c, v in sorted(r_c.items())},
            "images_before_per_class": dict(sorted(freq_before.items())),
            "boxes_added_per_class": dict(sorted(added.items())), "new_images": n_new}

# ---------- Copy-paste lỗi hiếm (chỉ cần nhãn bbox) ----------
def _feather_mask(h, w, box, margin):
    """Mặt nạ hòa trộn: =1 trên toàn bộ vùng lỗi, giảm dần về 0 trong phần viền nền."""
    bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(math.ceil(box[2])), int(math.ceil(box[3]))
    if margin <= 0:
        return np.ones((h, w), np.float32)
    inside = np.zeros((h, w), np.uint8)
    inside[by1:by2, bx1:bx2] = 1
    dist = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 3)   # khoảng cách tới vùng lỗi
    return np.clip(1.0 - dist / margin, 0.0, 1.0).astype(np.float32)

def _overlap(a, b, gap=4):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])

def copy_paste_rare(img_paths, label_dir, out_img_dir, out_lbl_dir, rare_classes,
                    n_new_images=200, max_paste=3, margin=6, seed=42):
    """
    Cắt lỗi thuộc `rare_classes` (kèm viền `margin` px) từ ảnh TRAIN, dán lên ảnh TRAIN khác
    ở vị trí không chồng lấn box có sẵn, hòa trộn biên mềm + khớp độ sáng trung bình.
    Không thu nhỏ patch (giữ nguyên kích thước lỗi thật).
    """
    rng = random.Random(seed)
    lf = lambda p: Path(label_dir) / f"{Path(p).stem}.txt"
    pool = []
    for p in img_paths:
        lbls = read_yolo(lf(p))
        if not any(c in rare_classes for c, *_ in lbls):
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED); H, W = img.shape[:2]
        for c, x1, y1, x2, y2 in yolo_to_xyxy(lbls, W, H):
            if c not in rare_classes:
                continue
            X1, Y1 = int(max(0, x1 - margin)), int(max(0, y1 - margin))
            X2, Y2 = int(min(W, x2 + margin)), int(min(H, y2 + margin))
            pool.append((c, img[Y1:Y2, X1:X2].copy(), (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1), Path(p).stem))
    if not pool:
        return {"pool": 0, "new_images": 0}
    Path(out_img_dir).mkdir(parents=True, exist_ok=True); Path(out_lbl_dir).mkdir(parents=True, exist_ok=True)
    added = Counter(); made = 0
    for k in range(n_new_images):
        p = rng.choice(img_paths)
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).copy(); H, W = img.shape[:2]
        boxes = yolo_to_xyxy(read_yolo(lf(p)), W, H)
        pasted = 0
        for _ in range(rng.randint(1, max_paste)):
            c, patch, (bx1, by1, bx2, by2), src = rng.choice(pool)
            if src == Path(p).stem:
                continue
            ph, pw = patch.shape[:2]
            if ph >= H or pw >= W:
                continue
            for _try in range(20):
                ox, oy = rng.randint(0, W - pw), rng.randint(0, H - ph)
                cand = [ox, oy, ox + pw, oy + ph]
                if all(not _overlap(cand, b[1:]) for b in boxes):
                    break
            else:
                continue
            region = img[oy:oy + ph, ox:ox + pw].astype(np.float32)
            if img.ndim == 3 and patch.ndim == 2:          # đồng bộ số kênh
                patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 2 and patch.ndim == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            pf = patch.astype(np.float32)
            ring = np.ones((ph, pw), bool)                # viền nền quanh lỗi trong patch
            ring[int(by1):int(math.ceil(by2)), int(bx1):int(math.ceil(bx2))] = False
            if ring.any():                                # khớp độ sáng NỀN của patch với vùng đích
                pf = pf + (region.mean() - pf[ring].mean())
            m = _feather_mask(ph, pw, (bx1, by1, bx2, by2), margin)
            if img.ndim == 3:
                m = m[..., None]
            img[oy:oy + ph, ox:ox + pw] = np.clip(m * pf + (1 - m) * region, 0, 255).astype(img.dtype)
            boxes.append([c, ox + bx1, oy + by1, ox + bx2, oy + by2])
            added[c] += 1; pasted += 1
        if pasted == 0:
            continue
        stem = f"{Path(p).stem}_cp{k}"
        cv2.imwrite(str(Path(out_img_dir) / f"{stem}{Path(p).suffix}"), img)
        (Path(out_lbl_dir) / f"{stem}.txt").write_text("\n".join(xyxy_to_yolo(boxes, W, H)))
        made += 1
    return {"pool": len(pool), "new_images": made, "boxes_added_per_class": dict(sorted(added.items()))}
```

**Cách dùng (chạy trong CPU notebook, sau khi đã chia dữ liệu):**
```python
from pathlib import Path
import shutil
from balance_aug import class_aware_augment, copy_paste_rare

ds = "gc10"
train_imgs = sorted(Path(f"data/yolo/{ds}/images/train").glob("*.jpg"))
lbl_dir    = Path(f"data/yolo/{ds}/labels/train")
out        = Path(f"data/yolo/{ds}_bal_v1")

# 1) Chép nguyên train/val/test gốc sang bản _bal (val/test KHÔNG đổi)
shutil.copytree(f"data/yolo/{ds}", out, dirs_exist_ok=True)

# 2) Class-aware augmentation (GC10: chỉ lật; ⏳ thêm rot90 sau khi kiểm tra trực quan)
stats_aug = class_aware_augment(train_imgs, lbl_dir, out/"images/train", out/"labels/train",
                                geo_ops=("hflip", "vflip"), t=0.2, cap=4.0, seed=42)

# 3) Copy-paste lớp hiếm (danh sách lớp hiếm lấy từ bảng 2.5.2)
RARE = {...}   # ⏳ vd. id các lớp rolled pit, crease, ...
stats_cp = copy_paste_rare(train_imgs, lbl_dir, out/"images/train", out/"labels/train",
                           rare_classes=RARE, n_new_images=300, max_paste=3, margin=6, seed=42)
print(stats_aug); print(stats_cp)   # ghi vào nhật ký thực nghiệm
```
> Điều chỉnh `t`, `cap`, `n_new_images` sao cho tổng ảnh train tăng ≤ 50% và lớp hiếm đạt ≥ 1/3 lớp lớn nhất. Đếm lại phân bố lớp sau khi sinh (dùng lại script EDA) trước khi upload.

