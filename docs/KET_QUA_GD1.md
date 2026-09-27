# Kết quả thực nghiệm – Giai đoạn 1 (Thường kỳ) và screening Giai đoạn 2

> Ghi lại toàn bộ những gì đã chạy tính đến lần tổng hợp đầu tiên: dữ liệu, quy trình, cấu hình, kết quả, nhận xét,
> hạn chế và các quyết định còn chờ. Kế hoạch gốc: [KE_HOACH_NGHIEN_CUU_LOI_BE_MAT_NHO.md](KE_HOACH_NGHIEN_CUU_LOI_BE_MAT_NHO.md).
> Số liệu lấy từ `results/master_results.csv`, `results/tables/*` và `results/eda/*` của phiên tổng hợp.
> **Mọi kết quả là 1 seed (seed 0)**; chênh lệch dưới ~1–1.5 điểm AP nên coi là nhiễu (xem Mục 8).

## 0. Tóm tắt

- **Mô hình chung: YOLO11n** (giữ nguyên). Kém mô hình tốt nhất 1–3 AP trên GC10/NEU/MT (7 AP trên PCB) nhưng
  nhẹ hơn ~16× về tham số và 16–70× về GFLOPs.
- **RT-DETR-l vượt rõ trên PKU-PCB** (bộ có lỗi cực nhỏ sau resize): AP 56.7 so với 49.7 (YOLO11n).
- **Nhánh P2 không giúp**: AP gần như không đổi trên GC10/PCB, **giảm 4.6 AP trên NEU**; chỉ AP_rel_small của GC10 tăng nhẹ (+2.4).
- **Cân bằng dữ liệu (tăng cường + copy-paste) không có lợi trên GC10**: AP_rare 35.6 so với 42.8 của dữ liệu gốc
  (số iteration như nhau); copy-paste kém hơn chỉ tăng cường. Trên MT không kết luận được (lớp hiếm chỉ 4 box test).
- **Screening GĐ2**: class-weighted BCE (`clsw`) tốt nhất (+2.3 AP GC10, +1.1 AP NEU, +8.7 AP_rel_small GC10);
  Coordinate Attention tốt nhất trong nhóm backbone (+2.6 AP GC10); Wise-IoU tốt nhất trong nhóm loss hộp (+1.4).
  Hai lựa chọn mặc định ban đầu **SimAM và NWD đều kém hơn mốc**.
- Chi phí thực tế: GĐ1 ≈ **16.8 giờ GPU** (kế hoạch ước 26 h), screening ≈ **16.4 giờ GPU** (ước 22 h).

---

## 1. Môi trường và tái lập

| Mục | Giá trị |
|---|---|
| Phần cứng | Kaggle, 2 × Tesla T4 16 GB mỗi phiên (một tiến trình / GPU), 4 vCPU |
| Phần mềm | Python 3.12, PyTorch 2.10 + CUDA 12.8, Ultralytics 8.4.163 (ghim phiên bản) |
| Code | https://github.com/Dlevinh755/small-defect-detection – train với code tới commit `2df7f89` (splits cố định), đánh giá lại với `3d6fab6` (sửa lỗi benchmark), báo cáo với `a4d8fbb` |
| Chia dữ liệu | `splits/<dataset>/{train,val,test}.txt`, seed 42, đã commit – mọi phiên / tài khoản dùng chung |
| Seed huấn luyện | 0 (GĐ1 và screening) |
| Cách chạy | Notebook `notebooks/sdd_end_to_end.ipynb`, 5 phiên song song trên 3 tài khoản (mỗi phiên một phần grid, `SESSION_SHARD`), sau đó 1 phiên tổng hợp đánh giá lại và làm báo cáo |

---

## 2. Dữ liệu

### 2.1. Nguồn (tải bằng `scripts/download_data.py`, kiểm tra tự động)

| id | Bộ dữ liệu | Nguồn | Giấy phép | Ảnh dùng được | Box |
|---|---|---|---|---|---|
| neu | NEU-DET | Kaggle `kaustubhdikshit/neu-surface-defect-database` | nghiên cứu | 1800 | 4189* |
| gc10 | GC10-DET | Kaggle `alex000kim/gc10det` | nghiên cứu | 2294 (ảnh có XML) | 3563 |
| pcb | PKU-Market-PCB | Kaggle `akhatova/pcb-defects` (bản gốc độ phân giải cao) | nghiên cứu | 693 | 2953 |
| mt | Magnetic Tile | GitHub `abin24/Magnetic-tile-defect-datasets.` | nghiên cứu | 1338 (386 lỗi / 952 sạch) | 443 |
| ksdd2 | KolektorSDD2 (đích GĐ3) | data.vicos.si `KolektorSDD2.zip` | CC BY-NC-SA 4.0 | 3335 (356 lỗi) | 391 |

\* Số của lần EDA, trước khi bỏ các box trùng (xem 2.2); dữ liệu dùng để train đã bỏ box trùng.

### 2.2. Vấn đề nhãn phát hiện trên dữ liệu thật và cách xử lý

| Bộ | Vấn đề | Xử lý |
|---|---|---|
| GC10 | Lớp 10 ghi là `10_yaozhed` (131 box) | Ánh xạ sang `waist_folding` |
| GC10 | 1 object nhãn `d` (lỗi gán nhãn) | Bỏ object (`ignore_classes`), có ghi lại |
| GC10 | ~1270 / ~3570 ảnh không có XML | Không dùng (đúng như bộ gốc) |
| MT | 6 ảnh `MT_Uneven` có mask rỗng (4) hoặc gần rỗng (26–45 px) | Bỏ ảnh – nếu giữ sẽ thành ảnh "sạch" sai nhãn; 19 thành phần mask < 9 px bị bỏ |
| KSDD2 | File rác `10301 (copy).png`, `10301_GT (copy).png` trong zip chính thức | Bỏ (ảnh không có mask) – còn đúng 3335 ảnh |
| NEU | Một số box trùng y hệt (cùng lớp, cùng tọa độ), ví dụ `crazing_120`, `inclusion_62`, `patches_198` | Giữ một bản (Ultralytics vốn bỏ khi train; ground truth test phải khớp) |

### 2.3. Chia dữ liệu (cố định, stratified theo lớp hiếm nhất trong ảnh, seed 42)

| Bộ | train | val | test | Ghi chú |
|---|---|---|---|---|
| NEU | 1440 | 180 | 180 | giống DAC-YOLO |
| GC10 | 1835 | 230 | 229 | |
| PCB | 555 | 69 | 69 | chia ngẫu nhiên (xem hạn chế 8.4) |
| MT | 1070 | 134 | 134 | |
| KSDD2 | 2098 | 233 | 1004 | test = test chính thức |

### 2.4. EDA – kích thước lỗi

| Bộ | Kích thước ảnh | Box "small" (<32²) ảnh gốc | Box "small" sau resize 640 | rel_small / rel_medium / rel_large |
|---|---|---|---|---|
| NEU | 200×200 | 9.5% | **0%** | 0% / 6.1% / 93.9% |
| GC10 | 2048×1000 | 0.4% | **24.2%** | 23.8% / 15.8% / 60.4% |
| PCB | ~2240–2530 px | 0.7% | **98.6%** | 99.9% / 0.1% / 0% |
| MT | ~105×290 | 49.9% | 32.3% | 36.1% / 24.8% / 39.1% |

- Nhóm kích thước tương đối chốt **`[0, 0.005, 0.02, 1]`** (diện tích box / diện tích ảnh): GC10 chia 80 / 51 / 227 box test (đều ≥ 50).
  Không có ngưỡng chung nào đạt ≥ 50 box / nhóm cho mọi bộ: PCB rơi hết vào nhóm nhỏ, NEU hết vào nhóm lớn, MT chỉ có 40 box test.
- **NEU gần như không có lỗi nhỏ** → chỉ dùng làm bộ đối chiếu, không dùng để kết luận về lỗi nhỏ.
- **PCB và GC10 là hai bộ mà lỗi bị co nhỏ khi resize** – ủng hộ giả thuyết H1.

### 2.5. Mất cân bằng (bảng 2.5.2 của kế hoạch, train gốc)

| Bộ | Lớp nhiều nhất (box) | Lớp ít nhất (box) | max/min | Ảnh lỗi : sạch | Box test lớp ít nhất | Mức độ | Lớp hiếm |
|---|---|---|---|---|---|---|---|
| NEU | inclusion (788) | pitted_surface (348) | 2.26 | chỉ ảnh lỗi | 44 | nhẹ | – |
| GC10 | silk_spot (704) | crease (61) | 11.54 | ~1 : 0 (2 ảnh sạch) | 8 | **nặng** | crescent_gap, rolled_pit, crease, waist_folding |
| PCB | spurious_copper (405) | spur (386) | 1.05 | chỉ ảnh lỗi | 52 | nhẹ | – |
| MT | break (96) | fray (30) | 3.20 | 1 : 2.47 | 4 | vừa | fray |
| KSDD2 | 1 lớp | – | – | ~1 : 8.4 (356 / 2979) | – | – | (mất cân bằng lỗi/sạch – xử lý ở GĐ3) |

Lớp hiếm = < 1/3 số box của lớp lớn nhất. Lớp có < 10 box test bị đánh dấu **không ổn định**: GC10 crease (8), rolled_pit (8); MT crack (6), fray (4), uneven (9).

---

## 3. Xử lý mất cân bằng (offline, chỉ tập train)

Cấu hình (`configs/protocol.yaml → balance`, phiên bản `v1`): repeat factor `r_c = min(cap, max(1, √(t / f_c)))` với
`t = 0.2`, `cap = 4`; tối đa +50% ảnh train (khi có copy-paste: ½ ngân sách cho tăng cường); copy-paste điền lớp hiếm tới
1/3 số ảnh của lớp lớn nhất; phép hình học GC10 = lật ngang/dọc (chưa bật xoay 90° – có thể phụ thuộc hướng cán), MT = lật + xoay 180°;
biến đổi quang học: tương phản ×[0.8, 1.2], độ sáng ±20, gamma [0.8, 1.25] (50%), nhiễu Gauss σ 2–6 (30%), blur 3×3 (20%); seed 42.

| Bộ | Áp dụng? | Ảnh thêm | Chi tiết |
|---|---|---|---|
| NEU | Không (max/min 2.26 < 3, chế độ auto) | 0 | `neu_bal_v1` = dữ liệu gốc |
| PCB | Không (1.05 < 3) | 0 | `pcb_bal_v1` = dữ liệu gốc |
| GC10 `bal_v1` | Có | +566 (+30.8%): 458 tăng cường + 108 copy-paste | rolled_pit 36→201 ảnh, crease 43→203, waist_folding 112→207, crescent_gap 210→293; tỉ lệ ảnh max/min 16.4× → 3.1× |
| GC10 `aug_v1` | Có (chỉ tăng cường) | +545 (+29.7%) | dùng cho run tùy chọn |
| MT `bal_v1` | Có | +251 (+23.5%), chỉ tăng cường | fray 26→73 ảnh đã vượt mục tiêu nên copy-paste không cần chạy |

Ghi chú: với `t = 0.2`, hầu hết lớp GC10 xuất hiện trong < 20% ảnh nên lớp phổ biến cũng tăng (vd. oil_spot 468→684 box);
ở MT tỉ lệ ảnh lỗi : sạch trong train đổi từ ~1:2.5 về ~1:1.6 (val/test giữ nguyên).

---

## 4. Quy trình thực nghiệm đã chạy

1. **Tải và kiểm tra dữ liệu** (`download_data.py`) → phát hiện và sửa các vấn đề nhãn ở Mục 2.2.
2. **Chuyển đổi** sang YOLO + COCO json, **chia dữ liệu cố định** (commit `splits/`).
3. **EDA** (Mục 2.4–2.5), chốt nhóm kích thước tương đối.
4. **Cân bằng** `_bal_v1` (+ `gc10_aug_v1`), kiểm tra QA bằng hình (`figures/balance/*`).
5. **5 phiên Kaggle song song** (3 tài khoản, mỗi phiên 2 GPU): 3 phiên chia 19 run GĐ1, 2 phiên chia 22 run screening.
   Sự cố: mọi run train xong nhưng lỗi ở bước đo GFLOPs (input trên GPU, model trên CPU) → sửa (`3d6fab6`), checkpoint giữ nguyên.
6. **Phiên tổng hợp**: gắn 5 output (41 run, không trùng), đánh giá lại tất cả trên tập test (không train lại), làm báo cáo.
   Sửa thêm 2 lỗi ở bước báo cáo (bảng AP theo lớp, biểu đồ > 8 mô hình).

### 4.1. Cấu hình huấn luyện và đánh giá (giống nhau cho mọi run)

| Mục | Giá trị |
|---|---|
| Kích thước ảnh | 640 (letterbox) |
| YOLO11n / -P2 | 100 epoch, batch 16, optimizer tự chọn của Ultralytics (AdamW lr 0.001 ở NEU), augmentation mặc định (mosaic, HSV, lật ngang), không early stopping |
| RT-DETR-l | 60 epoch, batch 8, mặc định Ultralytics |
| Faster R-CNN R50-FPN v2 | 24 epoch, batch 8, SGD lr 0.01, giảm ×0.1 ở epoch 16 và 22, warmup 500 iter, lật ngang; chọn checkpoint theo AP val |
| Khởi tạo | Pretrained COCO |
| Dữ liệu train | `<ds>_bal_v1` cho mọi mô hình; run đối chứng YOLO11n dùng dữ liệu gốc với epoch nhân theo \|train_bal\| / \|train\| (GC10: 131 epoch, MT: 123 epoch) |
| Cache | `cache='ram'` cho PCB và GC10 (ảnh gốc lớn, đọc qua ổ mạng chậm) |
| Chọn checkpoint | theo val; test đánh giá **một lần** |
| Dự đoán | conf 0.001, NMS IoU 0.7, tối đa 300 box / ảnh, tọa độ ảnh gốc |
| Chỉ số | COCO AP / AP50 / AP75 / AP_s,m,l; AP và recall theo nhóm kích thước tương đối; P / R tại conf 0.25, IoU 0.5; AP theo lớp; AP_rare / AP_common; TIDE |
| Hiệu năng | Params, GFLOPs @640 (torch FlopCounter), FPS = 1 / độ trễ forward batch 1, FP16, T4 (không gồm tiền/hậu xử lý) |

---

## 5. Kết quả GĐ1

### 5.1. Bảng A – so sánh mô hình (dữ liệu `_bal_v1`, seed 0)

AP, AP50, AP75, AP_s/m/l, AP_rel_small, R_rel_small tính bằng %.

| Bộ | Mô hình | AP | AP50 | AP75 | AP_s | AP_m | AP_l | AP_rel_small | R_rel_small | Params (M) | GFLOPs | FPS (FP16) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GC10 | Faster R-CNN | **37.4** | **73.1** | **38.6** | 70.0 | 19.8 | 37.0 | 16.8 | 66.2 | 43.3 | 450.9 | 22.2 |
| GC10 | RT-DETR-l | 32.4 | 66.6 | 25.6 | 30.0 | 18.4 | 32.0 | 20.2 | **72.5** | 32.8 | 105.3 | 29.0 |
| GC10 | YOLO11n | 34.3 | 68.7 | 28.5 | 20.0 | 24.0 | 34.2 | 23.5 | 56.2 | 2.6 | 6.4 | 78.4 |
| GC10 | YOLO11n-P2 | 34.0 | 70.5 | 28.0 | 10.0 | **32.5** | 33.4 | **25.9** | 53.8 | 2.7 | 10.3 | 60.9 |
| MT | Faster R-CNN | 66.8 | 93.4 | 80.1 | 61.8 | **85.6** | 65.5 | 59.5 | 100.0 | 43.3 | 450.8 | 22.9 |
| MT | RT-DETR-l | 68.4 | 91.5 | **82.3** | **70.8** | 82.2 | 59.2 | **74.6** | 100.0 | 32.8 | 105.3 | 28.3 |
| MT | YOLO11n | **69.5** | **96.2** | 81.2 | 65.2 | 67.9 | **75.1** | 67.7 | 100.0 | 2.6 | 6.4 | 89.2 |
| MT | YOLO11n-P2 | 68.9 | 94.4 | 80.1 | 64.3 | 82.9 | 69.4 | 67.2 | 93.8 | 2.7 | 10.3 | 90.6 |
| NEU | Faster R-CNN | 43.5 | **79.0** | 42.1 | **34.6** | **35.7** | 57.2 | – | – | 43.3 | 450.8 | 21.1 |
| NEU | RT-DETR-l | **43.9** | 75.9 | **42.8** | 26.2 | 35.4 | **59.6** | – | – | 32.8 | 105.3 | 28.0 |
| NEU | YOLO11n | 42.8 | 76.2 | 41.2 | 29.9 | 34.2 | 57.5 | – | – | 2.6 | 6.4 | 61.6 |
| NEU | YOLO11n-P2 | 38.2 | 74.0 | 30.5 | 32.2 | 31.1 | 49.5 | – | – | 2.7 | 10.3 | 65.8 |
| PCB | Faster R-CNN | 49.0 | 93.0 | 41.0 | 0.0 | 49.7 | 53.1 | 49.0 | 92.4 | 43.3 | 450.8 | 22.4 |
| PCB | RT-DETR-l | **56.7** | **98.6** | **59.8** | **36.8** | **56.8** | 55.2 | **56.7** | **97.9** | 32.8 | 105.3 | 27.1 |
| PCB | YOLO11n | 49.7 | 93.5 | 43.6 | 10.1 | 50.2 | 51.2 | 49.7 | 91.1 | 2.6 | 6.4 | 66.5 |
| PCB | YOLO11n-P2 | 49.6 | 92.2 | 46.3 | 8.9 | 50.1 | **60.0** | 49.6 | 89.0 | 2.7 | 10.3 | 66.2 |

"–": NEU không có box nào trong nhóm rel_small. **AP_s (chuẩn COCO) của GC10 và PCB dựa trên rất ít box** (ở độ phân giải gốc
chỉ 0.4% / 0.7% box là "small") nên không dùng để so sánh – dùng AP_rel_small / R_rel_small.

### 5.2. Bảng B – tác dụng của xử lý mất cân bằng (YOLO11n)

| Bộ | Dữ liệu train | Ảnh train | AP | AP_rare | AP_common | R_rel_small | AP các lớp hiếm |
|---|---|---|---|---|---|---|---|
| GC10 | gốc (epoch khớp iteration) | 1835 | **37.6** | **42.8** | 34.1 | 58.8 | crease 44.7, crescent_gap 63.8, rolled_pit 15.4, waist_folding 47.4 |
| GC10 | chỉ tăng cường `aug_v1` | 2380 | 36.6 | 39.8 | **34.5** | **66.2** | crease 27.5, crescent_gap 62.0, rolled_pit 20.7, waist_folding 48.9 |
| GC10 | tăng cường + copy-paste `bal_v1` | 2401 | 34.3 | 35.6 | 33.4 | 56.2 | crease 25.9, crescent_gap 62.3, rolled_pit 8.9, waist_folding 45.3 |
| MT | gốc (epoch khớp iteration) | 1070 | 66.9 | **85.0** | 62.4 | 93.8 | fray 85.0 |
| MT | `bal_v1` (chỉ tăng cường) | 1321 | **69.5** | 80.2 | **66.8** | **100.0** | fray 80.2 |

### 5.3. AP theo lớp (kèm số box test)

**GC10**

| Lớp | Box test | Cờ | YOLO11n [aug_v1] | Faster R-CNN | RT-DETR-l | YOLO11n | YOLO11n-P2 | YOLO11n [gốc] |
|---|---|---|---|---|---|---|---|---|
| crease | 8 | hiếm, không ổn định | 27.5 | 32.1 | 21.2 | 25.9 | 22.7 | 44.7 |
| crescent_gap | 26 | hiếm | 62.0 | 63.1 | 64.9 | 62.3 | 62.7 | 63.8 |
| inclusion | 41 | | 9.9 | 14.9 | 9.8 | 9.5 | 10.0 | 9.8 |
| oil_spot | 54 | | 28.2 | 26.3 | 26.6 | 31.1 | 27.4 | 30.2 |
| punching_hole | 30 | | 44.6 | 52.5 | 50.6 | 47.6 | 50.1 | 46.6 |
| rolled_pit | 8 | hiếm, không ổn định | 20.7 | 13.2 | 13.1 | 8.9 | 20.2 | 15.4 |
| silk_spot | 93 | | 25.3 | 23.7 | 20.0 | 25.1 | 25.2 | 27.9 |
| waist_folding | 14 | hiếm | 48.9 | 55.5 | 34.5 | 45.3 | 46.2 | 47.4 |
| water_spot | 34 | | 47.4 | 41.9 | 38.7 | 42.3 | 40.3 | 40.8 |
| welding_line | 50 | | 51.5 | 50.8 | 44.3 | 44.9 | 34.8 | 49.3 |

**MT**

| Lớp | Box test | Cờ | Faster R-CNN | RT-DETR-l | YOLO11n | YOLO11n-P2 | YOLO11n [gốc] |
|---|---|---|---|---|---|---|---|
| blowhole | 11 | | 71.0 | 75.7 | 73.8 | 73.2 | 71.4 |
| break | 10 | | 78.0 | 84.0 | 77.1 | 70.7 | 65.6 |
| crack | 6 | không ổn định | 56.7 | 59.9 | 54.8 | 59.8 | 51.2 |
| fray | 4 | hiếm, không ổn định | 82.6 | 72.1 | 80.2 | 83.9 | 85.0 |
| uneven | 9 | không ổn định | 46.0 | 50.4 | 61.4 | 57.0 | 61.3 |

**NEU**

| Lớp | Box test | Faster R-CNN | RT-DETR-l | YOLO11n | YOLO11n-P2 |
|---|---|---|---|---|---|
| crazing | 64 | 15.6 | 11.6 | 13.1 | 10.8 |
| inclusion | 111 | 41.6 | 44.7 | 44.1 | 41.5 |
| patches | 91 | 53.2 | 53.3 | 53.5 | 52.6 |
| pitted_surface | 44 | 53.6 | 63.7 | 54.9 | 54.4 |
| rolled-in_scale | 72 | 38.8 | 28.6 | 33.9 | 34.2 |
| scratches | 52 | 58.0 | 61.2 | 57.2 | 35.5 |

**PCB**

| Lớp | Box test | Faster R-CNN | RT-DETR-l | YOLO11n | YOLO11n-P2 |
|---|---|---|---|---|---|
| missing_hole | 42 | 62.1 | 64.6 | 56.9 | 61.7 |
| mouse_bite | 54 | 48.9 | 57.0 | 48.1 | 49.9 |
| open_circuit | 41 | 44.3 | 55.6 | 42.6 | 41.7 |
| short | 51 | 43.7 | 54.9 | 52.7 | 49.8 |
| spur | 52 | 49.0 | 55.2 | 48.0 | 48.1 |
| spurious_copper | 51 | 45.8 | 52.8 | 49.8 | 46.2 |

### 5.4. Chi phí huấn luyện thực tế (giờ GPU, 1 × T4)

| Mô hình | GC10 | MT | NEU | PCB | Tổng |
|---|---|---|---|---|---|
| Faster R-CNN | 1.49 | 0.86 | 0.98 | 0.66 | 3.99 |
| RT-DETR-l | 2.66 | 1.46 | 1.62 | 0.65 | 6.39 |
| YOLO11n (gồm run `_bal`, đối chứng gốc, `aug_v1`) | 2.11 | 0.98 | 0.67 | 0.22 | 3.98 |
| YOLO11n-P2 | 1.03 | 0.48 | 0.66 | 0.27 | 2.44 |
| **Tổng GĐ1** | | | | | **16.8** |

---

## 6. Screening GĐ2 (mỗi module riêng lẻ, dữ liệu `_bal_v1`, seed 0)

Mốc = YOLO11n của GĐ1 trên cùng dữ liệu. Δ = chênh lệch so với mốc (điểm %).

| Biến thể | Nhóm | GC10 AP | Δ | GC10 AP75 | GC10 AP_rel_small | Δ | NEU AP | Δ | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n (mốc) | – | 34.3 | – | 28.5 | 23.5 | – | 42.8 | – | 2.6 | 6.4 |
| YOLO11n-P2 (GĐ1) | neck | 34.0 | −0.3 | 28.0 | 25.9 | +2.4 | 38.2 | −4.6 | 2.7 | 10.3 |
| p2p4 | neck | 33.8 | −0.5 | 30.5 | 23.8 | +0.3 | 32.6 | −10.2 | 1.9 | 9.7 |
| **ca** (Coordinate Attention) | backbone | **36.9** | **+2.6** | 32.2 | 21.9 | −1.6 | 42.3 | −0.5 | 2.6 | 6.4 |
| cbam | backbone | 35.1 | +0.8 | 29.8 | 21.2 | −2.3 | 42.8 | 0.0 | 2.7 | 6.4 |
| simam | backbone | 33.6 | −0.7 | 27.7 | 19.5 | −4.0 | 39.8 | −3.0 | 2.6 | 6.4 |
| spd | backbone | 32.2 | −2.1 | 24.3 | 25.8 | +2.3 | 41.6 | −1.2 | 4.0 | 10.6 |
| nwd | loss hộp | 34.1 | −0.2 | 27.4 | 18.8 | −4.7 | 41.3 | −1.5 | 2.6 | 6.4 |
| nwd_assign | loss hộp + assigner | 34.1 | −0.2 | 30.1 | 22.7 | −0.8 | 39.9 | −2.9 | 2.6 | 6.4 |
| **wiou** (Wise-IoU v3) | loss hộp | 35.7 | +1.4 | 30.2 | 22.8 | −0.7 | 42.7 | −0.1 | 2.6 | 6.4 |
| inner (Inner-IoU) | loss hộp | 35.3 | +1.0 | 26.0 | 23.7 | +0.2 | 39.6 | −3.2 | 2.6 | 6.4 |
| **clsw** (class-weighted BCE, 1/√freq) | loss phân loại | 36.6 | +2.3 | **32.7** | **32.2** | **+8.7** | **43.9** | **+1.1** | 2.6 | 6.4 |
| focal | loss phân loại | 34.1 | −0.2 | 30.4 | 19.2 | −4.3 | 41.1 | −1.7 | 2.6 | 6.4 |

Chi phí screening: GC10 8.9 h + NEU 7.5 h = **16.4 giờ GPU** (22 run, ~0.5–1 h / run).

---

## 7. Nhận xét (điền Mục 4.7 của kế hoạch)

- **Mô hình chung:** YOLO11n. Tốt nhất ở MT, kém 1–3 AP trên GC10/NEU và 7 AP trên PCB, nhưng rẻ hơn nhiều (6.4 GFLOPs so với 105–451).
- **RQ1 / H1:** lỗi bị co nhỏ khi resize (PCB 98.6% box thành "small" ở 640, GC10 24.2%). Trên PCB, RT-DETR-l hơn các mô hình
  CNN khoảng 7 AP ở mọi lớp. Faster R-CNN mạnh nhất ở AP75 trên GC10 (38.6 so với ≤ 28.5) – định vị tốt hơn trên ảnh lớn.
- **TIDE:** ⏳ đọc từ `results/figures/p1/tide_*.png` rồi điền (loại lỗi chiếm ưu thế theo bộ).
- **H2 (P2):** không được ủng hộ – AP không tăng trên GC10/PCB, giảm 4.6 trên NEU (scratches 57.2 → 35.5; AP75 41.2 → 30.5);
  chỉ AP_rel_small của GC10 +2.4. Trên PCB, nơi lỗi nhỏ nhất, P2 cũng không giúp → thông tin có thể đã mất ngay khi
  thu ảnh 2500 px về 640; hướng cắt tile / SAHI (`p2_tiling`) đáng thử hơn.
- **Cân bằng dữ liệu:** không đạt tiêu chí "có lợi" của kế hoạch (AP_rare tăng và AP_common giảm ≤ 1). GC10: AP_rare
  giảm 7.2 (`bal_v1`) / 3.0 (`aug_v1`); copy-paste làm kém hơn chỉ tăng cường. MT: AP +2.6 nhưng lớp hiếm fray chỉ 4 box test.
  Ngược lại, cân bằng ở **mức loss** (`clsw`) là biến thể tốt nhất trong screening.
- **Hướng GĐ2 (đề xuất, chờ xác nhận – Mục 9):** backbone = Coordinate Attention; loss = class-weighted BCE (và/hoặc Wise-IoU);
  dữ liệu gốc thay cho `_bal_v1`; thêm PCB vào ablation.

---

## 8. Hạn chế và lưu ý khi trình bày

1. **1 seed** cho mọi run – chênh lệch < ~1–1.5 AP chưa có ý nghĩa thống kê; GĐ2 dùng 3 seed + t-test.
2. **Tập test nhỏ**: MT 40 box, nhiều lớp < 10 box test (đánh dấu "không ổn định"); AP_rare của GC10 phụ thuộc mạnh vào crease / rolled_pit (8 box mỗi lớp).
3. **AP_s chuẩn COCO không dùng được cho GC10 / PCB** (quá ít box small ở độ phân giải gốc) → dùng AP_rel_small.
4. **PCB chia ngẫu nhiên**: cùng một bo mạch có thể nằm ở cả train và test (rò rỉ theo bo). Có thể đánh giá phụ theo bo ở GĐ2 (`group_regex`).
5. **Screening chạy trên `_bal_v1`**, trong khi đề xuất GĐ2 dùng dữ liệu gốc – so sánh giữa các module vẫn hợp lệ (cùng dữ liệu), nhưng giá trị tuyệt đối sẽ khác.
6. `cache='ram'` (PCB, GC10) có thể làm kết quả không lặp lại bit-by-bit; áp dụng giống nhau cho mọi mô hình trên một bộ.
7. FPS là độ trễ forward batch 1 FP16 (không gồm tiền / hậu xử lý, NMS); FLOPs của Faster R-CNN đếm cả RPN + ROI head.
8. Số liệu EDA (Mục 2) được tính trước khi bỏ box trùng của NEU; dữ liệu train / test đã bỏ.

---

## 9. Quyết định cho GĐ2 (chờ xác nhận)

| # | Đề xuất | Căn cứ |
|---|---|---|
| 1 | Dữ liệu GĐ2 = gốc (`data_suffix: ""`) | Bảng B: cân bằng không có lợi |
| 2 | A2 = Coordinate Attention (thay SimAM) | Screening: +2.6 AP GC10; SimAM −0.7 / −3.0 |
| 3 | A3 = class-weighted BCE `clsw` (có thể thêm A8 = Wise-IoU) | Screening: +2.3 / +1.1 AP, +8.7 AP_rel_small; NWD không giúp |
| 4 | A1 = P2 giữ nguyên theo kế hoạch | Báo cáo trung thực tác dụng của neck |
| 5 | Bộ dữ liệu ablation: GC10 + PCB (± NEU) | PCB là bộ lỗi nhỏ thật sự, train ~0.2 h / run |
| 6 | Song song: `p2_tiling` (SAHI / tile) cho PCB, GC10; thêm 2 seed cho cặp đối chứng cân bằng GC10 | Kiểm chứng hướng độ phân giải và kết luận Bảng B |
