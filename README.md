# AI-Generated vs Real Images Evaluation

Hệ thống đánh giá chất lượng ảnh AI-generated so với ảnh thật cho bài toán phân biệt ảnh trẻ em.

## 📋 Mục tiêu

Chứng minh tính khả thi của việc sử dụng ảnh sinh từ AI để thay thế/bổ sung dữ liệu thật thông qua 3 metrics:

- **Metric A (FID)**: Đánh giá độ chân thực
- **Metric B (Cosine Similarity)**: Đánh giá tính nhất quán
- **Metric C (t-SNE)**: Đánh giá khả năng phân tách

## 🏗️ Cấu trúc Project

```
sd_stuff/
├── datahub/
│   ├── real_images/          # Ảnh thật
│   └── fake_images/          # Ảnh AI-generated
├── models/                   # Model layer (abstract pattern)
│   ├── __init__.py
│   ├── base_model.py        # Abstract base class
│   ├── inception_model.py   # InceptionV3 for FID
│   └── facenet_model.py     # FaceNet for similarity
├── metrics/                  # Metrics layer (abstract pattern)
│   ├── __init__.py
│   ├── base_metric.py       # Abstract base class
│   ├── fid_metric.py        # Metric A: FID
│   ├── cosine_similarity_metric.py  # Metric B: Cosine Similarity
│   └── tsne_metric.py       # Metric C: t-SNE
├── evaluate.py              # Main script (dependency injection)
├── example_usage.py         # Usage examples
├── requirements.txt
├── README.md
└── ARCHITECTURE.md          # Design pattern documentation
```

### 🎯 Design Pattern

- **Abstract Pattern**: Both Models and Metrics follow abstract base classes
- **Dependency Injection**: Metrics receive models from outside (not creating internally)
- **Separation of Concerns**: Models handle preprocessing/inference, Metrics handle evaluation logic

Xem chi tiết: [ARCHITECTURE.md](ARCHITECTURE.md)

## 🔧 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu trúc dữ liệu

#### Cho Metric A & B (FID và Cosine Similarity):
```
datahub/
├── real_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── fake_images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

#### Cho Metric C (t-SNE) - khuyến nghị:
```
datahub/
└── fake_images/
    ├── id_001/          # ID nhân vật 1
    │   ├── frontal.jpg
    │   ├── side_45.jpg
    │   └── ...
    ├── id_002/          # ID nhân vật 2
    │   ├── frontal.jpg
    │   ├── side_45.jpg
    │   └── ...
    └── ...
```

Hoặc dùng naming convention:
```
fake_images/
├── id_001_frontal.jpg
├── id_001_side_45.jpg
├── id_002_frontal.jpg
└── ...
```

## 🚀 Sử dụng

### Chạy tất cả metrics

```bash
python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images
```

### Chạy từng metric riêng lẻ

```bash
# Chỉ chạy FID
python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics fid

# Chỉ chạy Cosine Similarity
python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics cosine

# Chỉ chạy t-SNE
python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics tsne

# Chạy FID và t-SNE
python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics fid tsne
```

### Chỉ định output directory

```bash
python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --output ./my_results
```

## 📊 Metrics Chi tiết

### Metric A: FID (Fréchet Inception Distance)

**Mục đích**: Đánh giá độ chân thực của ảnh sinh so với ảnh thật

**Cách hoạt động**:
- Sử dụng InceptionV3 để trích xuất features
- Tính khoảng cách Fréchet giữa phân phối features của ảnh thật và ảnh sinh
- **Ảnh phải 1024x1024 selfie** (theo yêu cầu senior)

**Đánh giá**:
- `FID < 50`: ✓ **Đạt chuẩn** - Excellent
- `FID 50-100`: ✓ Good
- `FID 100-200`: ⚠ Acceptable
- `FID > 200`: ✗ **Thất bại** - Poor

**Output**: 
- `fid_results_TIMESTAMP.json`

### Metric B: Cosine Similarity

**Mục đích**: Đánh giá tính nhất quán của cùng một ID khi thay đổi góc chụp/ánh sáng

**Cách hoạt động**:
- Sử dụng FaceNet (InceptionResnetV1) để trích xuất face embeddings
- Tính cosine similarity giữa các cặp ảnh
- So sánh ảnh của cùng ID với các góc/điều kiện khác nhau

**Đánh giá**:
- `Similarity > 0.7`: ✓ **Đạt chuẩn** - Cùng một người, độ tin cậy cao
- `Similarity 0.5-0.7`: ⚠ **Cần kiểm tra** - Cùng người nhưng có biến thiên
- `Similarity < 0.5`: ✗ **Thất bại** - Model coi là 2 người khác nhau

**Output**: 
- `cosine_similarity_results_TIMESTAMP.json`

### Metric C: t-SNE Visualization

**Mục đích**: Đánh giá khả năng phân tách các ID nhân vật ảo khác nhau

**Cách hoạt động**:
- Trích xuất face embeddings cho tất cả ảnh
- Áp dụng t-SNE để giảm chiều xuống 2D
- Visualize và tính separation ratio

**Đánh giá**:
- `Separation Ratio > 2.0`: ✓ **Đạt chuẩn** - Excellent separation
- `Separation Ratio 1.5-2.0`: ✓ Good separation
- `Separation Ratio < 1.5`: ✗ **Thất bại** - Poor separation (overlap nhiều)

**Output**: 
- `tsne_results_TIMESTAMP.json`
- `tsne_visualization_TIMESTAMP.png` - Biểu đồ 2D

## 📝 Output Format

### JSON Results Example

```json
{
  "timestamp": "20231208_143022",
  "real_path": "./datahub/real_images",
  "fake_path": "./datahub/fake_images",
  "metrics": {
    "fid": {
      "metric": "FID",
      "score": 45.32,
      "num_real_images": 100,
      "num_fake_images": 100,
      "interpretation": "✓ ĐẠT CHUẨN - Excellent..."
    },
    "cosine_similarity": {
      "metric": "Cosine Similarity",
      "average_similarity": 0.78,
      "std_similarity": 0.12,
      "min_similarity": 0.65,
      "max_similarity": 0.92,
      "interpretation": "✓ ĐẠT CHUẨN..."
    },
    "tsne": {
      "metric": "t-SNE",
      "num_ids": 5,
      "total_images": 50,
      "cluster_metrics": {
        "avg_intra_cluster_distance": 0.45,
        "avg_inter_cluster_distance": 1.23,
        "separation_ratio": 2.73
      },
      "interpretation": "✓ ĐẠT CHUẨN - Excellent..."
    }
  }
}
```

## 🎯 Use Cases

### Use Case 1: Đánh giá chất lượng tổng thể
```bash
python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images
```
→ Chạy tất cả 3 metrics để có overview hoàn chỉnh

### Use Case 2: Test tính nhất quán của 1 ID
```bash
python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics cosine
```
→ Kiểm tra xem các ảnh của cùng ID có consistent không

### Use Case 3: Visualize phân bố các ID
```bash
python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics tsne
```
→ Tạo biểu đồ 2D để quan sát sự phân tách

## 🔬 Technical Notes

### Image Requirements

- **Format**: JPG, PNG, BMP, WEBP
- **Size**: Khuyến nghị 1024x1024 (theo yêu cầu của senior về pixel-to-pixel comparison)
- **Type**: Selfie, chụp khuôn mặt rõ ràng
- **Số lượng**: 
  - FID: 50-100 ảnh mỗi tập (real & fake)
  - Cosine: Tối thiểu 2 ảnh cùng ID
  - t-SNE: 5-10 IDs, mỗi ID ~10 ảnh

### Model Dependencies

- **FID**: InceptionV3 (pretrained on ImageNet) - khởi tạo qua `InceptionModel`
- **Cosine Similarity**: FaceNet (InceptionResnetV1 pretrained on VGGFace2) - khởi tạo qua `FaceNetModel`
- **t-SNE**: Same as Cosine Similarity

### Architecture

Models và Metrics được tách biệt:
- **Models** (`models/`): Xử lý preprocessing, inference, postprocessing
- **Metrics** (`metrics/`): Business logic để đánh giá
- **Dependency Injection**: Models được inject vào Metrics từ bên ngoài
## 🐛 Troubleshooting

### Lỗi: "FIDMetric requires an InceptionModel instance"
→ Metrics cần model được inject từ bên ngoài. Xem `example_usage.py`

### Lỗi: "No images found"
→ Kiểm tra đường dẫn và format file (jpg, png, etc.)

- GPU khuyến nghị cho xử lý nhanh
- CPU vẫn chạy được nhưng chậm hơn
- Batch processing để tối ưu memory

## 🐛 Troubleshooting

### Lỗi: "No images found"
→ Kiểm tra đường dẫn và format file (jpg, png, etc.)

### Lỗi: "Not enough images"
→ Đảm bảo có đủ số lượng ảnh theo yêu cầu mỗi metric

### Lỗi: "facenet_pytorch not found"
→ Code tự động fallback về ResNet50, vẫn chạy được nhưng accuracy có thể thấp hơn

### FID score quá cao
→ Kiểm tra:
- Ảnh có đúng size 1024x1024?
- Ảnh có cùng style/domain không?
- Chất lượng ảnh sinh có tốt không?

## 📚 References

- [FID Paper](https://arxiv.org/abs/1706.08500)
- [FaceNet Paper](https://arxiv.org/abs/1503.03832)
- [t-SNE Paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

## 👥 Contributors

Developed for AI-generated children images evaluation project.

---

**Note**: Đây là code evaluation, không bao gồm phần generation ảnh. Chỉ đánh giá ảnh có sẵn trong `datahub`.
