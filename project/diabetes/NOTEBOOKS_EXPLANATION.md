# Giải thích 3 notebook deep learning

Tài liệu này mô tả mục đích, quy trình xử lý và kết quả của ba notebook trong thư mục hiện tại:

- [`diabetes-deep-learning-from-scratch-exp1.ipynb`](./diabetes-deep-learning-from-scratch-exp1.ipynb)
- [`diabetes-deep-learning-from-scratch-exp2.ipynb`](./diabetes-deep-learning-from-scratch-exp2.ipynb)
- [`diabetes-deep-learning-from-scratch-exp3.ipynb`](./diabetes-deep-learning-from-scratch-exp3.ipynb)

## 1. Mục tiêu project

Các notebook xây dựng một mô hình mạng nơ-ron nhân tạo từ đầu bằng NumPy để dự đoán biến nhị phân `Diabetes_binary` trong bộ dữ liệu `data/diabetes.csv`.

Project minh họa toàn bộ quy trình deep learning mà không dùng sẵn mô hình machine learning:

1. Đọc và kiểm tra dữ liệu.
2. Tách dữ liệu thành train, validation và test theo phương pháp stratified split.
3. Chuẩn hóa feature bằng mean và standard deviation của tập train.
4. Tự cài đặt forward propagation, binary cross-entropy, backpropagation và cập nhật tham số.
5. Huấn luyện bằng mini-batch gradient descent.
6. Đánh giá bằng accuracy, precision, recall, F1-score và confusion matrix.
7. Lưu model NumPy cùng các tham số chuẩn hóa để dự đoán dữ liệu mới.

## 2. Quy trình dùng chung

### Dữ liệu

- Số mẫu: `70,692`.
- Số feature đầu vào ban đầu: `21`.
- Biến mục tiêu: `Diabetes_binary`.
- Nhãn `0` và `1` được giữ cân bằng trong các tập dữ liệu.
- Tỷ lệ chia: `70%` train, `15%` validation và `15%` test.
- Kích thước thực tế: train `49,484`, validation `10,602`, test `10,606`.
- Seed cố định là `42` để kết quả có thể tái lập.

### Kiến trúc mạng

```text
Input (21 hoặc 10 feature) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
```

- `ReLU` được dùng ở hai hidden layer.
- `Sigmoid` tạo xác suất thuộc lớp tiểu đường.
- Binary cross-entropy là hàm loss.
- Batch size là `512` và số epoch là `50`.
- Ngưỡng phân lớp mặc định là `0.5`: xác suất từ `0.5` trở lên được dự đoán là lớp `1`.

Mean và standard deviation chỉ được tính từ tập train, sau đó áp dụng cho validation, test và dữ liệu bệnh nhân mới. Cách này tránh làm rò rỉ thông tin từ validation/test vào quá trình huấn luyện.

## 3. Experiment 1 — baseline

File: [`diabetes-deep-learning-from-scratch-exp1.ipynb`](./diabetes-deep-learning-from-scratch-exp1.ipynb)

Experiment 1 là mô hình baseline sử dụng toàn bộ `21` feature. Learning rate được đặt là `0.001` để làm mốc so sánh cho các experiment sau.

Notebook thực hiện kiểm tra dữ liệu, stratified split, chuẩn hóa, huấn luyện mạng `21 -> 64 -> 32 -> 1` và lưu loss curve, confusion matrix cùng model đã huấn luyện.

### Kết quả đã ghi trong notebook

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.7202 |
| Precision | 0.7275 |
| Recall | 0.7039 |
| F1-score | 0.7155 |
| Train loss cuối | 0.5523 |
| Validation loss cuối | 0.5537 |

Confusion matrix: `[[3905, 1398], [1570, 3733]]`, trong đó hàng là nhãn thực tế `0, 1` và cột là nhãn dự đoán `0, 1`.

## 4. Experiment 2 — thay đổi learning rate

File: [`diabetes-deep-learning-from-scratch-exp2.ipynb`](./diabetes-deep-learning-from-scratch-exp2.ipynb)

Experiment 2 giữ nguyên dữ liệu, kiến trúc mạng và số epoch của Experiment 1, nhưng tăng learning rate từ `0.001` lên `0.01`. Mục đích là quan sát ảnh hưởng của tốc độ cập nhật tham số đến khả năng hội tụ.

### Kết quả đã ghi trong notebook

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.7427 |
| Precision | 0.7289 |
| Recall | 0.7728 |
| F1-score | 0.7502 |
| Train loss cuối | 0.5058 |
| Validation loss cuối | 0.5144 |

Confusion matrix: `[[3779, 1524], [1205, 4098]]`.

So với Experiment 1, learning rate lớn hơn giúp loss giảm nhanh hơn và cải thiện accuracy, recall cũng như F1-score trong lần chạy hiện tại. Learning rate quá lớn vẫn có thể làm quá trình học dao động hoặc không ổn định trong các bài toán khác.

## 5. Experiment 3 — chọn feature

File: [`diabetes-deep-learning-from-scratch-exp3.ipynb`](./diabetes-deep-learning-from-scratch-exp3.ipynb)

Experiment 3 kiểm tra liệu có thể giảm số feature mà vẫn duy trì hiệu năng hay không. Notebook tính Pearson correlation giữa từng feature và target **chỉ trên tập train**, xếp hạng theo trị tuyệt đối của correlation, rồi chọn 10 feature đứng đầu.

### 10 feature được chọn

1. `GenHlth`
2. `HighBP`
3. `HighChol`
4. `BMI`
5. `Age`
6. `DiffWalk`
7. `Income`
8. `HeartDiseaseorAttack`
9. `PhysHlth`
10. `Education`

Kích thước đầu vào giảm từ `21` xuống `10`, nên kiến trúc mạng trở thành `10 -> 64 -> 32 -> 1`. Learning rate được giữ ở mức `0.01` như Experiment 2.

### Kết quả đã ghi trong notebook

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.7434 |
| Precision | 0.7249 |
| Recall | 0.7847 |
| F1-score | 0.7536 |
| Train loss cuối | 0.5075 |
| Validation loss cuối | 0.5166 |

Confusion matrix: `[[3724, 1579], [1142, 4161]]`.

Kết quả cho thấy chỉ dùng 10 feature vẫn đạt hiệu năng tương đương, thậm chí F1-score và recall cao hơn nhẹ so với Experiment 2. Tuy nhiên, correlation đơn biến chỉ thể hiện quan hệ tuyến tính riêng lẻ; nó không chứng minh quan hệ nhân quả và có thể bỏ qua tương tác giữa các feature.

## 6. So sánh các experiment

| Experiment | Số feature | Learning rate | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| Exp 1 — baseline | 21 | 0.001 | 0.7202 | 0.7275 | 0.7039 | 0.7155 |
| Exp 2 — learning rate | 21 | 0.01 | 0.7427 | 0.7289 | 0.7728 | 0.7502 |
| Exp 3 — feature selection | 10 | 0.01 | 0.7434 | 0.7249 | 0.7847 | 0.7536 |

Theo kết quả hiện tại, Experiment 3 là cấu hình tốt nhất theo F1-score và recall, đồng thời dùng ít feature hơn. Đây là kết luận trong phạm vi lần chạy và cách chia dữ liệu hiện tại; nên dùng cross-validation hoặc nhiều seed nếu cần đánh giá chắc chắn hơn.

## 7. File output được tạo

Sau khi chạy notebook, các file chính được lưu trong các thư mục sau:

| Loại | Experiment 1 | Experiment 2 | Experiment 3 |
|---|---|---|---|
| Loss curve | [`figures/loss_curve-exp1.png`](./figures/loss_curve-exp1.png) | [`figures/loss_curve-exp2.png`](./figures/loss_curve-exp2.png) | [`figures/loss_curve-exp3.png`](./figures/loss_curve-exp3.png) |
| Confusion matrix | [`figures/confusion_matrix-exp1.png`](./figures/confusion_matrix-exp1.png) | [`figures/confusion_matrix-exp2.png`](./figures/confusion_matrix-exp2.png) | [`figures/confusion_matrix-exp3.png`](./figures/confusion_matrix-exp3.png) |
| Model | [`models/diabetes_numpy_mlp-exp1.npz`](./models/diabetes_numpy_mlp-exp1.npz) | [`models/diabetes_numpy_mlp-exp2.npz`](./models/diabetes_numpy_mlp-exp2.npz) | [`models/diabetes_numpy_mlp-exp3.npz`](./models/diabetes_numpy_mlp-exp3.npz) |

Model `.npz` chứa weight, bias, mean, standard deviation và threshold. Riêng Experiment 3 còn lưu `selected_indices` và `selected_features` để dữ liệu mới được xử lý đúng thứ tự và đúng 10 feature đã chọn.

## 8. Cách chạy

Chạy từ thư mục `project/diabetes` để các đường dẫn tương đối trong notebook hoạt động đúng:

```powershell
cd project/diabetes
.\\.venv\\Scripts\\Activate.ps1
jupyter notebook
```

Mở notebook tương ứng và chạy các cell theo thứ tự từ trên xuống dưới. Các notebook không phải công cụ chẩn đoán y khoa; kết quả chỉ phục vụ mục đích học tập và minh họa quy trình xây dựng mạng nơ-ron từ đầu.
