# TỔNG HỢP PROJECT MACHINE LEARNING VÀ DEEP LEARNING

## 1. Tổng quan

Project gồm ba bài toán độc lập:

| Thư mục | Bài toán | Loại bài toán | Target chính |
|---|---|---|---|
| `diabetes/` | Dự đoán bệnh tiểu đường | Binary classification | `Diabetes_binary` |
| `house-price-prediction/` | Dự đoán giá bất động sản | Regression | Tổng giá hoặc giá trên đơn vị diện tích, tùy nhóm notebook |
| `e-commerce-comment-analytis/` | Phân loại cảm xúc đánh giá sản phẩm | Multiclass text classification | `sentiment` |

Mỗi bài toán có hai nhóm notebook:

1. **Deep Learning from scratch:** tự xây dựng mạng neural bằng NumPy, gồm ba experiment.
2. **Classical Machine Learning:** so sánh ba mô hình scikit-learn, lưu toàn bộ pipeline để có thể tái sử dụng khi triển khai.

Các thư mục con dùng thống nhất:

- `data/`: dữ liệu đầu vào.
- `figures/`: biểu đồ đánh giá mô hình.
- `models/`: model và thông tin preprocessing đã lưu.
- `outputs/`: bảng metrics và file cấu hình experiment.

---

## 2. Bài toán Diabetes

### 2.1. Mục tiêu

Dự đoán một người có mắc bệnh tiểu đường hay không:

- `0`: không mắc bệnh.
- `1`: mắc bệnh.

Đây là bài toán **phân loại nhị phân**. Dataset `data/diabetes.csv` có 70.692 dòng, target `Diabetes_binary` và 21 biến đầu vào về sức khỏe, lối sống và nhân khẩu học.

### 2.2. Các notebook

| Notebook | Nội dung |
|---|---|
| `diabetes-deep-learning-from-scratch-exp1.ipynb` | Baseline MLP, sử dụng đủ 21 features, learning rate 0,001. |
| `diabetes-deep-learning-from-scratch-exp2.ipynb` | Giữ dữ liệu và kiến trúc, tăng learning rate lên 0,01. |
| `diabetes-deep-learning-from-scratch-exp3.ipynb` | Chọn 10 features có tương quan cao nhất với target trên tập train. |
| `diabetes-ml.ipynb` | So sánh Logistic Regression, Decision Tree và Random Forest. |

### 2.3. Deep Learning experiments

Kiến trúc tổng quát:

```text
Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
```

| Experiment | Số features | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Exp1 | 21 | 0,7202 | 0,7275 | 0,7039 | 0,7155 |
| Exp2 | 21 | 0,7427 | 0,7289 | 0,7728 | 0,7502 |
| Exp3 | 10 | 0,7434 | 0,7249 | 0,7847 | **0,7536** |

Exp3 là cấu hình tốt nhất trong nhóm MLP theo F1 và Recall, đồng thời giảm đầu vào từ 21 xuống 10 features.

### 2.4. Classical Machine Learning

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 0,7428 | **0,7567** | **0,8175** |
| Logistic Regression | **0,7449** | 0,7549 | 0,8174 |
| Decision Tree | 0,7187 | 0,7335 | 0,7873 |

Random Forest được chọn làm best model theo F1.

### 2.5. File đầu ra chính

- `models/diabetes_numpy_mlp-exp1.npz` đến `exp3.npz`: các MLP NumPy.
- `models/logistic_regression.joblib`, `decision_tree.joblib`, `random_forest.joblib`: các pipeline ML.
- `models/best_diabetes_model.joblib`: model ML tốt nhất.
- `figures/loss_curve-exp*.png`: quá trình học của MLP.
- `figures/confusion_matrix-exp*.png`: confusion matrix từng experiment.
- `figures/ml_model_comparison.png`, `ml_roc_curve.png`, `ml_best_confusion_matrix.png`: biểu đồ so sánh ML.
- `outputs/diabetes_ml_results.csv`: bảng kết quả ML.
- `outputs/model_config.json`: cấu hình và danh sách features của pipeline ML.

---

## 3. Bài toán House Price Prediction

### 3.1. Mục tiêu

Dự đoán giá bất động sản từ thông tin diện tích, số phòng tắm, ban công, tầng, vị trí, tình trạng nội thất và loại giao dịch.

Dataset `data/house_prices.csv` có 187.531 dòng và 21 cột. Dữ liệu chứa nhiều trường dạng text và missing values nên cần:

- Chuyển `Amount(in rupees)` thành tổng giá dạng số.
- Chuyển diện tích về square feet.
- Tách tầng hiện tại và tổng số tầng.
- Impute numeric features.
- One-hot encode categorical features.
- Chỉ học preprocessing từ tập train để tránh data leakage.

### 3.2. Lưu ý về hai target

Hai nhóm notebook đang giải quyết hai biến giá khác nhau:

- Nhóm MLP Exp1–Exp3 dự đoán **tổng giá căn nhà**, được parse từ `Amount(in rupees)` thành `price_clean`.
- Notebook `house-price-ml.ipynb` dự đoán cột số có sẵn `Price (in rupees)`, thực tế gần với **giá trên đơn vị diện tích** trong dataset.

Vì target và đơn vị khác nhau, không so sánh trực tiếp RMSE giữa hai nhóm notebook.

### 3.3. Deep Learning experiments

| Experiment | Điểm khác biệt | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Exp1 | Baseline, toàn bộ features, MLP 64–32 | 4.446.477 | 10.787.093 | 0,4626 |
| Exp2 | MLP 96–48, learning rate 0,003, batch 512 | **3.212.899** | **9.298.298** | **0,6007** |
| Exp3 | Feature selection trên train, MLP 48–24 | 4.091.003 | 9.871.748 | 0,5499 |

Exp2 là mô hình MLP tốt nhất. Exp3 dùng feature selection để kiểm tra khả năng giảm số thuộc tính nhưng vẫn giữ hiệu năng tương đối tốt.

Features được Exp3 chọn:

- Numeric: `Bathroom`, `Balcony`, `total_floors`.
- Categorical: `Status`, `Transaction`, `Furnishing`, `facing`, `Ownership`, `location`.

### 3.4. Classical Machine Learning

Notebook `house-price-ml.ipynb` so sánh:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 2.456,37 | **3.910,86** | **0,3575** |
| Random Forest Regressor | **1.069,08** | 7.450,06 | -1,3316 |
| Decision Tree Regressor | 1.336,68 | 9.876,89 | -3,0980 |

Linear Regression được chọn theo RMSE. Dataset có một số giá trị ngoại lệ rất lớn nên RMSE và R² của các mô hình cây bị ảnh hưởng mạnh dù MAE thấp hơn.

### 3.5. File đầu ra chính

- `house-price-prediction-exp1.ipynb` đến `exp3.ipynb`: ba MLP experiments.
- `house-price-ml.ipynb`: classical regression comparison.
- `models/house_price_numpy_mlp-exp*.npz`: model MLP.
- `models/house_price_category_maps-exp*.json`: category maps cần cho inference.
- `models/linear_regression.joblib`, `decision_tree_regressor.joblib`, `random_forest_regressor.joblib`.
- `models/best_house_price_model.joblib`: pipeline regression tốt nhất.
- `figures/house_price_loss_curve-exp*.png`: loss curve.
- `figures/house_price_actual_vs_predicted-exp*.png`: giá thực tế và dự đoán.
- `figures/house_price_residual_distribution-exp*.png`: phân phối residual.
- `figures/house_price_residual_vs_predicted-exp*.png`: residual theo dự đoán.
- `outputs/house_price_metrics-exp*.csv`: metrics từng MLP experiment.
- `outputs/house_price_config-exp*.json`: cấu hình và features từng experiment.
- `outputs/house_price_ml_results.csv`: kết quả classical ML.

---

## 4. Bài toán E-commerce Comment Analysis

### 4.1. Mục tiêu

Phân loại nội dung trong `Review Text` thành ba nhóm cảm xúc:

- Rating 1–2 → `Negative`.
- Rating 3 → `Neutral`.
- Rating 4–5 → `Positive`.

Đây là bài toán **multiclass text classification**. Dataset `data/Womens Clothing E-Commerce Reviews.csv` có 23.486 dòng.

`Rating` chỉ được dùng để tạo target và không được đưa vào feature huấn luyện, vì như vậy sẽ cung cấp trực tiếp đáp án cho model. Input chính là `Review Text` được chuyển thành vector TF-IDF.

### 4.2. Deep Learning experiments

| Experiment | Điểm khác biệt | Accuracy | Macro F1 |
|---|---|---:|---:|
| Exp1 | 2.000 TF-IDF features, MLP 128–64 | 0,7705 | 0,2901 |
| Exp2 | 3.000 TF-IDF features, MLP 160–64 | **0,8114** | **0,5277** |
| Exp3 | Chọn 1.000/3.000 TF-IDF features, MLP 96–48 | 0,7899 | 0,4135 |

Exp2 là MLP tốt nhất. Exp1 gần như chỉ dự đoán lớp Positive do dữ liệu mất cân bằng. Exp2 cải thiện khả năng nhận diện Negative và Neutral.

Exp3 thực hiện feature selection theo quy trình:

```text
3.000 TF-IDF candidates
        ↓
Tính between-class score chỉ trên training set
        ↓
Chọn 1.000 terms phân biệt nhất
        ↓
Train MLP 96–48
```

### 4.3. Classical Machine Learning

Notebook `ecommerce-ml.ipynb` lưu TF-IDF và classifier chung trong một pipeline.

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 0,7901 | **0,8067** | **0,6213** |
| Random Forest | **0,8105** | 0,7851 | 0,5501 |
| Decision Tree | 0,6256 | 0,6672 | 0,4539 |

Logistic Regression được chọn theo Macro F1 vì metric này đánh giá công bằng hơn giữa ba lớp khi dữ liệu mất cân bằng.

### 4.4. File đầu ra chính

- `e-commerce-analytics-exp1.ipynb` đến `exp3.ipynb`: ba MLP experiments.
- `ecommerce-ml.ipynb`: classical ML comparison.
- `models/ecommerce_sentiment_numpy_mlp-exp*.npz`: model MLP cùng vocabulary và IDF.
- `models/logistic_regression.joblib`, `decision_tree.joblib`, `random_forest.joblib`.
- `models/best_ecommerce_model.joblib`: pipeline sentiment tốt nhất.
- `figures/ecommerce_loss_curve-exp*.png`: loss curve.
- `figures/ecommerce_confusion_matrix-exp*.png`: confusion matrix.
- `figures/ecommerce_class_metrics-exp*.png`: Precision, Recall và F1 từng lớp.
- `outputs/ecommerce_class_metrics-exp*.csv`: metrics theo lớp.
- `outputs/ecommerce_summary_metrics-exp*.csv`: metrics tổng hợp.
- `outputs/ecommerce_config-exp*.json`: cấu hình, vocabulary size và trạng thái feature selection.
- `outputs/ecommerce_ml_results.csv`: kết quả classical ML.

---

## 5. So sánh ba bài toán

| Nội dung | Diabetes | House Price | E-commerce |
|---|---|---|---|
| Loại target | Nhị phân | Số liên tục | Ba lớp |
| Dữ liệu đầu vào | Numeric/ordinal | Numeric + categorical + text cần parse | Văn bản review |
| Output layer MLP | Sigmoid | Linear | Softmax |
| Loss | Binary cross-entropy | Mean squared error | Multiclass cross-entropy |
| Metrics chính | F1, ROC-AUC | MAE, RMSE, R² | Macro F1, Accuracy |
| Best MLP experiment | Exp3 | Exp2 | Exp2 |
| Best classical model | Random Forest | Linear Regression | Logistic Regression |

---

## 6. Quy trình chung

```text
Load dataset
    ↓
EDA và cleaning
    ↓
Feature engineering
    ↓
Train/validation/test split
    ↓
Fit preprocessing trên train
    ↓
Train model
    ↓
Đánh giá trên test
    ↓
Lưu model, figures, metrics và config
    ↓
Load lại model và kiểm tra inference
```

Các notebook sử dụng `SEED = 42` hoặc `random_state = 42` để kết quả có thể tái lập. Với Exp3, việc chọn features luôn dựa trên tập train, không sử dụng validation hoặc test.

## 7. Kết luận chung

- Diabetes cho thấy feature selection có thể giảm gần một nửa số biến mà vẫn duy trì hoặc cải thiện F1.
- House Price cho thấy việc tăng capacity và điều chỉnh optimization ở Exp2 hiệu quả hơn giảm features trong lần chạy hiện tại.
- E-commerce cho thấy dữ liệu mất cân bằng cần được đánh giá bằng Macro F1, không nên chỉ dựa trên Accuracy.
- Classical ML cung cấp baseline dễ triển khai; các pipeline `.joblib` đã chứa cả preprocessing và model.
- MLP NumPy giúp minh họa rõ forward propagation, loss, backpropagation và mini-batch gradient descent, nhưng cần lưu riêng metadata preprocessing để inference đúng.

Kết quả chỉ phản ánh lần chia dữ liệu với seed hiện tại. Nếu cần kết luận chắc chắn hơn, bước tiếp theo nên là cross-validation, chạy nhiều random seed và tuning hyperparameters có kiểm soát.
