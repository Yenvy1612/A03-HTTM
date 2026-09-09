# Web Demo

Web demo dùng trực tiếp ba classical ML pipeline tốt nhất trong thư mục `project/`:

- Diabetes: Random Forest.
- House Price: Linear Regression.
- E-commerce sentiment: Logistic Regression + TF-IDF.

## Price demo dự đoán `Price (in rupees)` (giá trên mỗi sqft theo cách diễn giải của dataset), sau đó nhân với diện tích để đưa ra tổng giá tham khảo.

## Chạy ứng dụng

Từ thư mục gốc Assignment 03:

```powershell
python web_demo/app.py
```

Mở `http://127.0.0.1:5000` trong trình duyệt.

Nếu môi trường chưa có thư viện:

```powershell
python -m pip install -r web_demo/requirements.txt
```

Các model phải được tạo trước bằng cách chạy notebook `diabetes-ml.ipynb`, `house-price-ml.ipynb` và `ecommerce-ml.ipynb`.
