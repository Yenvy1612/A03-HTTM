from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent / "project"

DIABETES_MODEL = PROJECT_DIR / "diabetes" / "models" / "best_diabetes_model.joblib"
HOUSE_MODEL = PROJECT_DIR / "house-price-prediction" / "models" / "best_house_price_model.joblib"
ECOMMERCE_MODEL = PROJECT_DIR / "e-commerce-comment-analytis" / "models" / "best_ecommerce_model.joblib"

app = Flask(__name__)

DIABETES_FIELDS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

HOUSE_FIELDS = [
    "Carpet Area Numeric", "Bathroom Numeric", "Balcony Numeric", "BHK",
    "location", "Status", "Transaction", "Furnishing", "facing", "Ownership",
]


@lru_cache(maxsize=3)
def load_model(path):
    return joblib.load(path)


def json_error(message, status=400):
    return jsonify({"ok": False, "error": message}), status


def as_float(payload, field, minimum=None, maximum=None):
    try:
        value = float(payload[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Trường '{field}' không hợp lệ.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Trường '{field}' phải lớn hơn hoặc bằng {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"Trường '{field}' phải nhỏ hơn hoặc bằng {maximum}.")
    return value


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    files = {
        "diabetes": DIABETES_MODEL.exists(),
        "house": HOUSE_MODEL.exists(),
        "ecommerce": ECOMMERCE_MODEL.exists(),
    }
    return jsonify({"ok": all(files.values()), "models": files})


@app.post("/api/predict/diabetes")
def predict_diabetes():
    payload = request.get_json(silent=True) or {}
    try:
        values = {field: as_float(payload, field) for field in DIABETES_FIELDS}
        values["BMI"] = as_float(payload, "BMI", 10, 100)
        values["GenHlth"] = as_float(payload, "GenHlth", 1, 5)
        values["MentHlth"] = as_float(payload, "MentHlth", 0, 30)
        values["PhysHlth"] = as_float(payload, "PhysHlth", 0, 30)
        values["Age"] = as_float(payload, "Age", 1, 13)
        values["Education"] = as_float(payload, "Education", 1, 6)
        values["Income"] = as_float(payload, "Income", 1, 8)
    except ValueError as exc:
        return json_error(str(exc))

    model = load_model(str(DIABETES_MODEL))
    frame = pd.DataFrame([values], columns=DIABETES_FIELDS)
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    return jsonify({
        "ok": True,
        "prediction": prediction,
        "label": "Nguy cơ tiểu đường" if prediction else "Không thuộc nhóm tiểu đường",
        "probability": probability,
        "model": "Random Forest",
        "note": "Kết quả minh họa học thuật, không thay thế chẩn đoán y khoa.",
    })


@app.post("/api/predict/house")
def predict_house():
    payload = request.get_json(silent=True) or {}
    try:
        values = {
            "Carpet Area Numeric": as_float(payload, "Carpet Area Numeric", 1),
            "Bathroom Numeric": as_float(payload, "Bathroom", 0),
            "Balcony Numeric": as_float(payload, "Balcony", 0),
            "BHK": as_float(payload, "BHK", 1),
            "location": str(payload.get("location", "")).strip(),
            "Status": str(payload.get("Status", "Ready to Move")).strip(),
            "Transaction": str(payload.get("Transaction", "Resale")).strip(),
            "Furnishing": str(payload.get("Furnishing", "Unfurnished")).strip(),
            "facing": str(payload.get("facing", "East")).strip(),
            "Ownership": str(payload.get("Ownership", "Freehold")).strip(),
        }
        if not values["location"]:
            raise ValueError("Vui lòng nhập địa điểm.")
    except ValueError as exc:
        return json_error(str(exc))

    model = load_model(str(HOUSE_MODEL))
    frame = pd.DataFrame([values], columns=HOUSE_FIELDS)
    prediction = max(0.0, float(model.predict(frame)[0]))
    estimated_total = prediction * values["Carpet Area Numeric"]
    return jsonify({
        "ok": True,
        "price_per_sqft": prediction,
        "estimated_total": estimated_total,
        "model": "Linear Regression",
        "unit_note": "Model dự đoán cột Price (in rupees), được diễn giải là giá trên mỗi sqft.",
    })


@app.post("/api/predict/sentiment")
def predict_sentiment():
    payload = request.get_json(silent=True) or {}
    review = str(payload.get("review", "")).strip()
    if len(review) < 3:
        return json_error("Nội dung đánh giá cần ít nhất 3 ký tự.")

    model = load_model(str(ECOMMERCE_MODEL))
    prediction = str(model.predict(pd.Series([review]))[0])
    probabilities = model.predict_proba(pd.Series([review]))[0]
    scores = {str(label): float(score) for label, score in zip(model.classes_, probabilities)}
    labels = {"Negative": "Tiêu cực", "Neutral": "Trung lập", "Positive": "Tích cực"}
    return jsonify({
        "ok": True,
        "prediction": prediction,
        "label": labels.get(prediction, prediction),
        "scores": scores,
        "model": "Logistic Regression + TF-IDF",
    })


@app.errorhandler(404)
def not_found(_error):
    return json_error("Không tìm thấy đường dẫn.", 404)


@app.errorhandler(500)
def server_error(_error):
    return json_error("Có lỗi khi chạy model. Hãy kiểm tra model đã được tạo đầy đủ.", 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
