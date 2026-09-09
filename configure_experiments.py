"""Make Exp1/2/3 materially distinct and add train-only feature selection to Exp3."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "project"


def source(cell):
    return "".join(cell.get("source", []))


def set_source(cell, value):
    cell["source"] = value.splitlines(True)


def replace(cell, old, new):
    value = source(cell)
    if old not in value:
        raise ValueError(f"Expected text not found: {old[:60]!r}")
    set_source(cell, value.replace(old, new))


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def clear_outputs(nb):
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def configure_house(exp):
    path = ROOT / "house-price-prediction" / f"house-price-prediction-exp{exp}.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    # Guarantee experiment-specific artifact names even if the source notebooks were copied.
    for cell in cells:
        if cell.get("cell_type") == "code":
            value = source(cell)
            for other in (1, 2, 3):
                value = value.replace(f"-exp{other}.", f"-exp{exp}.")
            set_source(cell, value)

    descriptions = {
        1: "Baseline: all engineered numeric and categorical features; MLP 64-32.",
        2: "Capacity experiment: all features; wider MLP 96-48, larger learning rate and smaller batch.",
        3: "Feature-selection experiment: select numeric features by train-only correlation and categorical features by train-only quality rules; compact MLP 48-24.",
    }
    cells.insert(0, markdown(f"# House Price Prediction — Experiment {exp}\n\n{descriptions[exp]}\n"))
    cells.insert(1, code(f'EXPERIMENT_NAME = "exp{exp}"\nEXPERIMENT_DESCRIPTION = {descriptions[exp]!r}\nprint(EXPERIMENT_NAME, EXPERIMENT_DESCRIPTION)\n'))

    # Locate cells by stable snippets after insertion.
    init_cell = next(c for c in cells if "def initialize_parameters" in source(c))
    train_cell = next(c for c in cells if "learning_rate =" in source(c) and "batch_size" in source(c))
    split_cell = next(c for c in cells if "train_df =" in source(c) and "test_df =" in source(c))
    metric_cell = next(c for c in cells if "house_metrics = pd.DataFrame" in source(c))

    if exp == 1:
        replace(train_cell, "epochs = 40", "epochs = 25")
    elif exp == 2:
        replace(init_cell, "hidden1=64,\n    hidden2=32", "hidden1=96,\n    hidden2=48")
        replace(train_cell, "learning_rate = 0.001", "learning_rate = 0.003")
        replace(train_cell, "epochs = 40", "epochs = 35")
        replace(train_cell, "batch_size = 1024", "batch_size = 512")
    else:
        replace(init_cell, "hidden1=64,\n    hidden2=32", "hidden1=48,\n    hidden2=24")
        replace(train_cell, "learning_rate = 0.001", "learning_rate = 0.002")
        replace(train_cell, "epochs = 40", "epochs = 35")
        selection = '''

# EXP3 FEATURE SELECTION (fit on training data only)
candidate_numeric_cols = numeric_cols.copy()
numeric_scores = {}
target_for_selection = np.log1p(train_df["price_clean"].to_numpy(dtype=float))
for col in candidate_numeric_cols:
    values = pd.to_numeric(train_df[col], errors="coerce")
    valid = values.notna()
    numeric_scores[col] = abs(np.corrcoef(values[valid], target_for_selection[valid])[0, 1]) if valid.sum() > 2 else 0.0
numeric_cols = sorted(candidate_numeric_cols, key=lambda c: np.nan_to_num(numeric_scores[c]), reverse=True)[:3]

candidate_categorical_cols = categorical_cols.copy()
categorical_cols = [
    col for col in candidate_categorical_cols
    if train_df[col].nunique(dropna=True) <= 100 and train_df[col].notna().mean() >= 0.60
]
selected_features = numeric_cols + categorical_cols
print("Train-only numeric correlation scores:", numeric_scores)
print("EXP3 selected numeric features:", numeric_cols)
print("EXP3 selected categorical features:", categorical_cols)
print("EXP3 final selected features:", selected_features)
'''
        set_source(split_cell, source(split_cell) + selection)

    # Replace the erroneous copied e-commerce final cell with experiment metadata.
    cells[-1] = code(f'''experiment_config = {{
    "experiment": "exp{exp}",
    "description": EXPERIMENT_DESCRIPTION,
    "numeric_features": list(numeric_cols),
    "categorical_features": list(categorical_cols),
    "feature_selection": {str(exp == 3)},
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "test_mae": float(test_mae),
    "test_rmse": float(test_rmse),
    "test_r2": float(test_r2),
}}
with open(OUTPUT_DIR / "house_price_config-exp{exp}.json", "w", encoding="utf-8") as f:
    json.dump(experiment_config, f, ensure_ascii=False, indent=2)
print("Saved experiment config:", experiment_config)
''')
    # Ensure metric name is experiment-specific.
    set_source(metric_cell, source(metric_cell).replace("house_price_metrics-exp1.csv", f"house_price_metrics-exp{exp}.csv"))
    clear_outputs(nb)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)


def configure_ecommerce(exp):
    path = ROOT / "e-commerce-comment-analytis" / f"e-commerce-analytics-exp{exp}.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    for cell in cells:
        if cell.get("cell_type") == "code":
            value = source(cell)
            for other in (1, 2, 3):
                value = value.replace(f"-exp{other}.", f"-exp{exp}.")
            set_source(cell, value)

    descriptions = {
        1: "Baseline: 2,000 most frequent TF-IDF terms; MLP 128-64.",
        2: "Representation/capacity experiment: 3,000 TF-IDF terms; wider MLP 160-64 and adjusted optimization.",
        3: "Feature-selection experiment: select 1,000 discriminative TF-IDF terms using train-only between-class scores; compact MLP 96-48.",
    }
    cells.insert(0, markdown(f"# E-commerce Sentiment — Experiment {exp}\n\n{descriptions[exp]}\n"))
    cells.insert(1, code(f'EXPERIMENT_NAME = "exp{exp}"\nEXPERIMENT_DESCRIPTION = {descriptions[exp]!r}\nprint(EXPERIMENT_NAME, EXPERIMENT_DESCRIPTION)\n'))
    vocab_cell = next(c for c in cells if "MAX_VOCAB =" in source(c))
    tfidf_cell = next(c for c in cells if "X_train = transform_tfidf" in source(c))
    init_cell = next(c for c in cells if "def initialize_parameters" in source(c))
    train_cell = next(c for c in cells if "learning_rate =" in source(c) and "batch_size" in source(c))

    if exp == 1:
        replace(train_cell, "epochs = 30", "epochs = 25")
        set_source(train_cell, source(train_cell).replace(" #exp2 0.05", ""))
    elif exp == 2:
        replace(vocab_cell, "MAX_VOCAB = 2000", "MAX_VOCAB = 3000")
        replace(init_cell, "hidden1=128,\n    hidden2=64", "hidden1=160,\n    hidden2=64")
        # Existing exp2 may already carry 0.05.
        value = source(train_cell)
        value = value.replace("learning_rate = 0.05 #exp2 0.05 ex3", "learning_rate = 0.03")
        value = value.replace("learning_rate = 0.01 #exp2 0.05", "learning_rate = 0.03")
        value = value.replace("epochs = 30", "epochs = 30").replace("batch_size = 256", "batch_size = 192")
        set_source(train_cell, value)
    else:
        replace(vocab_cell, "MAX_VOCAB = 2000", "MAX_VOCAB = 3000")
        replace(init_cell, "hidden1=128,\n    hidden2=64", "hidden1=96,\n    hidden2=48")
        value = source(train_cell)
        value = value.replace("learning_rate = 0.06 #exp2 0.05 ex3 0.06", "learning_rate = 0.025")
        value = value.replace("learning_rate = 0.01 #exp2 0.05", "learning_rate = 0.025")
        value = value.replace("epochs = 30", "epochs = 30").replace("batch_size = 256", "batch_size = 192")
        set_source(train_cell, value)
        selection = '''

# EXP3 FEATURE SELECTION (scores use training rows and labels only)
SELECTED_FEATURE_COUNT = 1000
class_means = np.vstack([X_train[y_train == cls].mean(axis=0) for cls in range(3)])
global_mean = X_train.mean(axis=0)
feature_scores = np.sum((class_means - global_mean) ** 2, axis=0)
selected_indices = np.argsort(feature_scores)[-SELECTED_FEATURE_COUNT:]
selected_indices.sort()

X_train = X_train[:, selected_indices]
X_val = X_val[:, selected_indices]
X_test = X_test[:, selected_indices]
vocab_words = vocab_words[selected_indices]
idf = idf[selected_indices]
vocab = {word: i for i, word in enumerate(vocab_words)}
selected_features = vocab_words.tolist()
print("EXP3 selected TF-IDF features:", len(selected_features))
print("Top selected terms:", sorted(zip(vocab_words, feature_scores[selected_indices]), key=lambda x: x[1], reverse=True)[:20])
print("Selected matrix shapes:", X_train.shape, X_val.shape, X_test.shape)
'''
        set_source(tfidf_cell, source(tfidf_cell) + selection)

    cells.extend([
        code(f'''ecommerce_metrics = pd.DataFrame({{
    "Class": class_names, "Precision": precision, "Recall": recall, "F1": f1
}})
ecommerce_metrics.to_csv(OUTPUT_DIR / "ecommerce_class_metrics-exp{exp}.csv", index=False)
summary_metrics = pd.DataFrame({{
    "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
    "Value": [accuracy, np.mean(precision), np.mean(recall), np.mean(f1)]
}})
summary_metrics.to_csv(OUTPUT_DIR / "ecommerce_summary_metrics-exp{exp}.csv", index=False)
display(ecommerce_metrics); display(summary_metrics)
'''),
        code(f'''experiment_config = {{
    "experiment": "exp{exp}",
    "description": EXPERIMENT_DESCRIPTION,
    "feature_selection": {str(exp == 3)},
    "input_features": ["Review Text"],
    "vocabulary_size": int(len(vocab_words)),
    "selected_feature_count": int(X_train.shape[1]),
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "accuracy": float(accuracy),
    "macro_f1": float(np.mean(f1)),
}}
with open(OUTPUT_DIR / "ecommerce_config-exp{exp}.json", "w", encoding="utf-8") as f:
    json.dump(experiment_config, f, ensure_ascii=False, indent=2)
print("Saved experiment config:", experiment_config)
''')
    ])
    clear_outputs(nb)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    for experiment in (1, 2, 3):
        configure_house(experiment)
        configure_ecommerce(experiment)
