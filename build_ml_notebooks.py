"""Generate the three consistent ML baseline notebooks for Assignment 03."""
from pathlib import Path
import textwrap
import json

try:
    import nbformat as nbf
except ModuleNotFoundError:
    class _V4:
        @staticmethod
        def new_markdown_cell(source):
            return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}

        @staticmethod
        def new_code_cell(source):
            return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}

        @staticmethod
        def new_notebook():
            return {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

    class _NBFormat:
        v4 = _V4()

        @staticmethod
        def write(notebook, path):
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    nbf = _NBFormat()

ROOT = Path(__file__).resolve().parent / "project"


def md(title, body=""):
    return nbf.v4.new_markdown_cell(f"## {title}\n\n{body}".strip())


def code(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def write_notebook(folder, filename, title, cells):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [nbf.v4.new_markdown_cell(f"# {title}\n\nA reproducible baseline experiment following the shared 25-section assignment workflow.")] + cells
    path = ROOT / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(path)


COMMON_IMPORTS = r'''
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import display

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
'''


def diabetes():
    features = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
                "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
                "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
    cells = [
        md("01. Problem Definition", "Predict whether a respondent has diabetes (binary classification) and compare three classical ML model families."),
        md("02. Import Libraries"), code(COMMON_IMPORTS + '''
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
'''),
        md("03. Experiment Configuration"), code(f'''
EXPERIMENT_NAME = "ml_baseline"
TARGET = "Diabetes_binary"
SELECTED_FEATURES = {features!r}
BEST_METRIC = "F1"
'''),
        md("04. Paths"), code('''
PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / "data" / "diabetes.csv").exists():
    PROJECT_DIR = Path("project/diabetes")
DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
for directory in (FIGURE_DIR, MODEL_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print("Project directory:", PROJECT_DIR.resolve())
'''),
        md("05. Load Dataset"), code('''
df = pd.read_csv(DATA_DIR / "diabetes.csv")
print("Shape:", df.shape)
display(df.head())
'''),
        md("06. Data Understanding"), code('''
df.info()
display(df.describe().T)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
display(df[TARGET].value_counts().rename_axis("class").to_frame("count"))
'''),
        md("07. Data Cleaning", "Zeros are valid for the binary health indicators in this BRFSS dataset. Missing-value handling remains inside each pipeline to prevent leakage."), code('''
df = df.drop_duplicates().copy()
assert TARGET in df and set(SELECTED_FEATURES).issubset(df.columns)
'''),
        md("08. Feature Engineering", "The supplied BRFSS columns are already encoded numeric/ordinal features; no target-derived features are created."),
        md("09. Feature Selection"), code('''
X = df[SELECTED_FEATURES].copy()
y = df[TARGET].astype(int)
print("Selected features ({}):".format(X.shape[1]), SELECTED_FEATURES)
display(y.value_counts(normalize=True).rename("proportion"))
'''),
        md("10. Train/Test Split"), code('''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)
'''),
        md("11. Preprocessing"), code('''
scaled_prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
tree_prep = Pipeline([("imputer", SimpleImputer(strategy="median"))])
'''),
        md("12. Define 3 ML Models"), code('''
models = {
    "Logistic Regression": Pipeline([("prep", scaled_prep), ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "Decision Tree": Pipeline([("prep", tree_prep), ("model", DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE))]),
    "Random Forest": Pipeline([("prep", tree_prep), ("model", RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE))]),
}
model_files = {"Logistic Regression": "logistic_regression.joblib", "Decision Tree": "decision_tree.joblib", "Random Forest": "random_forest.joblib"}
'''),
        md("13. Train Models"), code('''
predictions, probabilities = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
    probabilities[name] = model.predict_proba(X_test)[:, 1]
    print("Trained:", name)
'''),
        md("14. Evaluation Metrics"), code('''
rows = []
for name in models:
    pred, prob = predictions[name], probabilities[name]
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred, zero_division=0),
                 "Recall": recall_score(y_test, pred, zero_division=0),
                 "F1": f1_score(y_test, pred, zero_division=0),
                 "ROC-AUC": roc_auc_score(y_test, prob)})
'''),
        md("15. Model Comparison Table"), code('''
results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
display(results.style.format({c: "{:.4f}" for c in results.columns if c != "Model"}))
'''),
        md("16. Visualization"), code('''
ax = results.set_index("Model").plot(kind="bar", figsize=(11, 5), ylim=(0, 1), rot=0)
ax.set_title("Diabetes model comparison"); ax.set_ylabel("Score"); plt.tight_layout()
plt.savefig(FIGURE_DIR / "ml_model_comparison.png", dpi=160); plt.show()

plt.figure(figsize=(7, 6))
for name, prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC curves"); plt.legend(); plt.tight_layout()
plt.savefig(FIGURE_DIR / "ml_roc_curve.png", dpi=160); plt.show()
'''),
        md("17. Select Best Model"), code('''
best_name = results.iloc[0]["Model"]
best_model = models[best_name]
cm = confusion_matrix(y_test, predictions[best_name])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"Best model confusion matrix: {best_name}"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
plt.savefig(FIGURE_DIR / "ml_best_confusion_matrix.png", dpi=160); plt.show()
print("Best model by F1:", best_name)
'''),
        md("18. Save 3 Models"), code('''
model_paths = {}
for name, model in models.items():
    path = MODEL_DIR / model_files[name]
    joblib.dump(model, path); model_paths[name] = path
'''),
        md("19. Save Best Model"), code('''
best_path = MODEL_DIR / "best_diabetes_model.joblib"
joblib.dump(best_model, best_path)
print(best_path)
'''),
        md("20. Save Results"), code('''
results.to_csv(OUTPUT_DIR / "diabetes_ml_results.csv", index=False)
'''),
        md("21. Save Feature/Config"), code('''
config = {"experiment": EXPERIMENT_NAME, "task": "binary_classification", "target": TARGET,
          "features": SELECTED_FEATURES, "models": list(models), "best_metric": BEST_METRIC,
          "best_model": best_name, "random_state": RANDOM_STATE}
with open(OUTPUT_DIR / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
'''),
        md("22. Reload 3 Models"), code('''
loaded_models = {name: joblib.load(path) for name, path in model_paths.items()}
'''),
        md("23. Predict Test Sample"), code('''
sample = X_test.iloc[[0]]; actual = int(y_test.iloc[0])
verification = pd.DataFrame([{"Model": name, "Prediction": int(model.predict(sample)[0]), "Actual": actual}
                             for name, model in loaded_models.items()])
display(verification)
'''),
        md("24. Verify Saved Models"), code('''
for name in models:
    np.testing.assert_array_equal(loaded_models[name].predict(sample), models[name].predict(sample))
assert joblib.load(best_path).predict(sample).shape == (1,)
print("[OK] All saved models loaded and predicted successfully.")
'''),
        md("25. Conclusion"), code('''
print(f"Best model: {best_name} | F1={results.iloc[0]['F1']:.4f} | ROC-AUC={results.iloc[0]['ROC-AUC']:.4f}")
print("All metrics, plots, configuration, and deployable pipelines were saved inside the diabetes folder.")
'''),
    ]
    write_notebook("diabetes", "diabetes-ml.ipynb", "Diabetes — Classical ML Baselines", cells)


def house():
    cells = [
        md("01. Problem Definition", "Predict the numeric house price per square-foot field and compare three regression models."),
        md("02. Import Libraries"), code(COMMON_IMPORTS + '''
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
'''),
        md("03. Experiment Configuration"), code('''
EXPERIMENT_NAME = "ml_baseline"
TARGET = "Price (in rupees)"
NUMERIC_FEATURES = ["Carpet Area Numeric", "Bathroom Numeric", "Balcony Numeric", "BHK"]
CATEGORICAL_FEATURES = ["location", "Status", "Transaction", "Furnishing", "facing", "Ownership"]
SELECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
BEST_METRIC = "RMSE"
'''),
        md("04. Paths"), code('''
PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / "data" / "house_prices.csv").exists(): PROJECT_DIR = Path("project/house-price-prediction")
DATA_DIR, FIGURE_DIR, MODEL_DIR, OUTPUT_DIR = (PROJECT_DIR / x for x in ("data", "figures", "models", "outputs"))
for directory in (FIGURE_DIR, MODEL_DIR, OUTPUT_DIR): directory.mkdir(parents=True, exist_ok=True)
print("Project directory:", PROJECT_DIR.resolve())
'''),
        md("05. Load Dataset"), code('''
df = pd.read_csv(DATA_DIR / "house_prices.csv")
print("Shape:", df.shape); display(df.head())
'''),
        md("06. Data Understanding"), code('''
df.info(); display(df.describe(include="all").T)
print("Duplicate rows:", int(df.duplicated().sum()))
display(df.isna().sum().sort_values(ascending=False).head(12).to_frame("missing"))
'''),
        md("07. Data Cleaning"), code('''
df = df.drop_duplicates().copy()
df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
'''),
        md("08. Feature Engineering", "Parse numeric area/bathroom/balcony values and BHK from messy text. Parsing happens before splitting, but all learned imputing/encoding/scaling stays inside the pipelines."), code(r'''
df["Carpet Area Numeric"] = pd.to_numeric(df["Carpet Area"].str.extract(r"([\d,.]+)")[0].str.replace(",", "", regex=False), errors="coerce")
df["Bathroom Numeric"] = pd.to_numeric(df["Bathroom"].str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
df["Balcony Numeric"] = pd.to_numeric(df["Balcony"].str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
df["BHK"] = pd.to_numeric(df["Title"].str.extract(r"(\d+(?:\.\d+)?)\s*BHK", flags=2)[0], errors="coerce")
'''),
        md("09. Feature Selection"), code('''
X = df[SELECTED_FEATURES].copy(); y = df[TARGET].astype(float)
print("Rows:", len(X), "Features:", SELECTED_FEATURES); display(y.describe())
'''),
        md("10. Train/Test Split"), code('''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=RANDOM_STATE)
print("Train:", X_train.shape, "Test:", X_test.shape)
'''),
        md("11. Preprocessing"), code('''
numeric_scaled = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
numeric_tree = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))])
linear_prep = ColumnTransformer([("num", numeric_scaled, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])
tree_prep = ColumnTransformer([("num", numeric_tree, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])
'''),
        md("12. Define 3 ML Models"), code('''
models = {
 "Linear Regression": Pipeline([("prep", linear_prep), ("model", LinearRegression())]),
 "Decision Tree Regressor": Pipeline([("prep", tree_prep), ("model", DecisionTreeRegressor(max_depth=18, min_samples_leaf=5, random_state=RANDOM_STATE))]),
 "Random Forest Regressor": Pipeline([("prep", tree_prep), ("model", RandomForestRegressor(n_estimators=100, max_depth=22, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE))]),
}
model_files = {"Linear Regression":"linear_regression.joblib", "Decision Tree Regressor":"decision_tree_regressor.joblib", "Random Forest Regressor":"random_forest_regressor.joblib"}
'''),
        md("13. Train Models"), code('''
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train); predictions[name] = model.predict(X_test); print("Trained:", name)
'''),
        md("14. Evaluation Metrics"), code('''
rows=[]
for name, pred in predictions.items():
    rows.append({"Model":name, "MAE":mean_absolute_error(y_test,pred), "RMSE":mean_squared_error(y_test,pred,squared=False), "R2":r2_score(y_test,pred)})
'''),
        md("15. Model Comparison Table"), code('''
results=pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True); display(results)
'''),
        md("16. Visualization"), code('''
for metric, filename, color in [("MAE","mae_comparison.png","steelblue"),("RMSE","rmse_comparison.png","darkorange"),("R2","r2_comparison.png","seagreen")]:
    plt.figure(figsize=(9,4)); sns.barplot(data=results,x="Model",y=metric,color=color); plt.title(f"{metric} comparison"); plt.xticks(rotation=10); plt.tight_layout(); plt.savefig(FIGURE_DIR/filename,dpi=160); plt.show()
'''),
        md("17. Select Best Model"), code('''
best_name=results.iloc[0]["Model"]; best_model=models[best_name]; best_pred=predictions[best_name]
plt.figure(figsize=(6,6)); plt.scatter(y_test,best_pred,alpha=.15,s=10); bounds=[min(y_test.min(),best_pred.min()),max(y_test.max(),best_pred.max())]; plt.plot(bounds,bounds,"r--"); plt.xlabel("Actual price"); plt.ylabel("Predicted price"); plt.title(f"Actual vs predicted: {best_name}"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"actual_vs_predicted.png",dpi=160); plt.show()
print("Best model by RMSE:",best_name)
'''),
        md("18. Save 3 Models"), code('''
model_paths={}
for name,model in models.items():
    path=MODEL_DIR/model_files[name]; joblib.dump(model,path); model_paths[name]=path
'''),
        md("19. Save Best Model"), code('''
best_path=MODEL_DIR/"best_house_price_model.joblib"; joblib.dump(best_model,best_path)
'''),
        md("20. Save Results"), code('''
results.to_csv(OUTPUT_DIR/"house_price_ml_results.csv",index=False)
'''),
        md("21. Save Feature/Config"), code('''
config={"experiment":EXPERIMENT_NAME,"task":"regression","target":TARGET,"features":SELECTED_FEATURES,"models":list(models),"best_metric":BEST_METRIC,"best_model":best_name,"random_state":RANDOM_STATE}
with open(OUTPUT_DIR/"model_config.json","w",encoding="utf-8") as f: json.dump(config,f,indent=2,ensure_ascii=False)
'''),
        md("22. Reload 3 Models"), code('''
loaded_models={name:joblib.load(path) for name,path in model_paths.items()}
'''),
        md("23. Predict Test Sample"), code('''
sample=X_test.iloc[[0]]; actual=float(y_test.iloc[0]); verification=[]
for name,model in loaded_models.items():
    pred=float(model.predict(sample)[0]); verification.append({"Model":name,"Predicted Price":pred,"Actual Price":actual,"Absolute Error":abs(pred-actual)})
display(pd.DataFrame(verification))
'''),
        md("24. Verify Saved Models"), code('''
for name in models: np.testing.assert_allclose(loaded_models[name].predict(sample),models[name].predict(sample))
assert joblib.load(best_path).predict(sample).shape==(1,)
print("[OK] All saved models loaded and predicted successfully.")
'''),
        md("25. Conclusion"), code('''
print(f"Best model: {best_name} | RMSE={results.iloc[0]['RMSE']:.2f} | R²={results.iloc[0]['R2']:.4f}")
print("Saved deployable preprocessing+model pipelines; lower MAE/RMSE and higher R² are better.")
'''),
    ]
    write_notebook("house-price-prediction", "house-price-ml.ipynb", "House Price — Regression Baselines", cells)


def ecommerce():
    cells = [
        md("01. Problem Definition", "Infer Negative, Neutral, or Positive sentiment from review text and compare three classical classifiers."),
        md("02. Import Libraries"), code(COMMON_IMPORTS + '''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
'''),
        md("03. Experiment Configuration"), code('''
EXPERIMENT_NAME="ml_baseline"; TEXT_COL="Review Text"; RATING_COL="Rating"; TARGET="sentiment"
SELECTED_FEATURES=[TEXT_COL]; BEST_METRIC="Macro F1"
'''),
        md("04. Paths"), code('''
PROJECT_DIR=Path.cwd()
if not (PROJECT_DIR/"data"/"Womens Clothing E-Commerce Reviews.csv").exists(): PROJECT_DIR=Path("project/e-commerce-comment-analytis")
DATA_DIR,FIGURE_DIR,MODEL_DIR,OUTPUT_DIR=(PROJECT_DIR/x for x in ("data","figures","models","outputs"))
for directory in (FIGURE_DIR,MODEL_DIR,OUTPUT_DIR): directory.mkdir(parents=True,exist_ok=True)
print("Project directory:",PROJECT_DIR.resolve())
'''),
        md("05. Load Dataset"), code('''
df=pd.read_csv(DATA_DIR/"Womens Clothing E-Commerce Reviews.csv"); print("Shape:",df.shape); display(df.head())
'''),
        md("06. Data Understanding"), code('''
df.info(); display(df.describe(include="all").T); print("Missing review text:",int(df[TEXT_COL].isna().sum())); print("Duplicate review texts:",int(df[TEXT_COL].duplicated().sum()))
'''),
        md("07. Data Cleaning"), code('''
df=df.dropna(subset=[TEXT_COL,RATING_COL]).drop_duplicates(subset=[TEXT_COL]).copy(); df[TEXT_COL]=df[TEXT_COL].astype(str).str.strip(); df=df[df[TEXT_COL].str.len()>0]
print("Clean rows:",len(df))
'''),
        md("08. Feature Engineering", "Rating creates the research label only and is deliberately excluded from model inputs to prevent target leakage."), code('''
df[TARGET]=pd.cut(df[RATING_COL],bins=[0,2,3,5],labels=["Negative","Neutral","Positive"])
df["Review Length"]=df[TEXT_COL].str.len(); df["Word Count"]=df[TEXT_COL].str.split().str.len()
'''),
        md("09. Feature Selection"), code('''
X=df[TEXT_COL].copy(); y=df[TARGET].astype(str); print("Selected input:",SELECTED_FEATURES); display(y.value_counts().to_frame("count"))
plt.figure(figsize=(6,4)); sns.countplot(x=y,order=["Negative","Neutral","Positive"]); plt.title("Sentiment class distribution"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"class_distribution.png",dpi=160); plt.show()
'''),
        md("10. Train/Test Split"), code('''
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=RANDOM_STATE,stratify=y); print("Train:",len(X_train),"Test:",len(X_test))
'''),
        md("11. Preprocessing"), code('''
def make_tfidf(): return TfidfVectorizer(max_features=12000,ngram_range=(1,2),min_df=2,sublinear_tf=True,strip_accents="unicode")
'''),
        md("12. Define 3 ML Models"), code('''
models={
 "Logistic Regression":Pipeline([("tfidf",make_tfidf()),("model",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=RANDOM_STATE))]),
 "Decision Tree":Pipeline([("tfidf",make_tfidf()),("model",DecisionTreeClassifier(max_depth=40,min_samples_leaf=4,class_weight="balanced",random_state=RANDOM_STATE))]),
 "Random Forest":Pipeline([("tfidf",make_tfidf()),("model",RandomForestClassifier(n_estimators=120,max_depth=50,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=RANDOM_STATE))]),
}
model_files={"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib","Random Forest":"random_forest.joblib"}
'''),
        md("13. Train Models"), code('''
predictions={}
for name,model in models.items(): model.fit(X_train,y_train); predictions[name]=model.predict(X_test); print("Trained:",name)
'''),
        md("14. Evaluation Metrics"), code('''
rows=[]
for name,pred in predictions.items():
    rows.append({"Model":name,"Accuracy":accuracy_score(y_test,pred),"Weighted Precision":precision_score(y_test,pred,average="weighted",zero_division=0),"Weighted Recall":recall_score(y_test,pred,average="weighted",zero_division=0),"Weighted F1":f1_score(y_test,pred,average="weighted",zero_division=0),"Macro F1":f1_score(y_test,pred,average="macro",zero_division=0)})
'''),
        md("15. Model Comparison Table"), code('''
results=pd.DataFrame(rows).sort_values("Macro F1",ascending=False).reset_index(drop=True); display(results.style.format({c:"{:.4f}" for c in results.columns if c!="Model"}))
'''),
        md("16. Visualization"), code('''
results.set_index("Model").plot(kind="bar",figsize=(11,5),ylim=(0,1),rot=0); plt.ylabel("Score"); plt.title("E-commerce sentiment model comparison"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"ml_model_comparison.png",dpi=160); plt.show()
'''),
        md("17. Select Best Model"), code('''
best_name=results.iloc[0]["Model"]; best_model=models[best_name]; labels=["Negative","Neutral","Positive"]
cm=confusion_matrix(y_test,predictions[best_name],labels=labels); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels,cbar=False); plt.title(f"Best model confusion matrix: {best_name}"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(FIGURE_DIR/"best_confusion_matrix.png",dpi=160); plt.show(); print("Best model by Macro F1:",best_name)
'''),
        md("18. Save 3 Models"), code('''
model_paths={}
for name,model in models.items(): path=MODEL_DIR/model_files[name]; joblib.dump(model,path); model_paths[name]=path
'''),
        md("19. Save Best Model"), code('''
best_path=MODEL_DIR/"best_ecommerce_model.joblib"; joblib.dump(best_model,best_path)
'''),
        md("20. Save Results"), code('''
results.to_csv(OUTPUT_DIR/"ecommerce_ml_results.csv",index=False)
'''),
        md("21. Save Feature/Config"), code('''
config={"experiment":EXPERIMENT_NAME,"task":"multiclass_text_classification","target":TARGET,"target_definition":{"1-2":"Negative","3":"Neutral","4-5":"Positive"},"features":SELECTED_FEATURES,"excluded_to_prevent_leakage":[RATING_COL],"models":list(models),"best_metric":BEST_METRIC,"best_model":best_name,"random_state":RANDOM_STATE}
with open(OUTPUT_DIR/"model_config.json","w",encoding="utf-8") as f: json.dump(config,f,indent=2,ensure_ascii=False)
'''),
        md("22. Reload 3 Models"), code('''
loaded_models={name:joblib.load(path) for name,path in model_paths.items()}
'''),
        md("23. Predict Test Sample"), code('''
sample=X_test.iloc[[0]]; actual=y_test.iloc[0]; verification=pd.DataFrame([{"Model":name,"Prediction":model.predict(sample)[0],"Actual":actual} for name,model in loaded_models.items()]); display(sample.to_frame("Review Text")); display(verification)
'''),
        md("24. Verify Saved Models"), code('''
for name in models: np.testing.assert_array_equal(loaded_models[name].predict(sample),models[name].predict(sample))
assert joblib.load(best_path).predict(sample).shape==(1,); print("[OK] All saved models loaded and predicted successfully.")
'''),
        md("25. Conclusion"), code('''
print(f"Best model: {best_name} | Macro F1={results.iloc[0]['Macro F1']:.4f} | Accuracy={results.iloc[0]['Accuracy']:.4f}")
print("Each saved pipeline contains both TF-IDF and its classifier, ready for inference on raw review text.")
'''),
    ]
    write_notebook("e-commerce-comment-analytis", "ecommerce-ml.ipynb", "E-commerce Reviews — Sentiment ML Baselines", cells)


if __name__ == "__main__":
    diabetes(); house(); ecommerce()
