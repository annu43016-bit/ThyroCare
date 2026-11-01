import os
import io
import urllib.request
import pandas as pd
from flask import Flask, render_template, request, send_file
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "Thyroid_diagnosis.csv")

# Public mirror of the thyroid dataset sourced from the UCI Machine Learning Repository.
# If the CSV is not present, the app downloads it automatically.
DATA_URL = "https://raw.githubusercontent.com/Poojitha369/DSA6000/main/Thyroid_diagnosis.csv"

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(DATA_FILE):
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    except Exception as exc:
        raise RuntimeError(
            "Real dataset download failed. Please check internet connection, then run app.py again. "
            "You can also download Thyroid_diagnosis.csv from the dataset source listed in README.txt "
            f"and place it inside the data folder. Details: {exc}"
        )

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

# Keep a small, interpretable set of features for the college demonstration.
# The dataset's original diagnosis is converted to a binary screening target:
# negative = 0; any non-negative diagnosis label = 1.
required_columns = [
    "age", "sex", "on thyroxine", "query hypothyroid", "query hyperthyroid",
    "TSH", "T3", "TT4", "T4U", "FTI", "diagnosis"
]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

work = df[required_columns].copy()
work = work.replace("?", pd.NA)
work["diagnosis"] = work["diagnosis"].astype(str).str.strip().str.lower()
work = work[work["diagnosis"].notna() & (work["diagnosis"] != "nan")]
work["binaryClass"] = (work["diagnosis"] != "negative").astype(int)
work = work.drop(columns=["diagnosis"])

# Convert numeric features, preserve categorical fields.
numeric_features = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
categorical_features = ["sex", "on thyroxine", "query hypothyroid", "query hyperthyroid"]
for col in numeric_features:
    work[col] = pd.to_numeric(work[col], errors="coerce")
for col in categorical_features:
    work[col] = work[col].astype("string").fillna("Unknown").astype(str)

X = work.drop(columns=["binaryClass"])
y = work["binaryClass"].astype(int)

if y.nunique() < 2:
    raise ValueError("Dataset must contain both negative and positive diagnosis classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

model_definitions = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=8),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced")
}

results = {}
trained_models = {}
for name, classifier in model_definitions.items():
    model = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "precision": round(precision_score(y_test, pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, pred, zero_division=0) * 100, 2)
    }
    trained_models[name] = model

best_name = max(results, key=lambda name: results[name]["f1"])
best_model = trained_models[best_name]
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred, labels=[0, 1]).tolist()

probabilities = best_model.predict_proba(X_test)[:, 1]
roc_fpr, roc_tpr, _ = roc_curve(y_test, probabilities)
roc_auc = round(float(auc(roc_fpr, roc_tpr)), 3)

# Feature importance chart: use the fitted Decision Tree's encoded feature importances.
tree = trained_models["Decision Tree"].named_steps["classifier"]
encoded_names = trained_models["Decision Tree"].named_steps["preprocess"].get_feature_names_out()
importance_pairs = sorted(zip(encoded_names, tree.feature_importances_), key=lambda pair: pair[1], reverse=True)[:10]

dashboard = {
    "total": int(len(work)),
    "features": int(X.shape[1]),
    "positive": int((y == 1).sum()),
    "negative": int((y == 0).sum()),
    "results": results,
    "best_name": best_name,
    "best": results[best_name],
    "cm": cm,
    "importance_names": [str(name).replace("numeric__", "").replace("categorical__", "") for name, _ in importance_pairs],
    "importance_values": [round(float(value), 4) for _, value in importance_pairs],
    "roc": {
        "fpr": [round(float(value), 4) for value in roc_fpr],
        "tpr": [round(float(value), 4) for value in roc_tpr],
        "auc": roc_auc
    }
}

# Input form fields; categorical values use dropdowns.
form_fields = [
    {"name": "age", "label": "Age (years)", "kind": "number", "step": "1"},
    {"name": "sex", "label": "Sex", "kind": "select", "options": ["F", "M", "Unknown"]},
    {"name": "on thyroxine", "label": "On thyroxine?", "kind": "select", "options": ["f", "t", "Unknown"]},
    {"name": "query hypothyroid", "label": "Query hypothyroid?", "kind": "select", "options": ["f", "t", "Unknown"]},
    {"name": "query hyperthyroid", "label": "Query hyperthyroid?", "kind": "select", "options": ["f", "t", "Unknown"]},
    {"name": "TSH", "label": "TSH", "kind": "number", "step": "any"},
    {"name": "T3", "label": "T3", "kind": "number", "step": "any"},
    {"name": "TT4", "label": "TT4", "kind": "number", "step": "any"},
    {"name": "T4U", "label": "T4U", "kind": "number", "step": "any"},
    {"name": "FTI", "label": "FTI", "kind": "number", "step": "any"}
]

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    entered = {}
    if request.method == "POST":
        row = {}
        for field in form_fields:
            name = field["name"]
            raw = request.form.get(name, "").strip()
            entered[name] = raw
            if name in numeric_features:
                row[name] = pd.to_numeric(raw, errors="coerce") if raw else float("nan")
            else:
                row[name] = raw if raw else "Unknown"
        input_df = pd.DataFrame([row], columns=X.columns)
        predicted_class = int(best_model.predict(input_df)[0])
        prediction = {
            "class": predicted_class,
            "label": "Positive class predicted" if predicted_class == 1 else "Negative class predicted"
        }
    return render_template("index.html", d=dashboard, fields=form_fields, prediction=prediction, entered=entered)

@app.route("/download/csv")
def download_csv():
    report = pd.DataFrame(results).T
    report.index.name = "Model"
    output = io.BytesIO()
    output.write(report.to_csv().encode("utf-8"))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="thyroid_model_report.csv")

@app.route("/download/pdf")
def download_pdf():
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        return "PDF export ke liye run karein: python -m pip install reportlab", 500
    output = io.BytesIO()
    pdf = canvas.Canvas(output)
    pdf.drawString(50, 800, "ThyroCare AI - Thyroid Dataset Model Report")
    pdf.drawString(50, 778, f"Dataset records used: {dashboard['total']}")
    pdf.drawString(50, 758, f"Input features: {dashboard['features']}")
    pdf.drawString(50, 738, f"Best model by F1: {best_name}")
    pdf.drawString(50, 710, "Note: Educational prototype; not a medical diagnosis.")
    y_pos = 675
    for name, score in results.items():
        pdf.drawString(50, y_pos, f"{name}: Accuracy {score['accuracy']}%, Precision {score['precision']}%, Recall {score['recall']}%, F1 {score['f1']}%")
        y_pos -= 24
    pdf.save()
    output.seek(0)
    return send_file(output, mimetype="application/pdf", as_attachment=True, download_name="thyroid_model_report.pdf")

if __name__ == "__main__":
    app.run(debug=True)
