import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# Mlflow nosaukums
experiment_name = "Rezervāciju prognozēšana"

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "data/hotel.csv")
df = pd.read_csv(filename)

target_col = "booking_status"
X = df.drop(columns=[target_col])
y = df[target_col]

os.makedirs("artifacts", exist_ok=True)

plt.figure(figsize=(6,4))
df[target_col].value_counts().plot(kind="bar", color="skyblue")
plt.title("Booking Status Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("artifacts/data_booking_status_distribution.png")
plt.close()

# Apstrāde
num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(exclude=["number"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols),
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

classes = np.unique(y)

pos_label = classes[0]
cv_scoring = "roc_auc"


# Modeļu konfigurācijas
models = {
    "Dummy": {
        "model": DummyClassifier(strategy="most_frequent"),
        "params": {},
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "params": {
            "model__C": [0.01, 0.1, 1, 3, 10],
            "model__solver": ["liblinear", "saga"],
            "model__penalty": ["l1", "l2"],
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "params": {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [None, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.8, 1.0],
        },
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan", "minkowski"],
        },
    },
}

# Apmacība
results = []
trained_models = {}

best_model_name = None
best_cv = -np.inf

for name, cfg in models.items():
    print(f"\nModelis: {name} ")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", cfg["model"]),
    ])

    if cfg["params"]:
        grid = GridSearchCV(pipe, cfg["params"], scoring=cv_scoring, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_est = grid.best_estimator_
        cv_score = grid.best_score_
        trained_models[name] = grid
        print("Labākie parametri:", grid.best_params_)
    else:
        pipe.fit(X_train, y_train)
        best_est = pipe
        trained_models[name] = pipe

        cv_score = roc_auc_score(y_train, best_est.predict_proba(X_train)[:, 1])

    y_pred = best_est.predict(X_test)

    if hasattr(best_est, "predict_proba"):
        y_proba = best_est.predict_proba(X_test)
    else:
        y_proba = None


    f1 = f1_score(y_test, y_pred, pos_label=pos_label)
    auc_val = roc_auc_score(y_test, y_proba[:, np.where(classes == pos_label)[0][0]]) if y_proba is not None else np.nan

    acc = accuracy_score(y_test, y_pred)

    print("CV Score:", cv_score)
    print("Accuracy:", acc)
    print("F1:", f1)
    print("AUC:", auc_val)

    results.append({
        "model": name,
        "cv_score": cv_score,
        "acc": acc,
        "f1": f1,
        "auc": auc_val,
    })

    if cv_score > best_cv:
        best_cv = cv_score
        best_model_name = name

results_df = pd.DataFrame(results)
print("\nModeļu kopsavilkums ")
print(results_df)

plt.figure(figsize=(8,5))
sns.barplot(data=results_df, x="model", y="f1", palette="viridis")
plt.title("Model Comparison — F1 Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("artifacts/model_comparison_f1.png")
plt.close()

plt.figure(figsize=(8,5))
sns.barplot(data=results_df, x="model", y="acc", palette="magma")
plt.title("Model Comparison — Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("artifacts/model_comparison_acc.png")
plt.close()

plt.figure(figsize=(8,5))
sns.barplot(data=results_df, x="model", y="auc", palette="cubehelix")
plt.title("Model Comparison — AUC Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("artifacts/model_comparison_auc.png")
plt.close()


# Mlflow iestatījumi
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

# Metrikas 
def eval_metrics(y_true, y_pred, y_proba, name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)

    if y_proba is not None:
        auc_val = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc_val = np.nan


    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = f"artifacts/CM_{name}.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.savefig(cm_path)
    plt.close()

    return acc, f1, auc_val, cm_path

# Mlflow modeļu lejuplāde
def log_model_to_mlflow(model, name):
    with mlflow.start_run(run_name=name):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc, f1, auc_val, cm_path = eval_metrics(y_test, y_pred, y_proba, name)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        if not np.isnan(auc_val):
            mlflow.log_metric("auc", auc_val)

        mlflow.log_artifact(cm_path, name)

        mlflow.sklearn.log_model(model, name)

for name, model in trained_models.items():
    log_model_to_mlflow(model, name)

print(f"\n Labākais modelis: {best_model_name}, CV={best_cv:.4f}")
log_model_to_mlflow(trained_models[best_model_name], best_model_name)

