import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ===== Load data =====
df = pd.read_csv("dataset_preprocessing.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== MLflow =====
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="LogisticRegression_Basic"):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    print("Accuracy:", acc)
    print("F1:", f1)
    print("Model and mertics logged to MLflow.")
