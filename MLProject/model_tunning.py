import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Load data
df = pd.read_csv("dataset_preprocessing.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model + parameter grid
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20]
}

model = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="accuracy"
)

with mlflow.start_run(run_name="RF_Tuning"):
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
  
# PAKSA LOCAL FILE STORE
mlflow.set_tracking_uri("file:///C:/Users/ASUS/Documents/SMSML_Muhamad Rivaldo Maulana/Membangun Model/mlruns")
mlflow.set_experiment("Model_Tuning_RF")
with mlflow.start_run(run_name="RF_Tuning_Manual_Logging"):

    # Fit GridSearch
    grid.fit(X_train, y_train)

    # Ambil model terbaik
    best_model = grid.best_estimator_

    # Prediksi
    y_pred = best_model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    # ======================
    # MANUAL LOGGING (WAJIB SKILLED)
    # ======================
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model sebagai artifact
    mlflow.sklearn.log_model(best_model, artifact_path="model")

    print("Accuracy:", acc)
    print("F1-score:", f1)
    print("TRACKING URI:", mlflow.get_tracking_uri())
  
  with mlflow.start_run(run_name="RF_Tuning"):
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(best_model, "best_model")



