import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_model():
    # 1. Tentukan Path Data secara Dinamis
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir,
        "students_performance_preprocessing",
        "students_clean.csv"
    )

    # Cek apakah file ada sebelum dibaca
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File data tidak ditemukan di: {data_path}")
        
    # 2. Load Dataset
    df = pd.read_csv(data_path)

    # 3. Fitur dan Target
    X = df[["math score", "reading score", "writing score"]]
    y = df["test_prep"]

    # 4. Preprocessing (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Inisialisasi Model
    model = LogisticRegression(max_iter=1000)
    
    # 7. Training Model
    model.fit(X_train, y_train)

    # 8. Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 9. Logging ke MLflow dengan Proteksi Active Run
    # Ini adalah kunci agar tidak error di GitHub Actions
    run = mlflow.active_run()
    if run:
        # Jika dijalankan via 'mlflow run', gunakan sesi yang ada
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"Logged to active run: {run.info.run_id}")
    else:
        # Jika dijalankan manual (python modelling.py), buat sesi baru
        with mlflow.start_run(run_name="Manual Training"):
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, artifact_path="model")
            print("Logged to new manual run")

    print(f"Final Accuracy: {acc}")

if __name__ == "__main__":
    train_model()
