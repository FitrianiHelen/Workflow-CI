import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
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

    # 6. Aktifkan Autolog
    # Ini akan mencatat parameter, metrik, dan model secara otomatis ke MLflow
    mlflow.sklearn.autolog()

    # 7. Inisialisasi Model
    model = LogisticRegression(max_iter=1000)
    
    # 8. Proses Training & Logging
    run = mlflow.active_run()
    if run:
        # Jika dijalankan via 'mlflow run' (GitHub Actions)
        model.fit(X_train, y_train)
    else:
        # Jika dijalankan manual
        with mlflow.start_run(run_name="Manual Training with Autolog"):
            model.fit(X_train, y_train)

    # 9. Evaluasi Manual (Tambahan)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Final Accuracy: {acc}")

    # 10. SIMPAN MODEL & SCALER KE JOBLIB (Untuk inference.py)
    # Disimpan di folder yang sama dengan modelling.py
    model_save_path = os.path.join(base_dir, "model.pkl")
    scaler_save_path = os.path.join(base_dir, "scaler.pkl")

    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print(f"Model disimpan di: {model_save_path}")
    print(f"Scaler disimpan di: {scaler_save_path}")

if __name__ == "__main__":
    train_model()
