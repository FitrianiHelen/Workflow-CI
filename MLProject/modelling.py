import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_model():
    # Mengambil path data secara dinamis
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir,
        "students_performance_preprocessing",
        "students_clean.csv"
    )

    # Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File data tidak ditemukan di: {data_path}")
        
    df = pd.read_csv(data_path)

    # Fitur dan Target
    X = df[["math score", "reading score", "writing score"]]
    y = df["test_prep"]

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Inisialisasi Model
    model = LogisticRegression(max_iter=1000)
    
    # Training Model
    # Catatan: Kita tidak menggunakan 'with mlflow.start_run()' di sini 
    # karena sesi sudah otomatis dibuka oleh perintah 'mlflow run' di CI
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Logging ke MLflow (otomatis masuk ke run yang sedang aktif di CI)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model trained successfully. Accuracy: {acc}")

if __name__ == "__main__":
    train_model()
