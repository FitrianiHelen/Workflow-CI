import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir,
        "students_performance_preprocessing",
        "students_clean.csv"
    )

    df = pd.read_csv(data_path)

    X = df[["math score", "reading score", "writing score"]]
    y = df["test_prep"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)

    # JANGAN bikin run manual
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)

if __name__ == "__main__":
    train_model()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir,
        "students_performance_preprocessing",
        "students_clean.csv"
    )

    df = pd.read_csv(data_path)

    X = df[["math score", "reading score", "writing score"]]
    y = df["test_prep"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Accuracy:", acc)

if __name__ == "__main__":
    train_model()
