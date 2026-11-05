from pathlib import Path
from time import perf_counter

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TEST_SIZE = 0.2


def load_data():
    """Wczytuje Wine i tworzy wspólny podział dla obu modeli."""
    dataset = load_wine(as_frame=True)
    data_path = ROOT / "data" / "wine.csv"
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.frame.to_csv(data_path, index=False)

    X = dataset.data
    y = dataset.target
    split = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    return dataset, X, y, split


def build_sklearn_model():
    """Łączy skalowanie cech i regresję logistyczną w jeden Pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def train_sklearn(X_train, X_test, y_train, y_test):
    model = build_sklearn_model()
    start = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, predictions, accuracy, training_time


def main():
    dataset, X, y, (X_train, X_test, y_train, y_test) = load_data()
    print(f"Zbiór: Wine, klasy: {list(dataset.target_names)}")
    print(f"Cały zbiór: {X.shape}")
    print(f"Trening: {X_train.shape}, test: {X_test.shape}")

    _, _, sklearn_accuracy, sklearn_time = train_sklearn(
        X_train, X_test, y_train, y_test
    )
    print(
        f"scikit-learn - accuracy: {sklearn_accuracy:.4f}, "
        f"czas: {sklearn_time:.4f} s"
    )


if __name__ == "__main__":
    main()
