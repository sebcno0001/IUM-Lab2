from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TEST_SIZE = 0.2
EPOCHS = 200

torch.set_num_threads(1)


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


class WineNetwork(nn.Module):
    """Prosta sieć z dwiema warstwami ukrytymi."""

    def __init__(self, input_size, class_count):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, class_count),
        )

    def forward(self, X):
        return self.layers(X)


def train_torch(X_train, X_test, y_train, y_test, epochs=EPOCHS, seed=SEED):
    """Skaluje dane, trenuje nową sieć i ocenia ją na zbiorze testowym."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Skaler dopasowuje się wyłącznie do danych treningowych.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    X_train_tensor = torch.tensor(X_train_scaled)
    X_test_tensor = torch.tensor(X_test_scaled)
    y_train_tensor = torch.tensor(np.asarray(y_train), dtype=torch.long)

    model = WineNetwork(X_train.shape[1], len(np.unique(y_train)))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start = perf_counter()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = loss_function(logits, y_train_tensor)
        loss.backward()
        optimizer.step()
    training_time = perf_counter() - start

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).argmax(dim=1).numpy()
    accuracy = accuracy_score(y_test, predictions)
    return model, scaler, predictions, accuracy, training_time


def validate_sklearn(X, y):
    """Uruchamia 5-krotną walidację krzyżową Pipeline'u."""
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        build_sklearn_model(), X, y, cv=folds, scoring="accuracy"
    )
    return scores, scores.mean(), scores.std()


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

    _, _, _, torch_accuracy, torch_time = train_torch(
        X_train, X_test, y_train, y_test
    )
    print(
        f"PyTorch - accuracy: {torch_accuracy:.4f}, "
        f"czas: {torch_time:.4f} s"
    )

    sklearn_scores, sklearn_mean, sklearn_std = validate_sklearn(X, y)
    print(f"scikit-learn CV: {np.round(sklearn_scores, 4)}")
    print(f"Średnia: {sklearn_mean:.4f}, odchylenie: {sklearn_std:.4f}")


if __name__ == "__main__":
    main()
