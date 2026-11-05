from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


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


def main():
    dataset, X, y, (X_train, X_test, y_train, y_test) = load_data()
    print(f"Zbiór: Wine, klasy: {list(dataset.target_names)}")
    print(f"Cały zbiór: {X.shape}")
    print(f"Trening: {X_train.shape}, test: {X_test.shape}")


if __name__ == "__main__":
    main()
