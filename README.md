# Laboratorium 2 - Inżynieria uczenia maszynowego

Porównanie pipeline'u scikit-learn z prostą siecią neuronową PyTorch na zbiorze Wine.

## Uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

W Windows środowisko aktywuje polecenie `.venv\Scripts\activate`.

Skrypt wykonuje wspólny podział danych, trenuje oba modele, przeprowadza
5-krotną walidację krzyżową i zapisuje porównanie w `results/wyniki.txt`.

## Struktura

- `src/train.py` - kod obu modeli i walidacji
- `data/` - lokalny plik CSV zbioru Wine
- `results/wyniki.txt` - tabela wyników walidacji
- `models/` - katalog przygotowany na zapis modeli
- `requirements.txt` - zależności projektu

Dane, modele i środowisko `.venv` są wykluczone z repozytorium Git.
