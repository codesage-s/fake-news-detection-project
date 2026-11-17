# Reproducibility Checklist

## Data
- [x] Dataset source documented (Kaggle link in README)
- [x] Download script provided (`scripts/download_data.py`)
- [x] Duplicates removed before splitting
- [x] Source artifacts (Reuters tags) stripped to avoid leakage

## Code
- [x] Single entry point: `python main.py`
- [x] All random seeds fixed (`RANDOM_SEED = 42` in `config.py`)
- [x] Exact dependency versions pinned in `requirements.txt`
- [x] No non-deterministic operations (all solvers are deterministic)

## Splits
- [x] 70/10/20 train/val/test split with stratification
- [x] Split is deterministic (same seed → same split)
- [x] No data leakage between splits (duplicates removed first)

## Model selection
- [x] Models compared on validation set only
- [x] Best model selected by validation F1
- [x] Test set used exactly once for final evaluation

## Results
- [x] All metrics saved to `reports/metrics.json`
- [x] Plots saved to `reports/figures/`
- [x] Config snapshot saved alongside metrics
- [x] Running `python main.py` twice produces identical output
