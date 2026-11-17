# Data

Put the two CSV files from the Kaggle dataset here:

- `Fake.csv` — 23,481 fake news articles
- `True.csv` — 21,417 real news articles (Reuters wire stories)

Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Or use the helper script:

```
pip install kaggle
python scripts/download_data.py
```

The CSVs are ~116 MB total and are excluded from version control via `.gitignore`.
