# Fake News Detection - Report

## The task

Given a news article (title + body), decide whether it's real or fake.
"Fake" is treated as the positive class (label 1) since the point is to detect
fake news, so precision tells us how many of our "fake" flags are correct and
recall tells us how much fake news we actually catch.

## Looking at the data first

The Kaggle dataset has two files: `True.csv` (21,417 real Reuters articles) and
`Fake.csv` (23,481 fake ones), each with columns title, text, subject, date.

Before building anything, three things were found that would let a model score
~99-100% for the wrong reasons. Fixing these was the most important part:

1. The `subject` column gives the answer away. Real articles are only labelled
   `politicsNews` or `worldnews`, while fake ones use completely different
   subjects (`News`, `politics`, `left-news`, etc.) with no overlap. If you
   feed `subject` to the model it just memorises that mapping. So we don't use
   `subject` or `date` at all - only the title and body text.

2. Almost every real article starts with a Reuters dateline like
   "WASHINGTON (Reuters) - ...". 99.2% of real articles contain "Reuters" versus
   0.04% of fake ones. A model can get near-perfect accuracy just by spotting
   that word, which tells you the source, not whether the news is true. We strip
   the dateline and every "Reuters" mention in `text_clean.py`.

3. There are a lot of duplicate articles (about 6,000 repeated fake bodies and
   ~225 real). If the same article shows up in both train and test, the test
   score is inflated because the model has effectively seen it. So we remove
   duplicates before splitting.

After cleaning and dropping duplicates we're left with ~38,826 unique articles
out of 44,898, and the classes are close to balanced (~46% fake), so plain
accuracy is still a meaningful number.

Cleaning steps (in `text_clean.py`): lower-case, strip the Reuters tags, remove
URLs and emails, drop everything that isn't a letter, and collapse spaces. Title
and body are combined before cleaning, which also helps the ~800 fake rows that
have an empty body but a usable title. Stop-words are removed by the TF-IDF
vectoriser using scikit-learn's built-in English list.

## The 70/10/20 split

The data is split into 70% train, 10% validation, and 20% test, stratified by
label so the fake/real ratio stays the same in each part. Each split has one
job: the TF-IDF and the models are fit on the training set, the validation set
is used to compare models and choose one, and only the test set is touched at
the very end for the final numbers. The vectoriser is fit on the training text
only and then just transforms the validation and test text, so nothing from
those sets leaks into the features.

Sizes: train ~27,177, validation ~3,883, test ~7,766.

## Features: TF-IDF

Each article is represented as a TF-IDF vector over unigrams and bigrams. Term
frequency counts how often a word appears in the article (sublinear scaling,
1 + log(tf), is used so very repetitive words don't dominate), and inverse
document frequency down-weights words that appear in almost every article. TF-IDF
was chosen because it's fast, deterministic, a strong baseline for this kind of
text, and - importantly - interpretable: because every feature is a real word,
we can read off which words push a prediction toward fake or real.

The main limitation is that this is a bag-of-words model. Apart from the short
bigrams it ignores word order and context, and it treats synonyms as unrelated.

Settings: ngram_range (1,2), min_df 5, max_df 0.9, max_features 50,000.

## Models compared

Four linear classifiers were compared on the same TF-IDF features. Linear models
are the sensible choice for high-dimensional sparse text — they train in seconds
and don't overfit easily with regularisation.

- **Logistic Regression** — models the probability of "fake" as a sigmoid of a
  weighted sum of the word features. Gives proper probabilities and readable
  coefficients, so it's the easiest to explain.
- **Multinomial Naive Bayes** — the classic text baseline. Assumes words are
  independent given the class, which isn't really true but works well and is a
  good sanity check.
- **Linear SVM** — finds the maximum-margin boundary between the classes; usually
  one of the best for TF-IDF text.
- **Passive-Aggressive** — an online large-margin classifier that shows up a lot
  in fake-news examples, included for comparison.

The model with the best validation F1 is selected automatically.

## Results

Validation set (used to pick the model):

| Model               | Accuracy | Precision | Recall | F1     | AUC-ROC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.9853   | 0.9915    | 0.9765 | 0.9840 | 0.9989  |
| Multinomial NB      | 0.9601   | 0.9524    | 0.9615 | 0.9569 | 0.9913  |
| Linear SVC          | 0.9928   | 0.9949    | 0.9894 | 0.9922 | 0.9996  |
| Passive-Aggressive  | 0.9930   | 0.9961    | 0.9888 | 0.9924 | 0.9996  |

The top three are basically tied; Naive Bayes is a bit behind. Passive-Aggressive
has the best validation F1, so that's the one evaluated on the test set.

Test set (final numbers, computed once):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9912 |
| Precision | 0.9933 |
| Recall    | 0.9877 |
| F1-score  | 0.9905 |
| AUC-ROC   | 0.9995 |

The validation and test scores are very close, which is a good sign the model
isn't overfit to the validation set.

## What the model actually learned

The plot in `reports/figures/top_features.png` shows the most influential words.
The strongest "real" words are things like "said", "washington", weekdays, and
"factbox" — basically the house style of Reuters wire copy. The strongest "fake"
words are "video", "read", "breaking", "featured image", "getty images",
"watch" — the formatting habits of web and blog posts.

This is useful but it's also a warning: even after removing the explicit
"Reuters" tag, the model is mostly separating two writing styles / sources rather
than reasoning about whether a claim is true.

## Limitations

- This dataset really measures source, not truthfulness. Real news comes from
  one wire service and fake news from a set of flagged sites, so the two groups
  differ in style as much as in truth. A model this accurate on this dataset
  would not be equally accurate on fake news from a source it hasn't seen.
- A bag-of-words model can't check facts. It sees words, not meaning, so it
  can't verify a claim, spot satire, or catch fake news written in a polished,
  wire-like style.
- The data is from around 2016-2017 US politics, so the vocabulary is
  era-specific and performance would drop on newer or non-political news.
- It's English-only and uses no outside knowledge.
- The default 0.5 threshold is used; in a real deployment it should be tuned on
  the validation ROC depending on whether false positives or false negatives are
  worse.

## What to try next with more time

- Evaluate with a source-disjoint split (train on some outlets, test on others)
  and on a separate dataset like LIAR to get a fairer estimate of real
  generalisation.
- Try a fine-tuned transformer (e.g. DistilBERT) to capture meaning beyond
  bag-of-words.
- A small hyperparameter search over C, the n-gram range, and max_features.

## Reproducibility

One seed (42) fixes the split and all the models, the cleaning is plain regex,
the TF-IDF vocabulary is sorted, and the solvers are deterministic. Running
`python main.py` twice gives identical metrics. The exact package versions are
in `requirements.txt` and the exact config is saved into `reports/metrics.json`
on each run. See REPRODUCIBILITY_CHECKLIST.md.
