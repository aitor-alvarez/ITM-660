# ITM 660 — Class Materials

Class materials, exercises, and projects for ITM 660.

---

## Repository Structure
Future sessions will be added to the tree below
```
ITM-660/
├── Session 2/
│   ├── classification_models.py     # Classification pipeline (CLI + importable module)
│   └── classification_pipeline.ipynb  # Step-by-step notebook
└── .gitignore
```


---

## Sessions

### Session 2 — Classification Pipeline

Uses scikit-learn, Panda and Seaborn (for visualization) to build a pipeline that trains and compares multiple classifiers on any CSV dataset.

**Models included:**
- Logistic Regression
- Random Forest
- Gradient Boosted Trees

Any arbitrary model from scikit-learn can be added to the pipeline.

**Features:**
- Automatic preprocessing: categorical encoding, missing value imputation, feature scaling
- Hold-out evaluation: precision, recall, accuracy and weighted F1
- Stratified k-fold cross-validation
- Seaborn plots: model comparison bar chart + confusion matrix for the best model

**Usage:**

```bash
python "Session 2/classification_models.py" --csv data.csv --target label
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--test-size` | `0.2` | Fraction of data held out for testing |
| `--cv-folds` | `5` | Number of cross-validation folds |


## Setup

```bash
pip install 'r'
```

Python 3.10+ required.

Using it as a python standalone module
```python
from classification_models import run

results = run("data.csv", "target_column", test_size=0.25, cv_folds=10)
```


## Notebooks

Notebooks will be included on each session with step-by-step details