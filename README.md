# Deploying a Scalable ML Pipeline with FastAPI

Trains an income classifier on U.S. Census data, measures how its performance varies across demographic slices, and serves predictions through a FastAPI REST API, with unit tests and CI on GitHub Actions. Built from the Udacity starter code as a WGU course project.

## What it does
- **Model:** scikit-learn Random Forest (100 trees) predicting whether income exceeds $50K; categorical features are one-hot encoded
- **Evaluation:** 80/20 split (26,048 training rows, 6,513 test rows). Test precision **0.733**, recall **0.640**, F1 **0.683**. Precision, recall, and F1 are used instead of accuracy because the classes are imbalanced (76% / 24%)
- **Slice analysis:** performance on every value of each categorical feature (`slice_output.txt`). F1 ranges from 0.69 for some groups to 0.53 for others; see the model card
- **API:** `GET /` returns a welcome message; `POST /data/` returns a prediction
- **Tests and CI:** pytest unit tests for training, inference, metrics, and the data split, run with flake8 on every push through GitHub Actions

## Files
| File | Purpose |
|---|---|
| `train_model.py` | Loads data, trains and saves the model, writes slice metrics |
| `ml/` | Data processing and model functions |
| `main.py` | FastAPI app |
| `local_api.py` | Sends sample GET and POST requests to the running API |
| `test_ml.py` | Unit tests |
| `model_card.md` | Model details, metrics, ethical considerations |
| `screenshots/` | CI run, unit tests, local API output |

## Run it
```bash
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
python local_api.py
pytest test_ml.py
```

## Tools
Python · pandas · scikit-learn · FastAPI · pytest · flake8 · GitHub Actions
