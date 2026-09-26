from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.data_loader import load_data, split_data
from src.models import choose_model

# 1. Define file paths
DATA_PATH = Path("data") / "Heart_Disease_Cleveland.csv"

# 2. Load and split dataset
X, y = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = split_data(X, y)

# 3. Instantiate and fit model pipeline
model_name = "random_forest"
pipeline = choose_model(model_name)
pipeline.fit(X_train, y_train)

# 4. Predict probabilities and score
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, y_pred_proba)

print(f"Model: {model_name} | Test ROC-AUC Score: {score:.4f}")