import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import make_classification

# ---------------------------------------------------------
# 1. Generate dataset
# ---------------------------------------------------------
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
y = pd.Series(y, name="target")

# ---------------------------------------------------------
# 2. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# ---------------------------------------------------------
# 3. Train LightGBM
# ---------------------------------------------------------
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(stopping_rounds=30)]
)

# ---------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"\nAUC: {auc}\n")

print(classification_report(y_test, model.predict(X_test)))

# ---------------------------------------------------------
# 5. SHAP explainer
# ---------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# -------------------------------------------
# FIX: LightGBM returns a LIST → pick the first one
# -------------------------------------------
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# -------------------------------------------
# FIX: SHAP summary requires 2D matrix (samples × features)
# If 1D → reshape to (1, -1)
# -------------------------------------------
if shap_values.ndim == 1:
    shap_values = shap_values.reshape(-1, len(X_test.columns))

# ---------------------------------------------------------
# 6. Global SHAP summary plot
# ---------------------------------------------------------
print("\nGenerating SHAP summary plot...")
shap.summary_plot(shap_values, X_test, show=True)

# ---------------------------------------------------------
# 7. Local SHAP Waterfall Plot
# ---------------------------------------------------------
index_to_explain = 10
row = X_test.iloc[[index_to_explain]]

# SHAP row values (make 1D)
shap_row = explainer.shap_values(row)
if isinstance(shap_row, list):
    shap_row = shap_row[0]

shap_row = shap_row[0]  # extract vector

# expected value fix (scalar or array)
base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[0]

print(f"\nExplaining row index: {index_to_explain}")

shap.plots.waterfall(
    shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=row.values[0],
        feature_names=X_train.columns
    )
)

# ---------------------------------------------------------
# 8. Save model
# ---------------------------------------------------------
joblib.dump(model, "final_lgbm_model.pkl")
print("\nModel saved successfully as final_lgbm_model.pkl")

