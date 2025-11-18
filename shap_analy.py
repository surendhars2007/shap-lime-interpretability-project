import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import make_classification
from lime.lime_tabular import LimeTabularExplainer

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
# 5. SHAP Explainer
# ---------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# If list → binary classifier → take index 0
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Ensure SHAP matrix shape is valid
if shap_values.ndim == 1:
    shap_values = shap_values.reshape(-1, len(X_test.columns))

# ---------------------------------------------------------
# 6. Global SHAP summary
# ---------------------------------------------------------
print("\nGenerating SHAP summary plot...")
shap.summary_plot(shap_values, X_test, show=True)

# ---------------------------------------------------------
# 7. Local SHAP Explanation (Waterfall)
# ---------------------------------------------------------
index_to_explain = 10
row = X_test.iloc[[index_to_explain]]

local_shap = explainer.shap_values(row)
if isinstance(local_shap, list):
    local_shap = local_shap[0]

local_shap = local_shap[0]

base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[0]

print(f"\nExplaining row index (SHAP): {index_to_explain}")

shap.plots.waterfall(
    shap.Explanation(
        values=local_shap,
        base_values=base_value,
        data=row.values[0],
        feature_names=X_train.columns
    )
)

# ---------------------------------------------------------
# 8. LIME Explanation
# ---------------------------------------------------------
print("\nGenerating LIME explanation...")

lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=["Class 0", "Class 1"],
    mode="classification"
)

lime_exp = lime_explainer.explain_instance(
    data_row=row.values[0],
    predict_fn=model.predict_proba
)

lime_exp.show_in_notebook(show_table=True)

# Save LIME as HTML
lime_exp.save_to_file("lime_explanation.html")
print("\nLIME explanation saved as lime_explanation.html")

# ---------------------------------------------------------
# 9. Save model
# ---------------------------------------------------------
joblib.dump(model, "final_lgbm_model.pkl")
print("\nModel saved successfully as final_lgbm_model.pkl")

