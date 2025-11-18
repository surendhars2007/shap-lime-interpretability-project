# Advanced Interpretability of Black-Box Models Using SHAP and LIME

## Project Overview

This project demonstrates advanced model interpretability techniques applied to a non-linear, “black-box” classification model. Using **LightGBM** as the predictive model, we explore both **global** and **local** interpretability through **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations).  

The focus is not just on high predictive performance, but also on **understanding feature importance, interactions, and individual predictions**, providing actionable insights for business stakeholders.

---

## Repository Contents

| File | Description |
|------|-------------|
| `shap_analysis.py` | Main Python script to train LightGBM, generate SHAP and LIME explanations, and save the model. |
| `final_lgbm_model.pkl` | Trained LightGBM model serialized using Joblib. |
| `lime_explanation.html` | Interactive LIME explanation for a selected test instance. |
| `README.md` | Project overview, methodology, and results. |

---

## Dataset

- Synthetic classification dataset with **1000 samples and 20 features**.
- 10 informative features and 5 redundant features.
- Split into **train (50%)** and **test (50%)** sets.

*(In a real-world scenario, this could be a credit risk or customer churn dataset.)*

---

## Model Training

- **Model**: LightGBM Classifier  
- **Hyperparameters**:  
  - `n_estimators=500`  
  - `learning_rate=0.05`  
  - `subsample=0.8`  
  - `colsample_bytree=0.8`  
- **Evaluation Metric**: AUC (Area Under the ROC Curve)  
- **Early Stopping**: 30 rounds  

**Performance on Test Set:**

AUC: 0.981
Accuracy: 0.92
Precision, Recall, F1-score:
Class 0 -> Precision: 0.96, Recall: 0.88, F1: 0.92
Class 1 -> Precision: 0.88, Recall: 0.96, F1: 0.92

---

## Interpretability

### 1. SHAP (Global & Local)

- **Global Feature Importance:** SHAP summary plot highlights the **top influential features** across all predictions.  
- **Local Explanation:** SHAP waterfall plot explains individual predictions, showing **contributions of each feature**.  
- **Feature Interaction:** Dependency plots reveal interactions between features that drive model predictions.

**Example Global Top Features:**

| Feature | Mean Absolute SHAP |
|---------|------------------|
| f17     | 1.09             |
| f5      | 0.73             |
| f16     | 0.59             |
| f14     | 0.55             |
| f4      | 0.48             |

---

### 2. LIME (Local Explanation)

- **Local interpretability** for a specific instance (high-confidence correct prediction).  
- Provides **human-understandable explanation** by approximating the black-box model locally.  
- Output is saved as `lime_explanation.html` for interactive visualization.

---

## Comparative Analysis

| Aspect | SHAP | LIME |
|--------|------|------|
| Scope | Global + Local | Local only |
| Output | Contribution values of all features | Interpretable linear surrogate for one instance |
| Feature Interaction | Captured via dependency plots | Limited |
| Use Case | Identify global important features & individual prediction drivers | Explain single predictions to stakeholders |
| Strength | Consistent, theoretically sound (Shapley values) | Highly intuitive for non-technical stakeholders |

**Insights:**  
- SHAP and LIME explanations **converge** on the most influential features for the selected instance.  
- **Divergence** may appear for features with weak impact, as LIME only approximates locally.  
- SHAP’s global view helps **identify blind spots** and refine decision-making criteria.

---

## Usage

1. **Run the analysis script:**

```bash
python shap_analysis.py
Outputs generated:

final_lgbm_model.pkl → trained model

SHAP summary plot (interactive)

SHAP waterfall plot for selected instance

lime_explanation.html → LIME interactive explanationOutputs generated:

final_lgbm_model.pkl → trained model

SHAP summary plot (interactive)

SHAP waterfall plot for selected instance

lime_explanation.html → LIME interactive explanation
Dependencies

Python 3.8+

Libraries: numpy, pandas, lightgbm, scikit-learn, shap, joblib, lime, matplotlib

Install via:
pip install numpy pandas scikit-learn lightgbm shap joblib lime matplotlib

Conclusion

This project demonstrates how state-of-the-art interpretability techniques can be applied to black-box models, providing both global insights and local explanations. The combination of SHAP and LIME ensures a comprehensive understanding, suitable for both data scientists and business stakeholders.

