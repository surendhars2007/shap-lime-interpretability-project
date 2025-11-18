# 🔍 SHAP & LIME Explainability Project  
**Machine Learning Global & Local Interpretability Analysis**

This project demonstrates a complete workflow for **model explainability** using both **SHAP** (global & local explanations) and **LIME** (local explanations).  
It includes feature interaction analysis, global feature importance, and detailed comparison of SHAP vs LIME for a single prediction instance.

This project is suitable for academic submission, corporate ML explainability, and interview-ready portfolio work.

---

# 📘 Table of Contents
- [Introduction](#introduction)
- [Project Goal](#project-goal)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Model Training](#model-training)
- [Global SHAP Explanations](#global-shap-explanations)
- [Local Explanation (SHAP + LIME)](#local-explanation-shap--lime)
- [Comparative Analysis](#comparative-analysis)
- [Feature Interaction Analysis](#feature-interaction-analysis)
- [Results](#results)
- [Screenshots](#screenshots)
- [Conclusion](#conclusion)
- [Author](#author)

---

# 🧠 Introduction

Machine learning models often act like "black boxes," making predictions without explaining how they arrived at them.  
This project solves that by using:

- **SHAP** (SHapley Additive exPlanations) → mathematically consistent, global & local explanations  
- **LIME** (Local Interpretable Model-Agnostic Explanations) → local linear approximation for individual predictions  

Both techniques are implemented on a classification model and compared for interpretability.

---

# 🎯 Project Goal

This project aims to:

1. Train a machine learning model  
2. Generate global model explainability using SHAP  
3. Generate local instance-level explanations using SHAP and LIME  
4. Compare global vs local interpretability  
5. Analyze feature interactions using SHAP dependence & interaction values  
6. Produce a complete, submission-ready ML explainability report  

---

# 🚀 Features

### ✔ Model Training
- Loading dataset  
- Cleaning, preprocessing  
- Train-test split  
- Random Forest / LightGBM model training  
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1)

### ✔ SHAP Explainability
- Global SHAP summary plot  
- SHAP bar plot  
- SHAP force plot  
- SHAP waterfall plot  
- SHAP dependence plot  
- Feature interaction discovery  

### ✔ LIME Explainability
- Local linear explanation for 1 instance  
- Feature contribution visualization  
- Comparison with SHAP for the same row  

### ✔ Analysis Section
- SHAP vs LIME convergence  
- SHAP vs LIME divergence  
- Which technique is better for global vs local view  
- Feature interaction insights  

---

# 🛠 Technologies Used

| Component | Technology |
|----------|------------|
| Language | Python |
| ML Model | RandomForestClassifier / LightGBM |
| Explainability | SHAP, LIME |
| Libraries | Pandas, NumPy, Matplotlib, Scikit-Learn |
| Output | Plots + printed SHAP/LIME values |

---

# 📂 Project Structure

📦 SHAP-LIME-Explainability
┣ 📜 shap.py # Full working code
┣ 📜 README.md # Documentation (this file)
┗ 📁 data/ # (Optional) Your dataset
