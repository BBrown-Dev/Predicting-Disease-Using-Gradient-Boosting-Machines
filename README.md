# Predicting Disease Using Gradient Boosting Machines (GBM)

This project applies Gradient Boosting Machines (GBM) using scikit‑learn to build a predictive model for breast cancer diagnosis. The workflow includes data exploration, preprocessing, model training, evaluation, confusion matrix visualization, and feature‑importance analysis. The goal is to demonstrate how boosting, loss functions, and optimization techniques contribute to high‑performance medical classification models.

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**  
Provided by scikit‑learn, this dataset contains:

- 569 samples
- 30 numeric features
- **Binary target:** malignant or benign tumor

The dataset is widely used for benchmarking machine‑learning models in medical diagnosis.

---

## Data Preprocessing

- Train–test (80/20 split, stratified)
- Standardization using `StandardScaler`
- Basic inspection of feature names, shapes, and class distribution

---

## Gradient Boosting Model

The model is implemented using:

```python
from sklearn.ensemble import GradientBoostingClassifier

GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
```

### Why GBM?
- Sequential boosting of weak learners  
- Gradient‑based optimization of loss  
- Strong performance on structured/tabular data  
- Built‑in feature importance estimation  

---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1‑score
- Confusion Matrix

### Example Results
- **Accuracy:** 0.956  
- **Precision:** 0.947  
- **Recall:** 0.986  
- **F1‑score:** 0.966  

These metrics indicate excellent performance, especially the high recall, which is critical in medical diagnosis to minimize false negatives.

---

## Feature Importance

The model provides ranked feature importances.  
Top predictors include:

1. worst radius  
2. worst perimeter  
3. worst concave points  
4. mean concave points  
5. worst texture  

These features align with known clinical indicators of tumor malignancy.

---

## Visualizations

The project includes:

- Confusion matrix heatmap  
- Bar plot of top N most important features  

These visualizations help interpret model behavior and diagnostic reliability.

---

## How to Run

1. Install dependencies:

```bash
pip install scikit-learn matplotlib seaborn numpy
```

2. Run the script:

```bash
python gbm_breast_cancer.py
```

3. View printed metrics and generated plots.

---


## 📚 References (APA Style)

Scikit-learn. (n.d.). *GradientBoostingClassifier*. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

Saini, A. (2024). *Gradient Boosting Algorithm: A Complete Guide for Beginners*. Analytics Vidhya. [https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/](https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/)

GBM in Machine Learning. (n.d.). *Javatpoint*. [https://www.javatpoint.com/gbm-in-machine-learning](https://www.javatpoint.com/gbm-in-machine-learning)