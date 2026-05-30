"""
Gradient Boosting Machines (GBM) for Breast Cancer Diagnosis
Dataset: Breast Cancer Wisconsin (Diagnostic) from scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Set a random state for reproducibility
RANDOM_STATE = 42


def load_and_explore_data():
    """
    Load the Breast Cancer Wisconsin dataset and print basic info.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    print("-- Dataset Overview --")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"Target classes: {target_names}")
    print()

    return X, y, feature_names, target_names


def preprocess_data(X, y, test_size=0.2):
    """
    Split the data into train and test sets and scale features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("-- Data Preprocessing --")
    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
    print()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_gbm_classifier(
    X_train,
    y_train,
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
):
    """
    Train a Gradient Boosting Classifier with specified hyperparameters.
    """
    gbm = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
    )

    gbm.fit(X_train, y_train)

    print("-- Model Training --")
    print("Gradient Boosting Classifier trained with:")
    print(f"  n_estimators = {n_estimators}")
    print(f"  learning_rate = {learning_rate}")
    print(f"  max_depth = {max_depth}")
    print()

    return gbm


def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluate the model, then print classification metrics and confusion matrix.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("-- Model Evaluation --")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, target_names)

    return acc, prec, rec, f1, cm


def plot_confusion_matrix(cm, target_names):
    """
    Plot a confusion matrix heatmap.
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix - Gradient Boosting Classifier")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot the top N most important features from the trained GBM model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # descending order
    top_indices = indices[:top_n]

    top_features = feature_names[top_indices]
    top_importances = importances[top_indices]

    print("-- Feature Importance (Top Features) --")
    for rank, (feat, imp) in enumerate(zip(top_features, top_importances), start=1):
        print(f"{rank}. {feat}: {imp:.4f}")
    print()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=top_importances,
        y=top_features,
        orient="h",
        hue=top_features,
        palette="viridis",
        legend=False,
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Most Important Features - GBM")
    plt.tight_layout()
    plt.show()


def main():
    # Load and explore the dataset
    X, y, feature_names, target_names = load_and_explore_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Train the Gradient Boosting Classifier
    gbm_model = train_gbm_classifier(
        X_train,
        y_train,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
    )

    # Evaluate the model
    acc, prec, rec, f1, cm = evaluate_model(
        gbm_model,
        X_test,
        y_test,
        target_names,
    )

    # Plot feature importance
    plot_feature_importance(gbm_model, feature_names, top_n=10)


if __name__ == "__main__":
    main()