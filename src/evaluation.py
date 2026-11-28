import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def get_scoring_metrics():
    """
    Return the scoring metrics used during cross-validation.
    """

    scoring = {
        "balanced_accuracy": "balanced_accuracy",

        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),

        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),

        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),

        "roc_auc": "roc_auc",

        "pr_auc": "average_precision"
    }

    return scoring


def create_stratified_cv(
    y,
    n_splits=5,
    random_state=42
):
    """
    Create a StratifiedKFold cross-validation object.

    Reduce the number of folds if the minority class
    contains fewer than five samples.
    """

    smallest_class = int(
        y.value_counts().min()
    )

    actual_splits = min(
        n_splits,
        smallest_class
    )

    if actual_splits < 2:
        raise ValueError(
            "At least two samples per class are required "
            "for stratified cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=random_state
    )

    return cv


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate a fitted model on the testing dataset.

    Returns:
        metrics: Dictionary of evaluation metrics
        y_pred: Predicted class labels
        y_probability: Predicted positive-class probabilities
    """

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            y_probability
        ),

        "pr_auc": average_precision_score(
            y_test,
            y_probability
        )
    }

    return metrics, y_pred, y_probability