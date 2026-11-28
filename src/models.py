from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import (
    SelectKBest,
    f_classif
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# Number of features to select
DEFAULT_K_FEATURES = 50


def get_feature_selector(
    n_features,
    k=DEFAULT_K_FEATURES
):
    """
    Select the best numerical features using ANOVA F-value.
    """

    selected_features = min(
        k,
        n_features
    )

    return SelectKBest(
        score_func=f_classif,
        k=selected_features
    )


def build_logistic_model(
    n_features,
    C=0.01
):
    """
    Create the Logistic Regression pipeline.
    """

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "feature_selection",
            get_feature_selector(n_features)
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=5000,
                random_state=42
            )
        )
    ])


def build_random_forest_model(
    n_features
):
    """
    Create the Random Forest pipeline.
    """

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "feature_selection",
            get_feature_selector(n_features)
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            )
        )
    ])


def build_xgboost_model(
    n_features
):
    """
    Create the XGBoost pipeline.
    """

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "feature_selection",
            get_feature_selector(n_features)
        ),

        (
            "classifier",
            XGBClassifier(
                n_estimators=100,
                max_depth=1,
                learning_rate=0.03,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=6,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            )
        )
    ])