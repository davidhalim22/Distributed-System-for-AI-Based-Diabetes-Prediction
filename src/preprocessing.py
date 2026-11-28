import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    TARGET_COLUMN,
    ID_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    PROCESSED_DATA_DIR
)


def load_dataset(file_path=DATASET_PATH):
    """
    Load the processed GSE38642 dataset.

    Parameters:
        file_path: Location of the dataset CSV file.

    Returns:
        DataFrame containing the dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df


def prepare_features_target(df):
    """
    Separate the dataset into features and target.

    Diabetes_Status is the target.
    Sample_ID is excluded from the features.
    """

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "The target column contains missing values."
        )

    # The original project uses binary labels 0 and 1.
    unique_labels = set(df[TARGET_COLUMN].unique())

    if unique_labels != {0, 1}:
        raise ValueError(
            "Diabetes_Status must contain binary labels 0 and 1. "
            f"Found: {unique_labels}"
        )

    y = df[TARGET_COLUMN].astype(int)

    # Exclude the target and sample identifier.
    columns_to_drop = [TARGET_COLUMN]

    if ID_COLUMN in df.columns:
        columns_to_drop.append(ID_COLUMN)

    X = df.drop(
        columns=columns_to_drop
    ).copy()

    # Ensure all model features are numerical.
    non_numeric_columns = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric_columns:
        raise TypeError(
            "Non-numeric feature columns found: "
            f"{non_numeric_columns}"
        )

    # Replace infinite values with missing values.
    # The model pipelines will handle missing feature values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X.shape[1] == 0:
        raise ValueError(
            "No numerical features are available."
        )

    # Columns with no values cannot be meaningfully imputed.
    completely_empty_columns = X.columns[
        X.isnull().all()
    ].tolist()

    if completely_empty_columns:
        raise ValueError(
            "These feature columns contain only missing values: "
            f"{completely_empty_columns}"
        )

    print("Features and target separated successfully.")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


def split_dataset(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
):
    """
    Split the dataset into training and testing sets.

    Stratification preserves the class distribution.
    """

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each target class needs at least two samples "
            "for a stratified train/test split."
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def save_splits(
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir=PROCESSED_DATA_DIR
):
    """
    Save training and testing datasets as CSV files.

    Diabetes_Status is included in each saved dataset.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Combine features and target.
    train_data = X_train.copy()

    train_data[TARGET_COLUMN] = y_train.values

    test_data = X_test.copy()

    test_data[TARGET_COLUMN] = y_test.values

    # Save CSV files.
    train_path = output_dir / "train.csv"

    test_path = output_dir / "test.csv"

    train_data.to_csv(
        train_path,
        index=False
    )

    test_data.to_csv(
        test_path,
        index=False
    )

    print("Training data saved:", train_path)

    print("Testing data saved:", test_path)

    return train_path, test_path


def load_saved_split(file_path):
    """
    Load a previously saved training or testing dataset.

    Returns:
        X: Features
        y: Target
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Saved dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    y = df[TARGET_COLUMN].astype(int)

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
        raise TypeError(
            "The saved dataset contains non-numeric features."
        )

    return X, y