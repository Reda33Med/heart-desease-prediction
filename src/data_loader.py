from pathlib import Path
from typing import Tuple
import pandas as pd

from sklearn.model_selection import train_test_split

def load_data(
        file_path: Path,
        target_column: str = "target"
        ) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the heart disease dataset and split it into features (X) and target (y).

    Parameters:
        file_path (Path): Relative path to the CSV file, so we can import the data.
        target_column (str): Name of the target variable column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame (X) and Target Series (y).
    """
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    """Split the dataset into training and testing sets.

    Parameters:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target Series.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test