from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple

def prepare_test_train_data(
    mfccs_features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.10,
    val_size: float = 0.20,
    random_state: int = 10
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray]
]:
    """
    Splits MFCC features and corresponding labels into training, validation, and testing sets,
    then reshapes them for GRU model input.

    Parameters:
    -----------
    mfccs_features : np.ndarray
        Array of shape (num_samples, num_mfcc_features) containing MFCC features.
    
    labels : np.ndarray
        Array of shape (num_samples,) or (num_samples, 1) containing class labels.
    
    test_size : float, optional
        Proportion of data to reserve for testing (default is 0.10).
    
    val_size : float, optional
        Proportion of data to reserve for validation from the training set (default is 0.20).
    
    random_state : int, optional
        Random seed for reproducible splits (default is 10).

    Returns:
    --------
    A tuple containing:
        - (x_train, y_train): Training data and labels, shaped for GRU input
        - (x_val, y_val): Validation data and labels, shaped for GRU input
        - (x_test, y_test): Test data and labels, shaped for GRU input

    Example:
    --------
    >>> mfccs = np.random.rand(1000, 52)
    >>> labels = np.random.randint(0, 3, size=1000)
    >>> (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_gru_data(mfccs, labels)
    """

    # Step 1: Split into train+val and test
    mfcc_train_val, mfcc_test, labels_train_val, labels_test = train_test_split(
        mfccs_features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Preserve class distribution
    )

    # Step 2: Split train+val into train and val
    mfcc_train, mfcc_val, labels_train, labels_val = train_test_split(
        mfcc_train_val,
        labels_train_val,
        test_size=val_size / (1 - test_size),  # Adjusted ratio
        random_state=random_state,
        stratify=labels_train_val  # Preserve class distribution
    )

    # Reshape for GRU input [samples, timesteps, features]
    x_train = np.expand_dims(mfcc_train, axis=1)
    x_val = np.expand_dims(mfcc_val, axis=1)
    x_test = np.expand_dims(mfcc_test, axis=1)

    # Ensure labels are properly shaped
    y_train = np.expand_dims(labels_train, axis=1)
    y_val = np.expand_dims(labels_val, axis=1)
    y_test = np.expand_dims(labels_test, axis=1)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)