import numpy as np
def encode_labels_to_categorical(y: np.ndarray) -> np.ndarray:
    """
    Encode string labels to one-hot categorical representation.
    
    This function converts string labels ('Chronic', 'Acute', 'Healthy') into 
    one-hot encoded vectors suitable for multi-class classification. Each label 
    is mapped to a specific binary vector representation.
    
    Label Mapping:
    - 'Chronic' -> [1, 0, 0]
    - 'Acute'   -> [0, 1, 0] 
    - 'Healthy' -> [0, 0, 1]
    
    Parameters:
    -----------
    y : numpy.ndarray
        1D array containing string labels to be encoded. Expected values are 
        'Chronic', 'Acute', and/or 'Healthy'.
    
    Returns:
    --------
    Y_data : numpy.ndarray
        2D array of shape (n_samples, 3) containing one-hot encoded labels
        as float64 dtype. Each row represents one sample with exactly one 
        element set to 1.0 and others set to 0.0.
    
    
    Notes:
    ------
    - Input labels must be exactly 'Chronic', 'Acute', or 'Healthy' (case-sensitive)
    - Unknown labels will remain unchanged in the intermediate steps
    - Final output is converted to float64 for compatibility with neural networks
    """
    y_data_encode = y.reshape(y.shape[0], 1)
    y_data_encode = np.where(y_data_encode == 'Chronic', np.array([1, 0, 0]).reshape(1, 3), y_data_encode)
    y_data_encode = np.where(y_data_encode == 'Acute', np.array([0, 1, 0]).reshape(1, 3), y_data_encode)
    y_data_encode = np.where(y_data_encode == 'Healthy', np.array([0, 0, 1]).reshape(1, 3), y_data_encode)
    
    Y_data = y_data_encode.astype('float64')
    return Y_data