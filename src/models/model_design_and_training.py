import tensorflow as tf
from tensorflow.keras.models import  Model # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D # type: ignore
from tensorflow.keras.layers import Input, add, Dense, BatchNormalization, GRU # type: ignore
from tensorflow.keras.layers import LeakyReLU # type: ignore
import numpy as np

def create_model() -> Model:
    """
    Builds and returns a hybrid GRU-CNN deep learning model for sequence classification.
    
    The model architecture combines 1D convolutional layers for feature extraction 
    with bidirectional GRU layers (implemented via `go_backwards=True`) to capture 
    temporal dependencies in the input data. It also uses dense layers with LeakyReLU 
    activations for final classification. Multiple branches and skip connections 
    (via addition) are used to improve gradient flow and model performance.

    Input Shape:
        (batch_size, 1, 52): Assumes input sequences of length 52 with 1 channel

    Output Shape:
        (batch_size, 3): Softmax output probabilities for 3 classes

    Returns:
        A compiled Keras Model instance representing the GRU-CNN architecture.
    
    Architecture Summary:
        - Input Layer: Shape (1, 52)
        - Conv1D Layers: Two blocks of Conv1D + MaxPooling + BatchNorm
        - Parallel Bidirectional GRU Branches: Multi-scale temporal processing
        - Skip Connections: Additions between GRU outputs
        - Fully Connected Head: Multiple Dense + LeakyReLU layers
        - Output Layer: 3-node softmax for classification
    """
    Input_Sample = Input(shape=(1,52))

    model_conv = Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu')(Input_Sample)
    model_conv = MaxPooling1D(pool_size=2, strides = 2, padding = 'same')(model_conv)
    model_conv = BatchNormalization()(model_conv)

    model_conv = Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu')(model_conv)
    model_conv = MaxPooling1D(pool_size=2, strides = 2, padding = 'same')(model_conv)
    model_conv = BatchNormalization()(model_conv)

    model_2_1 = GRU(32,return_sequences=True,activation='tanh',go_backwards=True)(model_conv)
    model_2 = GRU(128,return_sequences=True, activation='tanh',go_backwards=True)(model_2_1)

    model_3 = GRU(64,return_sequences=True,activation='tanh',go_backwards=True)(model_conv)
    model_3 = GRU(128,return_sequences=True, activation='tanh',go_backwards=True)(model_3)

    model_x = GRU(64,return_sequences=True,activation='tanh',go_backwards=True)(model_conv)
    model_x = GRU(128,return_sequences=True, activation='tanh',go_backwards=True)(model_x)

    model_add_1 = add([model_3,model_2,model_x])

    model_5 = GRU(128,return_sequences=True,activation='tanh',go_backwards=True)(model_add_1)
    model_5 = GRU(32,return_sequences=True, activation='tanh',go_backwards=True)(model_5)

    model_6 = GRU(64,return_sequences=True,activation='tanh',go_backwards=True)(model_add_1)
    model_6 = GRU(32,return_sequences=True, activation='tanh',go_backwards=True)(model_6)

    model_add_2 = add([model_5,model_6,model_2_1])

    model_7 = Dense(32, activation=None)(model_add_2)
    model_7 = LeakyReLU()(model_7)
    model_7 = Dense(128, activation=None)(model_7)
    model_7 = LeakyReLU()(model_7)

    model_9 = Dense(64, activation=None)(model_add_2)
    model_9 = LeakyReLU()(model_9)
    model_9 = Dense(128, activation=None)(model_9)
    model_9 = LeakyReLU()(model_9)

    model_add_3 = add([model_7,model_9])

    model_10 = Dense(64, activation=None)(model_add_3)
    model_10 = LeakyReLU()(model_10)
    model_10 = Dense(32, activation=None)(model_10)
    model_10 = LeakyReLU()(model_10)
    model_10 = Dense(3, activation="softmax")(model_10)

    deeprespnet_model = Model(inputs=Input_Sample, outputs = model_10)
    return deeprespnet_model

import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import os

def train_model(model: Model, x_train: np.ndarray, y_train: np.ndarray, 
                x_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History:
    """
    Trains the GRU-CNN model on provided training data with validation and saves the best model.

    This function compiles the input Keras model with the Adam optimizer and categorical crossentropy loss,
    sets up training callbacks (early stopping and model checkpointing), and fits the model to the training data.
    
    Parameters:
    -----------
    model : keras.Model
        A compiled or uncompiled Keras model to be trained. Expected to be built using compatible input/output shapes.
    
    x_train : np.ndarray
        Training input data as a NumPy array.
        Shape: (num_samples, 1, 52) — batch of time-series sequences with one channel and 52 features.

    y_train : np.ndarray
        Training labels in one-hot encoded format.
        Shape: (num_samples, 3) — three-class classification problem.

    x_val : np.ndarray
        Validation input data as a NumPy array.
        Shape: (num_samples, 1, 52)

    y_val : np.ndarray
        Validation labels in one-hot encoded format.
        Shape: (num_samples, 3)

    Returns:
    --------
    history : tf.keras.callbacks.History
        A History object containing training metrics (loss, accuracy, val_loss, val_accuracy)
        recorded during training at each epoch.

    Training Configuration:
    -----------------------
    - Optimizer: Adam (learning rate = 0.0001)
    - Loss: Categorical Crossentropy
    - Metrics: Accuracy
    - Epochs: 5
    - Batch Size: 32
    - Callbacks:
        - EarlyStopping (patience=300 epochs, monitoring accuracy)
        - ModelCheckpoint (saves best model to './models/diagnosis_GRU_CNN_6.h5')

    Notes:
    ------
    - The model is not returned directly but the best weights are saved to disk.
    - EarlyStopping monitors 'accuracy' and restores the best weights upon stopping.
    - For reproducibility, ensure random seeds are set before calling this function.
    """
    
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    # Use the input model directly
    gru_model = model
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Compile the model
    gru_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training callbacks - REMOVED save_format parameter
    cb = [
        tf.keras.callbacks.EarlyStopping(
            patience=300,
            monitor='accuracy',
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./models/saved_model.keras",
            save_best_only=True,
            monitor='val_accuracy',  # Added monitor parameter
            mode='max',              # Added mode parameter
            verbose=1                # Added verbose for better feedback
        )
    ]

    # Train the model
    history = gru_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=cb,
        verbose=1  # Added verbose for training progress
    )
    
    print("Model training completed successfully!")
    return history