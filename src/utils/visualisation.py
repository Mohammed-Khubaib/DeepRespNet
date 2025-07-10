import matplotlib.pyplot as plt
import librosa
from sklearn.metrics import confusion_matrix
from features.audio_augmentations import add_noise,shift,stretch,pitch_shift
import numpy as np
from tensorflow.keras.models import  Model # type: ignore
import seaborn as sns

def show_audio(audio_path: str) -> None:
    """
    Display waveform visualizations of an audio file with various data augmentation techniques.
    
    This function loads an audio file and applies multiple audio augmentation techniques
    (noise addition, time shifting, time stretching, and pitch shifting) to demonstrate
    their effects on the original waveform. All variations are displayed in a 3x2 subplot
    grid for visual comparison.
    
    The function applies the following augmentations with fixed parameters:
    - Noise addition (factor: 0.0008)
    - Time shifting (shift: 3200 samples)
    - Time stretching (rates: 1.2x and 0.8x)
    - Pitch shifting (semitones: +3)
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file to be loaded and visualized. Should be a valid audio
        file format supported by librosa (e.g., .wav, .mp3, .flac).
    
    Returns
    -------
    None
        This function displays the plots directly and does not return any values.
    """
    y, sr = librosa.load(audio_path)
    y_noise = add_noise(y , 0.0008)
    y_shift = shift(y,3200)
    y_stretch_1 = stretch(y, 1.2)
    y_stretch_2 = stretch(y, 0.8)
    y_pitch_shift = pitch_shift(y, 3)
    
    plt.figure(figsize=(20, 8))
    
    plt.subplot(3,2,1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('orginal')

    plt.subplot(3,2,2)
    librosa.display.waveshow(y_noise, sr=sr)
    plt.title('noise')

    plt.subplot(3,2,3)
    librosa.display.waveshow(y_shift, sr=sr)
    plt.title('shift')
    
    plt.subplot(3,2,4)
    librosa.display.waveshow(y_stretch_1, sr=sr)
    plt.title('stretch 1')
    
    plt.subplot(3,2,5)
    librosa.display.waveshow(y_stretch_2, sr=sr)
    plt.title('stretch 2')
    
    plt.subplot(3,2,6)
    librosa.display.waveshow(y_pitch_shift, sr=sr)
    plt.title('pitch shift')

    plt.tight_layout()

def show_audio_features(audio_path: str) -> None:
    """
    Display MFCC feature visualizations of an audio file with various data augmentation techniques.
    
    This function loads an audio file, applies multiple audio augmentation techniques,
    extracts Mel-frequency Cepstral Coefficients (MFCC) features from each variation,
    and displays them as spectrograms in a 3x2 subplot grid. This visualization helps
    understand how different augmentations affect the extracted audio features used
    in machine learning models.
    
    The function applies the following augmentations with fixed parameters:
    - Noise addition (factor: 0.0008)
    - Time shifting (shift: 3200 samples)  
    - Time stretching (rates: 1.2x and 0.8x)
    - Pitch shifting (semitones: +3)
    
    For each variation, 50 MFCC coefficients are extracted and displayed as
    mel-scale spectrograms with power expressed in decibels.
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file to be loaded and analyzed. Should be a valid audio
        file format supported by librosa (e.g., .wav, .mp3, .flac).
    
    Returns
    -------
    None
        This function displays the MFCC spectrograms directly and does not return 
        any values.
    
    Notes
    -----
    - Requires librosa, matplotlib, numpy, and the custom augmentation functions 
      (add_noise, shift, stretch, pitch_shift) to be available in scope
    - Creates a matplotlib figure with 6 subplots arranged in a 3x2 grid
    - Each subplot shows MFCC features as a mel-scale spectrogram in dB
    - MFCC extraction parameters:
      * n_mfcc=50: Number of MFCC coefficients to extract
      * Uses default hop_length (512 samples) and n_fft (2048 samples)
    - Spectrogram display parameters:
      * y_axis='mel': Mel-frequency scale on y-axis
      * x_axis='time': Time scale on x-axis  
      * fmax=8000: Maximum frequency displayed (8 kHz)
      * Colorbar shows power in dB scale (+dB format)
    - All spectrograms use the same color scale for comparison
    
    Technical Details
    -----------------
    MFCC Shape: Each MFCC array has shape (n_mfcc, n_frames) where:
    - n_mfcc = 50 (number of coefficients)
    - n_frames depends on audio duration and hop_length
    
    Power Conversion: Uses librosa.power_to_db() with reference to maximum
    power for consistent dB scaling across all visualizations.
    """
    y, sr = librosa.load(audio_path)
    y_noise = add_noise(y , 0.0008)
    y_shift = shift(y,3200)
    y_stretch_1 = stretch(y, 1.2)
    y_stretch_2 = stretch(y, 0.8)
    y_pitch_shift = pitch_shift(y, 3)
    
    y = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
    y_noise = librosa.feature.mfcc(y=y_noise, sr=sr, n_mfcc=50)
    y_shift = librosa.feature.mfcc(y=y_shift, sr=sr, n_mfcc=50)
    y_stretch_1 = librosa.feature.mfcc(y=y_stretch_1, sr=sr, n_mfcc=50)
    y_stretch_2 = librosa.feature.mfcc(y=y_stretch_2, sr=sr, n_mfcc=50)
    y_pitch_shift = librosa.feature.mfcc(y=y_pitch_shift, sr=sr, n_mfcc=50)
    
    plt.figure(figsize=(20, 8))
    
    plt.subplot(3,2,1)
    librosa.display.specshow(librosa.power_to_db(y,ref=np.max),
                             y_axis='mel',
                             fmax=8000,
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('orginal')

    plt.subplot(3,2,2)
    librosa.display.specshow(librosa.power_to_db(y_noise,ref=np.max),
                             y_axis='mel',
                             fmax=8000,
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('noise')

    plt.subplot(3,2,3)
    librosa.display.specshow(librosa.power_to_db(y_shift,ref=np.max),
                             y_axis='mel',
                             fmax=8000,
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('shift')
    
    plt.subplot(3,2,4)
    librosa.display.specshow(librosa.power_to_db(y_stretch_1,ref=np.max),
                             y_axis='mel',
                             fmax=8000,
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('stretch 1')
    
    plt.subplot(3,2,5)
    librosa.display.specshow(librosa.power_to_db(y_stretch_2,ref=np.max),
                             y_axis='mel',
                             fmax=8000,
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('stretch 2')
    
    plt.subplot(3,2,6)
    librosa.display.specshow(librosa.power_to_db(y_pitch_shift,ref=np.max),
                             y_axis='mel',
                             fmax=8000,
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('pitch shift')
    
    

    plt.tight_layout()

def augmented_lables_count(lables):
    """
    Generate a bar plot showing the count of each unique label/diagnosis and print the count dictionary.
    
    This function takes an array of labels, counts the occurrences of each unique label,
    creates a horizontal bar chart visualization, and prints the count statistics.
    
    Parameters:
    -----------
    lables : numpy.ndarray or array-like
        Array containing labels/diagnoses to be counted. Can be strings, integers, 
        or any hashable data type.
    
    Returns:
    --------
    None
        This function does not return any value. It displays a matplotlib plot 
        and prints the count dictionary to the console.
    
    Side Effects:
    -------------
    - Displays a bar plot using matplotlib
    - Prints the count dictionary to console
    
    Dependencies:
    -------------
    - numpy (as np)
    - matplotlib.pyplot (as plt)
    
    """
    unique, counts = np.unique(lables, return_counts=True)
    data_count = dict(zip(unique, counts))
    data = data_count
    courses = list(data.keys())
    values = list(data.values())
    plt.figure(figsize = (10, 5))
    # creating the bar plot
    plt.bar(courses, values, color =['orange','green','blue'],
            width = 0.4)
    plt.xlabel("Diseases")
    plt.ylabel("Count")
    plt.grid(axis = 'y',color = 'green', linestyle = '--', linewidth = 0.5)
    plt.title("Count of each diagnosis")
    plt.show()
    print(data_count)

def plot_loss_curves(history):
    """
    Plot training and validation loss and accuracy curves from a Keras training history.
    
    This function creates two separate plots: one showing the loss curves and another 
    showing the accuracy curves over training epochs. Both training and validation 
    metrics are displayed for comparison.
    
    Parameters:
    -----------
    history : keras.callbacks.History or similar object
        The history object returned by model.fit() containing training metrics.
        Must have a .history attribute that is a dictionary containing:
        - 'loss': list of training loss values per epoch
        - 'val_loss': list of validation loss values per epoch  
        - 'accuracy': list of training accuracy values per epoch
        - 'val_accuracy': list of validation accuracy values per epoch
    
    Returns:
    --------
    None
        This function does not return any value. It displays two matplotlib plots.
    
    Side Effects:
    -------------
    - Displays two matplotlib figures (loss plot and accuracy plot)
    
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()

    # Plot accuracy
    plt.figure()
    plt.grid()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

def confusion_matrix_classes(deeprespnet_model: Model, x_test, y_test):
    classes = ['Chronic','Acute','Healthy']

    preds = deeprespnet_model.predict(x_test)
    classpreds = [np.argmax(t) for t in preds ]
    y_testclass = [np.argmax(t) for t in y_test]
    cm = confusion_matrix(y_testclass, classpreds)

    plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', xticklabels=classes, yticklabels=classes)

    plt.title('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.show(ax)

    return y_testclass, classpreds