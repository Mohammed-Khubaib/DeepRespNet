# import librosa
# import numpy as np
# from typing import List

# # Import augmentations from audio_augmentations.py
# from features.audio_augmentations import  stretch


# def gru_diagnosis_prediction(
#     test_audio: str,
#     model,
#     classes: List[str] = ['Chronic', 'Acute', 'Healthy']
# ) :
#     data_x, sampling_rate = librosa.load(test_audio)
#     data_x = stretch (data_x,1.2)

#     features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52).T,axis = 0)

#     features = features.reshape(1,52)

#     test_pred = model.predict(np.expand_dims(features, axis = 1))
#     classpreds = classes[np.argmax(test_pred[0], axis=1)[0]]
#     confidence = test_pred.T[test_pred[0].mean(axis=0).argmax()].mean()

#     print (classpreds , confidence)

#     return classpreds, confidence

import librosa
import numpy as np
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))
# Import augmentations from audio_augmentations.py
from src.features.audio_augmentations import  stretch

def audio_preprocessing(
    audio_input: str,
    n_mfcc: int = 52,
    stretch_rate: float = 1.2
) -> np.ndarray:
    """
    Load an audio file or file-like object, apply time-stretching, extract MFCC features,
    and return a feature vector.

    Args:
        audio_input (str or IO[bytes]): Path to the audio file or a file-like object (e.g., BytesIO).
        n_mfcc (int): Number of MFCC features to extract. Default is 52.
        stretch_rate (float): Rate to stretch the audio. Default is 1.2.

    Returns:
        np.ndarray: A 2D array of shape (1, n_mfcc) with dtype float32.
    """
    try:
        # If input is bytes, wrap it into a file-like object
        data_x, sampling_rate = librosa.load(audio_input, sr=None)
        data_x = librosa.util.normalize(data_x)
        data_x = stretch(data_x, stretch_rate)

        mfccs = librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=n_mfcc)
        features = np.mean(mfccs.T, axis=0)

        return features.reshape(1, n_mfcc).astype(np.float32)
    
    except Exception as e:
        raise RuntimeError(f"Error processing audio input: {e}")



def deeprespnet_diagnosis_prediction(
    features,
    model,
    classes: List[str] = ['Chronic', 'Acute', 'Healthy'],
    use_bento_model = False
) :
    if(use_bento_model):
        test_pred = model.predict.run(np.expand_dims(features, axis=1))
        predicted_class = classes[np.argmax(test_pred[0], axis=1)[0]]
        confidence = test_pred.T[test_pred[0].mean(axis=0).argmax()].mean()
    else:
        test_pred = model.predict(np.expand_dims(features, axis = 1))
        predicted_class = classes[np.argmax(test_pred[0], axis=1)[0]]
        confidence = test_pred.T[test_pred[0].mean(axis=0).argmax()].mean()

    # print (predicted_class , confidence)

    return predicted_class, confidence