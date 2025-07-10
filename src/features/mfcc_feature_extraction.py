# This is the New Function That must be used from now on and not this `mffcs_feature_exteraction`
import os
import pandas as pd
import numpy as np
import librosa
from features.audio_augmentations import add_noise, shift, stretch, pitch_shift

from typing import Tuple, Optional, Dict, Any
def mfcc_feature_extraction(
    dir_: str, 
    diagnosis_df: pd.DataFrame,
    n_mfcc: int = 52,
    max_chronic_per_patient: int = 2,
    augmentation_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract MFCC features from audio files with class-balanced data augmentation.
    
    This function processes audio files from a directory, extracts MFCC (Mel-frequency
    Cepstral Coefficients) features, and applies data augmentation selectively based
    on diagnosis class to create a balanced dataset. Chronic cases are limited to
    prevent class imbalance, while Acute and Healthy cases are augmented with multiple
    techniques to increase dataset size.
    
    The function implements different processing strategies per diagnosis:
    - **Chronic**: Limited sampling (max per patient) to prevent overrepresentation
    - **Acute/Healthy**: Full augmentation with 6 variations per original sample
    
    Data augmentation techniques applied to Acute/Healthy cases:
    - Original audio
    - Noise addition
    - Time shifting
    - Time stretching (2 rates)
    - Pitch shifting
    
    Parameters
    ----------
    dir_ : str
        Path to directory containing audio files (.wav format). Directory should
        contain audio files with naming convention where first 3 characters 
        represent patient ID (e.g., "001_recording.wav").
    diagnosis_df : pd.DataFrame
        DataFrame containing patient diagnosis information with columns:
        - 'pid': Patient ID (int) matching first 3 characters of audio filenames
        - 'diagnosis': Diagnosis label ('Chronic', 'Acute', or 'Healthy')
    n_mfcc : int, default=52
        Number of MFCC coefficients to extract from each audio file.
    max_chronic_per_patient : int, default=2
        Maximum number of chronic recordings to process per patient to prevent
        class imbalance.
    augmentation_params : dict, optional
        Dictionary containing augmentation parameters. If None, uses defaults:
        {
            'noise_factor': 0.001,
            'shift_samples': 1600, 
            'stretch_rate_1': 1.2,
            'stretch_rate_2': 0.8,
            'pitch_shift_steps': 3
        }
    
    Returns
    -------
    X_data : np.ndarray, shape=(n_samples, n_mfcc)
        Feature matrix where each row represents one audio sample's MFCC features.
        Features are averaged across time frames.
        Data type is float64.
    y_data : np.ndarray, shape=(n_samples,)
        Target labels corresponding to each sample in X_data.
        Contains strings: 'Chronic', 'Acute', or 'Healthy'.
    
    Notes
    -----
    **Class Balancing Strategy:**
    - Chronic cases: Limited to max recordings per patient to prevent overrepresentation
    - Acute/Healthy cases: Each original file generates 6 samples (1 original + 5 augmented)
    
    **Feature Extraction:**
    - Uses configurable number of MFCC coefficients (default: 52)
    - Features are averaged across time frames using np.mean(..., axis=0)
    - Librosa loads audio with 'kaiser_fast' resampling for speed
    
    **File Processing:**
    - Only processes files ending with '.wav'
    - Patient ID extracted from first 3 characters of filename
    - Chronic patient tracking uses first 7 characters of filename
    
    Examples
    --------
    >>> import pandas as pd
    >>> diagnosis_df = pd.read_csv('patient_diagnosis.csv')
    >>> X_features, y_labels = mfcc_feature_extraction(
    ...     'data/audio_files/', 
    ...     diagnosis_df
    ... )
    >>> print(f"Dataset shape: {X_features.shape}")
    >>> print(f"Label distribution: {np.unique(y_labels, return_counts=True)}")
    
    >>> # With custom parameters
    >>> custom_params = {
    ...     'noise_factor': 0.002,
    ...     'shift_samples': 2000,
    ...     'stretch_rate_1': 1.3,
    ...     'stretch_rate_2': 0.7,
    ...     'pitch_shift_steps': 4
    ... }
    >>> X_custom, y_custom = mfcc_feature_extraction(
    ...     'data/audio_files/', 
    ...     diagnosis_df,
    ...     n_mfcc=40,
    ...     max_chronic_per_patient=3,
    ...     augmentation_params=custom_params
    ... )
    
    See Also
    --------
    librosa.feature.mfcc : MFCC feature extraction
    add_noise : Noise augmentation function
    shift : Time shifting function
    stretch : Time stretching function
    pitch_shift : Pitch shifting function
    """
    
    # Validate inputs
    if not os.path.exists(dir_):
        raise FileNotFoundError(f"Directory {dir_} does not exist")
    
    if not {'pid', 'diagnosis'}.issubset(diagnosis_df.columns):
        raise ValueError("diagnosis_df must contain 'pid' and 'diagnosis' columns")
    
    # Set default augmentation parameters
    if augmentation_params is None:
        augmentation_params = {
            'noise_factor': 0.001,
            'shift_samples': 1600,
            'stretch_rate_1': 1.2,
            'stretch_rate_2': 0.8,
            'pitch_shift_steps': 3
        }
    
    # Initialize containers
    X_features = []
    y_labels = []
    
    # Track chronic patients to limit samples per patient
    # chronic_patients = []
    chronic_patient_counts = {}
    
    # Get list of wav files
    wav_files = [f for f in os.listdir(dir_) if f.endswith('.wav')]
    
    if not wav_files:
        raise ValueError(f"No .wav files found in directory {dir_}")
    
    print(f"Processing {len(wav_files)} audio files...")
    
    for i, sound_file in enumerate(wav_files):
        try:
            # Extract patient ID from filename (first 3 characters)
            patient_id = int(sound_file[:3])
            
            # Get diagnosis for this patient
            patient_data = diagnosis_df[diagnosis_df['pid'] == patient_id]
            if patient_data.empty:
                print(f"Warning: Patient ID {patient_id} not found in diagnosis_df, skipping {sound_file}")
                continue
                
            diagnosis = patient_data['diagnosis'].iloc[0]
            
            # Full file path
            file_path = os.path.join(dir_, sound_file)
            
            # Load audio
            audio_data, sampling_rate = librosa.load(file_path, res_type='kaiser_fast')
            
            if diagnosis == 'Chronic':
                # Handle chronic cases with limited sampling
                patient_key = sound_file[:7]  # First 7 characters for patient tracking
                
                # Initialize count for new patients
                if patient_key not in chronic_patient_counts:
                    chronic_patient_counts[patient_key] = 0
                
                # Only process if under the limit
                if chronic_patient_counts[patient_key] < max_chronic_per_patient:
                    mfcc_features = _extract_mfcc_features(audio_data, sampling_rate, n_mfcc)
                    X_features.append(mfcc_features)
                    y_labels.append(diagnosis)
                    chronic_patient_counts[patient_key] += 1
                    
            elif diagnosis in ['Acute', 'Healthy']:
                # Apply full augmentation for Acute and Healthy cases
                
                # 1. Original audio
                mfcc_original = _extract_mfcc_features(audio_data, sampling_rate, n_mfcc)
                X_features.append(mfcc_original)
                y_labels.append(diagnosis)
                
                # 2. Noise addition
                audio_noise = add_noise(audio_data, augmentation_params['noise_factor'])
                mfcc_noise = _extract_mfcc_features(audio_noise, sampling_rate, n_mfcc)
                X_features.append(mfcc_noise)
                y_labels.append(diagnosis)
                
                # 3. Time shifting
                audio_shift = shift(audio_data, augmentation_params['shift_samples'])
                mfcc_shift = _extract_mfcc_features(audio_shift, sampling_rate, n_mfcc)
                X_features.append(mfcc_shift)
                y_labels.append(diagnosis)
                
                # 4. Time stretching (rate 1)
                audio_stretch_1 = stretch(audio_data, augmentation_params['stretch_rate_1'])
                mfcc_stretch_1 = _extract_mfcc_features(audio_stretch_1, sampling_rate, n_mfcc)
                X_features.append(mfcc_stretch_1)
                y_labels.append(diagnosis)
                
                # 5. Time stretching (rate 2)
                audio_stretch_2 = stretch(audio_data, augmentation_params['stretch_rate_2'])
                mfcc_stretch_2 = _extract_mfcc_features(audio_stretch_2, sampling_rate, n_mfcc)
                X_features.append(mfcc_stretch_2)
                y_labels.append(diagnosis)
                
                # 6. Pitch shifting (FIXED: now uses MFCC instead of melspectrogram)
                audio_pitch_shift = pitch_shift(audio_data, augmentation_params['pitch_shift_steps'])
                mfcc_pitch_shift = _extract_mfcc_features(audio_pitch_shift, sampling_rate, n_mfcc)
                X_features.append(mfcc_pitch_shift)
                y_labels.append(diagnosis)
            
            else:
                print(f"Warning: Unknown diagnosis '{diagnosis}' for patient {patient_id}, skipping")
                
        except Exception as e:
            print(f"Error processing {sound_file}: {str(e)}")
            continue
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(wav_files)} files...")
    
    # Convert to numpy arrays
    X_data = np.array(X_features)
    y_data = np.array(y_labels)
    
    print("\nFeature extraction completed!")
    print(f"Final dataset shape: {X_data.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(y_data, return_counts=True)))}")
    
    return X_data, y_data


def _extract_mfcc_features(audio_data: np.ndarray, sampling_rate: int | float, n_mfcc: int) -> np.ndarray:
    """
    Helper function to extract MFCC features from audio data.
    
    Parameters
    ----------
    audio_data : np.ndarray
        Audio time series data
    sampling_rate : int  
        Sampling rate of the audio
    n_mfcc : int
        Number of MFCC coefficients to extract
        
    Returns
    -------
    np.ndarray
        Mean MFCC features across time frames, shape=(n_mfcc,)
    """
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc)
    return np.mean(mfcc_features.T, axis=0)
