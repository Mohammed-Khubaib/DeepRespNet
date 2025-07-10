import numpy as np
import librosa
def add_noise(data: np.ndarray, x: float) -> np.ndarray:
    """
    Add random Gaussian noise to an audio signal for data augmentation.
    
    This function generates random noise from a normal distribution and adds it
    to the input audio signal, scaled by a noise factor. This is commonly used
    in audio data augmentation to improve model robustness by simulating 
    real-world noise conditions.
    
    Parameters
    ----------
    data : np.ndarray, shape=(n,)
        Input audio time series as a 1D numpy array containing audio samples.
        Typically obtained from librosa.load().
    x : float
        Noise scaling factor that controls the intensity of added noise.
        Higher values produce more noise. Typical values range from 0.0001 to 0.01.
    
    Returns
    -------
    data_noise : np.ndarray, shape=(n,)
        Audio time series with added Gaussian noise, same shape as input.
        The output maintains the same data type as the input array.
    
    Notes
    -----
    - Uses np.random.randn() to generate standard normal distributed noise
    - The noise is scaled by factor x before being added to the original signal
    - No bounds checking is performed on the output signal amplitude
    - Random seed should be set externally for reproducible results
    
    Examples
    --------
    >>> y, sr = librosa.load("audio.wav")
    >>> y_noisy = add_noise(y, 0.001)
    >>> # Adds subtle noise to the audio signal
    
    >>> y_very_noisy = add_noise(y, 0.01)
    >>> # Adds more prominent noise to the audio signal
    """
    noise = np.random.randn(len(data))
    data_noise = data + x * noise
    return data_noise


def shift(data: np.ndarray, x: int) -> np.ndarray:
    """
    Apply time-domain shifting to an audio signal by rolling samples.
    
    This function shifts audio samples in the time domain by rolling the array
    elements. Positive values shift samples to the right (delayed), while 
    negative values shift samples to the left (advanced). This augmentation
    technique helps models become invariant to temporal shifts in audio data.
    
    Parameters
    ----------
    data : np.ndarray, shape=(n,)
        Input audio time series as a 1D numpy array containing audio samples.
        Typically obtained from librosa.load().
    x : int
        Number of samples to shift. Positive values shift right (delay),
        negative values shift left (advance). Values larger than array length
        will wrap around due to numpy.roll behavior.
    
    Returns
    -------
    shifted_data : np.ndarray, shape=(n,)
        Time-shifted audio signal with the same shape and data type as input.
        Samples that roll off one end appear at the other end (circular shift).
    
    Notes
    -----
    - Uses numpy.roll() which performs circular shifting
    - No audio samples are lost; they wrap around to the opposite end
    - For non-circular shifting, consider padding with zeros instead
    - Shift amount is in samples, not time units
    
    Examples
    --------
    >>> y, sr = librosa.load("audio.wav")
    >>> y_delayed = shift(y, 1000)
    >>> # Delays audio by 1000 samples (~45ms at 22050 Hz)
    
    >>> y_advanced = shift(y, -500)
    >>> # Advances audio by 500 samples (~23ms at 22050 Hz)
    """
    return np.roll(data, x)


def stretch(data: np.ndarray, rate: float) -> np.ndarray:
    """
    Apply time-stretching to an audio signal while preserving pitch.
    
    This function changes the duration of an audio signal without affecting
    its pitch using librosa's phase vocoder-based time stretching. Values
    greater than 1.0 make the audio faster (shorter), while values less than
    1.0 make it slower (longer). This augmentation helps models handle
    variations in speech rate and timing.
    
    Parameters
    ----------
    data : np.ndarray, shape=(n,)
        Input audio time series as a 1D numpy array containing audio samples.
        Typically obtained from librosa.load().
    rate : float
        Time-stretching factor. Must be positive.
        - rate > 1.0: Faster playback (shorter duration)
        - rate < 1.0: Slower playback (longer duration)  
        - rate = 1.0: No change
        Common values range from 0.8 to 1.2.
    
    Returns
    -------
    stretched_data : np.ndarray, shape=(m,)
        Time-stretched audio signal. Output length m = n / rate (approximately).
        Maintains the same data type as input but length will change.
    
    Raises
    ------
    librosa.util.exceptions.ParameterError
        If rate is not positive or if input data is invalid.
    
    Notes
    -----
    - Uses librosa.effects.time_stretch() with phase vocoder algorithm
    - Pitch is preserved during time stretching
    - Output array length will be different from input (length = n / rate)
    - Quality depends on the complexity of the audio signal
    
    Examples
    --------
    >>> y, sr = librosa.load("audio.wav")
    >>> y_fast = stretch(y, 1.2)
    >>> # Makes audio 20% faster (shorter duration)
    
    >>> y_slow = stretch(y, 0.8)  
    >>> # Makes audio 25% slower (longer duration)
    """
    data = librosa.effects.time_stretch(y=data, rate=rate)
    return data


def pitch_shift(data: np.ndarray, rate: float) -> np.ndarray:
    """
    Apply pitch shifting to an audio signal while preserving duration.
    
    This function shifts the pitch of an audio signal by a specified number
    of semitones without changing its duration. Positive values increase pitch
    (higher frequency), while negative values decrease pitch (lower frequency).
    This augmentation helps models become robust to pitch variations in audio.
    
    Parameters
    ----------
    data : np.ndarray, shape=(n,)
        Input audio time series as a 1D numpy array containing audio samples.
        Typically obtained from librosa.load().
    rate : float
        Number of semitones to shift the pitch. Can be positive or negative.
        - Positive values: Higher pitch (e.g., +12 = one octave up)
        - Negative values: Lower pitch (e.g., -12 = one octave down)
        - rate = 0: No change
        Common values range from -5 to +5 semitones.
    
    Returns
    -------
    pitch_shifted_data : np.ndarray, shape=(n,)
        Pitch-shifted audio signal with the same shape and data type as input.
        Duration remains unchanged, only pitch is modified.
    
    Raises
    ------
    librosa.util.exceptions.ParameterError
        If sampling rate is invalid or input data is malformed.
    
    Notes
    -----
    - Uses librosa.effects.pitch_shift() with sampling rate of 22050 Hz
    - WARNING: Hardcoded sampling rate (220250) appears to be a typo and 
      should likely be 22050 Hz (standard rate)
    - Duration is preserved during pitch shifting
    - Uses phase vocoder with time-domain pitch shifting
    - Quality may degrade with extreme pitch shifts (>±12 semitones)
    
    Examples
    --------
    >>> y, sr = librosa.load("audio.wav")
    >>> y_higher = pitch_shift(y, 3)
    >>> # Shifts pitch up by 3 semitones (minor third)
    
    >>> y_lower = pitch_shift(y, -2)
    >>> # Shifts pitch down by 2 semitones (whole tone)
    
    See Also
    --------
    librosa.effects.pitch_shift : Underlying librosa function
    stretch : For time-stretching without pitch change
    """
    data = librosa.effects.pitch_shift(data, sr=220250, n_steps=rate)
    return data