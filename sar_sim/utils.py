import numpy as np

def calculate_integration_angle(
    K_a: float = 1.2,
    wavelength: float = 0.03,
    azimuth_resolution: float = 1.0
) -> float:
    """
    Calculate the satellite integration angle (in radians) for SAR imaging.

    The integration angle determines the angular extent over which the target is
    illuminated to achieve the desired azimuth resolution.

    Parameters:
    - K_a (float): Azimuth broadening factor to account for windowing losses (typically 1.2).
    - wavelength (float): Radar wavelength in metres (e.g., 0.03 m for X-band).
    - azimuth_resolution (float): Desired azimuth resolution in metres.

    Returns:
    - float: Integration angle in radians.
    """
    if azimuth_resolution <= 0 or wavelength <= 0 or K_a <= 0:
        raise ValueError("All input parameters must be positive.")

    integration_angle = (K_a * wavelength) / (2 * azimuth_resolution)
    return integration_angle


def zero_pad_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    """
    Zero-pad a 1D signal to a specified target length.

    Parameters:
    - signal: np.ndarray
        Input signal to be zero-padded (1D).
    - target_length: int
        Desired total length after zero-padding (must be >= len(signal)).

    Returns:
    - np.ndarray: Zero-padded signal with shape (target_length,)
    """
    current_length = signal.shape[0]
    if target_length < current_length:
        raise ValueError("Target length must be greater than or equal to signal length.")

    padded_signal = np.zeros(target_length, dtype=signal.dtype)
    padded_signal[:current_length] = signal
    return padded_signal


def zero_pad_to_next_pow2(signal: np.ndarray) -> np.ndarray:
    """
    Zero-pad signal to the next power of 2 in length.
    """
    next_pow2 = int(2 ** np.ceil(np.log2(signal.shape[0])))
    return zero_pad_signal(signal, next_pow2)


def next_power_of_2(n: int) -> int:
    return int(2 ** np.ceil(np.log2(n)))