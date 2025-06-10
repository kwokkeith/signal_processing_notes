import numpy as np
import sar_sim.utils as utils

def matched_filter_frequency_domain(
    signal: np.ndarray,
    reference_chirp: np.ndarray,
    dtype=np.complex64,
    normalise: bool = False
) -> np.ndarray:
    """
    Performs matched filtering in frequency domain using conjugate dot product.

    Parameters:
    - signal: np.ndarray
        Received or simulated chirp signal in time domain (complex).
    - reference_chirp: np.ndarray
        Reference chirp signal in time domain (complex).
    - dtype: np.dtype
        Data type for internal and output arrays (default: complex64).
    - normalise: bool
        If True, normalises output to max magnitude of 1.

    Returns:
    - np.ndarray: Impulse response of matched filtering in time domain
    """
    signal = signal.astype(dtype, copy=False)
    reference_chirp = reference_chirp.astype(dtype, copy=False)

    n = signal.shape[0] + reference_chirp.shape[0] - 1
    n_padded = utils.next_power_of_2(n)

    # Zero-pad both signals
    signal_fft = np.fft.fft(signal, n=n_padded)
    ref_fft = np.fft.fft(reference_chirp, n=n_padded)

    # Matched filtering: conjugate dot product
    result_fft = signal_fft * np.conj(ref_fft)

    # Inverse FFT to get time-domain response
    impulse_response = np.fft.ifft(result_fft).astype(dtype)

    if normalise:
        impulse_response /= np.abs(impulse_response).max()

    return impulse_response