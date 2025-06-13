import numpy as np
import warnings

from scipy import signal
from scipy.signal import butter, filtfilt



def generate_chirp_time_domain(
    delay_s: float,
    sampling_frequency_hz: float,
    carrier_frequency_hz: float,
    bandwidth_hz: float,
    pulse_width_s: float,
    dtype=np.complex64
) -> np.ndarray:
    """
    Generates a time-delayed chirp signal for SAR in the time domain.

    Parameters:
    - delay_s: float, delay to apply (in seconds)
    - sampling_frequency_hz: float, sampling frequency in Hz
    - carrier_frequency_hz: float, carrier frequency in Hz
    - bandwidth_hz: float, chirp bandwidth in Hz
    - pulse_width_s: float, chirp duration in seconds
    - dtype: output data type

    Returns:
    - np.ndarray: complex time-delayed chirp
    """
    fs = sampling_frequency_hz
    fc = carrier_frequency_hz
    bw = bandwidth_hz
    Tp = pulse_width_s
    k = bw / Tp

    num_samples = int(np.floor(fs * Tp))

    # Chirp physically exists between delay_s and delay_s + Tp
    t = np.linspace(delay_s, delay_s + Tp, num_samples, endpoint=False)

    # Phase of LFM chirp centred at Tp/2 after delay
    phase = 2 * np.pi * (fc * t + 0.5 * k * (t - (delay_s + Tp/2))**2)
    chirp = np.exp(1j * phase).astype(dtype)

    return chirp



def apply_time_delay_frequency_domain(
    chirp: np.ndarray,
    time_delay_us: float,
    sampling_frequency_mhz: float = 360*2.5,
    dtype=np.complex64
) -> np.ndarray:
    """
    Applies a time delay (in microseconds) to a chirp using frequency domain phase shift.

    Parameters:
    - chirp: np.ndarray, complex chirp signal
    - time_delay_us: float, time delay in microseconds
    - sampling_frequency_mhz: float, sampling rate in MHz
    - dtype: np.dtype, optional (default: complex64)

    Returns:
    - np.ndarray: delayed chirp in time domain
    """
    time_delay = time_delay_us * 1e-6  # seconds
    fs = sampling_frequency_mhz * 1e6  # Hz
    n = chirp.shape[0]
    freqs = np.fft.fftfreq(n, d=1 / fs).astype(np.float32)

    chirp_fft = np.fft.fft(chirp)
    phase_shift = np.exp(-2j * np.pi * freqs * time_delay).astype(dtype)
    shifted_fft = chirp_fft * phase_shift

    return np.fft.ifft(shifted_fft).astype(dtype)



def iq_demodulate_chirp(
    chirp: np.ndarray,
    carrier_frequency_mhz: float = 360,
    sampling_frequency_mhz: float = 360 * 2.5,
    dtype=np.complex64
) -> np.ndarray:
    """
    Simulates analog IQ demodulation of a passband chirp signal using:
    - In-phase and quadrature mixing
    - Low-pass filtering to extract baseband signal

    Parameters:
    - chirp: np.ndarray
        Real-valued passband chirp signal
    - carrier_frequency_mhz: float
        Carrier frequency in MHz
    - sampling_frequency_mhz: float
        Sampling frequency in MHz
    - dtype: np.dtype
        Output data type (default: np.complex64)

    Returns:
    - np.ndarray: Baseband complex IQ signal (I + jQ)
    """
    fc = carrier_frequency_mhz * 1e6
    fs = sampling_frequency_mhz * 1e6
    n = chirp.shape[0]
    t = np.arange(n) / fs

    # Mix to baseband (I and Q separately)
    i_mixed = chirp * np.cos(2 * np.pi * fc * t)
    q_mixed = chirp * -np.sin(2 * np.pi * fc * t)

    # Design low-pass filter (cutoff = 1.2 * chirp bandwidth, conservatively < fs/2)
    nyq = fs / 2
    cutoff_hz = min(fc / 2, nyq * 0.8)
    b, a = butter(N=5, Wn=cutoff_hz / nyq, btype='low')

    # Apply LPF (simulate analog filtering)
    i_baseband = filtfilt(b, a, i_mixed)
    q_baseband = filtfilt(b, a, q_mixed)

    # Combine to form complex baseband signal
    baseband = (i_baseband + 1j * q_baseband).astype(dtype)
    return baseband



def apply_phase_alignment(
    data: np.ndarray,
    time_result_target: np.ndarray,
    time_result_ref: np.ndarray,
    sampling_frequency: float,
    chirp_start_time: float = 0.0,
) -> np.ndarray:
    """
    Applies range migration (phase-based) alignment to SAR data using time delay difference,
    corrected by the start time of the sampled window.

    Parameters:
    - data: np.ndarray
        Complex SAR data (azimuth × fast time).
    - time_result_target: np.ndarray
        1D array of time delays for the target (in seconds), one per azimuth line.
    - time_result_ref: np.ndarray
        1D array of reference time delays (in seconds), same length as target.
    - sampling_frequency: float
        Sampling frequency in Hz.
    - chirp_start_time: float, optional
        Start time of the chirp window (in seconds), default is 0.0.
        
    Returns:
    - np.ndarray: Phase-aligned SAR data.
    """
    num_azimuth, num_fast_time = data.shape
    if len(time_result_target) != num_azimuth or len(time_result_ref) != num_azimuth:
        raise ValueError("Time delay arrays must match number of azimuth lines.")

    # Frequency axis
    freqs = np.fft.fftfreq(num_fast_time, d=1 / sampling_frequency).astype(np.float32)
    
    # Align delays to fast-time window
    delta_t = time_result_ref - time_result_target # corrected difference

    # FFT along fast-time axis
    data_fft = np.fft.fft(data, axis=1)

    # Apply per-azimuth phase shift
    for i in range(num_azimuth):
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delta_t[i])
        data_fft[i, :] *= phase_shift.astype(data_fft.dtype)

    # IFFT back to time domain
    aligned_data = np.fft.ifft(data_fft, axis=1)
    return aligned_data



def apply_sample_shift_alignment(
    data: np.ndarray,
    time_result_target: np.ndarray,
    time_result_ref: np.ndarray,
    sampling_frequency: float
) -> np.ndarray:
    """
    Applies sample-shift-based range migration.

    Parameters:
    - data: np.ndarray [azimuth × fast time]
    - time_result_target: np.ndarray (s)
    - time_result_ref: np.ndarray (s)
    - sampling_frequency: float (Hz)

    Returns:
    - np.ndarray: Shifted SAR data
    """
    num_azimuth = data.shape[0]
    if len(time_result_target) != num_azimuth or len(time_result_ref) != num_azimuth:
        raise ValueError("Time delay arrays must match number of azimuth lines.")

    delta_samples = np.round((time_result_target - 2 * time_result_ref) * sampling_frequency).astype(int)
    aligned = np.zeros_like(data, dtype=data.dtype)

    for i in range(num_azimuth):
        aligned[i] = np.roll(data[i], -delta_samples[i])  # left shift = earlier arrival

    return aligned




def generate_baseband_chirp_time_domain(
    sampling_frequency_mhz: float,
    bandwidth_mhz: float,
    pulse_width_us: float,
    dtype=np.complex64
) -> np.ndarray:
    """
    Generate a complex baseband chirp signal in the time domain.

    This simulates an LFM chirp with a symmetric spectrum around 0 Hz,
    commonly used for SAR baseband simulation and matched filtering.

    Parameters:
    - sampling_frequency_mhz: float
        Sampling frequency in MHz.
    - bandwidth_mhz: float
        Chirp bandwidth in MHz.
    - pulse_width_us: float
        Chirp duration in microseconds.
    - dtype: np.dtype, optional
        Output data type, default is np.complex64.

    Returns:
    - np.ndarray: Complex-valued baseband chirp signal.
    """
    fs = sampling_frequency_mhz * 1e6     # Convert MHz → Hz
    bw = bandwidth_mhz * 1e6              # Convert MHz → Hz
    Tp = pulse_width_us * 1e-6            # Convert µs → s

    num_samples = int(np.floor(fs * Tp))
    t = np.linspace(-Tp / 2, Tp / 2, num_samples, endpoint=False)

    k = bw / Tp                           # Chirp rate in Hz/s
    phase = np.pi * k * t**2              # Symmetric baseband LFM chirp
    chirp = np.exp(1j * phase).astype(dtype)

    return chirp
