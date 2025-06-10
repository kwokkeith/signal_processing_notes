import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift

def plot_raw_sar_data(raw_data: np.ndarray, sampling_frequency_hz: float, downsample_fast: int = 10, xlim_microseconds: tuple = None, title="Simulated Raw SAR Data"):
    """
    Plot raw SAR data (Pulse Index vs Fast Time) in dB scale.
    """
    energy = np.abs(raw_data).max(axis=0)
    nonzero_indices = np.where(energy > 0.0)[0]

    if len(nonzero_indices) > 0 and xlim_microseconds is None:
        start_us = nonzero_indices[0] / sampling_frequency_hz * 1e6
        end_us = nonzero_indices[-1] / sampling_frequency_hz * 1e6
        xlim_microseconds = (start_us - 1, end_us + 1)
        print(f"Suggested xlim: ({start_us:.1f} μs, {end_us:.1f} μs)")
    elif xlim_microseconds is None:
        xlim_microseconds = (0, raw_data.shape[1] / sampling_frequency_hz * 1e6)

    if downsample_fast <= 0:
        downsample_fast = 1

    plt.figure(figsize=(10, 6))
    plt.imshow(
        20 * np.log10(np.abs(raw_data[:, ::downsample_fast]) + 1e-6),
        aspect='auto', cmap='viridis', interpolation='nearest',
        extent=[0, raw_data.shape[1] / sampling_frequency_hz * 1e6, 0, raw_data.shape[0]]
    )
    plt.colorbar(label='Amplitude (dB)')
    plt.xlim(xlim_microseconds)
    plt.xlabel('Fast Time (µs)')
    plt.ylabel('Pulse Index')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_slant_range_time_delays(time_delays: np.ndarray):
    """
    Plot Slant Range Time Delays vs Pulse Index.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(time_delays * 1e6)  # Convert to microseconds
    plt.xlabel('Pulse Index')
    plt.ylabel('Two-way Time Delay (µs)')
    plt.title('Slant Range Time Delays')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_phase_vs_azimuth(data, sampling_frequency, fast_time_index=None, fast_time_us=None):
    """
    Plots the unwrapped phase of the SAR signal at a given fast-time index or fast-time in microseconds.

    Parameters:
    - data: 2D complex SAR array (azimuth × fast time)
    - sampling_frequency: Hz
    - fast_time_index: Index into the fast-time axis (optional)
    - fast_time_us: Fast time (in μs) to be converted to index (optional)
    """
    if fast_time_index is None and fast_time_us is None:
        raise ValueError("You must provide either fast_time_index or fast_time_us.")

    if fast_time_us is not None:
        fast_time_index = int((fast_time_us * 1e-6) * sampling_frequency)

    if fast_time_index >= data.shape[1]:
        raise IndexError("fast_time_index out of bounds.")

    phase = np.unwrap(np.angle(data[:, fast_time_index]))

    plt.figure(figsize=(8, 5))
    plt.plot(phase)
    plt.xlabel("Azimuth Index (Pulse Number)")
    plt.ylabel("Unwrapped Phase (radians)")
    plt.title(f"Phase vs Azimuth at Fast Time Index {fast_time_index}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_chirp_frequency_spectrum(chirp: np.ndarray, sampling_frequency_hz: float, xlim_mhz: tuple = None):
    """
    Plot the frequency spectrum of a chirp signal.
    """
    n = len(chirp)
    chirp_fft = fftshift(fft(chirp))
    freqs = fftshift(fftfreq(n, d=1 / sampling_frequency_hz)) / 1e6  # MHz

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, 20 * np.log10(np.abs(chirp_fft) + 1e-6))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Chirp Frequency Spectrum')
    if xlim_mhz:
        plt.xlim(xlim_mhz)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_raw_sar_data_window(raw_data: np.ndarray, pulse_range: tuple = (0, 100), range_bin_range: tuple = (0, 1000)):
    """
    Plot a windowed portion of the raw SAR data.

    Parameters:
    - raw_data: np.ndarray, raw SAR data [pulses x fast-time]
    - pulse_range: tuple, (start_idx, end_idx) for pulses
    - range_bin_range: tuple, (start_idx, end_idx) for fast-time samples
    """
    data = raw_data[pulse_range[0]:pulse_range[1], range_bin_range[0]:range_bin_range[1]]
    plt.figure(figsize=(10, 6))
    plt.imshow(
        20 * np.log10(np.abs(data) + 1e-6),
        aspect='auto', cmap='viridis', interpolation='nearest'
    )
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('Fast Time Index (windowed)')
    plt.ylabel('Pulse Index (windowed)')
    plt.title('Windowed SAR Raw Data')
    plt.tight_layout()
    plt.show()


def plot_all_diagnostics(raw_data: np.ndarray, time_delays: np.ndarray, chirp: np.ndarray, sampling_frequency_hz: float, sample_index: int = 0):
    """
    Run all standard diagnostic plots for SAR simulation.
    """
    plot_raw_sar_data(raw_data, sampling_frequency_hz)
    plot_slant_range_time_delays(time_delays)
    plot_phase_vs_azimuth(raw_data, sample_index)
    plot_chirp_frequency_spectrum(chirp, sampling_frequency_hz)
