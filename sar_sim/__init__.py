# chirp.py
from .chirp import (
    generate_chirp_time_domain,
    generate_baseband_chirp_time_domain,
    apply_time_delay_frequency_domain,
    iq_demodulate_chirp,
    apply_phase_alignment,
    apply_sample_shift_alignment,
)

# geometry.py
from .geometry import (
    compute_target_time_delays,
)

# matched_filter.py
from .matched_filter import (
    matched_filter_frequency_domain,
)

# simulator.py
from .simulator import (
    SensorParameters,
    simulate_raw_data,
    simulate_range_compression,
    perform_range_migration,
    simulate_azimuth_compression,
)

# plotter.py
from .plotter import (
    plot_raw_sar_data,
    plot_slant_range_time_delays,
    plot_phase_vs_azimuth,
    plot_chirp_frequency_spectrum,
    plot_raw_sar_data_window,
    plot_all_diagnostics,
)

# utils.py
from .utils import (
    calculate_integration_angle,
    zero_pad_signal,
    zero_pad_to_next_pow2,
    next_power_of_2,
)
