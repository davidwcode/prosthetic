from scipy import signal
import numpy as np
import time

# --------------------------------------------
# Filtering function: notch and bandpass filters
# --------------------------------------------
def filter_emg_data(data, fs):
    """
    Apply a notch filter at 60 Hz and a bandpass filter from 10 Hz to 200 Hz.
    
    data: numpy array of shape (n_samples, n_channels)
    fs: sampling rate (Hz)
    """
    # --- Notch filter (60 Hz) ---
    notch_freq = 60.0  # Frequency to be removed from signal (Hz)
    Q = 30.0           # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)
    
    # --- Bandpass filter (10 Hz - 200 Hz) ---
    lowcut = 10.0
    highcut = 200.0
    order = 2
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b_band, a_band = signal.butter(order, [low, high], btype='band')
    
    # Apply filters channel-by-channel using zero-phase filtering.
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        # Apply notch filter
        channel_data = signal.filtfilt(b_notch, a_notch, channel_data)
        # Apply bandpass filter
        channel_data = signal.filtfilt(b_band, a_band, channel_data)
        filtered_data[:, ch] = channel_data
    return filtered_data

def record_emg(board_shim, num_samples, num_channels, fs):
    # Flush any residual data
    board_shim.get_board_data()
    
    collected_samples = 0
    chunks = []
    while collected_samples < num_samples:
        # Wait until at least one sample is available
        while board_shim.get_board_data_count() < 1:
            time.sleep(0.001)
        # Fetch available data but not more than needed
        num_to_fetch = min(num_samples - collected_samples, board_shim.get_board_data_count())
        chunks.append(board_shim.get_board_data(num_to_fetch)[:num_channels].T)
        collected_samples += num_to_fetch
    raw_data = np.concatenate(chunks, axis=0)
    return filter_emg_data(raw_data, fs)


def display_calibrate(state):
    """
    Collect calibration data for a given gesture state.
    This function waits for a countdown, collects NUM_SAMPLES samples,
    applies filtering, and returns the filtered data.
    """
    samples_list = []
    print("Calibrating starting for " + state + "...")
    time.sleep(2)
    print(state + " in 3 seconds...")
    time.sleep(1)
    print(state + " in 2 seconds...")
    time.sleep(1)
    print(state + " in 1 second...")
    time.sleep(1)
    print(state + " now!")