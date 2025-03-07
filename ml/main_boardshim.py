import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EMGDataset
from torchhd.models import Centroid
from tqdm import tqdm
import torchmetrics

# BoardShim imports from MindRove SDK
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

# For filtering, we use scipy.signal.
from scipy import signal
from model import Encoder

# -------------------------
# Parameters and settings
# -------------------------
NUM_SAMPLES = 10000          # Total samples to collect for each gesture calibration
NUM_CHANNELS = 4             # Number of EMG channels
# Note: The original LSL code segments data into windows of 6 samples. However, for filtering to work properly,
# a larger window may be needed. Adjust as needed:
SAMPLES_PER_POINT = 50       
BATCH_SIZE = 1
STATE_DICT = {0: "relax", 1: "clench", 2: "spiderman"}
SAMPLING_RATE = 500          # Hz, as assumed by your boardshim settings
DIMENSIONS = 10000

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

# --------------------------------------------
# Filtering function: notch and bandpass filters
# --------------------------------------------
def filter_emg_data(data, fs):
    """
    Apply a notch filter at 60 Hz and a bandpass filter from 4.5 Hz to 100 Hz.
    
    data: numpy array of shape (n_samples, n_channels)
    fs: sampling rate (Hz)
    """
    # --- Notch filter (60 Hz) ---
    notch_freq = 60.0  # Frequency to be removed from signal (Hz)
    Q = 30.0           # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)
    
    # --- Bandpass filter (4.5 Hz - 100 Hz) ---
    lowcut = 4.5
    highcut = 100.0
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

# -----------------------------
# Calibration function using boardshim
# -----------------------------
def calibrate(state, board_shim, fs):
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
    
    # Flush any residual data
    board_shim.get_board_data()
    
    collected_samples = 0
    while collected_samples < NUM_SAMPLES:
        # Wait until at least one sample is available
        while board_shim.get_board_data_count() < 1:
            time.sleep(0.001)
        # Fetch available data but not more than needed
        num_to_fetch = min(NUM_SAMPLES - collected_samples, board_shim.get_board_data_count())
        raw_data = board_shim.get_board_data(num_to_fetch)
        if len(raw_data) == 0:
            continue
        # board_shim.get_board_data() returns a list where each element corresponds to a channel.
        # For a 4-channel EMG device, take the first 4 channels and transpose.
        chunk = np.array(raw_data[:NUM_CHANNELS]).T  # Shape: (chunk_samples, NUM_CHANNELS)
        samples_list.append(chunk)
        collected_samples += chunk.shape[0]
    
    # Concatenate chunks and trim to NUM_SAMPLES
    raw_samples = np.concatenate(samples_list, axis=0)[:NUM_SAMPLES]
    
    # Apply filtering to the collected samples
    filtered_samples = filter_emg_data(raw_samples, fs)
    print("Calibration for " + state + " done!")
    return filtered_samples

# ----------------------------
# Main function: setup boardshim, calibration, and data loading
# ----------------------------
def main():
    # Initialize boardshim
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)
    
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)  # Start the stream (the parameter may be adjusted)
        print("Connected to MindRove board and started stream.")
        
        # Warm-up phase: wait for data to accumulate
        warmup_time = 10  # seconds
        print(f"Warming up for {warmup_time} seconds...")
        warmup_start = time.time()
        while time.time() - warmup_start < warmup_time:
            if board_shim.get_board_data_count() > 0:
                # Flush initial data
                board_shim.get_board_data(SAMPLING_RATE)
                break
        
        total_states = len(STATE_DICT)
        # Preallocate a large array to hold all calibration data
        data = np.zeros((total_states * NUM_SAMPLES, NUM_CHANNELS), dtype=np.float64)
        
        # Loop through each calibration gesture/state.
        for i in range(total_states):
            state = STATE_DICT[i]
            state_data = calibrate(state, board_shim, SAMPLING_RATE)
            data[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES] = state_data
        
        # Create labels: here, for each state we create labels for each window.
        num_points_per_state = NUM_SAMPLES // SAMPLES_PER_POINT
        labels = np.concatenate([np.full(num_points_per_state, i, dtype=int) for i in range(total_states)])
        
        # Reshape the data into windows of SAMPLES_PER_POINT samples.
        # Note: If you increase SAMPLES_PER_POINT to ensure good filtering, adjust your model input accordingly.
        data_windows = data.reshape(-1, SAMPLES_PER_POINT, NUM_CHANNELS)
        
        # Convert data to PyTorch tensors
        data_tensor = torch.tensor(data_windows, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Create dataset and dataloader
        dataset = EMGDataset(data_tensor, labels_tensor)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_data_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_data_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # For demonstration, print one batch
        for samples, batch_labels in train_data_loader:
            print("Sample batch of data:", samples)
            print("Data shape:", samples.size(), "Labels shape:", batch_labels.size())
            break

        encoder = Encoder(DIMENSIONS, SAMPLES_PER_POINT, NUM_CHANNELS)
        encoder = encoder.to(device)
        num_classes = total_states
        model = Centroid(DIMENSIONS, num_classes)
        model = model.to(device)

        with torch.no_grad():
            for samples, targets in tqdm(train_data_loader, desc="Training"):
                samples = samples.to(device)
                targets = targets.to(device)

                sample_hv = encoder(samples)
                model.add(sample_hv, targets)

        accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

        with torch.no_grad():
            model.normalize()

            for samples, targets in tqdm(test_data_loader, desc="Testing"):
                samples = samples.to(device)

                sample_hv = encoder(samples)
                output = model(sample_hv, dot=True)
                accuracy.update(output.cpu(), targets)

        print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")


    
    except Exception as e:
        print("Error during data collection:", e)
    
    finally:
        board_shim.stop_stream()
        board_shim.release_session()
        print("Disconnected from MindRove board.")

if __name__ == "__main__":
    main()
