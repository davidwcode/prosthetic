import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EMGDataset
from tqdm import tqdm
import torchmetrics
from util import *
import serial

# BoardShim imports from MindRove SDK
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

from model import HDClassifier

NUM_SAMPLES = 5000         # Total samples to collect for each gesture calibration
NUM_CHANNELS = 4             # Number of EMG channels
SAMPLES_PER_POINT = 50       
BATCH_SIZE = 1
STATE_DICT = {0: "relax", 1: "clench", 2: "spiderman", 3: "gun"}
SAMPLING_RATE = 500          # Hz
DIMENSIONS = 10000

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


def main():
    # Initialize boardshim
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)

    # Initalize serial port
    # ser = serial.Serial('COM3', 9600)
    time.sleep(2)
    
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
        
        num_classes = len(STATE_DICT)
        # Preallocate a large array to hold all calibration data
        data = np.zeros((num_classes * NUM_SAMPLES, NUM_CHANNELS), dtype=np.float32)
        
        # Loop through each calibration gesture/state.
        for i in range(num_classes):
            state = STATE_DICT[i]
            display_calibrate(state)
            num_points = NUM_SAMPLES // SAMPLES_PER_POINT
            for j in range(num_points):
                print(f"Collecting data for {state} ({j+1}/{num_points})")
                # Collect data for each window
                data[i*NUM_SAMPLES + j*SAMPLES_PER_POINT : i*NUM_SAMPLES + (j+1)*SAMPLES_PER_POINT] = record_emg(board_shim, SAMPLES_PER_POINT, NUM_CHANNELS, SAMPLING_RATE)
        
        # Create labels: here, for each state we create labels for each window.
        num_points_per_state = NUM_SAMPLES // SAMPLES_PER_POINT
        labels = np.concatenate([np.full(num_points_per_state, i, dtype=int) for i in range(num_classes)])
        
        # Reshape the data into windows of SAMPLES_PER_POINT samples.
        # Note: If you increase SAMPLES_PER_POINT to ensure good filtering, adjust your model input accordingly.
        data_windows = data.reshape(-1, SAMPLES_PER_POINT, NUM_CHANNELS)
        
        # Convert data to PyTorch tensors
        data_tensor = torch.tensor(data_windows, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        

        dataset = EMGDataset(data_tensor, labels_tensor)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_data_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_data_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # For demonstration
        for samples, batch_labels in train_data_loader:
            print("Sample batch of data:", samples)
            print("Data shape:", samples.size(), "Labels shape:", batch_labels.size())
            break

        model = HDClassifier(DIMENSIONS, num_classes, NUM_CHANNELS)
        model = model.to(device)

        with torch.no_grad():
            for samples, targets in tqdm(train_data_loader, desc="Building"):
                samples = samples.to(device)
                targets = targets.to(device)
                model.build(samples, targets)
            model.normalize()

        accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

        with torch.no_grad():

            for samples, targets in tqdm(test_data_loader, desc="Testing"):
                samples = samples.to(device)
                output = model(samples)
                accuracy.update(output.cpu(), targets)

        print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")

        print("Inference Time")
        # Flush any residual data
        board_shim.get_board_data()

        while True:
            votes = np.zeros(num_classes)
            num_votes = 0
            while num_votes < 20:
                filtered_data = record_emg(board_shim, SAMPLES_PER_POINT, NUM_CHANNELS, SAMPLING_RATE)
                data_tensor = torch.tensor(filtered_data, dtype=torch.float32)
                data_tensor = data_tensor.unsqueeze(0)
                data_tensor = data_tensor.to(device)
                output = model(data_tensor)
                votes[torch.argmax(output)] += 1
                num_votes += 1
            action = np.argmax(votes)
            print(f"Predicted action: {STATE_DICT[action]}")
            # ser.write(str(action).encode() + b'\n')




    
    except Exception as e:
        print("Error during data collection:", e)
    
    finally:
        board_shim.stop_stream()
        board_shim.release_session()
        print("Disconnected from MindRove board.")

if __name__ == "__main__":
    main()
