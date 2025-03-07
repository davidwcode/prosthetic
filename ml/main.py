import os
import sys

dll_path = os.path.join(os.path.dirname(__file__), "python")
os.environ["PATH"] += os.pathsep + dll_path
sys.path.append(dll_path)

import serial
import time
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EMGDataset
state_dict = {
    0: "relax",
    1: "clench",
    2: "spiderman"
}

NUM_SAMPLES = 2400
NUM_CHANNELS = 4
SAMPLES_PER_POINT = 6
BATCH_SIZE = 1

#Inlet
streams = resolve_stream()
for stream in streams:
    if stream.name() == "MindRoveStream":
        inlet = StreamInlet(stream,max_buflen=1)
        break
def calibrate(state):
    samples = np.zeros((NUM_SAMPLES, NUM_CHANNELS), dtype=np.float64)
    i = 0
    print("Calibrating starting for " + state + "...")
    time.sleep(2)
    print(state + " in 3 seconds...")
    time.sleep(1)
    print(state + " in 2 seconds...")
    time.sleep(1)
    print(state + " in 1 second...")
    time.sleep(1)
    print(state + " now!")
    while i < NUM_SAMPLES:
        sample, timestamp = inlet.pull_sample(timeout=0.0)
        if sample:
            samples[i] = sample[:NUM_CHANNELS]
            i += 1
    print("Calibration for " + state + " done!")
    return samples

ser = serial.Serial('COM3', 9600)

time.sleep(2)

labels = np.concatenate([np.zeros(NUM_SAMPLES // SAMPLES_PER_POINT, dtype=InterruptedError), 
                         np.ones(NUM_SAMPLES // SAMPLES_PER_POINT, dtype=int), 
                         np.full(NUM_SAMPLES // SAMPLES_PER_POINT, 2, dtype=int)])
data = np.zeros((3 * NUM_SAMPLES, NUM_CHANNELS), dtype=np.float64)

for i in range(3):
    data[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES] = calibrate(state_dict[i])

data.reshape(-1, SAMPLES_PER_POINT, 4)

data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

dataset = EMGDataset(data_tensor, labels_tensor)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for samples, labels in data_loader:
    print(samples)
    print(samples.size(), labels.size())
    break