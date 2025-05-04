import os
import sys

dll_path = os.path.join(os.path.dirname(__file__), "python")
os.environ["PATH"] += os.pathsep + dll_path
sys.path.append(dll_path)

import time
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchhd.models import Centroid
from tqdm import tqdm
import torchmetrics
from dataset import EMGDataset
from model import Encoder

STATE_DICT = {0: "relax", 1: "clench", 2: "spiderman"}

# STATE_DICT = {0: "relax", 1: "clench"}

NUM_SAMPLES = 2400
NUM_CHANNELS = 4
SAMPLES_PER_POINT = 6
BATCH_SIZE = 1
DIMENSIONS = 10000

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

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


time.sleep(2)

labels = np.concatenate([np.zeros(NUM_SAMPLES // SAMPLES_PER_POINT, dtype=int), 
                         np.ones(NUM_SAMPLES // SAMPLES_PER_POINT, dtype=int), 
                         np.full(NUM_SAMPLES // SAMPLES_PER_POINT, 2, dtype=int) if len(STATE_DICT) == 3 else np.array([])])
data = np.zeros((len(STATE_DICT) * NUM_SAMPLES, NUM_CHANNELS), dtype=np.float64)

for i in range(len(STATE_DICT)):
    data[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES] = calibrate(STATE_DICT[i])

data = data.reshape(-1, SAMPLES_PER_POINT, 4)

data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)
print("data tensor", data_tensor.size())
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
num_classes = len(STATE_DICT)
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