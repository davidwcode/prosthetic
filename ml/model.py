import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import embeddings

NUM_LEVELS = 21
N_GRAM_SIZE = 4


class Encoder(nn.Module):
    def __init__(self, out_features, timestamps, channels):
        super(Encoder, self).__init__()

        self.channels = embeddings.Random(channels, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features, high=20)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        signal = self.signals(input)
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))

        samples = torchhd.multiset(samples)
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE)
        return torchhd.hard_quantize(sample_hv)