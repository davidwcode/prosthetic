import torch
import torch.nn as nn

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid

NUM_LEVELS = 201
N_GRAM_SIZE = 4


class Encoder(nn.Module):
    def __init__(self, out_features, num_channels):
        super(Encoder, self).__init__()

        self.channels = embeddings.Random(num_channels, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features, low=-100.0, high=100.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        signal = self.signals(input)
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))

        samples = torchhd.multiset(samples)
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE)
        return torchhd.hard_quantize(sample_hv)
    
class HDClassifier(nn.Module):
    def __init__(self, high_dim, num_classes, num_channels):
        super(HDClassifier, self).__init__()
        
        self.encoder = Encoder(high_dim, num_channels)
        self.centroid = Centroid(high_dim, num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.centroid(self.encoder(input), dot=True)
    
    def build(self, input: torch.Tensor, target: torch.Tensor, lr=1.0):
        self.centroid.add(self.encoder(input), target, lr=lr)

    def build_online(self, input: torch.Tensor, target: torch.Tensor, lr=1.0):
        self.centroid.add_online(self.encoder(input), target, lr=lr)

    def build_adapt(self, input: torch.Tensor, target: torch.Tensor, lr=1.0):
        self.centroid.add_adapt(self.encoder(input), target, lr=lr)

    def normalize(self):
        self.centroid.normalize()