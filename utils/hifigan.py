import sys
import os
# __file__ == ./utils/hifigan.py
HIFI_PATH = os.path.join(os.path.dirname(__file__), '..', 'hifi-gan')
sys.path.append(HIFI_PATH)

from env import AttrDict
from models import Generator
sys.path.pop()

import json
from typing import Optional

import torch


class HiFiGANWrapper:
    """HiFi-GAN Wrapper
    """
    def __init__(self, config: str, ckpt: str, device: Optional[torch.device] = None):
        """Initializer.
        Args:
            config: path to the configuration.
            ckpt: path to the checkpoint.
            device: target computing device.
        """
        device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        with open(config) as f:
            # load configuration
            config = AttrDict(json.load(f))
        # load checkpoint
        ckpt = torch.load(ckpt, map_location='cpu')

        self.model = Generator(config)
        self.model.load_state_dict(ckpt['generator'])
        self.model.to(device)
        self.model.eval()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Inference the wav.
        Args:
            mel: [torch.float32; [B, mel, T]], spectrogram.
        Returns:
            [torch.float32; [B, T x hop]], generated wav signal, in range [-1, 1]
        """
        return self.model(mel).squeeze(dim=1)
