from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import ResBlock
from .config import Config


class Discriminator(nn.Module):
    """NANSY-discriminator.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: discriminator hyperparameters.
        """
        super().__init__()
        self.config = config
        self.proj = nn.Conv1d(
            config.mel, config.channels, config.kernels,
            padding=config.kernels // 2)

        self.blocks = nn.Sequential(*[
            ResBlock(
                config.channels,
                config.kernels,
                config.dilation,
                config.leak)
            for _ in range(config.blocks)])

        self.branch = nn.Sequential(
            ResBlock(
                config.channels,
                config.kernels,
                config.dilation,
                config.leak),
            nn.Conv1d(config.channels, config.spk, 1))

        self.proj_out = nn.Conv1d(
            config.channels, 1, config.kernels,
            padding=config.kernels // 2)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discriminates whether inputs are real of synthesized.
        Args:
            inputs: [torch.float32; [B, mel, T]], input spectrogram.
        Returns:
            [torch.float32; [B, T]], logits.
            [torch.float32; [B, spk, T]], expected speakers.
        """
        # [B, C, T]
        x = self.proj(inputs)
        # [B, C, T]
        x = self.blocks(x)
        # [B, spk, T]
        spk = self.branch(x)
        # [B, T], [B, spk, T]
        return self.proj_out(x).squeeze(dim=1), spk

    def save(self, path: str, optim: Optional[torch.optim.Optimizer] = None):
        """Save the models.
        Args:
            path: path to the checkpoint.
            optim: optimizer, if provided.
        """
        dump = {'model': self.state_dict(), 'config': vars(self.config)}
        if optim is not None:
            dump['optim'] = optim.state_dict()
        torch.save(dump, path)

    @classmethod
    def load(cls, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        config = Config()
        for key, val in states['config'].items():
            if not hasattr(config, key):
                import warnings
                warnings.warn(f'unidentified key {key}')
                continue
            setattr(config, key, val)
        # construct
        disc = cls(config)
        disc.load_state_dict(states['model'])
        if optim is not None:
            optim.load_state_dict(states['optim'])

        return disc
