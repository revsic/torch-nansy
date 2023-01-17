from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLayerNorm(nn.Module):
    """Conditional layer normalization.
    """
    def __init__(self, spk: int, channels: int):
        """Initializer.
        Args:
            spk: size of the speaker emebddings.
            chanensl: size of the input channels.
        """
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        self.proj = nn.Linear(spk, channels * 2)

    def forward(self, inputs: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """Whintening and colorizing the inputs adaptively.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensor.
            spk: [torch.float32; [B, S]], speaker embedding.
                where C = `channels`, S = `spk`.
        """
        # [B, C]
        scale, bias = self.proj(spk).chunk(2, dim=1)
        # [B, C, T]
        return self.norm(inputs) * scale[..., None] + bias[..., None]


class ConvGLU(nn.Module):
    """Convolution + Dropout + GLU
    """
    def __init__(self,
                 channels: int,
                 dropout: float,
                 kernels: int,
                 dilation: int,
                 blocks: int,
                 spk: Optional[int] = None):
        """Initializer.
        Args:
            channels: size of the input channels.
            dropout: dropout rate.
            kernels: size of the convolutional kernels.
            dilation: dilation factor.
            blocks: the number of the convolution blocks.
            spk: size of the speaker embedding,
                use conditional layer normalization if provided.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Conv1d(
                    channels, channels * 2, kernels, dilation=dilation ** i,
                    padding=(kernels - 1) * dilation ** i // 2),
                nn.GLU(dim=1))
            for i in range(blocks)])

        if spk is None:
            self.cln = [None for _ in range(blocks)]
        else:
            self.cln = nn.ModuleList([
                ConditionalLayerNorm(spk, channels)
                for _ in range(blocks)])

    def forward(self,
                inputs: torch.Tensor,
                spk: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensor.
            spk: [torch.float32; [B, S]], speaker embeeding,
                only valid on `any(cln is not None for cln in self.cln)`.
        Returns:
            [torch.float32; [B, C, T]], transformed inputs.
        """
        x = inputs
        for block, cln in zip(self.blocks, self.cln):
            # [B, C, T]
            x = x + block(x)
            if cln is not None:
                x = cln(x, spk)
        # [B, C, T]
        return x


class Synthesizer(nn.Module):
    """Source and filter syntheizer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channels: int,
                 spk: int,
                 kernels: int,
                 dilation: int,
                 leak: float,
                 dropout: float,
                 preconv_blocks: int,
                 preblocks: List[int],
                 postblocks: List[int]):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            out_channels: size of the output channels.
            channels: size of the hidden states.
            spk: size of the speaker embeddings.
            kernels: size of the convolutional kernels.
            dilation: dilation factor.
            leak: negative slope of leaky relu.
            dropout: dropout rates.
            preconv_blocks: the number of the convolution blocks before feed to ConvGLU.
            preblocks: the number of the ConvGLU blocks
                before conditioning the speaker and enrgy.
            postblocks: the number of the ConvGLU blocks with speaker conditioning.
        """
        super().__init__()
        # ref:NANSY, arXiv:2110.14513, Figure7
        # preconv_blocks=2, channels=128
        self.preconv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(in_channels, in_channels, 1),
                    nn.LeakyReLU(leak),
                    nn.Dropout(dropout))
                for _ in range(preconv_blocks)],
            nn.Conv1d(in_channels, channels, 1))
        # kernels=3, dilation=3, preblocks=[4, 4, 2]
        self.preblocks = nn.Sequential(
            *[
                ConvGLU(channels, dropout, kernels, dilation, blocks)
                for blocks in preblocks],
            nn.Conv1d(channels, channels, 1))

        self.cond_energy = nn.Sequential(
            nn.LeakyReLU(leak),
            nn.Conv1d(channels + 1, channels, 1))
        # postblocks=[4, 4, 2]
        self.postblocks = nn.Sequential(*[
            ConvGLU(channels, dropout, kernels, dilation, blocks, spk)
            for blocks in postblocks])

        self.proj = nn.Conv1d(channels, out_channels, 1)

    def forward(self,
                inputs: torch.Tensor,
                energy: torch.Tensor,
                spk: torch.Tensor) -> torch.Tensor:
        """Synthesize the signal.
        Args:
            inputs: [torch.float32; [B, I, T']], wav2vec2 features or yingram.
            energy: [torch.float32; [B, T]],
                energy values, mean of log-mel scale spectrogram on frequency axis.
            spk: [torch.float32; [B, S]], speaker embedding,
                where I = `in_channels`, S = `spk`.
        Returns:
            [torch.float32; [B, O, T]], synthesized signal,
                where O = `out_channels`.
        """
        # _, T
        _, timestep = energy.shape
        # since wav2vec2 feature is 50hz signal
        # and it mismatch the output spectrogram settings
        # , interpolate it based on the length of `energy`
        # [B, C, T']
        x = self.preconv(inputs)
        # [B, C, T']
        x = self.preblocks(x)
        # [B, I, T]
        x = F.interpolate(x, size=timestep, mode='linear')
        # [B, C, T], energy conditioning
        x = self.cond_energy(torch.cat([x, energy[:, None]], dim=1))
        for block in self.postblocks:
            # [B, C, T]
            x = block(x, spk=spk)
        # [B, O, T]
        return self.proj(x)
