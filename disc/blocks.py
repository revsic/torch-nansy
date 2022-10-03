import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block.
    """
    def __init__(self,
                 channels: int,
                 kernels: int,
                 dilation: int,
                 leak: float):
        """Initializer.
        Args:
            channels: size of the input channels.
            kernels: convolutional kernels.
            dilation: dilation factor.
            leak: negative slope for leaky relu.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(leak),
            nn.Conv1d(
                channels, channels, kernels, dilation=dilation,
                padding=(kernels - 1) * dilation // 2),
            nn.LeakyReLU(leak),
            nn.Conv1d(
                channels, channels, kernels, dilation=dilation,
                padding=(kernels - 1) * dilation // 2))

        self.through = nn.Conv1d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensor.
        Returns:
            [torch.float32; [B, C, T]], transformed.
        """
        return self.block(inputs) + self.through(inputs)
