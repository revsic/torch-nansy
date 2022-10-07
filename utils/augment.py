from typing import Union

import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from nansy.config import Config


class Augment:
    """Waveform augmentation.
    """
    def __init__(self, config: Config, device: torch.device):
        """Initializer.
        Args:
            config: Nansy configurations.
            device: torch device.
        """
        self.config = config
        self.device = device
        self.window = torch.hann_window(config.mel_windows, device=device)

    def _pitch_shift(self,
                     wavs: torch.Tensor,
                     steps: int,
                     bins_per_octave: int = 12) -> torch.Tensor:
        """Pitch shifting implementations.
        """
        # alias
        w, s = self.config.mel_windows, self.config.mel_strides
        return AF.pitch_shift(
            wavs, self.config.sr,
            steps, bins_per_octave=bins_per_octave,
            n_fft=w, hop_length=s, window=self.window)

    def pitch_shift(self,
                    wavs: torch.Tensor,
                    steps: Union[int, torch.Tensor],
                    bins_per_octave: int = 12) -> torch.Tensor:
        """Pitch shifts.
        Args:
            wavs: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
            steps: [B], pitch shifts.
            bins_per_octave: the number of steps per octave.
        Returns:
            [torch.float32; [B, T]], shifted.
        """
        # [B, T]
        if isinstance(steps, int):
            return self._pitch_shift(wavs, steps, bins_per_octave)
        # [B, T]
        return torch.cat([
            self._pitch_shift(wav[None], step.item(), bins_per_octave)
            for step, wav in zip(steps, wavs)], dim=0)

    def formant_shift(self,
                      wavs: torch.Tensor,
                      shifts: Union[float, torch.Tensor]) -> torch.Tensor:
        """Formant shifts, rough approximations of praat-parselmouth.
        Args:
            wavs: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
            shifts: [B], formant shifts.
        Returns:
            [torch.float32; [B, T]], shifted.
        """
        # alias
        windows, strides = self.config.mel_windows, self.config.mel_strides
        # [B, windows // 2 + 1, T / strides]
        stft = torch.stft(
            wavs,
            windows,
            self.config.mel_strides,
            window=torch.hann_window(windows, device=wavs.device),
            return_complex=True)
        # F = windows // 2 + 1
        num_freq = windows // 2 + 1
        # [F, F]
        mapper = torch.eye(num_freq // 2 + 1)
        def mapper_fn(shift: float) -> torch.Tensor:
            # [F, min(F x shift, F)]
            m = F.interpolate(
                mapper[None],
                scale_factor=shift,
                mode='linear')[0, :, :num_freq]
            # [F, F]
            return F.pad(m, [0, num_freq - m.shape[-1]]).T
        # [B, F, F]
        if isinstance(shifts, float):
            interp = mapper_fn(shifts)[None].repeat(stft.shape[0])
        else:
            interp = torch.stack([
                mapper_fn(shift.item()) for shift in shifts])
        # [B, T]
        return torch.istft(
            torch.matmul(interp.to(torch.complex64), stft),
            windows, strides, window=self.window)
