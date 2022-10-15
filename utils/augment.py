from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

from .lpc import LinearPredictiveCoding

from config import Config


class Augment(nn.Module):
    """Waveform augmentation.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: Nansy configurations.
        """
        super().__init__()
        self.config = config
        self.coder = LinearPredictiveCoding(
            config.train.num_code, config.data.win, config.data.hop)
        self.register_buffer(
            'window',
            torch.hann_window(config.data.win),
            persistent=False)

    def forward(self,
                wavs: torch.Tensor,
                pitch_shift: Optional[torch.Tensor] = None,
                formant_shift: Optional[torch.Tensor] = None,
                quality_power: Optional[torch.Tensor] = None,
                gain: Optional[torch.Tensor] = None,
                mode: str = 'linear') -> torch.Tensor:
        """Augment the audio signal, random pitch, formant shift and PEQ.
        Args:
            wavs: [torch.float32; [B, T]], audio signal.
            pitch_shift: [torch.float32; [B]], pitch shifts.
            formant_shift: [torch.float32; [B]], formant shifts.
            quality_power: [torch.float32; [B, num_peak + 2]],
                exponents of quality factor, for PEQ.
            gain: [torch.float32; [B, num_peak]], gain in decibel.
            mode: interpolation mode, `linear` or `nearest`.
        Returns:
            [torch.float32; [B, T]], augmented.
        """
        # random formant, pitch shifter
        if formant_shift is not None or pitch_shift is not None:
            # [B, F, T / S], complex64
            fft = torch.stft(
                wavs,
                self.config.data.fft,
                self.config.data.hop,
                self.config.data.win,
                self.window,
                return_complex=True)
            # [B, T / S, num_code]
            code = self.coder.from_stft(fft)
            # [B, T / S, F]
            filter_ = self.coder.envelope(code)
            source = fft.transpose(1, 2) / (filter_ + 1e-7)
            # [B, T / S, F]
            if formant_shift is not None:
                filter_ = self.interp(filter_, formant_shift, mode=mode)
            if pitch_shift is not None:
                source = self.interp(source, pitch_shift, mode=mode)
            # [B, T]
            wavs = torch.istft(
                (source * filter_).transpose(1, 2),
                self.config.data.fft,
                self.config.data.hop,
                self.config.data.win,
                self.window)
        # PEQ
        if quality_power is not None:
            # alias
            lowcut, highcut, num_peak, sr = \
                self.config.train.cutoff_lowpass, \
                self.config.train.cutoff_highpass, \
                self.config.train.num_peak, \
                self.config.data.sr
            # [B, num_peak + 2]
            qualities = self.config.train.q_min * (
                self.config.train.q_max / self.config.train.q_min) ** quality_power
            if gain is None:
                # [B, num_peak]
                gain = torch.zeros_like(qualities[:, :-2])
            # num_peak x ([B], [B])
            for i, (g, q) in enumerate(zip(gain.T, qualities.T)):
                center = lowcut * (highcut / lowcut) ** (i / (num_peak - 1))
                # [B, T]
                wavs = [
                    AF.equalizer_biquad(wav, sr, center, g.item(), q.item())
                    for wav in wavs]
            # [B, T], lowpass
            wavs = torch.stack([
                AF.lowpass_biquad(
                    AF.highpass_biquad(wav, sr, highcut, hq.item()),
                    sr, lowcut, lq.item())
                for wav, (lq, hq) in zip(wavs, qualities[:, -2:])], dim=0)
        return wavs

    @staticmethod
    def complex_interp(inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Interpolate the complex tensor.
        Args:
            inputs: [torch.complex64; [B, C, ...]], complex inputs.
        Returns:
            [torch.complex64; [B, C, ...]], interpolated.
        """
        mag = F.interpolate(inputs.abs(), *args, **kwargs)
        angle = F.interpolate(inputs.angle(), *args, **kwargs)
        return torch.polar(mag, angle)

    def interp(self,
               inputs: torch.Tensor,
               shifts: torch.Tensor,
               mode: str) -> torch.Tensor:
        """Interpolate the channel axis with dynamic shifts.
        Args:
            inputs: [torch.complex64; [B, T, C]], input tensor.
            shifts: [torch.float32; [B]], shift factor.
            mode: interpolation mode.
        Returns:
            [torch.complex64; [B, T, C]], interpolated.
        """
        # _, _, C
        _, _, channels = inputs.shape
        # B x [1, T, C x min(1., shifts)]
        interp = [
            Augment.complex_interp(
                f[None], scale_factor=s.item(), mode=mode)[..., :channels]
            for f, s in zip(inputs, shifts)]
        # [B, T, C]
        return torch.cat([
            F.pad(f, [0, channels - f.shape[-1]])
            for f in interp], dim=0)
