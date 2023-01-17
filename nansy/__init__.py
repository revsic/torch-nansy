from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import Config
from .melspec import MelSpec
from .synthesizer import Synthesizer
from .verifier import EcapaTdnn
from .wav2vec2 import Wav2Vec2Wrapper
from .yingram import Yingram


class Nansy(nn.Module):
    """Nansy generator.
    """
    def __init__(self, config: Config):
        """initializer.
        Args:
            config: nansy configurations, hyperparameters.
        """
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Wrapper(
            config.w2v_name,
            config.sr,
            config.w2v_spk,
            config.w2v_lin)

        self.verifier = EcapaTdnn(
            self.wav2vec2.channels,
            config.ver_out_channels,
            config.ver_channels,
            config.ver_prekernels,
            config.ver_scale,
            config.ver_kernels,
            config.ver_dilations,
            config.ver_bottleneck,
            config.ver_hiddens)

        self.melspec = MelSpec(
            config.mel_strides,
            config.mel_windows,
            config.mel_filters,
            config.mel_fmin,
            config.mel_fmax,
            config.sr)

        self.yingram = Yingram(
            config.yin_strides,
            config.yin_windows,
            config.yin_lmin,
            config.yin_lmax,
            config.yin_bins,
            config.sr)
        # compute channels
        self.yin_delta, self.yin_range, _ = config.yingram_channels()

        synth_fn = lambda in_channels: Synthesizer(
            in_channels,
            config.mel_filters,
            config.syn_channels,
            config.ver_out_channels,
            config.syn_kernels,
            config.syn_dilation,
            config.syn_leak,
            config.syn_dropout,
            config.syn_preconv_blocks,
            config.syn_preblocks,
            config.syn_postblocks)

        self.synth_src = synth_fn(self.yin_range)
        self.synth_fil = synth_fn(Wav2Vec2Wrapper.OUT_CHANNELS)

    def analyze_wav2vec2(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analyze the Wav2Vec2.0-relative features.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
        Returns:
            linguistic: [torch.float32; [B, 1024, S]], linguistic features.
            spk: [torch.float32; [B, ver_out_channels]], speaker embedding.
        """
        # [B, S, 1024], [B, S, 1024]
        spk, linguistic = self.wav2vec2.forward(audio)
        # [B, ver_out_channels]
        spk = self.verifier.forward(spk.transpose(1, 2))
        return linguistic.transpose(1, 2), spk

    def analyze_energy(self, audio: torch.Tensor) -> torch.Tensor:
        """Analyze the energy of spectrogram.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
        Returns:
            energy: [torch.float32; [B, T // mel_strides]], energy values.
            mel: [torch.float32; [B, mel_filters, T // mel_strides]], mel-spectrogram.
        """
        # [B, mel, T]
        mel = self.melspec.forward(audio)
        # [B, T]
        return mel.mean(dim=1), mel

    def analyze_yingram(self, audio: torch.Tensor) -> torch.Tensor:
        """Analyze the yingram.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
        Returns:
            [torch.float32; [B, Y, T // yin_strides]], yingram,
                where Y = yin_bins x (mmax - mmin + 1)
        """
        return self.yingram.forward(audio).transpose(1, 2)

    def sample_yingram(self,
                       yingram: torch.Tensor,
                       start: Optional[Union[int, torch.Tensor]] = None,
                       semitone: bool = False) \
            -> torch.Tensor:
        """Sample the yingram for synthesizer.
        Args:
            yingram: [torch.float32; [B, Y, T // yin_strides]], yingram,
                where Y = yin_bins x (mmax - mmin + 1)
            start: [torch.long; [B]], start position.
            semitone: treat the `start` as relative semitone steps
                otherwise absolute start index.
        Returns:
            [torch.float32; [B, Y', T // yin_strides]], sampled yingram,
                where Y' = yin_bins x (lmax - lmin + 1)
        """
        d, r = self.yin_delta, self.yin_range
        if start is None:
            return yingram[:, d:d + r]
        # semitone conversion
        if semitone:
            start = d + start * self.config.yin_bins
        # sampling
        if isinstance(start, int):
            return yingram[:, start:start + r]
        return torch.stack([y[s:s + r] for y, s in zip(yingram, start)], dim=0)

    def analyze(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode the features from audio.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
        Returns:
            analyzed features
                linguistic: [torch.float32; [B, 1024, S]], linguistic features.
                spk: [torch.float32; [B, ver_out_channels]], speaker embedding.
                energy: [torch.float32; [B, T // mel_strides]], energy values.
                yingram: [torch.float32; [B, Y, T // yin_strides]], full pitch-relative features.
                    where Y = yin_bins x (mmax - mmin + 1)
        """
        # [B, 1024, S], [B, ver_out_channels]
        linguistic, spk = self.analyze_wav2vec2(audio)
        # [B, T // mel_strides]
        energy, _ = self.analyze_energy(audio)
        # [B, Y, T // yin_strides]
        yingram = self.analyze_yingram(audio)
        return {
            'linguistic': linguistic,
            'spk': spk,
            'energy': energy,
            'yingram': yingram}

    def synthesize(self,
                   linguistic: torch.Tensor,
                   spk: torch.Tensor,
                   energy: torch.Tensor,
                   yingram: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize the mel-spectrogram.
        Args:
            linguistic: [torch.float32; [B, 1024, S]], linguistic features.
            spk: [torch.float32; [B, ver_out_channels]], speaker embedding.
            energy: [torch.float32; [B, T / mel_strides]], energy values.
            yingram: [torch.float32; [B, yin_range, T / yin_strides]],
                pitch-relative features.
        Returns:
            [torch.float32; [B, mel_filters, T / mel_strides]],
                synthesized mel-spectrogram, filter and source.
        """
        # [B, mel, T]
        filter_ = self.synth_fil.forward(linguistic, energy, spk)
        # [B, mel, T]
        source = self.synth_src.forward(yingram, energy, spk)
        return filter_ + source, filter_, source

    def forward(self, audio: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reconstruct the audio.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
        Returns:
            synth: [torch.float32; [B, mel_filters, T / mel_strides]],
                mel-spectrogram.
            aux: intermediate features, reference return values of `Nansy.analyze`.
        """
        feat = self.analyze(audio)
        # [B, mel, T']
        synth, filter_, source = self.synthesize(
            feat['linguistic'],
            feat['spk'],
            feat['energy'],
            self.sample_yingram(feat['yingram']))
        # add features
        feat['filter'] = filter_
        feat['source'] = source
        return synth, feat

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

    def load_(self, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints inplace.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        self.load_state_dict(states['model'])
        if optim is not None:
            optim.load_state_dict(states['optim'])

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
        nansy = cls(config)
        nansy.load_(states, optim)
        return nansy
