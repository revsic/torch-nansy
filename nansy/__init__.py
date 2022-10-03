from typing import Any, Dict, Optional, Tuple

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
            config.ver_in_channels,
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

    def analyze(self,
                audio: torch.Tensor,
                audiolen: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the features from audio.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
            audiolen: [torch.long; [B]], length of the audio.
        Returns:
            linguistic: [torch.float32; [B, L, T']], linguistic features.
            spk: [torch.float32; [B, S]], speaker embedding.
            energy: [torch.float32; [B, T]], energy values.
            yingram_full: [torch.float32; [B, Yf, T'']], full pitch-relative features.
            yingram: [torch.float32; [B, Y, T'']], part of yingram, synthesizing purpose.
                where L = `Wav2Vec2Wrapper.OUT_CHANNELS`
                      S = `config.ver_out_channels`
                      Yf = `Nansy.yingram.mmax - Nansy.yingram.mmin + 1`
                      Y = `Nansy.yin_range`.
        """
        # [B, T', L], [B, T', L]
        spk, linguistic = self.wav2vec2.forward(audio, audiolen)
        # [B, S]
        spk = self.verifier.forward(spk.transpose(1, 2))
        # [B, mel, T]
        mel = self.melspec.forward(audio)
        # [B, T]
        energy = mel.mean(dim=1)
        # [B, Yf, T']
        yingram_full = self.yingram.forward(audio).transpose(1, 2)
        # [B, Y, T']
        d, r = self.yin_delta, self.yin_range
        yingram = yingram_full[:, d:d + r]
        return linguistic.transpose(1, 2), spk, energy, yingram_full, yingram

    def synthesize(self,
                   linguistic: torch.Tensor,
                   spk: torch.Tensor,
                   energy: torch.Tensor,
                   yingram: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize the mel-spectrogram.
        Args:
            linguistic: [torch.float32; [B, L, T']], linguistic features.
            spk: [torch.float32; [B, S]], speaker embedding.
            energy: [torch.float32; [B, T]], energy values.
            yingram: [torch.float32; [B, Y, T'']], pitch-relative features,
                where L = `Wav2Vec2Wrapper.OUT_CHANNELS`
                      S = `config.ver_out_channels`
                      Y = `Nansy.yin_range`.
        Returns:
            [torch.float32; [B, M, T]], synthesized mel-spectrogram, filter and source.
                where M = `config.mel_filters`.
        """
        # [B, mel, T]
        filter_ = self.synth_fil.forward(linguistic, energy, spk)
        # [B, mel, T]
        source = self.synth_src.forward(yingram, energy, spk)
        return filter_ + source, filter_, source

    def forward(self,
                audio: torch.Tensor,
                audiolen: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reconstruct the audio.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
            audiolen: [torch.long; [B]], length of the audio.
        Returns:
            synth: [torch.float32; [B, mel, T']], mel-spectrogram.
                where T' = T / `Config.mel_strides`.
            aux: intermediate features, reference return values of `Nansy.analyze`.
        """
        linguistic, spk, energy, yingram_full, yingram = self.analyze(audio, audiolen)
        # [B, mel, T']
        synth, filter_, source = self.synthesize(linguistic, spk, energy, yingram)
        return synth, {
            'linguistic': linguistic,
            'spk': spk,
            'energy': energy,
            'yingram_full': yingram_full,
            'yingram': yingram,
            'synth': synth,
            'filter': filter_,
            'source': source}

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
        nansy = cls(config)
        nansy.load_state_dict(states['model'])
        if optim is not None:
            optim.load_state_dict(states['optim'])
