from typing import Dict, Tuple

from .yingram import Yingram


class Config:
    def yingram_channels(self) -> Tuple[int, int, Dict[str, int]]:
        # midi-range
        mmin, mmax = Yingram.midi_range(self.sr, self.yin_lmin, self.yin_lmax)
        # hertz to lag
        lmin, lmax = int(self.sr / self.yin_hmax), int(self.sr / self.yin_hmin)
        # for feeding synthesizer
        # WARNING: since integer quantization in `midi_range`
        # , it could occur the errors
        #  ex. 984-channels Yingram in paper, 980 in this repo
        s_mmin, s_mmax = Yingram.midi_range(self.sr, lmin, lmax)
        # for sampling
        bins = self.yin_bins
        delta, range_ = (s_mmin - mmin) * bins, (s_mmax - s_mmin + 1) * bins
        return delta, range_, {
            'yin-midi-min': mmin, 'yin-midi-max': mmax,
            'syn-midi-min': s_mmin, 'syn-midi-max': s_mmax}

    def __init__(self):
        # sample rate
        self.sr = 22050

        # yingram
        self.yin_strides = 450  # targeting 49hz on 22050hz sr
        self.yin_windows = 2048
        self.yin_lmin = 22      # 1000.40hz, 83midi(floor)
        self.yin_lmax = 2047    #   10.77hz,  5midi(ceil), 79-channel yingram

        self.yin_hmin = 25.11   # 878 lag, 20midi, +15 from lmin
        self.yin_hmax = 430.19  #  51 lag, 68midi, 49-channel

        self.yin_bins = 20  # the number of the bins per semitone

        # mel-spectrogram
        self.mel_strides = 256
        self.mel_windows = 1024
        self.mel_filters = 80
        self.mel_fmin = 0
        self.mel_fmax = 8000
    
        # wav2vec2
        self.w2v_name = 'facebook/wav2vec2-large-xlsr-53'
        self.w2v_spk = 1
        self.w2v_lin = 12

        # ecapa-tdnn
        self.ver_in_channels = 1024  # wav2vec2 output channels
        self.ver_out_channels = 192
        self.ver_channels = 512
        self.ver_prekernels = 5
        self.ver_scale = 8
        self.ver_kernels = 3
        self.ver_dilations = [2, 3, 4]
        self.ver_bottleneck = 128
        self.ver_hiddens = 1536

        # synthesizer
        # syn_in_channels = yingram or wa2vec2
        # syn_out_channels = mel_filters
        self.syn_channels = 128
        # syn_spk = ver_out_channels
        self.syn_kernels = 3
        self.syn_dilation = 3
        self.syn_leak = 0.2       # no ref, default nn.LeakyReLU = 0.01
        self.syn_dropout = 0.2    # no ref, default nn.Dropout   = 0.5
        self.syn_preconv_blocks = 2
        self.syn_preblocks = [4, 4, 2]
        self.syn_postblocks = [4, 4, 2]
