from .yingram import Yingram


class Config:
    """NANSY hyperparameters.
    """
    def yingram_channels(self) -> int:


    def __init__(self):
        # sample rate
        self.sr = 22050

        # yingram
        self.yin_strides = 450  # targeting 49hz on 22050hz sr
        self.yin_windows = 204
        self.yin_lmin = 22      # 1000.40hz, 83midi
        self.yin_lmax = 2047    #   10.77hz,  4midi, 79-channel yingram

        self.yin_hmin = 25.11   # 878 lag, 19midi, +15 from lmin
        self.yin_hmax = 430.19  #  51 lag, 68midi, 49-channel

        # mel-spectrogram
        self.mel_strides = 256
        self.mel_windows = 1024
        self.mel_filters = 80
        self.mel_fmin = 0
        self.mel_fmax = 8000
    
        # wav2vec2
        self.w2v_name = 'facebook/wav2vec2-large-xlsr-53'
        self.w2v_sr = 16000
        self.w2v_spk = 1
        self.w2v_lin = 12

        # ecapa-tdnn
        self.ver_in_channels = 1024  # wav2vec2 output channels
        self.ver_out_channels = 192
        self.ver_channels = 512
        self.ver_prekernels = 5
        self.ver_scale = 8
        self.ver_kenrels = 3
        self.ver_dilations = [2, 3, 4]
        self.ver_bottleneck = 128
        self.ver_hiddens = 1536

        # synthesizer
