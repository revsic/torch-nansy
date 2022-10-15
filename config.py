from speechset.config import Config as DataConfig
from disc.config import Config as DiscConfig
from nansy.config import Config as ModelConfig


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self, sr: int, hop: int):
        """Initializer.
        Args:
            sr: sample rate.
            hop: stft hop length.
        """
        # optimizer
        self.learning_rate = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.9

        # augment
        self.num_code = 32
        self.formant_shift = 1.4
        self.pitch_shift = 1.5
        self.cutoff_lowpass = 60
        self.cutoff_highpass = 10000
        self.q_min = 2
        self.q_max = 5
        self.num_peak = 8
        self.g_min = -12
        self.g_max = 12

        # loader settings
        self.batch = 32
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True

        # train iters
        self.epoch = 1000

        # segment length
        sec = 1.47
        self.seglen = int(sr * sec) // hop * hop

        # path config
        self.log = './log'
        self.ckpt = './ckpt'

        # model name
        self.name = 't1'

        # commit hash
        self.hash = 'unknown'


class Config:
    """Integrated configuration.
    """
    def __init__(self):
        self.data = DataConfig(batch=None)
        self.train = TrainConfig(self.data.sr, self.data.hop)
        self.model = ModelConfig()
        self.disc = DiscConfig(self.data.mel, self.model.ver_out_channels)

    def validate(self):
        assert (
            self.data.sr == self.model.sr
            and self.data.mel == self.model.mel_filters
            and self.data.hop == self.model.mel_strides
            and self.data.win == self.model.mel_windows
            and self.data.fft == self.model.mel_windows
            and self.data.fmin == self.model.mel_fmin
            and self.data.fmax == self.model.mel_fmax
            and self.data.win_fn == 'hann'), \
                'inconsistent data and model settings'

    def dump(self):
        """Dump configurations into serializable dictionary.
        """
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration.
        """
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes.
    """
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
