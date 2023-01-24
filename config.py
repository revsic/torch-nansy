from disc.config import Config as DiscConfig
from nansy.config import Config as ModelConfig


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self, hop: int):
        """Initializer.
        Args:
            hop: stft hop length.
        """
        # optimizer
        self.learning_rate = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.9

        # augment
        self.num_code = 32
        self.formant_shift = 1.4
        self.pitch_shift = 2.
        self.pitch_range = 1.5
        self.cutoff_lowpass = 60
        self.cutoff_highpass = 10000
        self.q_min = 2
        self.q_max = 5
        self.num_peak = 8
        self.g_min = -12
        self.g_max = 12

        # loader settings
        self.batch = 64
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True

        # train iters
        self.epoch = 1000

        # segment length
        frames = 128
        self.seglen = hop * frames

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
        self.model = ModelConfig()
        self.train = TrainConfig(self.model.mel_strides)
        self.disc = DiscConfig(self.model.mel_filters, self.model.ver_out_channels)

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
