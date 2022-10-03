from typing import Optional


class Config:
    """Discriminator hyperparameters.
    """
    def __init__(self,
                 mel: Optional[int] = None,
                 spk: Optional[int] = None):
        """Initializer.
        Args:
            mel: size of the mel filterbankss.
            spk: size of the speaker embeddings.
        """
        self.mel = mel
        self.spk = spk

        self.channels = 128
        self.kernels = 3
        self.dilation = 3
        self.leak = 0.2     # no ref 
        self.blocks = 5
