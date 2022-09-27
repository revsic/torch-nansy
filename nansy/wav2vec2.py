import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2Wrapper(nn.Module):
    """Wrapping huggingface wav2vec2.0.
    """
    DEFAULT = 'facebook/wav2vec2-large-xlsr-53'

    SPEAKER = 1
    LINGUISTIC = 12

    def __init__(self,
                 name: Optional[str] = None,
                 speaker: Optional[int] = None,
                 linguistic: Optional[int] = None):
        """Load the wav2vec2.0 pretrained model.
        Args:
            name: name of the model, default use facebook XLSR-53.
            sr: sample rates of the input audio, default 16khz for XLSR-53.
            speaker, linguistic: layer outputsfor speaker or linguistic features. 
        """
        super().__init__()
        name = name or Wav2Vec2Wrapper.DEFAULT
        self.model = Wav2Vec2Model.from_pretrained(name)
        self.eval()

        self.speaker = speaker or Wav2Vec2Wrapper.SPEAKER
        self.linguistic = linguistic or Wav2Vec2Wrapper.LINGUISTIC

    @torch.no_grad()
    def forward(self,
                audio: torch.Tensor,
                audiolen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the features from audio.
        Args:
            audio: [torch.float32; [B, T]], 16khz audio, [-1, 1]-ranged.
            audiolen: [torch.int32; [B]], length of the audios.
        Returns:

        """
        ## 1. zero-mean, unit-var
        normed = _
        ## 2. attention mask
        mask = _
        ## 3. inference
        self.model(normed, attention_mask=mask)
        ## 4. layer selection

    def train(self, _: bool = True):
        """Support only evaluation
        """
        pass

    def load_state_dict(self,
                        state_dict: Dict[str, torch.Tensor],
                        strict: bool = True):
        """Do not load state dict.
        """
        pass


if __name__ == '__main__':
    PATH = 'D:\\dataset\\LibriTTS\\test-clean\\61\\70970\\61_70970_000007_000001.wav'

    device = torch.device('cuda:0')

    import librosa
    out, _ = librosa.load(PATH, sr=16000)
    ptout = torch.tensor(out, device=device)

    model = Wav2Vec2Wrapper()
    p = model.preproc(
        [out, out[::2]], padding=True, return_tensors='pt')
    print(p)

    normed = (ptout - ptout.mean()) / (ptout.var() + 1e-7).sqrt()

    print(normed[:10])
