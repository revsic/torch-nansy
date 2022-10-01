from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2Wrapper(nn.Module):
    """Wrapping huggingface wav2vec2.0.
    """
    DEFAULT = 'facebook/wav2vec2-large-xlsr-53'
    # Since 0-th hidden state is poosition-informed convolution features
    # , one-base indexing required
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
        # warning can occurs since `Wav2Vec2Model` does not contain
        # quantization modules
        self.model = Wav2Vec2Model.from_pretrained(name)
        self.eval()

        self.speaker = speaker or Wav2Vec2Wrapper.SPEAKER
        self.linguistic = linguistic or Wav2Vec2Wrapper.LINGUISTIC

    @torch.no_grad()
    def forward(self,
                audio: torch.Tensor,
                audiolen: Optional[torch.Tensor] = None,
                return_all: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the features from audio.
        Args:
            audio: [torch.float32; [B, T]], 16khz audio, [-1, 1]-ranged.
            audiolen: [torch.long; [B]], length of the audios,
                masking the inputs if provided.
            return_all: return all hidden states if True, debugging purpose.
        Returns:
            speaker: [torch.float32; [B, C]], speaker embeddings.
            linguistic: [torch.float32; [B, S, C]], linguistic embeddings.
        """
        # B, T
        bsize, timestep = audio.shape
        if audiolen is None:
            audiolen = torch.full(
                (bsize,), timestep, dtype=torch.long, device=audio.device)
        # [B, T]
        mask = (
            torch.arange(timestep, device=audiolen.device)[None]
            < audiolen[:, None]).to(torch.float32)
        ## normalize the inputs before feed to wav2vec2
        ## , reference Wav2VecFeatureExtractor
        # [B]
        mean = (audio * mask).sum(dim=-1) / audiolen.to(torch.float32)
        # [B]
        var = (audio - mean[:, None]).square().sum(dim=-1) / audiolen.to(torch.float32)
        # [B, T], for numerical stability of square root
        normed = (audio - mean[:, None]) / (var[:, None] + 1e-7).sqrt()
        output = self.model(
            normed,
            attention_mask=mask.to(torch.long),
            output_hidden_states=True)
        if return_all:
            return output
        # [B, C(=1024)]
        speaker = output.hidden_states[self.speaker].mean(dim=1)
        # [B, S, C(=1024)]
        linguistic = output.hidden_states[self.linguistic]
        return speaker, linguistic

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
