from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from disc import Discriminator
from nansy import Nansy

from .augment import Augment


class TrainingWrapper:
    """Training wrapper.
    """
    def __init__(self,
                 model: Nansy,
                 disc: Discriminator,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: NANSY model.
            disc: discriminator.
            config: training configurations.
            device: torch device.
        """
        self.model = model
        self.disc = disc
        self.config = config
        self.device = device
        # augmentation
        self.aug = Augment(config)
        self.aug.to(device)
        # alias
        self.seglen = self.config.train.seglen

    def wrap(self, bunch: List[np.ndarray]) -> List[torch.Tensor]:
        """Wrap the array to torch tensor.
        Args:
            bunch: input tensors.
        Returns:
            wrapped.
        """
        return [torch.tensor(array, device=self.device) for array in bunch]

    def random_segment(self, bunch: List[np.ndarray]) -> List[np.ndarray]:
        """Segment the spectrogram and audio into fixed sized array.
        Args:
            bunch: input tensors.
                sid: [np.long; [B]], speaker id.
                length: [np.long; [B, 2]], speech lengths.
                speech1, speech2: [np.float32; [B, T]], speaches.
        Returns:
            randomly segmented spectrogram and audios.
        """
        # [B], [B, 2], [B, T], [B, T]
        sid, lengths, s1, s2 = bunch
        def segment(seq: np.ndarray, len_: np.ndarray) -> np.ndarray:
            # [B]
            start = np.random.randint(np.maximum(1, len_ - self.seglen))
            # [B, seglen]
            return np.array(
                [np.pad(q[s:s + self.seglen], [0, max(self.seglen - len(q), 0)])
                 for q, s in zip(seq, start)])
        # [B], [B, seglen], [B, seglen]
        return sid, segment(s1, lengths[:, 0]), segment(s2, lengths[:, 1])

    def sample_like(self, signal: torch.Tensor) -> List[torch.Tensor]:
        """Sample augmentation parameters.
        Args:
            signal: [torch.float32; [B, T]], speech signal.
        Returns:
            augmentation parameters.
        """
        # [B]
        bsize, _ = signal.shape
        def sampler(ratio):
            shifts = torch.rand(bsize, device=signal.device) * (ratio - 1.) + 1.
            # flip
            flip = torch.rand(bsize) < 0.5
            shifts[flip] = shifts[flip] ** -1
            return shifts
        # sample shifts
        fs = sampler(self.config.train.formant_shift)
        ps = sampler(self.config.train.pitch_shift)
        # parametric equalizer
        peaks = self.config.train.num_peak
        # quality factor
        power = torch.rand(bsize, peaks + 2, device=signal.device)
        # gains
        g_min, g_max = self.config.train.g_min, self.config.train.g_max
        gain = torch.rand(bsize, peaks + 2, device=signal.device) * (g_max - g_min) + g_min
        return fs, ps, power, gain

    def augment(self, signal: torch.Tensor, ps: bool = True) -> torch.Tensor:
        """Augment the speech.
        Args:
            signal: [torch.float32; [B, T]], segmented speech.
            ps: whether use pitch shift.
        Returns:
            [torch.float32; [B, T]], speech signal.
        """
        # B
        bsize, _ = signal.shape
        saves = None
        while saves is None or len(saves) < bsize:
            # [B] x 4
            fshift, pshift, power, gain = self.sample_like(signal)
            if not ps:
                pshift = None
            # [B, T]
            out = self.aug.forward(signal, pshift, fshift, power, gain)
            # for covering unexpected NaN
            nan = out.isnan().any(dim=-1)
            if not nan.all():
                # save the outputs for not-nan inputs
                if saves is None:
                    saves = out[~nan]
                else:
                    saves = torch.cat([saves, out[~nan]], dim=0)
        # [B, T]
        return saves[:bsize]

    def loss_discriminator(self,
                           sid: torch.Tensor,
                           s1: torch.Tensor,
                           s2: torch.Tensor,
                           r1: bool = True) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the discriminator loss.
        Args:
            sid: [torch.long; [B]], speaker id.
            s1, s2: [torch.float32; [B, seglen]], segmented speech.
            r1: whether use r1-regularization, for evaluation loss.
        Returns:
            loss and disctionaries.
        """
        with torch.no_grad():
            # augmentation
            s1_f = self.augment(s1)
            s1_g = self.augment(s1, ps=False)
            # _, [B, S, 1024]
            _, linguistic = self.model.wav2vec2.forward(s1_f)
            # [B, ver_out_channels]
            _, spk1 = self.model.analyze_wav2vec2(s1)
            _, spk2 = self.model.analyze_wav2vec2(s2)
            # [B, T / mel_strides]
            energy, mel = self.model.analyze_energy(s1)
            # [B, Y, T / yin_strides]
            yingram = self.model.sample_yingram(
                self.model.analyze_yingram(s1_g))
            # [B, mel, T]
            rctor, _, _ = self.model.synthesize(linguistic, spk1, energy, yingram)

        # for gradient penalty
        mel.requires_grad_(r1)
        # [B, T], [B, spk, T]
        d_real, spk_real = self.disc.forward(mel)
        # [B, T], [B, spk, T]
        d_fake, spk_fake = self.disc.forward(rctor)

        bsize = spk_real.shape[0]
        # [], range [1, B - 1]
        start = np.random.randint(bsize - 1) + 1
        # [B], for shuffling
        indices = (np.arange(bsize) + start) % bsize
        # [B, T]
        pos_real = torch.matmul(spk2[:, None], spk_real).squeeze(dim=1)
        neg_real = torch.matmul(spk2[indices, None], spk_real).squeeze(dim=1)
        # [B, T]
        pos_fake = torch.matmul(spk2[:, None], spk_fake).squeeze(dim=1)
        neg_fake = torch.matmul(spk2[indices, None], spk_fake).squeeze(dim=1)

        # [B, T]
        logits_real = d_real + pos_real - neg_real
        logits_fake = d_fake + pos_fake - neg_fake

        if r1:
            # gradient penalty
            r1_grads = torch.autograd.grad(
                outputs=[d_real.sum()],
                inputs=[mel],
                create_graph=True,
                only_inputs=True)[0]
            # []
            r1_penalty = r1_grads.square().sum(dim=[1, 2]).mean()
        else:
            # no loss
            r1_penalty = torch.tensor([0.], device=mel.device)

        # masking if fail to construct negative pair
        logits_real = logits_real[sid != sid[indices]]
        logits_fake = logits_fake[sid != sid[indices]]
        disc_real = F.softplus(-logits_real).mean()
        disc_fake = -F.softplus(-logits_fake).mean()
        loss = disc_real + disc_fake + r1_penalty * 10
        losses = {
            'disc/loss': loss.item(),
            'disc/r1': r1_penalty.item(),
            'disc/disc-real': disc_real.item(),
            'disc/disc-fake': disc_fake.item(),
            'disc-aux/d-real': d_real.mean().item(),
            'disc-aux/d-fake': d_fake.mean().item(),
            'disc-aux/pos-real': pos_real.mean().item(),
            'disc-aux/pos-fake': pos_fake.mean().item(),
            'disc-aux/neg-real': neg_real[sid != sid[indices]].mean().item(),
            'disc-aux/neg-fake': neg_fake[sid != sid[indices]].mean().item()}
        # for visualization
        # [B, T, Y] -> [B, T, Y // bins] -> [B, Y // bins, T]
        yingram = F.interpolate(
            yingram.transpose(1, 2),
            scale_factor=self.config.model.yin_bins ** -1, mode='linear').transpose(1, 2)
        return loss, losses, {
            'yingram': yingram.cpu().detach().numpy(),
            'mel': mel.cpu().detach().numpy(),
            'rctor': rctor.cpu().detach().numpy()}

    def loss_generator(self, sid: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the generator loss.
        Args:
            sid: [torch.long; [B]], speaker id.
            s1, s2: [torch.float32; [B, seglen]], segmented speech.
        Returns:
            loss and disctionaries.
        """
        with torch.no_grad():
            # augmentation
            s1_f = self.augment(s1)
            s1_g = self.augment(s1, ps=False)
            # [B, S, 1024]
            _, linguistic = self.model.wav2vec2.forward(s1_f)
            # [B, T / mel_strides]
            energy, mel = self.model.analyze_energy(s1)
            # [B, Y, T / yin_strides]
            yingram = self.model.sample_yingram(
                self.model.analyze_yingram(s1_g))
        # [B, ver_out_channels]
        _, spk1 = self.model.analyze_wav2vec2(s1)
        _, spk2 = self.model.analyze_wav2vec2(s2)
        # [B, mel, T]
        rctor, filter_, source = self.model.synthesize(linguistic, spk1, energy, yingram)
        # []
        rctor_loss = (mel - rctor).abs().mean()

        # [B, T], [B, spk, T]
        d_fake, spk = self.disc.forward(rctor)

        bsize = spk.shape[0]        
        # [], range [1, B - 1]
        start = np.random.randint(bsize - 1) + 1
        # [B], for shuffling
        indices = (np.arange(bsize) + start) % bsize
        # [B, T]
        pos = torch.matmul(spk2[:, None], spk).squeeze(dim=1)
        neg = torch.matmul(spk2[indices, None], spk).squeeze(dim=1)
        # [B, T]
        logits = d_fake + pos - neg
        # masking if fail to construct negative pair
        logits = logits[sid != sid[indices]]
        disc = F.softplus(-logits).mean()

        # [B, B], metric purpose
        confusion = torch.matmul(spk1, spk2.T)
        mask = (sid[:, None] == sid) * (
            1 - torch.eye(bsize, device=self.device))
        # placeholder
        metric_pos, metric_neg = None, None
        if mask.sum() > 0:
            metric_pos = (confusion * mask).mean().item()
        if (1 - mask).sum() > 0:
            metric_neg = (confusion * (1 - mask)).mean().item()

        # []
        loss = disc + rctor_loss
        losses = {
            'gen/loss': loss.item(),
            'gen/rctor': rctor_loss.item(),
            'gen/disc': disc.item(),
            'gen-aux/d-fake': d_fake.mean().item(),
            'gen-aux/pos': pos.mean().item(),
            'gen-aux/neg': neg[sid != sid[indices]].mean().item()}
        # for visualization
        # [B, T, Y] -> [B, T, Y // bins] -> [B, Y // bins, T]
        yingram = F.interpolate(
            yingram.transpose(1, 2),
            scale_factor=self.config.model.yin_bins ** -1, mode='linear').transpose(1, 2)
        # conditional metric
        if metric_pos is not None:
            losses['metric/pos'] = metric_pos
        if metric_neg is not None:
            losses['metric/neg'] = metric_neg
        return loss, losses, {
            'yingram': yingram.cpu().detach().numpy(),
            'mel': mel.cpu().detach().numpy(),
            'rctor': rctor.cpu().detach().numpy(),
            'filter': filter_.cpu().detach().numpy(),
            'source': source.cpu().detach().numpy()}
