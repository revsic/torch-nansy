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
                speeches: [np.float32; [B, T]], speeches.
                lengths: [np.long; [B]], speech lengths.
        Returns:
            randomly segmented spectrogram and audios.
        """
        # [B], [B, 2], [B, T], [B, T]
        _, speeches, lengths = bunch
        def segment(seq: np.ndarray, len_: np.ndarray) -> np.ndarray:
            # [B]
            start = np.random.randint(np.maximum(1, len_ - self.seglen))
            # [B, seglen]
            return np.array(
                [np.pad(q[s:s + self.seglen], [0, max(self.seglen - len(q), 0)])
                 for q, s in zip(seq, start)])
        # [B], [B, seglen]
        return segment(speeches, lengths)

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
        pr = sampler(self.config.train.pitch_range)
        # parametric equalizer
        peaks = self.config.train.num_peak
        # quality factor
        power = torch.rand(bsize, peaks + 2, device=signal.device)
        # gains
        g_min, g_max = self.config.train.g_min, self.config.train.g_max
        gain = torch.rand(bsize, peaks + 2, device=signal.device) * (g_max - g_min) + g_min
        return fs, ps, pr, power, gain

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
            fshift, pshift, prange, power, gain = self.sample_like(signal)
            if not ps:
                pshift = None
            # [B, T]
            out = self.aug.forward(signal, pshift, prange, fshift, power, gain)
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

    def loss_discriminator(self, seg: torch.Tensor, r1: bool = True) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the discriminator loss.
        Args:
            seg: [torch.float32; [B, seglen]], segmented speech.
            r1: whether use r1-regularization, for evaluation loss.
        Returns:
            loss and disctionaries.
        """
        with torch.no_grad():
            # augmentation
            seg_f = self.augment(seg)
            seg_g = self.augment(seg, ps=False)
            # _, [B, S, 1024]
            _, linguistic = self.model.wav2vec2.forward(seg_f)
            # [B, 1024, S]
            linguistic = linguistic.transpose(1, 2)
            # [B, T / mel_strides]
            energy, mel = self.model.analyze_energy(seg)
            # [B, Y, T / yin_strides]
            yingram = self.model.sample_yingram(self.model.analyze_yingram(seg_g))
            # [B, ver_out_channels]
            _, spk = self.model.analyze_wav2vec2(seg)
        # [B, mel, T]
        rctor, _, _ = self.model.synthesize(linguistic, spk, energy, yingram)
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
        # [B, ver_out_channels]
        spk_pos, spk_neg = spk, spk[indices]

        # [B, T]
        pos_real = torch.matmul(spk_pos[:, None], spk_real).squeeze(dim=1)
        neg_real = torch.matmul(spk_neg[:, None], spk_real).squeeze(dim=1)
        # [B, T]
        pos_fake = torch.matmul(spk_pos[:, None], spk_fake).squeeze(dim=1)
        neg_fake = torch.matmul(spk_neg[:, None], spk_fake).squeeze(dim=1)

        # [B, T], average is not necessary since generation is conditioned.
        # use Choi et al., ICLR2020, instead of NANSY objective
        rel_real = d_real - d_fake + pos_real - pos_fake
        # NANSY: neg_fake - neg_real
        rel_cont = pos_real - neg_real
        # [B, T]
        logits = rel_real + rel_cont
        # [B, T]
        disc = F.softplus(-logits).mean()

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
        loss = disc + r1_penalty * 10
        losses = {
            'disc/loss': loss.item(),
            'disc/r1': r1_penalty.item(),
            'disc/disc': disc.item(),
            'disc-aux/logits': logits.mean().item(),
            'disc-aux/d-real': d_real.mean().item(),
            'disc-aux/d-fake': d_fake.mean().item(),
            'disc-aux/pos-real': pos_real.mean().item(),
            'disc-aux/pos-fake': pos_fake.mean().item(),
            'disc-aux/neg-real': neg_real.mean().item(),
            'disc-aux/neg-fake': neg_fake.mean().item()}
        return loss, losses

    def loss_generator(self, seg: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
        """Compute the generator loss.
        Args:
            seg: [torch.float32; [B, seglen]], segmented speech.
        Returns:
            loss and disctionaries.
        """
        with torch.no_grad():
            # augmentation
            seg_f = self.augment(seg)
            seg_g = self.augment(seg, ps=False)
            # [B, S, 1024]
            _, linguistic = self.model.wav2vec2.forward(seg_f)
            # [B, 1024, S]
            linguistic = linguistic.transpose(1, 2)
            # [B, T / mel_strides]
            energy, mel = self.model.analyze_energy(seg)
            # [B, Y, T / yin_strides]
            yingram = self.model.sample_yingram(
                self.model.analyze_yingram(seg_g))
        # [B, ver_out_channels]
        _, spk = self.model.analyze_wav2vec2(seg)
        # [B, mel, T]
        rctor, filter_, source = self.model.synthesize(linguistic, spk, energy, yingram)
        # []
        rctor_loss = (mel - rctor).abs().mean()

        # [B, T], [B, spk, T]
        d_real, spk_real = self.disc.forward(mel)
        # [B, T], [B, spk, T]
        d_fake, spk_fake = self.disc.forward(rctor)

        bsize = spk_real.shape[0]
        # [], range [1, B - 1]
        start = np.random.randint(bsize - 1) + 1
        # [B], for shuffling
        indices = (np.arange(bsize) + start) % bsize
        # [B, ver_out_channels]
        spk_pos, spk_neg = spk, spk[indices]

        # [B, T]
        pos_real = torch.matmul(spk_pos[:, None], spk_real).squeeze(dim=1)
        neg_real = torch.matmul(spk_neg[:, None], spk_real).squeeze(dim=1)
        # [B, T]
        pos_fake = torch.matmul(spk_pos[:, None], spk_fake).squeeze(dim=1)
        neg_fake = torch.matmul(spk_neg[:, None], spk_fake).squeeze(dim=1)

        # [B, T]
        rel_fake = d_fake - d_real + pos_fake - pos_real
        rel_cont = pos_fake - neg_fake
        # [B, T]
        logits = rel_fake + rel_cont
        # []
        disc = F.softplus(-logits).mean()

        # [B, B], metric purpose
        confusion = torch.matmul(spk, spk.T)
        # [B, B]
        mask = torch.eye(bsize, device=self.device)
        # []
        metric_pos = (confusion * mask).sum().item() / bsize
        metric_neg = (confusion * (1 - mask)).sum().item() / (bsize * (bsize - 1))

        # []
        loss = disc + rctor_loss
        losses = {
            'gen/loss': loss.item(),
            'gen/rctor': rctor_loss.item(),
            'gen/disc': disc.item(),
            'gen-aux/logits': logits.mean().item(),
            'gen-aux/d-real': d_real.mean().item(),
            'gen-aux/d-fake': d_fake.mean().item(),
            'gen-aux/pos-real': pos_real.mean().item(),
            'gen-aux/pos-fake': pos_fake.mean().item(),
            'gen-aux/neg-real': neg_real.mean().item(),
            'gen-aux/neg-fake': neg_fake.mean().item(),
            'metric/pos': metric_pos,
            'metric/neg': metric_neg}
        # for visualization
        # [B, T, Y] -> [B, T, Y // bins] -> [B, Y // bins, T]
        yingram = F.interpolate(
            yingram.transpose(1, 2),
            scale_factor=self.config.model.yin_bins ** -1, mode='linear').transpose(1, 2)
        return loss, losses, {
            'yingram': yingram.cpu().detach().numpy(),
            'mel': mel.cpu().detach().numpy(),
            'rctor': rctor.cpu().detach().numpy(),
            'filter': filter_.cpu().detach().numpy(),
            'source': source.cpu().detach().numpy()}
