import argparse
import json
import os

import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

import speechset
from config import Config
from disc import Discriminator
from nansy import Nansy
from utils.dataset import PairedDataset
from utils.wrapper import TrainingWrapper


class Trainer:
    """TacoSpawn trainer.
    """
    LOG_IDX = 0

    def __init__(self,
                 model: Nansy,
                 disc: Discriminator,
                 dataset: PairedDataset,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: Nansy model.
            disc: discriminator.
            dataset: dataset.
            config: unified configurations.
            device: target computing device.
        """
        self.model = model
        self.disc = disc
        self.dataset = dataset
        self.config = config
        # train-test split
        self.testset = self.dataset.split(config.train.split)

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.train.batch,
            shuffle=config.train.shuffle,
            collate_fn=self.dataset.collate,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=config.train.batch,
            collate_fn=self.dataset.collate,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        # training wrapper
        self.wrapper = TrainingWrapper(model, disc, config, device)

        self.optim_g = torch.optim.Adam(
            self.model.parameters(),
            config.train.learning_rate,
            (config.train.beta1, config.train.beta2))

        self.optim_d = torch.optim.Adam(
            self.disc.parameters(),
            config.train.learning_rate,
            (config.train.beta1, config.train.beta2))

        self.train_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def train(self, epoch: int = 0):
        """Train wavegrad.
        Args:
            epoch: starting step.
        """
        self.model.train()
        step = epoch * len(self.loader)
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=len(self.loader), leave=False) as pbar:
                # random pairing before sampling
                self.dataset.random_pairing(seed=epoch)
                for it, bunch in enumerate(self.loader):
                    sid, s1, s2 = self.wrapper.wrap(
                        self.wrapper.random_segment(bunch))
                    loss_g, losses_g, aux_g = \
                        self.wrapper.loss_generator(sid, s1, s2)
                    # update
                    self.optim_g.zero_grad()
                    loss_g.backward()
                    self.optim_g.step()

                    loss_d, losses_d, aux_d = \
                        self.wrapper.loss_discriminator(sid, s1, s2)
                    # update
                    self.optim_d.zero_grad()
                    loss_d.backward()
                    self.optim_d.step()

                    step += 1
                    pbar.update()
                    pbar.set_postfix({'loss': loss_d.item(), 'step': step})

                    for key, val in {**losses_g, **losses_d}.items():
                        self.train_log.add_scalar(f'{key}', val, step)

                    with torch.no_grad():
                        grad_norm = np.mean([
                            torch.norm(p.grad).item()
                            for p in self.model.parameters() if p.grad is not None])
                        param_norm = np.mean([
                            torch.norm(p).item()
                            for p in self.model.parameters() if p.dtype == torch.float32])

                    self.train_log.add_scalar('common/grad-norm', grad_norm, step)
                    self.train_log.add_scalar('common/param-norm', param_norm, step)
                    self.train_log.add_scalar(
                        'common/learning-rate-g', self.optim_g.param_groups[0]['lr'], step)
                    self.train_log.add_scalar(
                        'common/learning-rate-d', self.optim_d.param_groups[0]['lr'], step)

                    if (it + 1) % (len(self.loader) // 50) == 0:
                        self.train_log.add_image(
                            'train/gt',
                            self.mel_img(aux_g['mel'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            'train/rctor',
                            self.mel_img(aux_g['rctor'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            'train/filter',
                            self.mel_img(aux_g['filter'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            'train/source',
                            self.mel_img(aux_g['source'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            'train/yingram',
                            self.mel_img(aux_g['yingram'][Trainer.LOG_IDX]), step)

            self.model.save(f'{self.ckpt_path}_{epoch}.ckpt', self.optim_g)
            self.disc.save(f'{self.ckpt_path}_{epoch}.ckpt-disc', self.optim_d)

            losses = {
                key: [] for key in {**losses_d, **losses_g}}
            COND_KEYS = ['metric/pos', 'metric/neg']
            for key in COND_KEYS:
                losses[key] = []

            with torch.no_grad():
                for bunch in self.testloader:
                    sid, s1, s2 = self.wrapper.wrap(
                        self.wrapper.random_segment(bunch))
                    _, losses_g, _ = self.wrapper.loss_generator(sid, s1, s2)
                    _, losses_d, _ = self.wrapper.loss_discriminator(sid, s1, s2, r1=False)
                    for key, val in {**losses_g, **losses_d}.items():
                        losses[key].append(val)
                # test log
                for key, val in losses.items():
                    self.test_log.add_scalar(f'{key}', np.mean(val), step)

    def mel_img(self, mel: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram images.
        Args:
            signal: [np.float32; [mel, T]], speech signal.
        Returns:
            [np.float32; [3, mel, T]], mel-spectrogram in viridis color map.
        """
        # minmax norm in range(0, 1)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-7)
        # in range(0, 255)
        mel = (mel * 255).astype(np.uint8)
        # [mel, T, 3]
        mel = self.cmap[mel]
        # [3, mel, T], make origin lower
        mel = np.flip(mel, axis=0).transpose(2, 0, 1)
        return mel


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-epoch', default=0, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--auto-rename', default=False, action='store_true')
    args = parser.parse_args()

    # seed setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # configurations
    config = Config()
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    if args.name is not None:
        config.train.name = args.name

    log_path = os.path.join(config.train.log, config.train.name)
    # auto renaming
    if args.auto_rename and os.path.exists(log_path):
        config.train.name = next(
            f'{config.train.name}_{i}' for i in range(1024)
            if not os.path.exists(f'{log_path}_{i}'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # prepare datasets
    dataset = PairedDataset(
        speechset.utils.DumpReader(args.data_dir))

    # model definition
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Nansy(config.model)
    model.to(device)

    disc = Discriminator(config.disc)
    disc.to(device)

    trainer = Trainer(model, disc, dataset, config, device)

    # loading
    if args.load_epoch > 0:
        # find checkpoint
        ckpt_path = os.path.join(
            config.train.ckpt,
            config.train.name,
            f'{config.train.name}_{args.load_epoch}.ckpt')
        # load checkpoint
        ckpt = torch.load(ckpt_path)
        model.load_(ckpt, trainer.optim_g)
        # discriminator checkpoint
        ckpt_disc = torch.load(f'{ckpt_path}-disc')
        disc.load_(ckpt_disc, trainer.optim_d)
        print('[*] load checkpoint: ' + ckpt_path)
        # since epoch starts with 0
        args.load_epoch += 1

    # git configuration
    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    # start train
    trainer.train(args.load_epoch)
