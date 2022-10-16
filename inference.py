import argparse
import os

import librosa
import numpy as np
import torch

from nansy import Nansy
from utils.hifigan import HiFiGANWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=None)
parser.add_argument('--hifi-ckpt', default=None)
parser.add_argument('--hifi-config', default=None)
parser.add_argument('--wav1', default=None)
parser.add_argument('--wav2', default=None)
parser.add_argument('--out-dir', default='./outputs')
args = parser.parse_args()

# load checkpoint
ckpt = torch.load(args.ckpt, map_location='cpu')
nansy = Nansy.load(ckpt)

device = torch.device('cuda:0')
nansy.to(device)
nansy.eval()

hifigan = HiFiGANWrapper(args.hifi_config, args.hifi_ckpt, device)

# load wav
SR = nansy.config.sr
wav1, _ = librosa.load(args.wav1, sr=SR)
wav2, _ = librosa.load(args.wav2, sr=SR)
wavs = [wav1, wav2]
# pack
wavlen = np.array([len(w) for w in wavs])
wav = np.stack([np.pad(w, [0, wavlen.max() - len(w)]) for w in wavs], axis=0)
# convert
wavlen = torch.tensor(wavlen, device=device)
wav = torch.tensor(wav, device=device)

with torch.no_grad():
    # [B, mel, T] reconstruction
    out, aux = nansy.forward(wav, wavlen)
    print(f'[*] reconstruct {out.shape}')
    # vocoding
    out = hifigan.forward(out)
    out = out / out.max(dim=-1).values.clamp_min(1e-7)[:, None]
    print(f'[*] done, {out.shape}')

    HOP = 256
    for i, (w, l) in enumerate(zip(out, wavlen)):
        librosa.output.write_wav(
            os.path.join(args.out_dir, f'rctor{i}.wav'),
            w.cpu().numpy()[:l.item() * HOP],
            sr=SR)

    spk = aux['spk']
    spk = torch.flip(spk, dims=(0,))
    # vc
    out, _, _ = nansy.synthesize(
        aux['linguistic'], spk, aux['energy'], aux['yingram'])
    print(f'[*] vc {out.shape}')

    out = hifigan.forward(out)
    out = out / out.max(dim=-1).values.clamp_min(1e-7)[:, None]
    print(f'[*] done, {out.shape}')

    HOP = 256
    for i, (w, l) in enumerate(zip(out, wavlen)):
        librosa.output.write_wav(
            os.path.join(args.out_dir, f'vc{i}.wav'),
            w.cpu().numpy()[:l.item() * HOP],
            sr=SR)

