import argparse

import librosa
import torch

from nansy import Nansy
from utils.hifigan import HiFiGANWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=None)
parser.add_argument('--hifi-ckpt', default=None)
parser.add_argument('--hifi-config', default=None)
parser.add_argument('--wav', default=None)
args = parser.parse_args()

# load checkpoint
ckpt = torch.load(args.ckpt, map_location='cpu')
nansy = Nansy.load(ckpt)

device = torch.device('cuda:0')
nansy.to(device)
nansy.eval()

hifigan = HiFiGANWrapper(args.hifi_config, args.hifi_ckpt, device)

# load wav
wav, _ = librosa.load(args.wav, sr=nansy.config.sr)
wav = torch.tensor(wav, device=device)

with torch.no_grad():
    # reconstruction
    out, _ = nansy.forward(wav[None])
    print(f'[*] reconstruct {out.shape}')
    # vocoding
    out = hifigan.forward(out)
    out = out.squeeze(dim=0).cpu().numpy()
    print(f'[*] done, {out.shape}')

librosa.output.write_wav('rctor.wav', out, sr=nansy.config.sr)
