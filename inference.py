import argparse

import librosa
import torch

from nansy import Nansy
from utils.hifigan import HiFiGANWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=None)
parser.add_argument('--hifi-ckpt', default=None)
parser.add_argument('--hifi-config', default=None)
parser.add_argument('--context', default=None)
parser.add_argument('--identity', default=None)
parser.add_argument('--out-path', default='./outputs/out.wav')
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
context, _ = librosa.load(args.context, sr=SR)
identity, _ = librosa.load(args.identity, sr=SR)

with torch.no_grad():
    context = torch.tensor(context[None], device=device)
    # TODO: pitch shift to match media pitch
    feat = nansy.analyze(context)

    # [1, T]
    identity = torch.tensor(identity[None], device=device)
    # [1, ver_out_channels]
    _, spk = nansy.analyze_wav2vec2(identity)

    # [1, mel_filters, T / mel_strides]
    mel, _, _ = nansy.synthesize(
        feat['linguistic'],
        spk,
        feat['energy'],
        nansy.sample_yingram(feat['yingram']))

    # vocoding
    out = hifigan.forward(mel).clamp(-1, 1).squeeze(dim=0)
    # []
    maxval = out.abs().max(dim=-1).values
    # [T]
    out = out / maxval.clamp_min(1e-7)
    print(f'[*] done, {out.shape}')

    librosa.output.write_wav(args.out_path, out.cpu().numpy(), sr=SR)
