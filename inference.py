import argparse

import librosa
import numpy as np
import torch

from nansy import Nansy
from utils.hifigan import HiFiGANWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=None)
parser.add_argument('--hifi-ckpt', default=None)
parser.add_argument('--hifi-config', default=None)
parser.add_argument('--context', default=None)
parser.add_argument('--identity', default=None)
parser.add_argument('--median-shift', default=False, action='store_true')
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
if args.median_shift:
    import parselmouth
    extract_pitch = lambda x: parselmouth.praat.call(
        parselmouth.Sound(x, sampling_frequency=SR),
        'To Pitch', 0.01, 75, 600).selected_array['frequency']
    nonzero_median = lambda x: np.median(x[x > 1e-5]).item()
    # [P]
    pc_median = nonzero_median(extract_pitch(context))
    pi_median = nonzero_median(extract_pitch(identity))
    # midi steps
    BINS_PER_OCTAVE = 12
    steps = BINS_PER_OCTAVE * (np.log2(pi_median) - np.log2(pc_median))
    print(f'[*] shift {int(steps)} semitone steps')
    # moving median
    context_p = librosa.effects.pitch_shift(
        context, sr=SR, n_steps=int(steps), bins_per_octave=BINS_PER_OCTAVE)
    librosa.output.write_wav(f'{args.out_path}.shifted.wav', context_p, sr=SR)
else:
    context_p = context

with torch.no_grad():
    context = torch.tensor(context[None], device=device)
    # extracted features
    feat = nansy.analyze(context)
    # [1, Y, S]
    yingram = nansy.analyze_yingram(
        torch.tensor(context_p[None], device=device))

    # [1, T]
    identity = torch.tensor(identity[None], device=device)
    # [1, ver_out_channels]
    _, spk = nansy.analyze_wav2vec2(identity)

    # [1, mel_filters, T / mel_strides]
    mel, _, _ = nansy.synthesize(
        feat['linguistic'],
        spk,
        feat['energy'],
        nansy.sample_yingram(yingram))

    # vocoding
    out = hifigan.forward(mel).clamp(-1, 1).squeeze(dim=0)
    # [T]
    out = out / out.abs().amax().clamp_min(1e-7)
    print(f'[*] done, {out.shape}')

    librosa.output.write_wav(args.out_path, out.cpu().numpy(), sr=SR)
