from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

import speechset

 
class RealtimeWavDataset(speechset.WavDataset):
    """Realtime-loading scheme.
    """
    def __init__(self,
                 reader: speechset.datasets.DataReader,
                 device: torch.device,
                 verbose: bool = False):
        """Initializer.
        """
        succeed = RealtimeWavDataset.hook_reader(reader)
        if verbose:
            print(f'[*] RealtimeWavDataset: {succeed} hook installed')
        # super call hooked reader
        super().__init__(reader)
        self.device = device

    def normalize(self, sid: int, text: str, speech: Tuple[torch.Tensor, int, int]) \
            -> Tuple[torch.Tensor, int, int]:
        """Normalize datum.
        Args:
            sid: speaker id.
            text: transcription.
            speech: speech signal and sampling rates.
        """
        return speech

    def collate(self, bunch: List[Tuple[torch.Tensor, int, int]]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x tuple, loaded speech signals.
        Returns:
            batch data.
                speeches: [torch.float32; [B, T]], speech signal.
                lengths: [torch.long; [B]], speech lengths.
        """
        # resample on gpu
        audios = [
            torchaudio.functional.resample(audio.to(self.device), prev, tgt)
            for audio, prev, tgt in bunch]
        # [B]
        lengths = torch.tensor([len(a) for a in audios], dtype=torch.long, device=self.device)
        # []
        maxlen = lengths.amax()
        # [B, T]
        speeches = torch.stack([F.pad(a, [0, maxlen - len(a)])  for a in audios], dim=0)
        return speeches, lengths

    @staticmethod
    def load_audio(path: str, sr: int) -> Tuple[torch.Tensor, int, int]:
        """Load audio with real-time action.
        Args:
            path: path to the audio.
            sr: sampling rate.
        Returns:
            [torch.float32; [T]], audio signal in original sampling rate, [-1, 1]-ranged.
            int, original sampling rate.
            int, target sampling rate, same as `sr`.
        """
        # [C, T]
        audio, prev = torchaudio.load(path)
        # [T]
        return audio[0].to(torch.float32), prev, sr

    @classmethod
    def hook_reader(cls, reader: speechset.datasets.DataReader) -> int:
        """Reader hook for preventing resampling on reading with CPU.
        Args:
            reader: data reader.
        """
        if isinstance(reader, speechset.datasets.ConcatReader):
            succ = 0
            for subreader in reader.readers:
                succ += RealtimeWavDataset.hook_reader(subreader)
                if isinstance(subreader, speechset.utils.DumpReader):
                    # HACK: recaching the mapper
                    loader = subreader.preproc()
                    for path in subreader.dataset():
                        reader.mapper[path] = loader
            return succ

        if isinstance(reader, speechset.utils.DumpReader):
            def load_dump(path: str):
                sid, text, audio = np.load(path, allow_pickle=True)
                return sid, text, (torch.tensor(audio), reader.prev_sr, reader.sr)
            reader.preprocessor = load_dump
        else:
            reader.load_audio = RealtimeWavDataset.load_audio
        return 1


class WeightedRandomWrapper(speechset.speeches.SpeechSet):
    """Speechset wrapper for weighted sampling. 
    """
    def __init__(self, wrapped: speechset.speeches.SpeechSet, subepoch: Optional[int] = None):
        """Initializer.
        Args:
            wrapped: speechset.
            subepoch: the number of the iterations for 1-epoch, use `len(wrapped)` as default.
        """
        super().__init__(wrapped.reader)
        # hold
        self.speechset = wrapped
        # grouping
        self.groups = {}
        for path, (sid, _) in tqdm(self.dataset.items()):
            if sid not in self.groups:
                self.groups[sid] = []
            self.groups[sid].append(path)
        # queing
        self.sids = list(self.groups.keys())
        self.queue = {sid: self.shuffle(paths) for sid, paths in self.groups.items()}
        # iteration checks
        self.subepoch = subepoch or (len(wrapped) // len(self.sids))

    def shuffle(self, lists: List[str], seed: Optional[int] = None):
        """Shuffle the queue
        """
        rng = np.random.default_rng(seed)
        return [lists[i] for i in rng.permutation(len(lists))]

    def sample(self, sid: int):
        # assign if empty
        if len(self.queue[sid]) == 0:
            self.queue[sid] = self.shuffle(self.groups[sid])
        # pop
        path = self.queue[sid].pop()
        return self.normalize(*self.preproc(path))

    def __len__(self) -> int:
        """Return length of the dataset.
        Returns:
            length.
        """
        return len(self.groups) * self.subepoch

    def __getitem__(self, index: Union[int, slice]):
        """Weighted sampling.
        Args:
            index: input index.
        Returns:
            normalized.
        """
        num = len(self.sids)
        if isinstance(index, int):
            # cyclic
            return self.sample(self.sids[index % num])
        # pack
        return self.collate([
            self.sample(self.sids[i % num])
            for i in range(index.start, index.stop, index.step or 1)])

    def normalize(self, *args, **kwargs):
        """Forward to given.
        """
        return self.speechset.normalize(*args, **kwargs)

    def collate(self, *args, **kwargs):
        """Forward to given.
        """
        return self.speechset.collate(*args, **kwargs)
