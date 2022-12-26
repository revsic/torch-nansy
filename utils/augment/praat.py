import os
import multiprocessing as mp
from typing import Callable, Optional, Tuple, Union

import numpy as np
import parselmouth

import speechset
from config import Config


class PraatAugment:
    """Praat based augmentation.
    """
    def __init__(self,
                 config: Config,
                 pitch_steps: float = 0.01,
                 pitch_floor: float = 75,
                 pitch_ceil: float = 600):
        """Initializer.
        Args:
            config: configurations.
            pitch_steps: pitch measurement intervals.
            pitch_floor: minimum pitch.
            pitch_ceil: maximum pitch.
        """
        self.config = config
        self.pitch_steps = pitch_steps
        self.pitch_floor = pitch_floor
        self.pitch_ceil = pitch_ceil

    def augment(self,
                snd: Union[parselmouth.Sound, np.ndarray],
                formant_shift: float = 1.,
                pitch_shift: float = 1.,
                pitch_range: float = 1.,
                duration_factor: float = 1.) -> np.ndarray:
        """Augment the sound signal with praat.
        """
        if not isinstance(snd, parselmouth.Sound):
            snd = parselmouth.Sound(snd, sampling_rate=self.config.model.sr)
        pitch = parselmouth.praat.call(
            snd, 'To Pitch', self.pitch_steps, self.pitch_floor, self.pitch_ceil)
        median = parselmouth.praat.call(
            pitch, 'Get quantile', 0., 0., 0.5, 'Hertz')
        out, = parselmouth.praat.call(
            (snd, pitch), 'Change gender',
            formant_shift,
            median * pitch_shift,
            pitch_range,
            duration_factor).values
        return out

    def augment_wrap(self, wrap: Tuple[np.ndarray, float, float]) -> np.ndarray:
        audio, fs, ps = wrap
        return self.augment(audio, fs, ps)

    def cache(self,
              out_dir: str,
              reader: speechset.datasets.DataReader,
              num_ps: int = 5,
              num_fs: int = 5,
              verbose: Optional[Callable] = None,
              pool: Optional[Union[int, mp.Pool]] = None,
              chunksize: int = 1):
        """Cache the contextual features.
        Args:
            out_dir: output directory.
            reader: data reader.
            num_ps: the number of the candidates for pitch shifting.
            num_ps: the number of the candidates for formant shifting.
            verbose: whether write the progressbar or not.
        """
        dataset, preproc = reader.dataset(), reader.preproc()
        if verbose is not None:
            dataset = verbose(dataset)

        ps = np.linspace(1., self.config.train.pitch_shift, num_ps)
        fs = np.linspace(1., self.config.train.formant_shift, num_fs)

        close_pool = False
        if pool is not None and isinstance(pool, int):
            close_pool = True
            pool = mp.Pool(pool)

        os.makedirs(out_dir, exist_ok=True)
        for path in dataset:
            # [T]
            outputs = preproc(path)
            assert len(outputs) in [2, 3]
            if len(outputs) == 2:
                _, audio = outputs
            else:
                _, _, audio = outputs
            # cache the augment
            augmented = []
            if pool is None:
                # wrap
                snd = parselmouth.Sound(audio, sampling_frequency=self.config.model.sr)
                for p in ps:
                    for f in fs:
                        augmented.extend([
                            self.augment(snd, f, p),
                            self.augment(snd, 1 / f, p),
                            self.augment(snd, f, 1 / p),
                            self.augment(snd, 1 / f, 1 / p)])
            else:
                for out in pool.imap_unordered(
                        self.augment_wrap,
                        [
                            (audio, f_, p_)
                            for p in ps
                            for f in fs
                            for f_, p_ in [
                                (f, p),
                                (1 / f, p),
                                (f, 1 / p),
                                (1 / f, 1 / p)]],
                        chunksize=chunksize):
                    augmented.append(out)

            # write
            name, _ = os.path.splitext(os.path.basename(path))
            np.save(os.path.join(out_dir, f'{name}.npy'), np.stack(augmented, axis=0))

        if close_pool:
            pool.close()


if __name__ == '__main__':
    def main():
        import argparse
        from tqdm import tqdm

        parser = argparse.ArgumentParser()
        parser.add_argument('--data-dir', required=True)
        parser.add_argument('--out-dir', required=True)
        parser.add_argument('--num-ps', default=3, type=int)
        parser.add_argument('--num-fs', default=3, type=int)
        parser.add_argument('--pool', default=None, type=int)
        parser.add_argument('--chunksize', default=1, type=int)
        args = parser.parse_args()

        # hard code the reader
        reader = speechset.utils.DumpReader(args.data_dir)

        config = Config()
        aug = PraatAugment(config)
        aug.cache(
            args.out_dir,
            reader,
            args.num_ps,
            args.num_fs,
            tqdm,
            args.pool,
            args.chunksize)

    main()
