import os
import hydra
from hydra import utils
from pathlib import Path
import librosa
import scipy
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def process_wav(wav_path, out_path, sr=160000, preemph=0.97, n_fft=2048, n_fft2=1024, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path.with_suffix(".wav"), sr=sr,
                          offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999

    mel = librosa.feature.melspectrogram(preemphasis(wav, preemph),
                                         sr=sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         fmin=fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1

    mag = np.abs(librosa.stft(preemphasis(wav, preemph),
                              n_fft=n_fft,
                              hop_length=hop_length,
                              win_length=win_length))
    logmag = librosa.amplitude_to_db(mag, top_db=top_db)
    logmag = logmag / top_db + 1

    # wav = mulaw_encode(wav, mu=2**bits)

    np.save(out_path.with_suffix(".wav.npy"), wav)
    np.save(out_path.with_suffix(".mel.npy"), logmel)
    np.save(out_path.with_suffix(".mag.npy"), logmag)

    return out_path, logmel.shape[-1]

@hydra.main(config_path="config/preprocessing.yaml")
def preprocess_dataset(cfg):
    in_dir = Path(utils.to_absolute_path(cfg.in_dir))

    out_train_dir = Path(utils.to_absolute_path("datasets")) / str(cfg.dataset.dataset) / "train"
    out_train_dir.mkdir(parents=True, exist_ok=True)

    out_test_dir = Path(utils.to_absolute_path("datasets")) / str(cfg.dataset.dataset) / "test"
    out_test_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for root, _, files in os.walk(in_dir):
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.wav':
                names.append(name)

    num_train_names = int(len(names) * (1 - cfg.split))
    # num_test_names = len(names) - num_train_names
    train_names = names[:num_train_names]
    test_names = names[num_train_names:]

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    print("Extracting features")
    futures = []

    for name in train_names:
        wav_path = in_dir / name
        out_path = out_train_dir / name
        futures.append(executor.submit(partial(process_wav, wav_path, out_path, **cfg.preprocessing)))

    for name in test_names:
        wav_path = in_dir / name
        out_path = out_test_dir / name
        futures.append(executor.submit(partial(process_wav, wav_path, out_path, **cfg.preprocessing)))

    results = [future.result() for future in tqdm(futures)]

    lengths = [x[-1] for x in results]
    frames = sum(lengths)
    frame_shift_ms = cfg.preprocessing.hop_length / cfg.preprocessing.sr
    hours = frames * frame_shift_ms / 3600
    print("Wrote {} utterances, {} frames ({:.2f} hours)".format(len(lengths), frames, hours))


if __name__ == "__main__":
    preprocess_dataset()
