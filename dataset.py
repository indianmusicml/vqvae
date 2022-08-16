import os
import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, root, hop_length, sample_frames):
        self.root = Path(root)
        self.hop_length = hop_length
        self.sample_frames = sample_frames

        self.metadata = []
        for root, _, files in os.walk(self.root):
            for filename in files:
                name, ext = os.path.splitext(filename)
                name2, ext2 = os.path.splitext(name)
                if ext2 == '.wav':
                    self.metadata.append(name2)

        print("loaded dataset with {} samples".format(len(self.metadata)))

        # min_duration = (sample_frames + 2) * hop_length / sr
        # with open(self.root / "train.json") as file:
        #     metadata = json.load(file)
        #     self.metadata = [
        #         Path(out_path) for _, _, duration, out_path in metadata
        #         if duration > min_duration
        #     ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        path = self.root / path

        audio = np.load(path.with_suffix(".wav.npy"))
        # mel = np.load(path.with_suffix(".mel.npy"))
        # mag = np.load(path.with_suffix(".mag.npy"))

        # pos = randint(1, mag.shape[-1] - self.sample_frames - 2)
        # # mel = mel[:, pos - 1:pos + self.sample_frames + 1]
        # mag = mag[1:, pos - 1:pos + self.sample_frames + 1]  # throw away DC
        # audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length]

        pos = randint(1, audio.shape[-1] - self.sample_frames)
        audio = audio[pos:(pos + self.sample_frames)]

        # # vocoder training
        # pos = randint(0, mag.shape[-1] - self.sample_frames - 1)
        # mag = mag[1:, pos:(pos + self.sample_frames)]  # throw away DC
        # audio = audio[(pos * self.hop_length):((pos + self.sample_frames) * self.hop_length)]

        # mag = torch.FloatTensor(mag)
        audio = torch.FloatTensor(audio)

        # print('path: %s, pos: %d, len: %d, mag %d, audio %d' % (path, pos, mag.shape[-1], mag.size(-1), audio.size(-1)))

        return audio
        # return audio, mag


class AudioFileDataset(Dataset):
    def __init__(self, root, sample_frames):
        self.root = Path(root)
        self.sample_frames = sample_frames

        self.metadata = {}
        self.audio = {}
        for root, _, files in os.walk(self.root):
            for filename in files:
                name, ext = os.path.splitext(filename)
                name2, ext2 = os.path.splitext(name)
                if ext2 == '.wav':
                    path = self.root / filename
                    audio = np.load(path)
                    self.metadata[filename] = len(audio)
                    self.audio[filename] = audio

        self.total_frames = 0
        for name in self.metadata:
            self.total_frames += self.metadata[name]

        print("loaded dataset with {} files, {} total frames".format(len(self.metadata), self.total_frames))

    def __len__(self):
        return int(np.floor(self.total_frames / self.sample_frames))

    def __getitem__(self, index):
        index = index * self.sample_frames
        offset = 0
        buffer = np.empty((self.sample_frames, ), dtype=np.float32)
        buffer_pos = 0
        while buffer_pos < self.sample_frames:
            for name, n_frames in sorted(self.metadata.items()):
                if index > offset + n_frames:
                    offset += n_frames
                    continue

                buffer_len = self.sample_frames - buffer_pos
                start = index - offset
                if start + buffer_len > n_frames:
                    buffer_len = n_frames - start

                audio = self.audio[name]
                buffer[buffer_pos:(buffer_pos + buffer_len)] = audio[start:(start + buffer_len)]
                buffer_pos += buffer_len

                if buffer_pos >= self.sample_frames:
                    break

                # print('index: %d, offset: %d, pos: %d, len: %d, name: %s' % (index, offset, buffer_pos, buffer_len, name))
            if buffer_pos < self.sample_frames:
                print('circling dataset')

        return torch.FloatTensor(buffer)
