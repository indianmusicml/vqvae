import os
from pathlib import Path
from collections import OrderedDict
import hydra
from hydra import utils
from tqdm import tqdm
import numpy as np
import torch

# from model import Encoder, Decoder2
from jukebox_vqvae.vqvae import VQVAE, calculate_strides


def clean_state_dict(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict_new[name] = v
    return state_dict_new


# @hydra.main(config_path="config/preprocessing_prior.yaml")
# def preprocess_dataset(cfg):
#     in_dir = Path(utils.to_absolute_path(cfg.in_dir))
#     out_dir = Path(utils.to_absolute_path(cfg.out_dir))
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     names = []
#     for root, _, files in os.walk(in_dir):
#         for filename in files:
#             subname, ext = os.path.splitext(filename)
#             name, ext2 = os.path.splitext(subname)
#             if ext == '.npy' and ext2 == '.wav':
#                 names.append(name)
#     print(names)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # encoder = Encoder(**cfg.model.encoder)
#     # decoder = Decoder2(**cfg.model.decoder)
#     # encoder.to(device)
#     # decoder.to(device)
#
#     print("Load checkpoint from: {}:".format(cfg.vqvae_checkpoint))
#     checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
#     checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#     encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]))
#     decoder.load_state_dict(clean_state_dict(checkpoint["decoder"]))
#
#     encoder.eval()
#     decoder.eval()
#
#     block_size = 16000
#     with torch.no_grad():
#         for i, name in enumerate(tqdm(names)):
#             wav_path = in_dir / name
#             wav_path = wav_path.with_suffix(".wav.npy")
#             audio = np.load(wav_path)
#             audio = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device).contiguous()
#
#             indices = np.empty(int(audio.size(-1)/block_size * 8000), dtype=np.long)
#             c = 0
#             for j in range(0, audio.size(-1), block_size):
#                 _, indices_chunk = encoder.encode(audio[:, :, j:j+block_size])
#                 indices_chunk = indices_chunk.squeeze().cpu().numpy()
#                 # print(indices_chunk)
#                 # print(indices_chunk.shape)
#                 indices[c:c+indices_chunk.shape[0]] = indices_chunk
#                 c += indices_chunk.shape[0]
#             indices = indices[:c]
#
#             out_path = out_dir / name
#             np.save(out_path.with_suffix(".idx.npy"), indices)
#
#     print("processed {} audio files".format(len(names)))


# @hydra.main(config_path="config/preprocessing_prior.yaml")
# def preprocess_dataset(cfg):
#     in_dir = Path(utils.to_absolute_path(cfg.in_dir))
#     out_dir = Path(utils.to_absolute_path(cfg.out_dir))
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     names = []
#     for root, _, files in os.walk(in_dir):
#         for filename in files:
#             subname, ext = os.path.splitext(filename)
#             name, ext2 = os.path.splitext(subname)
#             if ext == '.npy' and ext2 == '.mag':
#                 names.append(name)
#     print(names)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     encoder = Encoder(**cfg.model.encoder)
#     decoder = Decoder2(**cfg.model.decoder)
#     encoder.to(device)
#     decoder.to(device)
#
#     print("Load checkpoint from: {}:".format(cfg.vqvae_checkpoint))
#     checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
#     checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#     encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]))
#     decoder.load_state_dict(clean_state_dict(checkpoint["decoder"]))
#
#     encoder.eval()
#     decoder.eval()
#
#     # block_size = 16000
#     # with torch.no_grad():
#     #     for i, name in enumerate(tqdm(names)):
#     #         mag_path = in_dir / name
#     #         mag_path = mag_path.with_suffix(".mag.npy")
#     #         mag = np.load(mag_path)
#     #         mag = torch.FloatTensor(mag).unsqueeze(0).to(device).contiguous()
#     #
#     #         indices = np.empty(int(mag.size(-1)/block_size * 8000), dtype=np.long)
#     #         c = 0
#     #         for j in range(0, mag.size(-1), block_size):
#     #             _, indices_chunk = encoder(mag[:, :, j:j+block_size])
#     #             indices_chunk = indices_chunk.squeeze().cpu().numpy()
#     #             print(indices_chunk)
#     #             print(indices_chunk.shape)
#     #             indices[c:c+indices_chunk.shape[0]] = indices_chunk
#     #             c += indices_chunk.shape[0]
#     #         indices = indices[:c]
#     #
#     #         out_path = out_dir / name
#     #         np.save(out_path.with_suffix(".idx.npy"), indices)
#
#     with torch.no_grad():
#         for i, name in enumerate(tqdm(names)):
#             mag_path = in_dir / name
#             mag_path = mag_path.with_suffix(".mag.npy")
#             mag = np.load(mag_path)
#             mag = mag[1:, :]  # throw away DC
#             mag = torch.FloatTensor(mag).unsqueeze(0).to(device).contiguous()
#
#             _, indices = encoder.encode(mag)
#             indices = indices.squeeze().cpu().numpy()
#             print(indices)
#             print(indices.shape)
#
#             out_path = out_dir / name
#             np.save(out_path.with_suffix(".idx.npy"), indices)
#
#     print("processed {} mag files".format(len(names)))


# using jukebox vqvae
class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

@hydra.main(config_path="config/preprocessing_prior.yaml")
def preprocess_dataset(cfg):
    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for root, _, files in os.walk(in_dir):
        for filename in files:
            subname, ext = os.path.splitext(filename)
            name, ext2 = os.path.splitext(subname)
            if ext == '.npy' and ext2 == '.wav':
                names.append(name)
    print(names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hps = Hyperparams(
        sample_length=cfg.training.sample_frames,
        sr=cfg.preprocessing.sr,
        levels = 1,
        downs_t = (8,),
        strides_t = (2,),
        emb_width = 64,
        l_bins = 256,
        l_mu = 0.99,
        commit = 0.02,
        spectral = 0.0,
        multispectral = 1.0,
        loss_fn = 'l2',
        width = 32,
        depth = 4,
        m_conv = 1.0,
        dilation_growth_rate = 3,
        hvqvae_multipliers=None,
        lmix_l1=0.0,
        lmix_l2 = 1.0,
        lmix_linf=0.02,
        use_bottleneck=True,
        dilation_cycle=None,
        vqvae_reverse_decoder_dilation=True,
        revival_threshold=0.1,#1.0,
        linf_k=2048,
    )

    print(hps)

    block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv,
                        dilation_growth_rate=hps.dilation_growth_rate,
                        dilation_cycle=hps.dilation_cycle,
                        reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

    vqvae = VQVAE(input_shape=(hps.sample_length, 1), levels=hps.levels, downs_t=hps.downs_t, strides_t=hps.strides_t,
                  emb_width=hps.emb_width, l_bins=hps.l_bins,
                  mu=hps.l_mu, commit=hps.commit,
                  spectral=hps.spectral, multispectral=hps.multispectral,
                  multipliers=hps.hvqvae_multipliers, use_bottleneck=hps.use_bottleneck,
                  **block_kwargs)

    resume_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
    vqvae.load_state_dict(checkpoint["model"])

    block_size = 16000
    with torch.no_grad():
        for i, name in enumerate(tqdm(names)):
            wav_path = in_dir / name
            wav_path = wav_path.with_suffix(".wav.npy")
            audio = np.load(wav_path)
            audio = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device).contiguous()

            indices = np.empty(int(audio.size(-1)/block_size * 8000), dtype=np.long)
            c = 0
            for j in range(0, audio.size(-1), block_size):
                x_out, loss, metrics = vqvae(audio, hps, hps.loss_fn)
                _, indices_chunk = vqvae.encode(audio[:, :, j:j+block_size])
                indices_chunk = indices_chunk.squeeze().cpu().numpy()
                # print(indices_chunk)
                # print(indices_chunk.shape)
                indices[c:c+indices_chunk.shape[0]] = indices_chunk
                c += indices_chunk.shape[0]
            indices = indices[:c]

            out_path = out_dir / name
            np.save(out_path.with_suffix(".idx.npy"), indices)

    print("processed {} audio files".format(len(names)))

if __name__ == "__main__":
    preprocess_dataset()
