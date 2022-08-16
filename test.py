import os
import random

import hydra
import hydra.utils as utils

from collections import OrderedDict
from pathlib import Path
import torch
import numpy as np
import scipy
import librosa
import pyloudnorm
from tqdm import tqdm

from preprocess import preemphasis
from model import Encoder, Decoder2
from model2.vqvae import VQVAE

from random import randint, seed

from train import mel_to_audio, mag_to_audio

from prior import TransformerModel, generate_square_subsequent_mask

from torch import nn, Tensor


def clean_state_dict(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict_new[name] = v
    return state_dict_new


@hydra.main(config_path="config/test.yaml")
def main(cfg):
    in_path = Path(utils.to_absolute_path(cfg.in_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    # decoder = Decoder(**cfg.model.decoder)
    decoder = Decoder2(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]))
    decoder.load_state_dict(clean_state_dict(checkpoint["decoder"]))

    encoder.eval()
    decoder.eval()

    meter = pyloudnorm.Meter(cfg.preprocessing.sr)

    wav, _ = librosa.load(in_path, sr=cfg.preprocessing.sr)
    ref_loudness = meter.integrated_loudness(wav)
    wav = wav / np.abs(wav).max() * 0.999

    print(wav.shape)

    start = int(len(wav)/2 - 10*cfg.preprocessing.sr)
    end = int(len(wav)/2 + 10*cfg.preprocessing.sr)
    wav = wav[start:end]

    print(wav.shape)

    mel = librosa.feature.melspectrogram(
        preemphasis(wav, cfg.preprocessing.preemph),
        sr=cfg.preprocessing.sr,
        n_fft=cfg.preprocessing.n_fft,
        n_mels=cfg.preprocessing.n_mels,
        hop_length=cfg.preprocessing.hop_length,
        win_length=cfg.preprocessing.win_length,
        fmin=cfg.preprocessing.fmin,
        power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
    logmel = logmel / cfg.preprocessing.top_db + 1

    mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
    print(mel.size())

    with torch.no_grad():

        for i in range(10):
            z, _ = encoder.encode(mel)

            z = z + torch.rand(z.size()).to(device)*0.001

            print(z.size())
            output = decoder.generate(z)
            print(output.shape)

            # output_loudness = meter.integrated_loudness(output)
            # output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
            # librosa.output.write_wav(out_path, output.astype(np.float32), sr=cfg.preprocessing.sr)

            audio = mel_to_audio(output.squeeze().cpu().numpy(),
                                 sr=cfg.preprocessing.sr,
                                 n_fft=cfg.preprocessing.n_fft,
                                 hop_length=cfg.preprocessing.hop_length,
                                 win_length=cfg.preprocessing.win_length,
                                 top_db=cfg.preprocessing.top_db,
                                 preemph=cfg.preprocessing.preemph)
            out_path = Path(utils.to_absolute_path(cfg.out_path + str(i)))
            librosa.output.write_wav(out_path, audio.astype(np.float32), sr=cfg.preprocessing.sr)


@hydra.main(config_path="config/test.yaml")
def main2(cfg):
    in_path = Path(utils.to_absolute_path(cfg.in_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VQVAE(cfg.model.encoder.channels, 32, 2, cfg.model.encoder.n_embeddings, cfg.model.encoder.embedding_dim, 0.25).to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(clean_state_dict(checkpoint["model"]))

    model.eval()

    meter = pyloudnorm.Meter(cfg.preprocessing.sr)

    wav, _ = librosa.load(in_path, sr=cfg.preprocessing.sr)
    ref_loudness = meter.integrated_loudness(wav)
    wav = wav / np.abs(wav).max() * 0.999

    print(wav.shape)

    start = int(len(wav)/2 - 10*cfg.preprocessing.sr)
    end = int(len(wav)/2 + 10*cfg.preprocessing.sr)
    wav = wav[start:end]

    print(wav.shape)

    mel = librosa.feature.melspectrogram(
        preemphasis(wav, cfg.preprocessing.preemph),
        sr=cfg.preprocessing.sr,
        n_fft=cfg.preprocessing.n_fft,
        n_mels=cfg.preprocessing.n_mels,
        hop_length=cfg.preprocessing.hop_length,
        win_length=cfg.preprocessing.win_length,
        fmin=cfg.preprocessing.fmin,
        power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
    logmel = logmel / cfg.preprocessing.top_db + 1

    mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
    print(mel.size())

    with torch.no_grad():
        for i in range(1):
            _, mels_hat, _, _ = model(mel, jitter=False)
            print(mels_hat.shape)

            # output_loudness = meter.integrated_loudness(output)
            # output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
            # librosa.output.write_wav(out_path, output.astype(np.float32), sr=cfg.preprocessing.sr)

            audio = mel_to_audio(mels_hat.squeeze().cpu().numpy(),
                                 sr=cfg.preprocessing.sr,
                                 n_fft=cfg.preprocessing.n_fft,
                                 hop_length=cfg.preprocessing.hop_length,
                                 win_length=cfg.preprocessing.win_length,
                                 top_db=cfg.preprocessing.top_db,
                                 preemph=cfg.preprocessing.preemph)
            out_path = Path(utils.to_absolute_path((cfg.out_path + str(i)))).with_suffix(".wav")
            librosa.output.write_wav(out_path, audio.astype(np.float32), sr=cfg.preprocessing.sr)


@hydra.main(config_path="config/test.yaml")
def main3(cfg):
    in_dir = Path(utils.to_absolute_path(cfg.in_path))
    out_dir = Path(utils.to_absolute_path(cfg.out_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for root, _, files in os.walk(in_dir):
        for filename in files:
            subname, ext = os.path.splitext(filename)
            name, ext2 = os.path.splitext(subname)
            if ext == '.npy' and ext2 == '.mel':
                names.append(name)
    print(names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VQVAE(cfg.model.encoder.channels, 32, 2, cfg.model.encoder.n_embeddings, cfg.model.encoder.embedding_dim, 0.25).to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(clean_state_dict(checkpoint["model"]))

    model.eval()

    with torch.no_grad():
        for name in names:
            mel_path = in_dir / name
            mel_path = mel_path.with_suffix(".mel.npy")
            mel = np.load(mel_path)
            mel = torch.FloatTensor(mel).unsqueeze(0).to(device)
            print(mel.size())

            _, _, indices, _ = model(mel)
            print(indices.size())

            indices = indices.squeeze().cpu().numpy()

            out_path = out_dir / name
            np.save(out_path.with_suffix(".idx.npy"), indices)


@hydra.main(config_path="config/test.yaml")
def main4(cfg):
    in_path = Path(utils.to_absolute_path(cfg.in_path_idx))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = cfg.model.encoder.n_embeddings  # size of vocabulary
    emsize = 256   # embedding dimension
    d_hid = 256    # ?? dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2    # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2      # number of heads in nn.MultiheadAttention
    prior = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout=0.).to(device)

    print("Load prior checkpoint from: {}:".format(cfg.prior_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.prior_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    prior.load_state_dict(checkpoint["model"])

    encoder = Encoder(**cfg.model.encoder).to(device)
    decoder = Decoder2(**cfg.model.decoder).to(device)

    print("Load checkpoint from: {}:".format(cfg.vqvae_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]))
    decoder.load_state_dict(clean_state_dict(checkpoint["decoder"]))

    prior.eval()
    encoder.eval()
    decoder.eval()

    indices = torch.from_numpy(np.load(in_path)).to(device)
    print(indices.size())

    context_length = 8192
    # num_tokens = 0
    num_tokens = 65536

    seed(4567)  # -> pos=1799920
    # seed(8901)    # -> pos=2186495

    with torch.no_grad():
        # decoded_tokens = torch.tensor(random.randint(1, cfg.model.encoder.n_embeddings)).unsqueeze(-1).to(device)
        # decoded_tokens = torch.tensor(i).unsqueeze(-1).to(device)

        pos = random.randint(1, indices.size(0) - context_length - num_tokens)
        print('pos: %d' % pos)
        context_tokens = indices[pos:pos+(context_length + num_tokens)]
        context_tokens = context_tokens.unsqueeze(-1).to(device)

        output_tokens = torch.empty((context_length + num_tokens, 1), dtype=torch.int64).to(device)
        output_tokens[:context_length] = context_tokens[:context_length]

        src_mask = generate_square_subsequent_mask(context_length).to(device)

        for j in tqdm(range(num_tokens)):
            input_tokens = context_tokens[j:j+context_length]
            output = prior(input_tokens, src_mask)
            top_indices = torch.argmax(output, dim=-1)
            top_indices_last_token = top_indices[-1:][0]  # we only care about the last token that was decoded
            output_tokens[context_length + j] = top_indices_last_token
            # print(top_indices_last_token)
            # print(output.size())

        print(output_tokens.size())
        print(output_tokens)
        z = encoder.quantize(output_tokens)
        print(z.size())
        output = decoder(z)
        print(output.size())
        audio = output.squeeze().cpu().numpy()
        out_path = Path(utils.to_absolute_path(cfg.out_path))
        librosa.output.write_wav(out_path, audio.astype(np.float32), sr=cfg.preprocessing.sr)

        # for j in range(100):
        #     pos = random.randint(1, indices.size(0) - bptt - 2)
        #     decoded_tokens = indices[pos:pos+bptt].to(device)
        #     src_mask = generate_square_subsequent_mask(decoded_tokens.size(0)).to(device)
        #     output = prior(decoded_tokens.unsqueeze(-1), src_mask)
        #     # print(output.size())
        #
        #     top_indices = torch.argmax(output, dim=-1)
        #     # we only care about the last token that was decoded
        #     top_indices_last_token = top_indices[-1:][0]
        #
        #     print('predicted %d %f' % (top_indices_last_token.item(), output[-1, :, top_indices_last_token.item()].item()))
        #     print('truth %d %f' % (indices[pos+bptt+1].item(), output[-1, :, indices[pos+bptt+1]].item()))
        #     print('-------')




@hydra.main(config_path="config/test.yaml")
def test2(cfg):
    in_path = Path(utils.to_absolute_path(cfg.in_path))
    mel = np.load(in_path)

    sample_frames = 256
    seed(1234)
    pos = randint(1, mel.shape[-1] - sample_frames - 2)
    mel = mel[:, pos - 1:pos + sample_frames + 1]

    mel = (mel - 1) * cfg.preprocessing.top_db
    mel = librosa.db_to_amplitude(mel, cfg.preprocessing.top_db)
    mel *= 0.01

    wav = librosa.feature.inverse.mel_to_audio(mel,
                                               sr=cfg.preprocessing.sr,
                                               n_fft=cfg.preprocessing.n_fft,
                                               hop_length=cfg.preprocessing.hop_length,
                                               win_length=cfg.preprocessing.win_length,
                                               power=1.)

    zi = ((2 - cfg.preprocessing.preemph) * wav[0] - wav[1]) / (3 - cfg.preprocessing.preemph)
    wav[0] -= zi
    wav = scipy.signal.lfilter([1.], [1., -cfg.preprocessing.preemph], wav)

    out_path = Path(utils.to_absolute_path(cfg.out_path))
    librosa.output.write_wav(out_path, wav.astype(np.float32), sr=cfg.preprocessing.sr)


def recon(mels, cfg, device):
    model = VQVAE(cfg.model.encoder.channels, 32, 2, cfg.model.encoder.n_embeddings, cfg.model.encoder.embedding_dim, 0.25).to(device)

    print("Load checkpoint from: {}:".format(cfg.vqvae_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(clean_state_dict(checkpoint["model"]))

    model.eval()

    with torch.no_grad():
        _, x_hat, z_q, indices, _ = model(mels.unsqueeze(0))

    return x_hat, z_q, indices


def generate(indices, cfg, device):
    model = VQVAE(cfg.model.encoder.channels, 32, 2, cfg.model.encoder.n_embeddings, cfg.model.encoder.embedding_dim, 0.25).to(device)

    print("Load checkpoint from: {}:".format(cfg.vqvae_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(clean_state_dict(checkpoint["model"]))

    model.eval()

    with torch.no_grad():
        output, z_q = model.generate(indices)

    return output, z_q


@hydra.main(config_path="config/test.yaml")
def test3(cfg):
    in_path_mel = Path(utils.to_absolute_path(cfg.in_path_mel))
    in_path_idx = Path(utils.to_absolute_path(cfg.in_path_idx))
    out_path = Path(utils.to_absolute_path(cfg.out_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # indices0 = torch.from_numpy(np.load(in_path)).to(device)
    # print(indices0)
    # output0, z_q0 = generate(indices0, cfg, device)

    # mels0 = torch.from_numpy(np.load(in_path_mel)).to(device)
    # output0, z_q0, indices0 = recon(mels0, cfg, device)

    # print('----------')

    indices1 = torch.from_numpy(np.load(in_path_idx)).to(device)
    # indices1 = indices1[:-20]

    # n = indices1.shape[0]
    # print(n)
    # n20 = int(indices1.shape[0]/20)
    # indices1_trim = np.empty((n-40,), dtype=np.long)
    # c = 0
    # for i in range(n):
    #     if (i % n20) < n20-2:
    #         indices1_trim[c] = indices1[i].item()
    #         c += 1
    #     else:
    #         print('skip! %d' % i)
    # print(c)
    # indices1 = torch.from_numpy(indices1_trim).cuda()

    # reshape into a 20xN grid, remove last column, flatten
    indices1 = indices1.view(20, -1)
    indices1 = indices1[:, :-1]
    indices1 = indices1.contiguous().view(-1)

    # print(indices1[:40])
    print(indices1[-100:-80])
    print(indices1.size())
    output1, z_q1 = generate(indices1, cfg, device)

    print('----------')

    mels1 = torch.from_numpy(np.load(in_path_mel)).to(device)
    mels1 = mels1[:, :-4]
    output2, z_q2, indices2 = recon(mels1, cfg, device)
    indices2 = indices2.squeeze()
    # print(indices2[:40])
    print(indices2[-100:-80])
    print(indices2.size())

    q_loss = torch.mean((z_q1 - z_q2)**2)
    print(q_loss)

    recon_loss = torch.mean((output1 - output2)**2)
    print(recon_loss)

    # idx_loss = torch.sum(torch.abs(indices1 - indices2[:indices1.size(0)]))
    # print(idx_loss)
    #
    # non_zero_mask = torch.nonzero(indices1 != indices2[:indices1.size(0)]).squeeze()
    # print(non_zero_mask)

    # idx_loss = torch.sum(torch.abs(indices1[:indices2.size(0)] - indices2))
    # print(idx_loss)

    torch.set_printoptions(threshold=10000)
    non_zero_mask = torch.nonzero(indices1[:indices2.size(0)] != indices2).squeeze()
    print(non_zero_mask)


    # idx_loss = torch.sum((indices0[:indices1.size(0)] - indices1))
    # print(idx_loss)

    # q_loss = torch.mean((z_q0[:, :, :, :z_q2.size(3)] - z_q2)**2)
    # print(q_loss)
    #
    # recon_loss = torch.mean((output0[:, :, :output2.size(2)] - output2)**2)
    # print(recon_loss)
    #
    # out_path0 = Path(utils.to_absolute_path(cfg.out_path + "_recon.wav"))
    # audio = mel_to_audio(output0[:, :, :500].squeeze().cpu().numpy(),
    #                      sr=cfg.preprocessing.sr,
    #                      n_fft=cfg.preprocessing.n_fft,
    #                      hop_length=cfg.preprocessing.hop_length,
    #                      win_length=cfg.preprocessing.win_length,
    #                      top_db=cfg.preprocessing.top_db,
    #                      preemph=cfg.preprocessing.preemph)
    # librosa.output.write_wav(out_path0, audio.astype(np.float32), sr=cfg.preprocessing.sr)

    out_path1 = Path(utils.to_absolute_path(cfg.out_path + "_recon-20.wav"))
    audio = mel_to_audio(output1[:, :, :500].squeeze().cpu().numpy(),
                         sr=cfg.preprocessing.sr,
                         n_fft=cfg.preprocessing.n_fft,
                         hop_length=cfg.preprocessing.hop_length,
                         win_length=cfg.preprocessing.win_length,
                         top_db=cfg.preprocessing.top_db,
                         preemph=cfg.preprocessing.preemph)
    librosa.output.write_wav(out_path1, audio.astype(np.float32), sr=cfg.preprocessing.sr)


    # model = VQVAE(cfg.model.encoder.channels, 32, 2, cfg.model.encoder.n_embeddings, cfg.model.encoder.embedding_dim, 0.25).to(device)
    #
    # print("Load checkpoint from: {}:".format(cfg.vqvae_checkpoint))
    # checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    # model.load_state_dict(clean_state_dict(checkpoint["model"]))
    #
    # model.eval()
    #
    # with torch.no_grad():
    #
    #     # indices = torch.from_numpy(np.load(in_path)).to(device)
    #     # print(indices)
    #
    #     mel = torch.FloatTensor(np.load(in_path)).unsqueeze(0).to(device)
    #     print(mel.size())
    #
    #     # mel = mel[:, :, :-4]
    #
    #     _, mel_hat, indices, _ = model(mel)
    #     # indices = indices.squeeze()
    #     # print(indices)
    #     # print(indices.size())
    #
    #     # print('------')
    #     # _, _, indices1, _ = model(mel[:, :, :-4])
    #     # indices1 = indices1.squeeze()
    #     # print(indices1)
    #     # print(indices1.size())
    #     print('------')
    #     # print(indices)
    #     output, z_q0 = model.generate(indices)
    #     # print(output.shape)
    #
    #     recon_loss = torch.mean((mel_hat - output)**2)
    #     print(recon_loss)
    # #
    # #     # test_len = 30000
    # #     # pos = random.randint(1, indices.size(0) - test_len)
    # #     # indices = indices[pos:pos+test_len]
    # #
    #     print('------')
    #     indices = indices[:-20]
    #     # indices = torch.narrow(indices, 0, 0, indices.size(0) - 20)
    #     # indices[::100] = 0
    #     # indices[-20:] = 0
    #     # print(indices)
    #     # print(indices.size())
    #
    #     output2, z_q2 = model.generate(indices)
    #     # print(output.shape)
    #
    #     recon_loss = torch.mean((output[:, :, :output2.size(2)] - output2)**2)
    #     print(recon_loss)
    #
    #     # recon_loss = torch.mean((z_q0[:, :, :, :z_q2.size(3)] - z_q2)**2)
    #     # print(recon_loss)
    #
    # #     # print(mel_hat[:, 0, :5])
    # #     # print(output[:, 0, :5])
    # #     # print(output2[:, 0, :5])
    # #
    # #     # print(mel_hat[:, 0, :5])
    # #     print(z_q0[0, 0, 2, :5])
    # #     print(z_q2[0, 0, 2, :5])
    # #
    # #     # output = output2
    # #
    # #     # out_path1 = Path(utils.to_absolute_path(cfg.out_path + "_recon0"))
    # #     # np.save(out_path1.with_suffix(".mel.npy"), output1.squeeze().cpu().numpy())
    # #     # out_path2 = Path(utils.to_absolute_path(cfg.out_path + "_recon-20"))
    # #     # np.save(out_path2.with_suffix(".mel.npy"), output2.squeeze().cpu().numpy())
    # #
    # #     # output = output[:, :, :500]
    # #     # audio = mel_to_audio(output.squeeze().cpu().numpy(),
    # #     #                      sr=cfg.preprocessing.sr,
    # #     #                      n_fft=cfg.preprocessing.n_fft,
    # #     #                      hop_length=cfg.preprocessing.hop_length,
    # #     #                      win_length=cfg.preprocessing.win_length,
    # #     #                      top_db=cfg.preprocessing.top_db,
    # #     #                      preemph=cfg.preprocessing.preemph)
    # #     # librosa.output.write_wav(out_path, audio.astype(np.float32), sr=cfg.preprocessing.sr)


@hydra.main(config_path="config/test.yaml")
def main5(cfg):
    in_path = Path(utils.to_absolute_path(cfg.in_path_mag))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    # decoder = Decoder(**cfg.model.decoder)
    decoder = Decoder2(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    print("Load checkpoint from: {}:".format(cfg.vqvae_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]))
    decoder.load_state_dict(clean_state_dict(checkpoint["decoder"]))

    encoder.eval()
    decoder.eval()

    mag = np.load(in_path)
    mag = mag[1:, :]  # throw away DC

    # start = int(mag.shape[1]/2 - 10*cfg.preprocessing.sr/cfg.preprocessing.hop_length)
    # end = int(mag.shape[1]/2 + 10*cfg.preprocessing.sr/cfg.preprocessing.hop_length)
    # mag = mag[:, start:end]

    # audio = mag_to_audio(mag,
    #                      hop_length=cfg.preprocessing.hop_length,
    #                      win_length=cfg.preprocessing.win_length,
    #                      top_db=cfg.preprocessing.top_db,
    #                      preemph=cfg.preprocessing.preemph)
    # out_path = Path(utils.to_absolute_path(cfg.out_path + '_orig')).with_suffix('.wav')
    # librosa.output.write_wav(out_path, audio.astype(np.float32), sr=cfg.preprocessing.sr)

    mag = torch.FloatTensor(mag).unsqueeze(0).to(device)
    print(mag.size())

    with torch.no_grad():

        z, _, _, _ = encoder(mag)
        print(z.size())
        output = decoder(z)
        print(output.shape)

        out_path = Path(utils.to_absolute_path(cfg.out_path + '_recon')).with_suffix('.mag.npy')
        np.save(out_path, output.squeeze().cpu().numpy())

        # audio = mag_to_audio(output.squeeze().cpu().numpy(),
        #                      hop_length=cfg.preprocessing.hop_length,
        #                      win_length=cfg.preprocessing.win_length,
        #                      top_db=cfg.preprocessing.top_db,
        #                      preemph=cfg.preprocessing.preemph)
        # out_path = Path(utils.to_absolute_path(cfg.out_path + '_recon')).with_suffix('.wav')
        # librosa.output.write_wav(out_path, audio.astype(np.float32), sr=cfg.preprocessing.sr)


        # for i in range(1):
        #
        #     # z_jit = z + torch.rand(z.size()).to(device)*0.001
        #
        #     output = decoder.generate(z_jit)
        #     print(output.shape)
        #
        #     audio = mel_to_audio(output.squeeze().cpu().numpy(),
        #                          sr=cfg.preprocessing.sr,
        #                          n_fft=cfg.preprocessing.n_fft,
        #                          hop_length=cfg.preprocessing.hop_length,
        #                          win_length=cfg.preprocessing.win_length,
        #                          top_db=cfg.preprocessing.top_db,
        #                          preemph=cfg.preprocessing.preemph)
        #     out_path = Path(utils.to_absolute_path(cfg.out_path + str(i)))
        #     librosa.output.write_wav(out_path, audio.astype(np.float32), sr=cfg.preprocessing.sr)


@hydra.main(config_path="config/test.yaml")
def main6(cfg):
    in_path = Path(utils.to_absolute_path(cfg.in_path_wav))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    # decoder = Decoder(**cfg.model.decoder)
    decoder = Decoder2(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    print("Load checkpoint from: {}:".format(cfg.vqvae_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]))
    decoder.load_state_dict(clean_state_dict(checkpoint["decoder"]))

    encoder.eval()
    decoder.eval()

    wav = np.load(in_path)

    # start = int(wav.shape[-1]/2 - 10*cfg.preprocessing.sr)
    # end = int(wav.shape[-1]/2 + 10*cfg.preprocessing.sr)

    start = int(wav.shape[-1]/2 - 512)
    end = int(wav.shape[-1]/2 + 512)

    audio = wav[start:end]
    print(audio)

    out_path = Path(utils.to_absolute_path(cfg.out_path + '_orig')).with_suffix('.wav')
    librosa.output.write_wav(out_path, audio.astype(np.float32), sr=cfg.preprocessing.sr)

    audio_out = np.empty(audio.shape, dtype=np.float32)

    audio_in = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
    print(audio_in.size())

    # block_size = 16000
    block_size = audio_in.size(-1)
    # block_size = 8192

    with torch.no_grad():

        hist = {}
        for i in range(0, audio_in.size(-1), block_size):
            print('-------------')
            print(audio_in[:, :, i:i+block_size].size())
            z, indices = encoder.encode(audio_in[:, :, i:i+block_size])
            print(indices.size())
            print(z.size())

            indices = indices.squeeze().cpu().numpy()
            # print(indices)

            output = decoder(z)
            print(output.size())
            audio_out[i:i+block_size] = output.squeeze().cpu().numpy()

            for j in range(len(indices)):
                if indices[j] in hist:
                    hist[indices[j]] = hist[indices[j]] + 1
                else:
                    hist[indices[j]] = 1
        print(audio_out)
        print(hist)
        out_path = Path(utils.to_absolute_path(cfg.out_path + '_recon')).with_suffix('.wav')
        librosa.output.write_wav(out_path, audio_out, sr=cfg.preprocessing.sr)


        # for i in range(1):
        #
        #     # z_jit = z + torch.rand(z.size()).to(device)*0.001
        #
        #     output = decoder.generate(z_jit)
        #     print(output.shape)
        #
        #     audio = mel_to_audio(output.squeeze().cpu().numpy(),
        #                          sr=cfg.preprocessing.sr,
        #                          n_fft=cfg.preprocessing.n_fft,
        #                          hop_length=cfg.preprocessing.hop_length,
        #                          win_length=cfg.preprocessing.win_length,
        #                          top_db=cfg.preprocessing.top_db,
        #                          preemph=cfg.preprocessing.preemph)
        #     out_path = Path(utils.to_absolute_path(cfg.out_path + str(i)))
        #     librosa.output.write_wav(


def main7():

    in_channels = 1
    channels = 512
    embedding_dim = 64
    encoder = nn.Sequential(
        nn.Conv1d(in_channels, channels, 3, 1, 0, bias=False),
        nn.BatchNorm1d(channels),
        nn.ReLU(True),
        # nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        # nn.BatchNorm1d(channels),
        # nn.ReLU(True),
        nn.Conv1d(channels, channels, 4, 2, 1, bias=False),
        nn.BatchNorm1d(channels),
        nn.ReLU(True),
        # nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        # nn.BatchNorm1d(channels),
        # nn.ReLU(True),
        # nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        # nn.BatchNorm1d(channels),
        # nn.ReLU(True),
        nn.Conv1d(channels, embedding_dim, 1)
    )
    len = 1024
    x = torch.rand(1, in_channels, len)
    print(x.size())
    y = encoder(x)
    print(y.size())


if __name__ == "__main__":
    # main()
    # main2()
    # main3()
    # main4()
    # test2()
    # test3()
    main5()
    # main6()
    # main7()