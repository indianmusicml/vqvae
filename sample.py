import random
import gc

import hydra
import hydra.utils as utils

from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from model import Encoder, Decoder, Decoder2
from model2.vqvae import VQVAE

from train import mel_to_audio, mag_to_audio
from prior import TransformerModel, generate_square_subsequent_mask
from transformers import OpenAIGPTConfig, OpenAIGPTLMHeadModel
from melgan.generator import Generator
from melgan.multiscale import MultiScaleDiscriminator


def clean_state_dict(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict_new[name] = v
    return state_dict_new


def generate_trajectory(n_iter, dim, _z0=None, mov_last=None, jump=0.3, smooth=0.3, include_z0=True):
    _z = np.empty((n_iter + int(not include_z0), dim), dtype=np.float32)
    _z[0] = _z0 if _z0 is not None else np.random.random(dim) * 2 - 1
    mov = mov_last if mov_last is not None else (np.random.random(dim) * 2 - 1) * jump
    for i in range(1, len(_z)):
        mov = mov * smooth + (np.random.random(dim) * 2 - 1) * jump * (1 - smooth)
        mov -= (np.abs(_z[i-1] + mov) > 1) * 2 * mov
        _z[i] = _z[i-1] + mov
    return _z[-n_iter:], mov


@hydra.main(config_path="config/sample.yaml")
def sample(cfg):
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]))
    decoder.load_state_dict(clean_state_dict(checkpoint["decoder"]))

    encoder.eval()
    decoder.eval()

    ###
    # Move along local random trajectory in the latent space
    # (endless generative "improvisation" with smoothly changing breaks)
    n_generate = 2048

    jump = 0.4  # factor of distance between adjacent trajectory points (speed of changing)
    smooth = 0.25  # smoothing the trajectory turns, [0, 1]

    # seed1 = 562  # change this to change starting point
    # np.random.seed(seed1)
    z0 = np.random.random(cfg.model.encoder.embedding_dim) * 2 - 1

    # seed2 = 377  # change this to change trajectory
    # np.random.seed(seed2)
    z, _ = generate_trajectory(n_generate, dim=cfg.model.encoder.embedding_dim, _z0=z0, include_z0=True,
                               jump=jump, smooth=smooth)
    z = torch.from_numpy(z).unsqueeze(0).cuda()
    print(z.size())
    print(z)
    # z.to(device)

    zq, indices = encoder.codebook.encode(z)

    print(zq.size())
    print(zq)

    print(indices)

    e_latent_loss = F.mse_loss(z, zq)
    print(e_latent_loss)

    output = decoder.generate(zq)

    path = out_dir / 'out'
    librosa.output.write_wav(path.with_suffix(".wav"), output.astype(np.float32), sr=cfg.preprocessing.sr)


@hydra.main(config_path="config/sample.yaml")
def sample2(cfg):
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VQVAE(cfg.model.encoder.channels, 32, 2, cfg.model.encoder.n_embeddings, cfg.model.encoder.embedding_dim, 0.25).to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(clean_state_dict(checkpoint["model"]))

    model.eval()

    # ###
    # # Move along local random trajectory in the latent space
    # # (endless generative "improvisation" with smoothly changing breaks)
    # n_generate = 512
    #
    # jump = 0.4  # factor of distance between adjacent trajectory points (speed of changing)
    # smooth = 0.25  # smoothing the trajectory turns, [0, 1]
    #
    # # seed1 = 562  # change this to change starting point
    # # np.random.seed(seed1)
    # # z0 = np.random.random(cfg.model.encoder.embedding_dim) * 2 - 1
    # z0 = np.random.random(cfg.model.encoder.embedding_dim) * 2 - 1
    #
    # # seed2 = 377  # change this to change trajectory
    # # np.random.seed(seed2)
    # z, _ = generate_trajectory(n_generate, dim=cfg.model.encoder.embedding_dim, _z0=z0, include_z0=True,
    #                            jump=jump, smooth=smooth)

    # z = np.empty((64, 20, 50), dtype=np.float32)
    # z[...] = np.random.random_sample(z.shape) * 2. - 1.
    # z = torch.from_numpy(z).unsqueeze(0).cuda()
    # print(z.size())
    # print(z)
    # z.to(device)

    with torch.no_grad():
        # for i in range(10):
        #     indices = np.random.randint(0, 512, (10000, 1))
        #     indices = torch.from_numpy(indices).cuda()
        #     print(indices.size())
        #     mels = model.generate(indices)
        #     print(mels.size())
        #
        #     path = out_dir / ('out%d' % i)
        #     audio = mel_to_audio(mels.squeeze().cpu().numpy(),
        #                          sr=cfg.preprocessing.sr,
        #                          n_fft=cfg.preprocessing.n_fft,
        #                          hop_length=cfg.preprocessing.hop_length,
        #                          win_length=cfg.preprocessing.win_length,
        #                          top_db=cfg.preprocessing.top_db,
        #                          preemph=cfg.preprocessing.preemph)
        #     print(audio.shape)
        #     librosa.output.write_wav(path.with_suffix(".wav"), audio.astype(np.float32), sr=cfg.preprocessing.sr)

        for i in range(512):
            indices = np.empty((20, 1), dtype=np.int)
            indices[...] = i
            print(indices)
            indices = torch.from_numpy(indices).cuda()
            print(indices.size())
            mels = model.generate(indices)
            print(mels.size())

            path = out_dir / ('codebook%d' % i)
            audio = mel_to_audio(mels.squeeze().cpu().numpy(),
                                 sr=cfg.preprocessing.sr,
                                 n_fft=cfg.preprocessing.n_fft,
                                 hop_length=cfg.preprocessing.hop_length,
                                 win_length=cfg.preprocessing.win_length,
                                 top_db=cfg.preprocessing.top_db,
                                 preemph=cfg.preprocessing.preemph)
            print(audio.shape)
            librosa.output.write_wav(path.with_suffix(".wav"), audio.astype(np.float32), sr=cfg.preprocessing.sr)


@hydra.main(config_path="config/sample.yaml")
def sample3(cfg):
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae = VQVAE(cfg.model.encoder.channels, 32, 2, cfg.model.encoder.n_embeddings, cfg.model.encoder.embedding_dim, 0.25).to(device)
    print("Load vqvae checkpoint from: {}:".format(cfg.vqvae_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    vqvae.load_state_dict(clean_state_dict(checkpoint["model"]))

    ntokens = cfg.model.encoder.n_embeddings  # size of vocabulary
    emsize = 200   # embedding dimension
    d_hid = 200    # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2    # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2      # number of heads in nn.MultiheadAttention
    prior = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers).to(device)

    print("Load prior checkpoint from: {}:".format(cfg.prior_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.prior_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    prior.load_state_dict(checkpoint["model"])

    vqvae.eval()
    prior.eval()

    seq_len = 1200

    random.seed(1234)

    with torch.no_grad():
        for i in range(512):
            # decoded_tokens = torch.tensor(random.randint(1, cfg.model.encoder.n_embeddings)).unsqueeze(-1).to(device)
            decoded_tokens = torch.tensor(i).unsqueeze(-1).to(device)
            for j in tqdm(range(seq_len-1)):
                src_mask = generate_square_subsequent_mask(j+1).to(device)
                output = prior(decoded_tokens.unsqueeze(-1), src_mask)

                top_indices = torch.argmax(output, dim=-1)
                # we only care about the last token that was decoded
                top_indices_last_token = top_indices[-1:][0]
                # add most likely token to the already decoded tokens
                decoded_tokens = torch.cat([decoded_tokens, top_indices_last_token])
            print(decoded_tokens)
            decoded_tokens = decoded_tokens.unsqueeze(-1)
            print(decoded_tokens.size())

            mels = vqvae.generate(decoded_tokens)
            print(mels.size())

            path = out_dir / ('noprime%d' % i)
            audio = mel_to_audio(mels.squeeze().cpu().numpy(),
                                 sr=cfg.preprocessing.sr,
                                 n_fft=cfg.preprocessing.n_fft,
                                 hop_length=cfg.preprocessing.hop_length,
                                 win_length=cfg.preprocessing.win_length,
                                 top_db=cfg.preprocessing.top_db,
                                 preemph=cfg.preprocessing.preemph)
            print(audio.shape)
            librosa.output.write_wav(path.with_suffix(".wav"), audio.astype(np.float32), sr=cfg.preprocessing.sr)


@hydra.main(config_path="config/sample.yaml")
def sample4(cfg):
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    in_path = Path(utils.to_absolute_path(cfg.in_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae = VQVAE(cfg.model.encoder.channels, 32, 2, cfg.model.encoder.n_embeddings, cfg.model.encoder.embedding_dim, 0.25).to(device)
    print("Load vqvae checkpoint from: {}:".format(cfg.vqvae_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    vqvae.load_state_dict(clean_state_dict(checkpoint["model"]))

    ntokens = cfg.model.encoder.n_embeddings  # size of vocabulary
    emsize = 200   # embedding dimension
    d_hid = 200    # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2    # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2      # number of heads in nn.MultiheadAttention
    prior = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers).to(device)

    print("Load prior checkpoint from: {}:".format(cfg.prior_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.prior_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    prior.load_state_dict(checkpoint["model"])

    vqvae.eval()
    prior.eval()

    indices = torch.from_numpy(np.load(in_path)).to(device)
    print(indices.size())

    # mel = np.load(in_path)
    # mel = torch.FloatTensor(mel).unsqueeze(0).to(device)
    # print(mel.size())

    test_len = 10000

    # with torch.no_grad():
    #
    #     _, _, _, indices, _ = vqvae(mel)
    #     print(indices.size())
    #
    #     pos = random.randint(1, indices.size(0) - test_len - 1)
    #     # indices = indices[pos:pos+test_len]
    #     mels, _ = vqvae.generate(indices)
    #     print(mels.size())
    #
    #     path = out_dir / 'orig'
    #     audio = mel_to_audio(mels.squeeze().cpu().numpy(),
    #                          sr=cfg.preprocessing.sr,
    #                          n_fft=cfg.preprocessing.n_fft,
    #                          hop_length=cfg.preprocessing.hop_length,
    #                          win_length=cfg.preprocessing.win_length,
    #                          top_db=cfg.preprocessing.top_db,
    #                          preemph=cfg.preprocessing.preemph)
    #     print(audio.shape)
    #     librosa.output.write_wav(path.with_suffix(".wav"), audio.astype(np.float32), sr=cfg.preprocessing.sr)


    prime_len = 200
    seq_len = 5000

    random.seed(4567)

    with torch.no_grad():
        for i in range(10):

            gc.collect()
            torch.cuda.empty_cache()

            # decoded_tokens = torch.tensor(random.randint(1, cfg.model.encoder.n_embeddings)).unsqueeze(-1).to(device)
            # decoded_tokens = torch.tensor(i).unsqueeze(-1).to(device)

            # decoded_tokens = indices[:prime_len].to(device)

            pos = random.randint(1, int((indices.size(0) - prime_len)/20)-1)
            decoded_tokens = indices.view(20, -1).to(device)
            decoded_tokens = decoded_tokens[:, pos:pos+int(prime_len/20)]
            # decoded_tokens = decoded_tokens.contiguous().view(-1)
            decoded_tokens = decoded_tokens.t().contiguous().view(-1)

            print(decoded_tokens.squeeze())
            print(decoded_tokens.size())
            for j in tqdm(range(seq_len - prime_len)):
                # src_mask = generate_square_subsequent_mask(decoded_tokens.size(0)).to(device)
                # output = prior(decoded_tokens.unsqueeze(-1), src_mask)

                src_mask = generate_square_subsequent_mask(prime_len).to(device)
                output = prior(decoded_tokens[-prime_len:].unsqueeze(-1), src_mask)

                # print(output.size())

                output = output[-1, :, :]  # only care about last frame
                output = output / 0.99
                top_indices_last_token = torch.distributions.Categorical(logits=output).sample()

                # top_indices = torch.argmax(output, dim=-1)
                # # we only care about the last token that was decoded
                # top_indices_last_token = top_indices[-1:]
                # # add most likely token to the already decoded tokens

                # print(top_indices_last_token)
                # print(top_indices_last_token.size())
                # print(decoded_tokens.size())

                # print(decoded_tokens.squeeze())
                decoded_tokens = torch.cat([decoded_tokens.squeeze(), top_indices_last_token])
                # print(decoded_tokens)
                # print(decoded_tokens.size())
            print(decoded_tokens)
            decoded_tokens = decoded_tokens.unsqueeze(-1)

            decoded_tokens = decoded_tokens.view(-1, 20).t().contiguous().view(-1)

            print(decoded_tokens.size())

            mels, _ = vqvae.generate(decoded_tokens)
            print(mels.size())

            path = out_dir / ('%02d' % i)
            audio = mel_to_audio(mels.squeeze().cpu().numpy(),
                                 sr=cfg.preprocessing.sr,
                                 n_fft=cfg.preprocessing.n_fft,
                                 hop_length=cfg.preprocessing.hop_length,
                                 win_length=cfg.preprocessing.win_length,
                                 top_db=cfg.preprocessing.top_db,
                                 preemph=cfg.preprocessing.preemph)
            print(audio.shape)
            librosa.output.write_wav(path.with_suffix(".wav"), audio.astype(np.float32), sr=cfg.preprocessing.sr)


@hydra.main(config_path="config/sample.yaml")
def sample5(cfg):
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    in_path = Path(utils.to_absolute_path(cfg.in_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder2(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    print("Load vqvae checkpoint from: {}:".format(cfg.vqvae_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.vqvae_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]))
    decoder.load_state_dict(clean_state_dict(checkpoint["decoder"]))

    # trying to match jukebox "tiny_prior"
    bptt = 200     # should be ~2 sec for hopsize=160 at sr=16k
    ntokens = cfg.model.encoder.n_embeddings  # size of vocabulary
    emsize = 512
    nlayers = 12
    nhead = 1
    config = OpenAIGPTConfig(vocab_size=ntokens,
                             n_positions=bptt,
                             n_embd=emsize,
                             n_layer=nlayers,
                             n_head=nhead)
    prior = OpenAIGPTLMHeadModel(config).to(device)

    print("Load prior checkpoint from: {}:".format(cfg.prior_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.prior_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    prior.load_state_dict(clean_state_dict(checkpoint["model"]))
    # prior = OpenAIGPTLMHeadModel.from_pretrained(checkpoint_path)

    vocoder = Generator(cfg.model.encoder.in_channels).to(device)

    print("Load vocoder checkpoint from: {}:".format(cfg.vocoder_checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.vocoder_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    vocoder.load_state_dict(clean_state_dict(checkpoint["model_g"]))

    encoder.eval()
    decoder.eval()
    prior.eval()
    vocoder.eval()

    indices = torch.from_numpy(np.load(in_path)).to(device)
    print(indices.size())

    prime_len = 200
    seq_len = 800

    random.seed(4567)

    with torch.no_grad():
        for i in range(10):

            gc.collect()
            torch.cuda.empty_cache()

            # decoded_tokens = indices[:prime_len].to(device)

            pos = random.randint(1, indices.size(0) - prime_len - 1)
            decoded_tokens = indices.to(device)
            decoded_tokens = decoded_tokens[pos:pos+prime_len]

            print(decoded_tokens.squeeze())
            print(decoded_tokens.size())
            for j in tqdm(range(seq_len - prime_len)):
                output = prior.forward(decoded_tokens[-prime_len:].unsqueeze(-1))
                output = output.logits

                # print(output.size())

                output = output[-1, :, :]  # only care about last frame
                output = output / 0.99
                top_indices_last_token = torch.distributions.Categorical(logits=output).sample()

                # top_indices = torch.argmax(output, dim=-1)
                # # we only care about the last token that was decoded
                # top_indices_last_token = top_indices[-1:]
                # # add most likely token to the already decoded tokens

                # print(top_indices_last_token)
                # print(top_indices_last_token.size())
                # print(decoded_tokens.size())

                # print(decoded_tokens.squeeze())
                decoded_tokens = torch.cat([decoded_tokens.squeeze(), top_indices_last_token])
                # print(decoded_tokens)
                # print(decoded_tokens.size())
            print(decoded_tokens)
            decoded_tokens = decoded_tokens.unsqueeze(-1)

            # decoded_tokens = decoded_tokens.view(-1, 20).t().contiguous().view(-1)

            print(decoded_tokens.size())

            z = encoder.quantize(decoded_tokens)
            mags = decoder(z)
            print(mags.size())

            # audio = mag_to_audio(mags.squeeze().cpu().numpy(),
            #                      hop_length=cfg.preprocessing.hop_length,
            #                      win_length=cfg.preprocessing.win_length,
            #                      top_db=cfg.preprocessing.top_db,
            #                      preemph=cfg.preprocessing.preemph)

            audio = vocoder(mags)
            audio = audio.squeeze().cpu().numpy()
            print(audio.shape)

            path = out_dir / ('%02d' % i)
            sf.write(path.with_suffix(".wav"), audio.astype(np.float32), cfg.preprocessing.sr, 'PCM_24')
            # librosa.output.write_wav(path.with_suffix(".wav"), audio.astype(np.float32), sr=cfg.preprocessing.sr)

if __name__ == "__main__":
    # sample()
    # sample2()
    # sample3()
    # sample4()
    sample5()
