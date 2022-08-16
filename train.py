import gc
from collections import OrderedDict
from pathlib import Path
from itertools import chain

import hydra
from hydra import utils

from tqdm import tqdm

import numpy as np
import scipy

# import apex.amp as amp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from dataset import SpeechDataset
from dataset import AudioFileDataset

# from model import Encoder, Decoder
# from model import RosinalityEncoder, RosinalityDecoder
# from model import Encoder, Decoder2
# from model2.vqvae import VQVAE
from jukebox_vqvae.vqvae import VQVAE, calculate_strides

from audio_utils import calculate_bandwidth

import librosa
import soundfile as sf


def clean_state_dict(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict_new[name] = v
    return state_dict_new

def mel_to_audio(mel, sr, n_fft, hop_length, win_length, top_db, preemph):
    mel = (mel - 1) * top_db
    mel = librosa.db_to_amplitude(mel, top_db)
    mel *= 0.01  # hmmmm
    wav = librosa.feature.inverse.mel_to_audio(mel,
                                               sr=sr,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               win_length = win_length,
                                               power=1)
    wav[0] -= ((2. - preemph) * wav[0] - wav[1]) / (3. - preemph)
    wav = scipy.signal.lfilter([1.], [1., -preemph], wav)
    return wav


def mag_to_audio(mag, hop_length, win_length, top_db, preemph):
    mag = (mag - 1) * top_db
    mag = librosa.db_to_amplitude(mag, top_db)
    mag *= 0.01  # hmmmm

    # add back DC
    n_fft2 = mag.shape[0]  # should be nfft/2 = 1024
    mag_pad = np.zeros((n_fft2 + 1, mag.shape[1]))
    mag_pad[1:, :] = mag
    mag = mag_pad

    wav = librosa.griffinlim(mag,
                             hop_length=hop_length,
                             win_length=win_length)
    wav[0] -= ((2. - preemph) * wav[0] - wav[1]) / (3. - preemph)
    wav = scipy.signal.lfilter([1.], [1., -preemph], wav)
    return wav


def _loss_fn(loss_fn, x_target, x_pred):
    if loss_fn == 'l1':
        return torch.mean(torch.abs(x_pred - x_target)) #/ hps.bandwidth['l1']
    elif loss_fn == 'l2':
        return torch.mean((x_pred - x_target) ** 2) #/ hps.bandwidth['l2']
    elif loss_fn == 'linf':
        residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
        values, _ = torch.topk(residual, 2048, dim=1)
        return torch.mean(values) #/ hps.bandwidth['l2']
    else:
        assert False, f"Unknown loss_fn {loss_fn}"


def generate_trajectory(n_iter, dim, _z0=None, mov_last=None, jump=0.3, smooth=0.3, include_z0=True):
    _z = np.empty((n_iter + int(not include_z0), dim), dtype=np.float32)
    _z[0] = _z0 if _z0 is not None else np.random.random(dim) * 2 - 1
    mov = mov_last if mov_last is not None else (np.random.random(dim) * 2 - 1) * jump
    for i in range(1, len(_z)):
        mov = mov * smooth + (np.random.random(dim) * 2 - 1) * jump * (1 - smooth)
        mov -= (np.abs(_z[i-1] + mov) > 1) * 2 * mov
        _z[i] = _z[i-1] + mov
    return _z[-n_iter:], mov


# def save_checkpoint(encoder, decoder, optimizer, scheduler, step, checkpoint_dir):
#     checkpoint_state = {
#         "encoder": encoder.state_dict(),
#         "decoder": decoder.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         # "amp": amp.state_dict(),
#         "scheduler": scheduler.state_dict(),
#         "step": step}
#     checkpoint_dir.mkdir(exist_ok=True, parents=True)
#     checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
#     torch.save(checkpoint_state, checkpoint_path)
#     print("Saved checkpoint: {}".format(checkpoint_path.stem))

def save_checkpoint2(model, optimizer, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # "amp": amp.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


# def eval(encoder, decoder, dataloader, device, writer, global_step, cfg):
#     encoder.eval()
#     decoder.eval()
#     average_recon_loss = average_multispec_loss = average_vq_loss = 0
#     average_loss = 0
#     # for i, (audio, mels, mags) in enumerate(tqdm(dataloader)):
#     # for i, (audio, mags) in enumerate(tqdm(dataloader)):
#     # for i, mags in enumerate(tqdm(dataloader)):
#     for i, audio in enumerate(tqdm(dataloader)):
#         # audio, mels = audio.to(device), mels.to(device)
#         # audio, mags = audio.to(device), mags.to(device)
#         # mags = mags.to(device)
#         # audio = audio.to(device)
#         audio = audio.unsqueeze(1).to(device)
#
#         # z, vq_loss, _ = encoder(mels)
#         # z, vq_loss, _, _ = encoder(mags)
#         z, vq_loss, _, _ = encoder(audio)
#         # output = decoder(audio[:, :-1], z)
#         output = decoder(z)
#
#         # recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
#
#         # d = mels - output
#         # d = mags - output
#         d = audio - output
#         recon_loss = (d.view(d.shape[0], -1) ** 2).sum(dim=-1).sqrt()
#
#         # recon_loss = _loss_fn('l1', audio[:, 1:], output.transpose(1, 2))
#         # spec_loss = spectral_loss(x_target, x_out, hps)
#         multispec_loss = multispectral_loss(audio, output)
#
#         average_recon_loss += (recon_loss.mean().item() - average_recon_loss) / (i+1)
#         average_multispec_loss += (multispec_loss.mean().item() - average_multispec_loss) / (i+1)
#         average_vq_loss += (vq_loss.mean().item() - average_vq_loss) / (i+1)
#
#         if i == 0:
#             print('running eval generator')
#             with torch.no_grad():
#                 # z, _ = encoder0.encode(mels)
#                 # print(z.size())
#                 for j in range(len(z)):
#                     output = decoder(z[j].unsqueeze(0))
#                     # print(output.size())
#
#                     # audio = mel_to_audio(output.squeeze().cpu().numpy(),
#                     #                      sr=cfg.preprocessing.sr,
#                     #                      n_fft=cfg.preprocessing.n_fft,
#                     #                      hop_length=cfg.preprocessing.hop_length,
#                     #                      win_length=cfg.preprocessing.win_length,
#                     #                      top_db=cfg.preprocessing.top_db,
#                     #                      preemph=cfg.preprocessing.preemph)
#
#                     # audio = mag_to_audio(output.squeeze().cpu().numpy(),
#                     #                      hop_length=cfg.preprocessing.hop_length,
#                     #                      win_length=cfg.preprocessing.win_length,
#                     #                      top_db=cfg.preprocessing.top_db,
#                     #                      preemph=cfg.preprocessing.preemph)
#
#                     audio = output.squeeze().cpu().numpy()
#
#                     writer.add_audio("recon_{}".format(j),
#                                      audio,
#                                      global_step=global_step,
#                                      sample_rate=cfg.preprocessing.sr)
#
#     writer.add_scalar("recon_loss/test", average_recon_loss, global_step)
#     writer.add_scalar("multispec_loss/test", average_multispec_loss, global_step)
#     writer.add_scalar("vq_loss/test", average_vq_loss, global_step)
#
#     print("eval:{}, recon loss:{:.2E}, multispec loss:{:.2E}, vq loss:{:.2E}".format(global_step,
#                                                                                      average_recon_loss,
#                                                                                      average_multispec_loss,
#                                                                                      average_vq_loss))
#
#     # print("eval:{}, recon loss:{:.2E}, vq loss:{:.2E}".format(global_step, average_recon_loss, average_vq_loss))


# @hydra.main(config_path="config/train.yaml")
# def train_model(cfg):
#     tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
#     checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
#     writer = SummaryWriter(tensorboard_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     encoder = Encoder(**cfg.model.encoder)
#     # decoder = Decoder(**cfg.model.decoder)
#     decoder = Decoder2(**cfg.model.decoder)
#
#     # encoder = RosinalityEncoder()
#     # decoder = RosinalityDecoder()
#
#     encoder.to(device)
#     decoder.to(device)
#
#     optimizer = optim.Adam(
#         chain(encoder.parameters(), decoder.parameters()),
#         lr=cfg.training.optimizer.lr)
#     # [encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level="O1")
#     scheduler = optim.lr_scheduler.MultiStepLR(
#         optimizer, milestones=cfg.training.scheduler.milestones,
#         gamma=cfg.training.scheduler.gamma)
#
#     encoder = torch.nn.DataParallel(encoder)
#     decoder = torch.nn.DataParallel(decoder)
#     encoder.to(device)
#     decoder.to(device)
#
#     if cfg.resume:
#         print("Resume checkpoint from: {}:".format(cfg.resume))
#         resume_path = utils.to_absolute_path(cfg.resume)
#         checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
#         encoder.load_state_dict(checkpoint["encoder"])
#         decoder.load_state_dict(checkpoint["decoder"])
#         optimizer.load_state_dict(checkpoint["optimizer"])
#         # amp.load_state_dict(checkpoint["amp"])
#         scheduler.load_state_dict(checkpoint["scheduler"])
#         global_step = checkpoint["step"]
#     else:
#         global_step = 0
#
#     train_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "train"
#     train_dataset = SpeechDataset(
#         root=train_root_path,
#         hop_length=cfg.preprocessing.hop_length,
#         sample_frames=cfg.training.sample_frames)
#     train_dataloader = DataLoader(train_dataset,
#                                   batch_size=cfg.training.batch_size,
#                                   shuffle=True,
#                                   num_workers=cfg.training.n_workers,
#                                   pin_memory=True,
#                                   drop_last=True)
#
#     n_epochs = cfg.training.n_steps // len(train_dataloader) + 1
#     start_epoch = global_step // len(train_dataloader) + 1
#
#     test_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "test"
#     test_dataset = SpeechDataset(
#         root=test_root_path,
#         hop_length=cfg.preprocessing.hop_length,
#         sample_frames=cfg.training.sample_frames)
#     test_dataloader = DataLoader(test_dataset,
#                               batch_size=len(test_dataset),
#                               shuffle=False,
#                               num_workers=cfg.training.n_workers,
#                               pin_memory=True,
#                               drop_last=True)
#
#     num_paths = 5
#     # n_generate = 128
#     n_generate = 16000
#     jump = 0.4  # factor of distance between adjacent trajectory points (speed of changing)
#     smooth = 0.25  # smoothing the trajectory turns, [0, 1]
#     seed1 = 562  # change this to change starting point
#     np.random.seed(seed1)
#     z0 = np.random.random(cfg.model.encoder.embedding_dim) * 2 - 1
#     seed2 = 377  # change this to change trajectory
#     np.random.seed(seed2)
#     z_paths = []
#     for i in range(num_paths):
#         z, _ = generate_trajectory(n_generate, dim=cfg.model.encoder.embedding_dim, _z0=z0, include_z0=True,
#                                     jump=jump, smooth=smooth)
#         z_paths.append(z)
#     z_paths = torch.stack([torch.from_numpy(x).float() for x in z_paths]).cuda()
#
#     for epoch in range(start_epoch, n_epochs + 1):
#         average_recon_loss = average_multispec_loss = average_vq_loss = average_perplexity = average_usage = 0
#         # for i, (audio, mags) in enumerate(tqdm(train_dataloader)):
#         # for i, (audio, mels, mags) in enumerate(train_dataloader):
#         for i, audio in enumerate(tqdm(train_dataloader)):
#             # audio, mags = audio.to(device), mags.to(device)
#
#             # mags = mags.to(device)
#             audio = audio.unsqueeze(1).to(device)
#
#             # print(audio.size())
#
#             encoder.train()
#             decoder.train()
#             optimizer.zero_grad()
#
#             # z, vq_loss, perplexity = encoder(mels)
#             # z, vq_loss, perplexity, usage = encoder(mags)
#             z, vq_loss, perplexity, usage = encoder(audio)
#             output = decoder(z)
#             # output = decoder(audio[:, :-1], z)
#
#             # print(z.size())
#             # print(output.size())
#
#             # print('--------')
#             # print(mags.size())
#             # print(z.size())
#             # print('--------')
#
#             # print(audio.size())
#
#             # print(output.size())
#             # print('---------')
#
#             # recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
#             # loss = recon_loss + vq_loss
#
#             # print(audio[:, 1:].size())
#             # print(output.size())
#             # recon_loss = _loss_fn('l1', audio[:, 1:], output)
#
#             # print(mels.size())
#             # print(output.size())
#
#             # d = mels - output
#             # d = mags - output
#             d = audio - output
#             # print(d.size())
#             recon_loss = (d.view(d.shape[0], -1) ** 2).sum(dim=-1).sqrt()
#
#             # # spec_loss = spectral_loss(x_target, x_out, hps)
#             multispec_loss = multispectral_loss(audio, output)
#             # loss = recon_loss + multispec_loss + vq_loss
#
#             # print(recon_loss.size())
#             # print(vq_loss.size())
#
#             loss = recon_loss.mean() + vq_loss.mean() + multispec_loss.mean()
#             # loss = recon_loss.mean() + vq_loss.mean()
#
#             # loss = loss.mean()
#
#             # with amp.scale_loss(loss, optimizer) as scaled_loss:
#             #     scaled_loss.backward()
#
#             loss.backward()
#
#             # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
#             optimizer.step()
#             scheduler.step()
#
#             average_recon_loss += (recon_loss.mean().item() - average_recon_loss) / (i+1)
#             average_multispec_loss += (multispec_loss.mean().item() - average_multispec_loss) / (i+1)
#             average_vq_loss += (vq_loss.mean().item() - average_vq_loss) / (i+1)
#             average_perplexity += (perplexity.mean().item() - average_perplexity) / (i+1)
#             average_usage += (usage.mean().item() - average_usage) / (i+1)
#
#             if global_step > 0:
#                 if global_step % cfg.training.checkpoint_interval == 0:
#                     save_checkpoint(encoder, decoder, optimizer, scheduler, global_step, checkpoint_dir)
#
#                 if global_step % cfg.training.generate_sample_interval == 0:
#                     with torch.no_grad():
#                         zq, _ = encoder.module.codebook.encode(z_paths)
#                         for j in range(len(zq)):
#                             output = decoder(zq[j].unsqueeze(0))
#                             # print(output.size())
#
#                             # audio = mel_to_audio(output.squeeze().cpu().numpy(),
#                             #                      sr=cfg.preprocessing.sr,
#                             #                      n_fft=cfg.preprocessing.n_fft,
#                             #                      hop_length=cfg.preprocessing.hop_length,
#                             #                      win_length=cfg.preprocessing.win_length,
#                             #                      top_db=cfg.preprocessing.top_db,
#                             #                      preemph=cfg.preprocessing.preemph)
#
#                             # audio = mag_to_audio(output.squeeze().cpu().numpy(),
#                             #                      hop_length=cfg.preprocessing.hop_length,
#                             #                      win_length=cfg.preprocessing.win_length,
#                             #                      top_db=cfg.preprocessing.top_db,
#                             #                      preemph=cfg.preprocessing.preemph)
#
#                             audio = output.squeeze().cpu().numpy()
#
#                             writer.add_audio("rand_{}".format(j),
#                                                      audio,
#                                                      global_step=global_step,
#                                                      sample_rate=cfg.preprocessing.sr)
#
#                 if global_step % cfg.training.eval_interval == 0:
#                     eval(encoder, decoder, test_dataloader, device, writer, global_step, cfg)
#
#             global_step += 1
#
#         writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
#         writer.add_scalar("multispec_loss/train", average_multispec_loss, global_step)
#         writer.add_scalar("vq_loss/train", average_vq_loss, global_step)
#         writer.add_scalar("average_perplexity", average_perplexity, global_step)
#         writer.add_scalar("average_usage", average_usage, global_step)
#
#         if epoch % 10 == 0:
#             print("epoch:{}, recon loss:{:.2E}, multispec loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}, usage:{:.3f}"
#                   .format(epoch, average_recon_loss, average_multispec_loss, average_vq_loss, average_perplexity, average_usage))
#             # print("epoch:{}, recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
#             #       .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))
#
#
# def save_checkpoint2(model, optimizer, step, checkpoint_dir):
#     checkpoint_state = {
#         "model": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         "step": step}
#     checkpoint_dir.mkdir(exist_ok=True, parents=True)
#     checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
#     torch.save(checkpoint_state, checkpoint_path)
#     print("Saved checkpoint: {}".format(checkpoint_path.stem))


# @hydra.main(config_path="config/train.yaml")
# def train_model2(cfg):
#     tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
#     checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
#     writer = SummaryWriter(tensorboard_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     model = VQVAE(cfg.model.encoder.channels, 32, 2, cfg.model.encoder.n_embeddings, cfg.model.encoder.embedding_dim, 0.25).to(device)
#
#     optimizer = optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr, amsgrad=True)
#
#     model = torch.nn.DataParallel(model).to(device)
#
#     if cfg.resume:
#         print("Resume checkpoint from: {}:".format(cfg.resume))
#         resume_path = utils.to_absolute_path(cfg.resume)
#         checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
#         model.load_state_dict(checkpoint["model"])
#         optimizer.load_state_dict(checkpoint["optimizer"])
#         global_step = checkpoint["step"]
#     else:
#         global_step = 0
#
#     train_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "train"
#     train_dataset = SpeechDataset(
#         root=train_root_path,
#         hop_length=cfg.preprocessing.hop_length,
#         sample_frames=cfg.training.sample_frames)
#     train_dataloader = DataLoader(train_dataset,
#                                   batch_size=cfg.training.batch_size,
#                                   shuffle=True,
#                                   num_workers=cfg.training.n_workers,
#                                   pin_memory=True,
#                                   drop_last=True)
#
#     n_epochs = cfg.training.n_steps // len(train_dataloader) + 1
#     start_epoch = global_step // len(train_dataloader) + 1
#
#     test_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "test"
#     test_dataset = SpeechDataset(
#         root=test_root_path,
#         hop_length=cfg.preprocessing.hop_length,
#         sample_frames=cfg.training.sample_frames)
#     test_dataloader = DataLoader(test_dataset,
#                               batch_size=len(test_dataset),
#                               shuffle=False,
#                               num_workers=cfg.training.n_workers,
#                               pin_memory=True,
#                               drop_last=True)
#
#     num_paths = 5
#     n_generate = 128
#     jump = 0.4  # factor of distance between adjacent trajectory points (speed of changing)
#     smooth = 0.25  # smoothing the trajectory turns, [0, 1]
#     seed1 = 562  # change this to change starting point
#     np.random.seed(seed1)
#     z0 = np.random.random(cfg.model.encoder.embedding_dim) * 2 - 1
#     seed2 = 377  # change this to change trajectory
#     np.random.seed(seed2)
#     z_paths = []
#     for i in range(num_paths):
#         z, _ = generate_trajectory(n_generate, dim=cfg.model.encoder.embedding_dim, _z0=z0, include_z0=True,
#                                     jump=jump, smooth=smooth)
#         z_paths.append(z)
#     z_paths = torch.stack([torch.from_numpy(x).float() for x in z_paths]).cuda()
#
#     for epoch in range(start_epoch, n_epochs + 1):
#         average_recon_loss = average_multispec_loss = average_vq_loss = average_perplexity = 0
#         for i, (audio, mels) in enumerate(tqdm(train_dataloader)):
#
#             audio, mels = audio.to(device), mels.to(device)
#
#             model.train()
#             optimizer.zero_grad()
#
#             vq_loss, mels_hat, indices, perplexity = model(mels)
#
#             mels = mels[:, :, 1:-1]
#
#             recon_loss = torch.mean((mels_hat - mels)**2)
#             loss = recon_loss.mean() + vq_loss.mean()
#
#             loss.backward()
#             optimizer.step()
#
#             average_recon_loss += (recon_loss.mean().item() - average_recon_loss) / (i+1)
#             average_vq_loss += (vq_loss.mean().item() - average_vq_loss) / (i+1)
#             average_perplexity += (perplexity.mean().item() - average_perplexity) / (i+1)
#
#             if global_step > 0:
#                 if global_step % cfg.training.checkpoint_interval == 0:
#                     save_checkpoint2(model, optimizer, global_step, checkpoint_dir)
#
#                 # if global_step % cfg.training.generate_sample_interval == 0:
#                 #     with torch.no_grad():
#                 #         print('running rand generator')
#                 #         _, zq, _ = model.module.vector_quantization(z_paths)
#                 #         for j in range(len(zq)):
#                 #             output = model.module.decoder(zq[j].unsqueeze(0))
#                 #             # print(output.size())
#                 #             audio = mel_to_audio(output.squeeze().cpu().numpy(),
#                 #                                  sr=cfg.preprocessing.sr,
#                 #                                  n_fft=cfg.preprocessing.n_fft,
#                 #                                  hop_length=cfg.preprocessing.hop_length,
#                 #                                  win_length=cfg.preprocessing.win_length,
#                 #                                  top_db=cfg.preprocessing.top_db,
#                 #                                  preemph=cfg.preprocessing.preemph)
#                 #             writer.add_audio("rand_{}".format(j),
#                 #                              audio,
#                 #                              global_step=global_step,
#                 #                              sample_rate=cfg.preprocessing.sr)
#
#                 if global_step % cfg.training.eval_interval == 0:
#                     eval2(model, test_dataloader, device, writer, global_step, cfg)
#
#             global_step += 1
#
#         writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
#         writer.add_scalar("vq_loss/train", average_vq_loss, global_step)
#         writer.add_scalar("average_perplexity", average_perplexity, global_step)
#
#         print("epoch:{}, recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
#               .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))
#
# def eval2(model, dataloader, device, writer, global_step, cfg):
#     model.eval()
#     average_recon_loss = average_vq_loss = 0
#     for i, (audio, mels) in enumerate(tqdm(dataloader)):
#         audio, mels = audio.to(device), mels.to(device)
#
#         vq_loss, mels_hat, indices, perplexity, usage = model(mels)
#
#         mels = mels[:, :, 1:-1]
#
#         recon_loss = torch.mean((mels_hat - mels)**2)
#
#         average_recon_loss += (recon_loss.mean().item() - average_recon_loss) / (i+1)
#         average_vq_loss += (vq_loss.mean().item() - average_vq_loss) / (i+1)
#
#         if i == 0:
#             print('running eval generator')
#             with torch.no_grad():
#                 for j in range(len(mels_hat)):
#                     audio = mel_to_audio(mels_hat[j].squeeze().cpu().numpy(),
#                                          sr=cfg.preprocessing.sr,
#                                          n_fft=cfg.preprocessing.n_fft,
#                                          hop_length=cfg.preprocessing.hop_length,
#                                          win_length=cfg.preprocessing.win_length,
#                                          top_db=cfg.preprocessing.top_db,
#                                          preemph=cfg.preprocessing.preemph)
#                     writer.add_audio("recon_{}".format(j),
#                                      audio,
#                                      global_step=global_step,
#                                      sample_rate=cfg.preprocessing.sr)
#
#     writer.add_scalar("recon_loss/test", average_recon_loss, global_step)
#     # writer.add_scalar("multispec_loss/test", average_multispec_loss, global_step)
#     writer.add_scalar("vq_loss/test", average_vq_loss, global_step)
#
#     # print("eval:{}, recon loss:{:.2E}, multispec loss:{:.2E}, vq loss:{:.2E}".format(global_step,
#     #                                                                                  average_recon_loss,
#     #                                                                                  average_multispec_loss,
#     #                                                                                  average_vq_loss))
#
#     print("eval:{}, recon loss:{:.2E}, vq loss:{:.2E}".format(global_step, average_recon_loss, average_vq_loss))

# @hydra.main(config_path="config/train.yaml")
# def test(cfg):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     encoder = Encoder(**cfg.model.encoder)
#     # decoder = Decoder(**cfg.model.decoder)
#     decoder = Decoder2(**cfg.model.decoder)
#
#     encoder.to(device)
#     decoder.to(device)
#
#     encoder = torch.nn.DataParallel(encoder)
#     decoder = torch.nn.DataParallel(decoder)
#     encoder.to(device)
#     decoder.to(device)
#
#     audio = torch.rand(cfg.training.sample_frames)
#     audio = audio.unsqueeze(0).unsqueeze(0).to(device)
#
#     # train_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "train"
#     # train_dataset = SpeechDataset(
#     #     root=train_root_path,
#     #     hop_length=cfg.preprocessing.hop_length,
#     #     sample_frames=cfg.training.sample_frames)
#     # train_dataloader = DataLoader(train_dataset,
#     #                               batch_size=cfg.training.batch_size,
#     #                               shuffle=True,
#     #                               num_workers=cfg.training.n_workers,
#     #                               pin_memory=True,
#     #                               drop_last=True)
#     #
#     # for audio in train_dataloader:
#     #     audio = audio.unsqueeze(1).to(device)
#     #     print(audio.size())
#     #
#     #     encoder.train()
#     #     decoder.train()
#     #
#     #     # z, vq_loss, perplexity = encoder(mels)
#     #     # z, vq_loss, perplexity, usage = encoder(mags)
#     #     z, vq_loss, perplexity, usage, indices = encoder(audio)
#     #     output = decoder(z)
#     #     # output = decoder(audio[:, :-1], z)
#     #
#     #     print(z.size())
#     #     print(output.size())
#     #     print(indices.size()) ######## seems important
#     #
#     #     break
#
#     print(audio.size())
#
#     encoder.train()
#     decoder.train()
#
#     # z, vq_loss, perplexity = encoder(mels)
#     # z, vq_loss, perplexity, usage = encoder(mags)
#     z, vq_loss, perplexity, usage = encoder(audio)
#     output = decoder(z)
#     # output = decoder(audio[:, :-1], z)
#
#     print(z.size())
#     print(output.size())


class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


@hydra.main(config_path="config/train.yaml")
def train_model3(cfg):

    np.random.seed(1234)

    train_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "train"
    test_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "test"
    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    writer = SummaryWriter(tensorboard_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hps = Hyperparams(
        sample_length=cfg.training.sample_frames,
        sr=cfg.preprocessing.sr,
        levels = 1,
        downs_t = (8,),
        strides_t = (2,),
        emb_width = 64,
        l_bins = 256,
        l_mu = 0.1,
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
        revival_threshold=1.0,
        linf_k=2048,
    )

    print(hps)

    block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv,
                        dilation_growth_rate=hps.dilation_growth_rate,
                        dilation_cycle=hps.dilation_cycle,
                        reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

    vqvae = VQVAE(input_shape=(hps.sample_length,1), levels=hps.levels, downs_t=hps.downs_t, strides_t=hps.strides_t,
                  emb_width=hps.emb_width, l_bins=hps.l_bins,
                  mu=hps.l_mu, commit=hps.commit,
                  spectral=hps.spectral, multispectral=hps.multispectral,
                  multipliers=hps.hvqvae_multipliers, use_bottleneck=hps.use_bottleneck,
                  **block_kwargs)

    vqvae.bandwidth = calculate_bandwidth(AudioFileDataset(root=train_root_path,
                                                           sample_frames=cfg.training.sample_frames))
    print(vqvae.bandwidth)
    # print(vqvae.bandwidth['spec'].size())

    optimizer = optim.Adam(vqvae.parameters(), lr=cfg.training.optimizer.lr)
    # [encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level="O1")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)

    vqvae = torch.nn.DataParallel(vqvae)
    vqvae.to(device)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        vqvae.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]

        for level, bottleneck in enumerate(vqvae.bottleneck.level_blocks):
            num_samples = hps.sample_length
            downsamples = calculate_strides(hps.strides_t, hps.downs_t)
            raw_to_tokens = np.prod(downsamples[:level + 1])
            num_tokens = (num_samples // raw_to_tokens)
            bottleneck.restore_k(num_tokens=num_tokens, threshold=hps.revival_threshold)
    else:
        global_step = 0

    train_dataset = AudioFileDataset(
        root=train_root_path,
        sample_frames=cfg.training.sample_frames)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.training.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.training.n_workers,
                                  pin_memory=True,
                                  drop_last=False)

    n_epochs = cfg.training.n_steps // len(train_dataloader) + 1
    start_epoch = global_step // len(train_dataloader) + 1


    test_dataset = AudioFileDataset(root=test_root_path, sample_frames=cfg.training.sample_frames)
    test_dataloader = DataLoader(test_dataset,
                              batch_size=cfg.training.batch_size,#len(test_dataset),
                              shuffle=False,
                              num_workers=cfg.training.n_workers,
                              pin_memory=True,
                              drop_last=False)

    print('%d training batches' % len(train_dataloader))
    print('%d test batches' % len(test_dataloader))

    num_paths = 5
    # n_generate = 128
    n_generate = 16000
    jump = 0.4  # factor of distance between adjacent trajectory points (speed of changing)
    smooth = 0.25  # smoothing the trajectory turns, [0, 1]
    seed1 = 562  # change this to change starting point
    np.random.seed(seed1)
    z0 = np.random.random(cfg.model.encoder.embedding_dim) * 2 - 1
    seed2 = 377  # change this to change trajectory
    np.random.seed(seed2)
    z_paths = []
    for i in range(num_paths):
        z, _ = generate_trajectory(n_generate, dim=cfg.model.encoder.embedding_dim, _z0=z0, include_z0=True,
                                    jump=jump, smooth=smooth)
        z_paths.append(z)
    z_paths = torch.stack([torch.from_numpy(x).float() for x in z_paths]).cuda()

    for epoch in range(start_epoch, n_epochs + 1):

        gc.collect()
        torch.cuda.empty_cache()

        average_recon_loss = average_multispec_loss = average_vq_loss = average_usage = average_usage_smoothed = 0
        # for i, (audio, mags) in enumerate(tqdm(train_dataloader)):
        # for i, (audio, mels, mags) in enumerate(train_dataloader):
        for i, audio in enumerate(tqdm(train_dataloader)):
            # audio = audio.unsqueeze(1).to(device)
            audio = audio.unsqueeze(-1).to(device, non_blocking=True)

            # print(audio.size())

            vqvae.train()
            optimizer.zero_grad()

            x_out, loss, metrics = vqvae(audio, hps, hps.loss_fn)

            # print(x_out.size())
            # print(loss)
            # print(metrics)

            vq_loss = metrics['commit_loss']
            recon_loss = metrics['recons_loss']
            multispec_loss = metrics['multispectral_loss']
            usage_smoothed = metrics['usage']
            usage = metrics['used_curr']

            loss.mean().backward()

            # grad_norm = 0.0
            # for p in list(vqvae.parameters()):
            #     if p.grad is not None:
            #         grad_norm += p.grad.norm(p=2, dtype=torch.float32)**2
            # grad_norm = float(grad_norm**0.5)

            # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            scheduler.step()

            average_recon_loss += (recon_loss.mean().item() - average_recon_loss) / (i+1)
            average_multispec_loss += (multispec_loss.mean().item() - average_multispec_loss) / (i+1)
            average_vq_loss += (vq_loss.mean().item() - average_vq_loss) / (i+1)
            average_usage += (usage.mean().item() - average_usage) / (i+1)
            average_usage_smoothed += (usage_smoothed.mean().item() - average_usage) / (i+1)

            # idx = np.random.randint(len(audio))
            # print(audio[idx])
            # zs = vqvae.module.encode(audio[idx].unsqueeze(0))
            # # zs = vqvae.encode(audio[idx].unsqueeze(0))
            # print(zs)

            if global_step > 0:
                if global_step % cfg.training.checkpoint_interval == 0:
                    save_checkpoint2(vqvae, optimizer, scheduler, global_step, checkpoint_dir)

                # if global_step % cfg.training.generate_sample_interval == 0:
                #     with torch.no_grad():
                #         zq, _ = encoder.module.codebook.encode(z_paths)
                #         for j in range(len(zq)):
                #             output = decoder(zq[j].unsqueeze(0))
                #             # print(output.size())
                #
                #             # audio = mel_to_audio(output.squeeze().cpu().numpy(),
                #             #                      sr=cfg.preprocessing.sr,
                #             #                      n_fft=cfg.preprocessing.n_fft,
                #             #                      hop_length=cfg.preprocessing.hop_length,
                #             #                      win_length=cfg.preprocessing.win_length,
                #             #                      top_db=cfg.preprocessing.top_db,
                #             #                      preemph=cfg.preprocessing.preemph)
                #
                #             # audio = mag_to_audio(output.squeeze().cpu().numpy(),
                #             #                      hop_length=cfg.preprocessing.hop_length,
                #             #                      win_length=cfg.preprocessing.win_length,
                #             #                      top_db=cfg.preprocessing.top_db,
                #             #                      preemph=cfg.preprocessing.preemph)
                #
                #             audio = output.squeeze().cpu().numpy()
                #
                #             writer.add_audio("rand_{}".format(j),
                #                                      audio,
                #                                      global_step=global_step,
                #                                      sample_rate=cfg.preprocessing.sr)

                if global_step % cfg.training.eval_interval == 0:
                    eval3(vqvae, test_dataloader, device, writer, global_step, cfg, hps)

            global_step += 1

            # writer.add_scalar("recon_loss/train", recon_loss.mean().item(), global_step)
            # writer.add_scalar("multispec_loss/train", multispec_loss.mean().item(), global_step)
            # writer.add_scalar("vq_loss/train", vq_loss.mean().item(), global_step)
            # # writer.add_scalar("average_perplexity", average_perplexity, global_step)
            # writer.add_scalar("average_usage", usage.mean().item(), global_step)

        writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
        writer.add_scalar("multispec_loss/train", average_multispec_loss, global_step)
        writer.add_scalar("vq_loss/train", average_vq_loss, global_step)
        writer.add_scalar("usage/train", average_usage, global_step)
        writer.add_scalar("usage_smoothed/train", average_usage_smoothed, global_step)

        if True:#epoch % 10 == 0:
            print("epoch:{}, recon loss:{:.3f}, multispec loss:{:.3f}, vq loss:{:.3f}, usage:{:.3f}, usage smooth:{:.3f}"
                  .format(epoch, average_recon_loss, average_multispec_loss, average_vq_loss, average_usage, average_usage_smoothed))


def eval3(vqvae, dataloader, device, writer, global_step, cfg, hps):
    vqvae.eval()
    average_recon_loss = average_multispec_loss = average_vq_loss = average_usage = 0
    for i, audio in enumerate(dataloader):  # probably only one batch...
        audio = audio.unsqueeze(-1).to(device, non_blocking=True)

        x_out, loss, metrics = vqvae(audio, hps, hps.loss_fn)

        vq_loss = metrics['commit_loss']
        recon_loss = metrics['recons_loss']
        multispec_loss = metrics['multispectral_loss']
        # usage = metrics['used_curr']

        average_recon_loss += (recon_loss.mean().item() - average_recon_loss) / (i+1)
        average_multispec_loss += (multispec_loss.mean().item() - average_multispec_loss) / (i+1)
        average_vq_loss += (vq_loss.mean().item() - average_vq_loss) / (i+1)
        # average_usage += (usage.mean().item() - average_usage) / (i+1)

        with torch.no_grad():
            # first N samples from each batch
            n_audio = 10
            for j in range(n_audio):
                k = j * int(len(x_out) / n_audio)
                output = x_out[k]
                audio = output.squeeze().cpu().numpy()
                writer.add_audio("recon_{}".format(i*len(x_out) + j),
                                 audio,
                                 global_step=global_step,
                                 sample_rate=cfg.preprocessing.sr)

    writer.add_scalar("recon_loss/test", average_recon_loss, global_step)
    writer.add_scalar("multispec_loss/test", average_multispec_loss, global_step)
    writer.add_scalar("vq_loss/test", average_vq_loss, global_step)
    # writer.add_scalar("usage/test", average_usage, global_step)

    print("eval:{}, recon loss:{:.2E}, multispec loss:{:.2E}, vq loss:{:.2E}".format(global_step,
                                                                                     average_recon_loss,
                                                                                     average_multispec_loss,
                                                                                     average_vq_loss))


@hydra.main(config_path="config/train.yaml")
def test2(cfg):
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
        revival_threshold=1.0,
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
    vqvae = vqvae.to(device)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        vqvae.load_state_dict(clean_state_dict(checkpoint["model"]))

        for level, bottleneck in enumerate(vqvae.bottleneck.level_blocks):
            num_samples = hps.sample_length
            downsamples = calculate_strides(hps.strides_t, hps.downs_t)
            raw_to_tokens = np.prod(downsamples[:level + 1])
            num_tokens = (num_samples // raw_to_tokens)
            bottleneck.restore_k(num_tokens=num_tokens, threshold=hps.revival_threshold)

        # for level, bottleneck in enumerate(vqvae.bottleneck.level_blocks):
        #     bottleneck.mu = 0.1


    train_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "train"
    train_dataset = AudioFileDataset(
        root=train_root_path,
        sample_frames=cfg.training.sample_frames)

    # audio = torch.rand(cfg.training.sample_frames)
    # audio = audio.unsqueeze(0).unsqueeze(-1).to(device)

    np.random.seed(1234)
    rand_idx = np.random.permutation(len(train_dataset))
    n_iters = 100
    n_samples = 10
    for _ in range(n_iters):
        for i in range(n_samples):
            audio_in = train_dataset[rand_idx[i]].unsqueeze(0).unsqueeze(-1).to(device)
            print(audio_in.size())

            vqvae.train()

            audio_out, loss, metrics = vqvae(audio_in, hps, hps.loss_fn)
            print(metrics)

            # zs = vqvae.encode(audio_in)
            # print(zs[0].size())
            # print(zs)
            # audio_out = vqvae.decode(zs)

            print(audio_out.size())

            audio_out = audio_out.detach().squeeze().cpu().numpy()
            print(audio_out.shape)

            # audio_out = audio_in.detach().squeeze().cpu().numpy()
            # print(audio_out.shape)

            out_dir = Path(utils.to_absolute_path(cfg.out_dir))
            out_dir.mkdir(exist_ok=True, parents=True)
            path = out_dir / ('%02d' % i)
            sf.write(path.with_suffix(".wav"), audio_out.astype(np.float32), cfg.preprocessing.sr, 'PCM_24')

            print('---------')

    print('done!')


@hydra.main(config_path="config/train.yaml")
def test3(cfg):
    from audio_utils import spectral_loss, multispectral_loss, norm, spec, squeeze, stft
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "train"
    train_dataset = AudioFileDataset(
        root=train_root_path,
        sample_frames=cfg.training.sample_frames)

    # x = train_dataset[0]
    # y = train_dataset[0]
    # spec_norm = spec(x, n_fft=2048, hop_length=160, win_length=400)
    # spec_norm_diff = spec(x, n_fft=2048, hop_length=160, win_length=400) - spec(y, n_fft=2048, hop_length=160, win_length=400)
    # print(spec_norm.size())
    # print(spec_norm_diff.size())
    # print('---------')
    #
    # x = train_dataset[0].unsqueeze(0).unsqueeze(-1).to(device)
    # y = train_dataset[1].unsqueeze(0).unsqueeze(-1).to(device)
    # spec_in = spec(squeeze(x.float()), n_fft=2048, hop_length=160, win_length=400)
    # spec_out = spec(squeeze(y.float()), n_fft=2048, hop_length=160, win_length=400)
    # print(spec_in.size())
    # print((spec_in - spec_out).size())
    # print(norm(spec_in - spec_out).size())

    # x = train_dataset[0]
    # xs = spec(x, n_fft=2048, hop_length=160, win_length=400)
    # xs_norm = torch.linalg.norm(xs)
    # # xs_flat = xs.view(xs.shape[0], -1) ** 2
    # # s_norm = xs_flat.sum(dim=-1).sqrt()
    # # print(xs_flat.size())
    # print(xs.size())
    # print(xs_norm.size())
    # print(xs_norm)

    # bandwidth = calculate_bandwidth(train_dataset, duration=10)
    # # for key in bandwidth:
    # #     bandwidth[key] = bandwidth[key].to(device)
    # print(bandwidth)
    # print(bandwidth['spec'].size())

    # x = train_dataset[0].unsqueeze(0).unsqueeze(-1).to(device)
    # y = train_dataset[1].unsqueeze(0).unsqueeze(-1).to(device)
    # print(multispectral_loss(x, y).size())
    # print(spectral_loss(x, y).size())
    # sl = multispectral_loss(x, y) / bandwidth['spec']
    # print(sl.size())
    # sl = torch.mean(sl)
    # print(sl)

    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=cfg.training.batch_size,
    #                               shuffle=True,
    #                               num_workers=cfg.training.n_workers,
    #                               pin_memory=True,
    #                               drop_last=True)
    # for i, audio in enumerate(tqdm(train_dataloader)):
    #     audio = audio.unsqueeze(-1).to(device, non_blocking=True)
    #     print(audio.size())


if __name__ == "__main__":
    # train_model()
    # train_model2()
    train_model3()

    # test()
    # test2()
    # test3()
