import hydra
from hydra import utils
import itertools
from pathlib import Path
import math
import traceback

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from melgan.generator import Generator
from melgan.multiscale import MultiScaleDiscriminator

from dataset import SpeechDataset


def eval(generator, discriminator, dataloader, device, writer, step, cfg):
    generator.eval()
    discriminator.eval()
    # torch.backends.cudnn.benchmark = False

    feat_match = 10.0

    loss_g_sum = 0.0
    loss_d_sum = 0.0
    for i, (audio, mags) in enumerate(dataloader):
        mags = mags.to(device)
        audio = audio.to(device)

        audio = audio.unsqueeze(1)

        # generator
        fake_audio = generator(mags)
        disc_fake = discriminator(fake_audio[:, :, :audio.size(2)])
        disc_real = discriminator(audio)
        loss_g = 0.0
        loss_d = 0.0
        for (feats_fake, score_fake), (feats_real, score_real) in zip(disc_fake, disc_real):
            loss_g += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
            for feat_f, feat_r in zip(feats_fake, feats_real):
                loss_g += feat_match * torch.mean(torch.abs(feat_f - feat_r))
            loss_d += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
            loss_d += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))

        loss_g_sum += loss_g.item()
        loss_d_sum += loss_d.item()

    loss_g_avg = loss_g_sum / len(dataloader)
    loss_d_avg = loss_d_sum / len(dataloader)

    audio = audio[0][0].cpu().detach().numpy()
    fake_audio = fake_audio[0][0].cpu().detach().numpy()

    writer.add_audio("orig/test",
                     audio,
                     global_step=step,
                     sample_rate=cfg.preprocessing.sr)

    writer.add_audio("recon/test",
                     fake_audio,
                     global_step=step,
                     sample_rate=cfg.preprocessing.sr)

    writer.add_scalar("loss_g/test", loss_g_avg, step)
    writer.add_scalar("loss_d/test", loss_d_avg, step)

    print("eval:{}, gen loss:{:.2E}, disc loss:{:.2E},".format(step,
                                                               loss_g_avg,
                                                               loss_d))


    # torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config/train.yaml")
def train_model(cfg):
    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    writer = SummaryWriter(tensorboard_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_match = 10.0

    model_g = Generator(cfg.model.encoder.in_channels)
    model_d = MultiScaleDiscriminator()

    model_g = torch.nn.DataParallel(model_g).to(device)
    model_d = torch.nn.DataParallel(model_d).to(device)

    optim_g = torch.optim.Adam(model_g.parameters(),
                               lr=cfg.training.optimizer.lr, betas=(0.5, 0.9))
    optim_d = torch.optim.Adam(model_d.parameters(),
                               lr=cfg.training.optimizer.lr, betas=(0.5, 0.9))

    init_epoch = -1
    step = 0

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

    # # this accelerates training when the size of minibatch is always consistent.
    # # if not consistent, it'll horribly slow down.
    # torch.backends.cudnn.benchmark = True

    train_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "train"
    train_dataset = SpeechDataset(
        root=train_root_path,
        hop_length=cfg.preprocessing.hop_length,
        sample_frames=cfg.training.sample_frames)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.training.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.training.n_workers,
                                  pin_memory=True,
                                  drop_last=True)

    n_epochs = cfg.training.n_steps // len(train_dataloader) + 1
    # start_epoch = global_step // len(train_dataloader) + 1

    test_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "test"
    test_dataset = SpeechDataset(
        root=test_root_path,
        hop_length=cfg.preprocessing.hop_length,
        sample_frames=cfg.training.sample_frames)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=len(test_dataset),
                                 shuffle=False,
                                 num_workers=cfg.training.n_workers,
                                 pin_memory=True,
                                 drop_last=True)

    try:
        for epoch in itertools.count(init_epoch+1):

            # shuffle data?

            for i, (audio, mags) in enumerate(train_dataloader):
                audio, mags = audio.to(device), mags.to(device)

                audio = audio.unsqueeze(1)

                # print('input audio: %d' % audio.size(-1))
                # print('input mags: %d' % audio.size(-1))

                # generator
                model_g.train()
                optim_g.zero_grad()
                fake_audio = model_g(mags)

                # hmmm
                fake_audio = fake_audio[:, :, :(cfg.training.sample_frames * cfg.preprocessing.hop_length)]

                # print('output audio: %d' % audio.size(-1))

                disc_fake = model_d(fake_audio)
                disc_real = model_d(audio)

                loss_g = 0.0
                for (feats_fake, score_fake), (feats_real, _) in zip(disc_fake, disc_real):
                    loss_g += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                    for feat_f, feat_r in zip(feats_fake, feats_real):
                        loss_g += feat_match * torch.mean(torch.abs(feat_f - feat_r))

                loss_g.backward()
                optim_g.step()

                # discriminator
                fake_audio = fake_audio.detach()
                model_d.train()
                optim_d.zero_grad()
                disc_fake = model_d(fake_audio)
                disc_real = model_d(audio)

                loss_d = 0.0
                for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                    loss_d += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                    loss_d += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))

                loss_d.backward()
                optim_d.step()

                step += 1

                # logging
                loss_g = loss_g.item()
                loss_d = loss_d.item()
                if any([loss_g > 1e8, math.isnan(loss_g), loss_d > 1e8, math.isnan(loss_d)]):
                    print("loss_g %.01f loss_d_avg %.01f at step %d!" % (loss_g, loss_d, step))
                    raise Exception("Loss exploded")

                if step % 100 == 0:
                    print("train:{}, gen loss:{:.2E}, disc loss:{:.2E},".format(step, loss_g, loss_d))

                writer.add_scalar("loss_g/train", loss_g, step)
                writer.add_scalar("loss_d/train", loss_d, step)

                if step % cfg.training.generate_sample_interval == 0:
                    for j in range(0, fake_audio.size(0), 10):
                        writer.add_audio("orig_{}/train".format(j),
                                         audio[j, :, :].cpu().numpy(),
                                         global_step=step,
                                         sample_rate=cfg.preprocessing.sr)
                        writer.add_audio("recon_{}/train".format(j),
                                         fake_audio[j, :, :].cpu().numpy(),
                                         global_step=step,
                                         sample_rate=cfg.preprocessing.sr)

                if step % cfg.training.eval_interval == 0:
                    eval(model_g, model_d, test_dataloader, device, writer, step, cfg)

                if step % cfg.training.checkpoint_interval == 0:
                    checkpoint_state = {
                        'model_g': model_g.state_dict(),
                        'model_d': model_d.state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'step': step,
                        'epoch': epoch
                    }
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)
                    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
                    torch.save(checkpoint_state, checkpoint_path)
                    print("Saved checkpoint: {}".format(checkpoint_path.stem))

    except Exception as e:
        print("Exiting due to exception: %s" % e)
        traceback.print_exc()


if __name__ == "__main__":
    train_model()
