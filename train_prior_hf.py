import os
from typing import Tuple
import time
import random
from pathlib import Path
from tqdm import tqdm

import hydra
from hydra import utils

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import OpenAIGPTConfig, OpenAIGPTLMHeadModel

class VQCodebookDataset(Dataset):
    def __init__(self, data_path, bptt):
        self.bptt = bptt

        data = []
        for root, _, files in os.walk(data_path):
            for filename in files:
                subname, ext = os.path.splitext(filename)
                name, ext2 = os.path.splitext(subname)
                if ext == '.npy' and ext2 == '.idx':
                    path = data_path / name
                    indices = torch.from_numpy(np.load(path.with_suffix(".idx.npy")))
                    data.append(indices)

        data = torch.cat(data, 0)   # flattens into one long sequence
        self.data = data
        print("loaded dataset with {} samples".format(data.size(0)))

    def __len__(self):
        return self.data.size(0) - self.bptt - 1

    def __getitem__(self, index):
        seq_len = min(self.bptt, len(self.data) - 1 - index)
        data = self.data[index:index+seq_len]
        return data


def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


@hydra.main(config_path="config/train.yaml")
def train_prior(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(1234)

    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    writer = SummaryWriter(tensorboard_path)

    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))

    train_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "train"
    test_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "test"

    train_batch_size = 1024
    eval_batch_size = 1024

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
    model = OpenAIGPTLMHeadModel(config).to(device)

    model = torch.nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300000, 400000], gamma=0.5)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    n_epochs = cfg.training.n_steps
    start_epoch = global_step

    best_val_loss = float('inf')
    # best_model = None

    log_interval = 10
    checkpoint_interval = 5000

    train_dataset = VQCodebookDataset(
        data_path=train_root_path,
        bptt=bptt)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=False)

    test_dataset = VQCodebookDataset(
        data_path=test_root_path,
        bptt=bptt)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=eval_batch_size,
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True,
                                 drop_last=False)

    for epoch in range(start_epoch, n_epochs + 1):
        epoch_start_time = time.time()

        model.train()
        # total_loss = 0.
        average_loss = 0.
        start_time = time.time()

        num_batches = len(train_dataloader)
        for i, batch in enumerate(tqdm(train_dataloader)):
            # inputs, labels = batch[..., :-1], batch[..., -1]
            # labels = labels.unsqueeze(-1)
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            # print(inputs.size())
            # print(labels.size())
            # outputs = model.forward(inputs, labels=labels)

            inputs = batch
            inputs = inputs.to(device)
            outputs = model.forward(inputs, labels=inputs)
            loss = outputs.loss.mean()

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # total_loss += loss.item()
            average_loss += (loss.item() - average_loss) / (i + 1)

            if global_step % log_interval == 0 and global_step > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                ppl = np.exp(average_loss)
                print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                      f'lr {lr:.4e} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {average_loss:5.4f} | ppl {ppl:8.2f}')
                start_time = time.time()
                writer.add_scalar("loss/train", average_loss, global_step)
                writer.add_scalar("perplexity/train", ppl, global_step)

                # generate(model, device)

            if global_step % checkpoint_interval == 0 and global_step > 0:
                save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_dir)

            global_step += 1

        model.eval()  # turn on evaluation mode
        val_loss = 0.
        num_val_batches = len(test_dataloader)
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_dataloader)):
                # inputs, labels = batch[..., :-1], batch[..., -1]
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                # outputs = model.forward(inputs, labels=labels)

                inputs = batch
                inputs = inputs.to(device)
                outputs = model.forward(inputs, labels=inputs)

                loss = outputs.loss.mean()
                val_loss += loss.item()
                if i > 0 and i % log_interval == 0:
                    print(f'| epoch {epoch:3d} | {i:5d}/{num_val_batches:5d} val batches | val loss {(val_loss / (i + 1)):5.4f}')
        val_loss = val_loss / num_val_batches
        val_ppl = np.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)
        writer.add_scalar("loss/test", val_loss, global_step)
        writer.add_scalar("perplexity/test", val_ppl, global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # best_model = copy.deepcopy(model)

        scheduler.step()


if __name__ == "__main__":
    train_prior()
