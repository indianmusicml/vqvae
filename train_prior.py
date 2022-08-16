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
from torch import nn, Tensor
# from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from prior import TransformerModel, generate_square_subsequent_mask


# class VQCodebookDataset(Dataset):
#     def __init__(self, data_path, bsz, bptt):
#         self.bptt = bptt
#
#         data = []
#         for root, _, files in os.walk(data_path):
#             for filename in files:
#                 subname, ext = os.path.splitext(filename)
#                 name, ext2 = os.path.splitext(subname)
#                 if ext == '.npy' and ext2 == '.idx':
#                     path = data_path / name
#                     indices = torch.from_numpy(np.load(path.with_suffix(".idx.npy")))
#                     data.append(indices)
#
#         # batchify
#         data = torch.cat(data, 0)   # flattens into one long sequence
#         seq_len = data.size(0) // bsz
#         data = data[:seq_len * bsz]
#         data = data.view(bsz, seq_len).t().contiguous()  # shape [seq_len, batch_size]
#         # data = batchify(data, bsz)
#
#         self.data = data
#         print("loaded dataset with {} samples".format(data.size(0)))
#
#     def __len__(self):
#         return self.data.size(0) - self.bptt - 1
#
#     def __getitem__(self, index):
#         seq_len = min(self.bptt, len(self.data) - 1 - index)
#         data = self.data[index:index+seq_len]
#         target = self.data[index+1:index+1+seq_len].reshape(-1)
#         return data, target


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


def batchify(data: list, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.
    Args:
        data: list, shape [N]
        bsz: int, batch size
    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    data = torch.cat(data, 0)   # flattens into one long sequence
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def create_dataset(data_path: Path, bsz: int) -> Tensor:
    data = []
    for root, _, files in os.walk(data_path):
        for filename in files:
            subname, ext = os.path.splitext(filename)
            name, ext2 = os.path.splitext(subname)
            if ext == '.npy' and ext2 == '.idx':
                path = data_path / name
                indices = torch.from_numpy(np.load(path.with_suffix(".idx.npy")))
                # reshape to 20xN, transpose, flatten
                # indices = indices.view(20, -1).t().contiguous().view(-1)
                data.append(indices)
    data = batchify(data, bsz)  # shape [seq_len, batch_size]
    return data


def get_batch(source: Tensor, bptt: int, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        bptt: int
        i: int
    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data.t(), target


def get_batch_rand(source: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        seq_len: int
    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    i = random.randint(1, len(source) - seq_len - 1)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def generate(model, device):
    output_to_decode = 512
    decoded_tokens = torch.tensor(random.randint(1, 512)).unsqueeze(-1).to(device)
    for i in range(output_to_decode):
        src_mask = generate_square_subsequent_mask(i+1).to(device)
        output = model(decoded_tokens.unsqueeze(-1), src_mask)

        top_indices = torch.argmax(output, dim=-1)
        # we only care about the last token that was decoded
        top_indices_last_token = top_indices[-1:][0]
        # add most likely token to the already decoded tokens
        decoded_tokens = torch.cat([decoded_tokens, top_indices_last_token])
    return decoded_tokens


@hydra.main(config_path="config/train.yaml")
def train_prior(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(1234)

    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.checkpoint_dir
    writer = SummaryWriter(tensorboard_path)

    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))

    train_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "train"
    test_root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path / "test"

    train_batch_size = 256
    eval_batch_size = 256

    # bptt = 256

    # ntokens = cfg.model.encoder.n_embeddings  # size of vocabulary
    # emsize = 256   # embedding dimension
    # d_hid = 12     # dimension of the feedforward network model in nn.TransformerEncoder
    # # emsize = 200   # embedding dimension
    # # d_hid = 200     # dimension of the feedforward network model in nn.TransformerEncoder
    # nlayers = 2    # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    # nhead = 2      # number of heads in nn.MultiheadAttention
    # dropout = 0.2  # dropout probability
    # model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    # # model = torch.nn.DataParallel(model).to(device)

    # # trying to match jukebox "tiny_prior"
    # # bptt = 8192
    # bptt = 200  # should be ~2 sec for hopsize=160 at sr=16k
    # ntokens = cfg.model.encoder.n_embeddings  # size of vocabulary
    # emsize = 256   # embedding dimension
    # d_hid = 256    # ?? dimension of the feedforward network model in nn.TransformerEncoder
    # nlayers = 2    # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    # nhead = 2      # number of heads in nn.MultiheadAttention
    # dropout = 0.1  # dropout probability
    # model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    # # model = torch.nn.DataParallel(model).to(device)

    # trying to match jukebox "tiny_prior"
    bptt = 8192      # should be 0.5 sec for
    # bptt = 200     # should be ~2 sec for hopsize=160 at sr=16k
    ntokens = cfg.model.encoder.n_embeddings  # size of vocabulary
    emsize = 256   # embedding dimension
    d_hid = 256    # ?? dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 12   # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 1      # number of heads in nn.MultiheadAttention
    dropout = 0.1  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    # model = torch.nn.DataParallel(model).to(device)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)

    criterion = nn.CrossEntropyLoss()

    # lr = 5.0  # learning rate
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

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

    train_data = create_dataset(train_root_path, train_batch_size).to(device)
    test_data = create_dataset(test_root_path, eval_batch_size).to(device)
    print(train_data.shape)
    print(test_data.shape)

    # train_dataset = VQCodebookDataset(
    #     data_path=train_root_path,
    #     bsz=train_batch_size,
    #     bptt=bptt)
    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=train_batch_size,
    #                               shuffle=True,
    #                               num_workers=8,
    #                               pin_memory=True,
    #                               drop_last=False)

    # test_dataset = VQCodebookDataset(
    #     data_path=test_root_path,
    #     bsz=eval_batch_size,
    #     bptt=bptt)
    # test_dataloader = DataLoader(test_dataset,
    #                              batch_size=eval_batch_size,
    #                              shuffle=False,
    #                              num_workers=8,
    #                              pin_memory=True,
    #                              drop_last=False)

    for epoch in range(start_epoch, n_epochs + 1):
        epoch_start_time = time.time()

        model.train()
        # total_loss = 0.
        average_loss = 0.
        start_time = time.time()


        num_batches = train_data.size(0) - bptt - 1
        # num_batches = len(train_dataloader)
        batch_idx = np.random.permutation(num_batches)  # shuffle the data
        for i in range(num_batches):
        # for i, data_targets in enumerate(tqdm(train_dataloader)):
            #data, targets = data_targets
            # data, targets = get_batch(train_data, bptt, i)
            data, targets = get_batch(train_data, bptt, batch_idx[i])
            # data, targets = get_batch_rand(train_data, bptt)

            data = data.to(device)
            targets = targets.to(device)

            # seq_len = data.size(0)
            # seq_len = data.size(1)
            # src_mask = generate_square_subsequent_mask(seq_len).to(device)

            # print(data)
            # print(data.shape)
            # print(src_mask.shape)

            # output = model(data, src_mask)
            output = model(data)

            output = output.permute(1, 0, 2)
            output_flat = output.view(-1, ntokens)
            loss = criterion(output_flat, targets)

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
        num_val_batches = test_data.size(0) - bptt - 1
        # num_val_batches = len(test_dataloader)
        with torch.no_grad():
            for i in range(num_val_batches):
            # for i, data_targets in enumerate(tqdm(test_dataloader)):
                data, targets = get_batch(test_data, bptt, i)
                # data, targets = data_targets

                data = data.to(device)
                targets = targets.to(device)

                # seq_len = data.size(0)
                # seq_len = data.size(1)
                # src_mask = generate_square_subsequent_mask(seq_len).to(device)
                # output = model(data, src_mask)

                output = model(data)

                output = output.permute(1, 0, 2)
                output_flat = output.view(-1, ntokens)
                val_loss += criterion(output_flat, targets).item()
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
