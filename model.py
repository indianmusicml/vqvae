import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tqdm import tqdm
import numpy as np
from preprocess import mulaw_decode


class RosinalityResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        # residual connection
        out += input
        return out


class RosinalityEncoder(nn.Module):
    def __init__(self, in_channel=1, n_hidden_channels=768, n_res_block=2,
                 n_res_channel=32, resolution_factor=2, groups=1,
                 n_embeddings=512, embedding_dim=64,
                 use_local_kernels=False):
        super(RosinalityEncoder, self).__init__()
        self.use_local_kernels = use_local_kernels

        downsampling_stride = 2
        if not use_local_kernels:
            # downsampling using overlapping kernels
            downsampling_kernel_size = 2 * downsampling_stride
        else:
            # downsampling kernels do not overlap
            downsampling_kernel_size = downsampling_stride
        # downsampling module
        if resolution_factor == 16:
            blocks = [
                nn.Conv2d(in_channel, n_hidden_channels // 4,
                          downsampling_kernel_size,
                          stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hidden_channels // 4, n_hidden_channels // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hidden_channels // 2, 3 * n_hidden_channels // 4,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(3 * n_hidden_channels // 4, n_hidden_channels,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1,
                          groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hidden_channels, n_hidden_channels, 3, padding=1, groups=groups),
            ]
        elif resolution_factor == 8:
            blocks = [
                nn.Conv2d(in_channel, n_hidden_channels // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hidden_channels // 2, n_hidden_channels // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    n_hidden_channels // 2, n_hidden_channels,
                    downsampling_kernel_size, stride=downsampling_stride,
                    padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hidden_channels, n_hidden_channels, 3, padding=1, groups=groups),
            ]
        elif resolution_factor == 4:
            blocks = [
                nn.Conv2d(in_channel, n_hidden_channels // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hidden_channels // 2, n_hidden_channels,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hidden_channels, n_hidden_channels, 3, padding=1,
                          groups=groups),
            ]
        elif resolution_factor == 2:
            blocks = [
                nn.Conv2d(in_channel, n_hidden_channels // 2,
                          downsampling_kernel_size, stride=downsampling_stride,
                          padding=1, groups=groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hidden_channels // 2, n_hidden_channels, 3, padding=1, groups=groups),
            ]
        else:
            raise ValueError(
                f"Unexpected resolution factor {resolution_factor}")

        for i in range(n_res_block):
            blocks.append(RosinalityResBlock(n_hidden_channels, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

        self.quantize_conv_t = nn.Conv2d(n_hidden_channels,
                                         embedding_dim, 1)

        self.codebook = VQEmbeddingEMA(n_embeddings, embedding_dim)
        self.jitter = Jitter(0.5)

        # print(in_channel)
        # print(channel // 2)
        # self.conv1 =  nn.Conv2d(in_channel, channel // 2,
        #                         downsampling_kernel_size, stride=downsampling_stride,
        #                         padding=1, groups=groups)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(channel // 2, channel, 3, padding=1, groups=groups)

    def forward(self, mels):
        mels = torch.unsqueeze(mels, 1)
        z = self.blocks(mels)
        # quant_t = self.quantize_conv_t(z).permute(0, 2, 3, 1)
        print(z.size())
        z = self.quantize_conv_t(z)
        print(z.size())
        z, loss, perplexity, usage = self.codebook(z.transpose(1, 2))
        # z = self.jitter(z)
        return z, loss, perplexity, usage


class RosinalityDecoder(nn.Module):
    def __init__(self, in_channel=64, out_channel=768, channel=128,
                 n_res_block=2, n_res_channel=32, resolution_factor=2,
                 groups=1, use_local_kernels=False):
        super(RosinalityDecoder, self).__init__()
        self.use_local_kernels = use_local_kernels

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(RosinalityResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        upsampling_stride = 2
        if not use_local_kernels:
            # upsampling using overlapping kernels
            upsampling_kernel_size = 2 * upsampling_stride
        else:
            # upsampling kernels do not overlap
            upsampling_kernel_size = upsampling_stride
        # upsampling module
        if resolution_factor == 16:
            blocks.extend(
                [
                    nn.ConvTranspose2d(
                        channel, 3 * channel // 4,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        3 * channel // 4, channel // 2,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, channel // 4,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 4, out_channel,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups)
                ]
            )
        elif resolution_factor == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2,
                                       upsampling_kernel_size,
                                       stride=upsampling_stride,
                                       padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, channel // 2,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups)
                ]
            )
        elif resolution_factor == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2,
                                       upsampling_kernel_size,
                                       stride=upsampling_stride,
                                       padding=1, groups=groups),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel,
                        upsampling_kernel_size, stride=upsampling_stride,
                        padding=1, groups=groups),
                ]
            )
        elif resolution_factor == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel,
                                   upsampling_kernel_size,
                                   stride=upsampling_stride,
                                   padding=1, groups=groups)
            )
        else:
            raise ValueError(
                f"Unexpected resolution factor {resolution_factor}")

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)



def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


class Encoder(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, embedding_dim, jitter=0):
        super(Encoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(in_channels, channels, 3, 1, 0, bias=False),
        #     nn.BatchNorm1d(channels),
        #     nn.ReLU(True),
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.BatchNorm1d(channels),
        #     nn.ReLU(True),
        #     nn.Conv1d(channels, channels, 4, 2, 1, bias=False),
        #     nn.BatchNorm1d(channels),
        #     nn.ReLU(True),
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.BatchNorm1d(channels),
        #     nn.ReLU(True),
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.BatchNorm1d(channels),
        #     nn.ReLU(True),
        #     nn.Conv1d(channels, embedding_dim, 1)
        # )

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, embedding_dim, 1)
        )

        self.codebook = VQEmbeddingEMA(n_embeddings, embedding_dim)
        self.jitter = Jitter(jitter)

    def forward(self, x):
        z = self.encoder(x)
        z_q, loss, perplexity, usage = self.codebook(z.transpose(1, 2))
        # z = self.jitter(z)
        return z_q, loss, perplexity, usage

    def encode(self, x):
        # print('in: %s' % str(x.shape))
        z = self.encoder(x)
        # print('out: %s' % str(z.shape))
        z_q, indices = self.codebook.encode(z.transpose(1, 2))
        return z_q, indices

    def quantize(self, indices):
        z_q = self.codebook.quantize(indices)
        return z_q


class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        else:
            batch_size, sample_size, channels = x.size()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=2.0, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # init_bound = 1 / 512
        # embedding = torch.Tensor(n_embeddings, embedding_dim)
        # embedding.uniform_(-init_bound, init_bound)
        # self.register_buffer("embedding", embedding)
        # self.register_buffer("ema_count", torch.zeros(n_embeddings))
        # self.register_buffer("ema_weight", self.embedding.clone())

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1./n_embeddings, 1./n_embeddings)

    def encode(self, x):
        # M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, self.embedding_dim)

        distances = torch.addmm(torch.sum(self.embedding.weight ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.weight.t(),
                                alpha=-2.0, beta=1.0)

        # indices = torch.argmin(distances.float(), dim=-1)
        # quantized = F.embedding(indices, self.embedding)
        # quantized = quantized.view_as(x)

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # print(encoding_indices.size())

        encodings = torch.zeros(encoding_indices.shape[0], self.n_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # print(encodings.size())

        quantized = torch.matmul(encodings, self.embedding.weight).view_as(x)

        # is this necessary??
        quantized = x + (quantized - x).detach()

        # x64 = torch.tensor(x, dtype=torch.double, device=x.device)
        # quantized64 = torch.tensor(quantized, dtype=torch.double, device=quantized.device)
        # quantized64 = x64 + (quantized64 - x64).detach()
        # print(quantized64)
        # quantized = torch.tensor(quantized64, dtype=torch.float, device=quantized64.device)
        # print(quantized)

        return quantized, encoding_indices

    def forward(self, x):
        # M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, self.embedding_dim)

        distances = torch.addmm(torch.sum(self.embedding.weight ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.weight.t(),
                                alpha=-2.0, beta=1.0)

        # indices = torch.argmin(distances.float(), dim=-1)
        # encodings = F.one_hot(indices, M).float()
        # quantized = F.embedding(indices, self.embedding)
        # quantized = quantized.view_as(x)

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view_as(x)

        # if self.training:
        #
            # self.ema_count = torch.sum(encodings, dim=0)
            # self.ema_weight = torch.matmul(encodings.t(), x_flat)
            # self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            # # reset unused codebook vectors to random input vectors ala jukebox
            # unused = torch.sum(encodings, dim=0) < 1.0
            # num_empty = torch.sum(unused).item()
            # if num_empty > 0 and torch.rand(1) < 0.1:
            #     print('resetting unused! %d' % num_empty)
            #     x_rand = x.detach()[torch.randint(x.size(0), (1,)), torch.randint(x.size(1), (num_empty,)), :]
            #     self.embedding[unused] = x_rand

            # self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            #
            # n = torch.sum(self.ema_count)
            # self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            #
            # dw = torch.matmul(encodings.t(), x_flat)
            # self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            #
            # self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # e_latent_loss = F.mse_loss(x, quantized.detach())
        # loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        # quantized = quantized.detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage = torch.sum(torch.sum(encodings, dim=0) >= 1.0).float()

        return quantized, loss, perplexity, usage

    def quantize(self, encoding_indices):
        encodings = torch.zeros(encoding_indices.shape[0], self.n_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(1, -1, self.embedding_dim)  # necessary?

        return quantized


class Decoder(nn.Module):
    def __init__(self, in_channels,
                 conditioning_channels, mu_embedding_dim, rnn_channels,
                 fc_channels, bits, hop_length):
        super().__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**bits
        self.hop_length = hop_length

        self.rnn1 = nn.GRU(in_channels, conditioning_channels,
                           num_layers=2, batch_first=True, bidirectional=True)
        self.mu_embedding = nn.Embedding(self.quantization_channels, mu_embedding_dim)
        self.rnn2 = nn.GRU(mu_embedding_dim + 2*conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, z):
        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)

        z, _ = self.rnn1(z)

        z = F.interpolate(z.transpose(1, 2), scale_factor=self.hop_length)
        z = z.transpose(1, 2)

        x = self.mu_embedding(x)
        x, _ = self.rnn2(torch.cat((x, z), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, z):
        output = []
        cell = get_gru_cell(self.rnn2)

        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)

        z, _ = self.rnn1(z)

        z = F.interpolate(z.transpose(1, 2), scale_factor=self.hop_length)
        z = z.transpose(1, 2)

        batch_size, sample_size, _ = z.size()

        h = torch.zeros(batch_size, self.rnn_channels, device=z.device)
        x = torch.zeros(batch_size, device=z.device).fill_(self.quantization_channels // 2).long()

        for m in tqdm(torch.unbind(z, dim=1), leave=False):
            x = self.mu_embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)
            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            dist = Categorical(logits=logits)
            x = dist.sample()
            output.append(2 * x.float().item() / (self.quantization_channels - 1.)  - 1.)

        output = np.asarray(output, dtype=np.float64)
        output = mulaw_decode(output, self.quantization_channels)
        return output


class Decoder2(nn.Module):
    def __init__(self, in_channels, channels, embedding_dim):
        super(Decoder2, self).__init__()
        # self.decoder = nn.Sequential(nn.ConvTranspose1d(embedding_dim, channels, 1),
        #                              nn.ReLU(True),
        #                              nn.ConvTranspose1d(channels, channels, 3, 1, 1, bias=False),
        #                              nn.ReLU(True),
        #                              nn.ConvTranspose1d(channels, channels, 3, 1, 1, bias=False),
        #                              nn.ReLU(True),
        #                              nn.ConvTranspose1d(channels, channels, 4, 2, 1, bias=False),
        #                              nn.ReLU(True),
        #                              nn.ConvTranspose1d(channels, channels, 3, 1, 1, bias=False),
        #                              nn.ReLU(True),
        #                              nn.ConvTranspose1d(channels, in_channels, 3, 1, 0, bias=False),
        #                              )

        self.decoder = nn.Sequential(nn.ConvTranspose1d(embedding_dim, channels, 1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(channels, channels, 3, 2, 1, bias=False, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(channels, channels, 3, 2, 1, bias=False, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(channels, channels, 3, 2, 1, bias=False, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(channels, channels, 3, 2, 1, bias=False, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(channels, channels, 3, 2, 1, bias=False, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(channels, channels, 3, 2, 1, bias=False, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(channels, channels, 3, 2, 1, bias=False, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose1d(channels, in_channels, 3, 2, 1, bias=False, output_padding=1),
                                     )

    def forward(self, z):
        z = z.transpose(1, 2)
        return self.decoder(z)

