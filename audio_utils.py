import torch as t
import numpy as np

def stft(sig, n_fft, hop_length, win_length):
    return t.stft(sig, n_fft, hop_length, win_length=win_length, window=t.hann_window(win_length, device=sig.device))

def spec(x, n_fft, hop_length, win_length):
    return t.norm(stft(x, n_fft, hop_length, win_length), p=2, dim=-1)

def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()

def squeeze(x):
    if len(x.shape) == 3:
        assert x.shape[-1] in [1, 2]
        x = t.mean(x, -1)
    if len(x.shape) != 2:
        raise ValueError(f'Unknown input shape {x.shape}')
    return x

def calculate_bandwidth(dataset, sr=16000, n_fft=2048, hop_length=160, win_length=400, duration=600):
    n_samples = int(sr * duration)
    l1, total, total_sq, n_seen, idx = 0.0, 0.0, 0.0, 0.0, 0
    spec_norm_total, spec_nelem = 0.0, 0.0
    rand_idx = np.random.permutation(len(dataset))
    while n_seen < n_samples:
        samples = dataset[rand_idx[idx]]
        spec_norm = t.linalg.norm(spec(samples, n_fft, hop_length, win_length))
        spec_norm_total += spec_norm
        spec_nelem += 1
        n_seen += int(np.prod(samples.shape))
        l1 += t.sum(t.abs(samples))
        total += t.sum(samples)
        total_sq += t.sum(samples ** 2)
        idx += 1

    mean = total / n_seen
    bandwidth = dict(l2=total_sq / n_seen - mean ** 2,
                     l1=l1 / n_seen,
                     spec=spec_norm_total / spec_nelem)
    return bandwidth

def spectral_loss(x_in, x_out, n_fft=2048, hop_length=160, win_length=400):
    spec_in = spec(squeeze(x_in.float()), n_fft, hop_length, win_length)
    spec_out = spec(squeeze(x_out.float()), n_fft, hop_length, win_length)
    return norm(spec_in - spec_out)

def multispectral_loss(x_in, x_out, n_fft=(2048,1024,512), hop_length=(160,80,40), win_length=(400,200,100)):
    losses = []
    assert len(n_fft) == len(hop_length) == len(win_length)
    args = [n_fft,
            hop_length,
            win_length]
    for n_fft, hop_length, win_length in zip(*args):
        spec_in = spec(squeeze(x_in.float()), n_fft, hop_length, win_length)
        spec_out = spec(squeeze(x_out.float()), n_fft, hop_length, win_length)
        # spec_in = spec(x_in.squeeze(1), n_fft, hop_length, win_length)
        # spec_out = spec(x_out.squeeze(1), n_fft, hop_length, win_length)
        losses.append(norm(spec_in - spec_out))
    return sum(losses) / len(losses)

def spectral_convergence(x_in, x_out, n_fft=2048, hop_length=160, win_length=400, epsilon=2e-3):
    spec_in = spec(squeeze(x_in.float()), n_fft, hop_length, win_length)
    spec_out = spec(squeeze(x_out.float()), n_fft, hop_length, win_length)

    gt_norm = norm(spec_in)
    residual_norm = norm(spec_in - spec_out)
    mask = (gt_norm > epsilon).float()
    return (residual_norm * mask) / t.clamp(gt_norm, min=epsilon)