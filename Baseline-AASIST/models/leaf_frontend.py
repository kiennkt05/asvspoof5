import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def differentiable_ema(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # Fast GPU-friendly EMA using depthwise 1D Convolution
    # x: (B, C, T)
    # s: (1, C)
    B, C, T = x.shape
    
    # K determines how far back the EMA remembers. 
    # Since s >= 0.01, (1-0.01)^1024 is practically zero.
    K = min(T, 1024) 
    
    # Construct exponential decay kernel: w[tau] = s * (1-s)^tau
    k = torch.arange(K, dtype=x.dtype, device=x.device).view(1, 1, K)
    s_view = s.view(1, C, 1)
    
    w = s_view * (1.0 - s_view)**k
    
    # Flip for cross-correlation in F.conv1d
    w = torch.flip(w, dims=[-1]).view(C, 1, K)
    
    # Causal padding
    x_padded = F.pad(x, (K - 1, 0))
    out = F.conv1d(x_padded, w, groups=C)
    
    # Add initial state correction (1-s)^(t+1) * x_0 to match exact sequential EMA
    out[:, :, :K] += ((1.0 - s_view)**(k + 1)) * x[:, :, 0:1]
    
    return out

class sPCEN(nn.Module):
    def __init__(self, num_filters, alpha=0.96, smooth_coef=0.04, delta=2.0, root=2.0, floor=1e-6, trainable=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(num_filters).fill_(alpha), requires_grad=trainable)
        self.delta = nn.Parameter(torch.empty(num_filters).fill_(delta), requires_grad=trainable)
        self.root = nn.Parameter(torch.empty(num_filters).fill_(root), requires_grad=trainable)
        self.s = nn.Parameter(torch.empty(num_filters).fill_(smooth_coef), requires_grad=trainable)
        self.floor = floor

    def forward(self, x):
        # x shape: (B, C, T)
        # Clamp s >= 0.01 to ensure the EMA decays to zero within K=1024 steps
        s = torch.clamp(self.s, min=0.01, max=0.99).view(1, -1)
        alpha = torch.clamp(self.alpha, min=0.0, max=1.0).view(1, -1, 1)
        root = torch.clamp(self.root, min=1.0).view(1, -1, 1)
        delta = torch.clamp(self.delta, min=0.0).view(1, -1, 1)

        ema = differentiable_ema(x, s)
        
        one_over_root = 1.0 / root
        out = ((x / (self.floor + ema)**alpha + delta)**one_over_root) - delta**one_over_root
        return out

class GaborConv1D(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate

        # 1. Initialize with Mel scale
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = 2595 * np.log10(1 + f / 700)
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 1)
        filbandwidthsf = 700 * (10**(filbandwidthsmel / 2595) - 1)
        
        # Normalized center frequencies [0, 0.5]
        init_freqs = filbandwidthsf[:-1] / self.sample_rate
        # Initial bandwidths
        init_bw = (filbandwidthsf[1:] - filbandwidthsf[:-1]) / self.sample_rate

        sigma_min = 4 * math.sqrt(2 * math.log(2)) / self.kernel_size
        
        # Initialize center_freqs with logit (inverse of sigmoid)
        init_freqs_tensor = torch.tensor(init_freqs, dtype=torch.float32)
        init_freqs_scaled = torch.clamp(init_freqs_tensor / 0.5, min=1e-4, max=0.9999)
        self.center_freqs = nn.Parameter(torch.logit(init_freqs_scaled))

        # Initialize bandwidths with inverse softplus
        # inverse_softplus(y) = log(exp(y) - 1). Use expm1 for numerical stability.
        init_bw_tensor = torch.tensor(init_bw, dtype=torch.float32)
        target_softplus_val = torch.clamp(init_bw_tensor - sigma_min, min=1e-5)
        self.bandwidths = nn.Parameter(torch.log(torch.expm1(target_softplus_val)))

    def get_filters(self):
        # Constrain center frequencies to [0, 0.5]
        freqs = 0.5 * torch.sigmoid(self.center_freqs)
        
        # Frequency bandwidth B_norm.
        sigma_min = 4.0 * math.sqrt(2.0 * math.log(2.0)) / self.kernel_size
        B_norm = F.softplus(self.bandwidths) + sigma_min
        
        # FIX for AASIST: Frequency bandwidth (B_norm) is inversely proportional to
        # temporal standard deviation (sigma_t) via sigma_t = 1 / (2*pi*B_norm).
        # The original AASIST used B_norm directly as sigma_t, resulting in sigma_t = 0.003,
        # creating a Delta function instead of a Gabor filter!
        sigmas = 1.0 / (2.0 * math.pi * B_norm)
        
        # Ensure sigma_t >= 2.0 (filter doesn't collapse to 1 sample)
        # and not too wide (sigma_t <= kernel_size / 2)
        sigmas = torch.clamp(sigmas, min=2.0, max=self.kernel_size / 2.0)

        t = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1, device=freqs.device)
        t = t.view(1, -1)
        freqs = freqs.view(-1, 1)
        sigmas = sigmas.view(-1, 1)

        # Gabor = Gaussian * Sinusoid
        gaussian = (1 / (math.sqrt(2 * math.pi) * sigmas)) * torch.exp(-0.5 * (t / sigmas)**2)
        sinusoid_real = torch.cos(2 * math.pi * freqs * t)
        sinusoid_imag = torch.sin(2 * math.pi * freqs * t)

        real_filters = gaussian * sinusoid_real
        imag_filters = gaussian * sinusoid_imag
        
        filters = torch.cat([real_filters, imag_filters], dim=0).unsqueeze(1)
        return filters

    def forward(self, x):
        filters = self.get_filters()
        out = F.conv1d(x, filters, stride=1, padding=self.kernel_size//2)
        out_real, out_imag = torch.chunk(out, 2, dim=1)
        out_complex = torch.cat([out_real, out_imag], dim=1)
        return out_complex
