"""
Bayesian layers for Bayes by Backpropagation (Blundell et al., 2015).

Uses the local reparameterization trick (Kingma et al., 2015):
  instead of sampling weights w ~ q(w|μ,σ) and computing conv(x, w),
  we compute mean and variance of the output analytically:
    μ_out = conv(x, μ_w)
    σ²_out = conv(x², σ²_w)
  then sample:  out = μ_out + sqrt(σ²_out + ε) * N(0,1)

This reduces gradient variance vs weight-space sampling and avoids storing
sampled weight tensors (saves memory for large conv kernels).

KL divergence uses a scale mixture prior p(w) = π·N(0,σ₁²) + (1-π)·N(0,σ₂²)
(Blundell et al., 2015 §3.3). KL is estimated with a single MC sample per
forward call, which is unbiased and sufficient with SGD.

Only Conv2d and ConvTranspose2d are converted to Bayesian variants.
The LTAE's Conv1d and Linear layers remain deterministic (temporal attention
mechanism; converting them adds instability without changing the dominant
parameter mass).
"""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Prior helpers
# ---------------------------------------------------------------------------

def _log_gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return -0.5 * math.log(2 * math.pi) - sigma.log() - 0.5 * ((x - mu) / sigma) ** 2


def _log_scale_mixture(x: torch.Tensor, sigma1: float, sigma2: float, pi: float) -> torch.Tensor:
    """log p(x) under scale-mixture Gaussian: π·N(0,σ₁²) + (1-π)·N(0,σ₂²)."""
    log_p1 = math.log(pi) - 0.5 * math.log(2 * math.pi) - math.log(sigma1) - 0.5 * (x / sigma1) ** 2
    log_p2 = math.log(1 - pi) - 0.5 * math.log(2 * math.pi) - math.log(sigma2) - 0.5 * (x / sigma2) ** 2
    return torch.logaddexp(log_p1, log_p2)


# ---------------------------------------------------------------------------
# Bayesian Conv2d
# ---------------------------------------------------------------------------

class BayesConv2d(nn.Module):
    """Bayesian Conv2d (local reparameterization trick, scale-mixture prior)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        prior_sigma1: float = 1.0,
        prior_sigma2: float = 0.002,
        prior_pi: float = 0.5,
        init_rho: float = -3.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.prior_pi = prior_pi

        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight_mu = nn.Parameter(torch.empty(weight_shape))
        self.weight_rho = nn.Parameter(torch.full(weight_shape, init_rho))
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_rho = nn.Parameter(torch.full((out_channels,), init_rho))

        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @property
    def weight_sigma(self) -> torch.Tensor:
        return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self) -> torch.Tensor:
        return F.softplus(self.bias_rho)

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        """Apply spatial padding; return (padded_x, effective_padding_for_conv)."""
        if self.padding_mode == "zeros":
            return x, self.padding
        ph, pw = self.padding
        return F.pad(x, (pw, pw, ph, ph), mode=self.padding_mode), (0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_p, pad = self._pad(x)
        mu_out = F.conv2d(x_p, self.weight_mu, self.bias_mu,
                          self.stride, pad, self.dilation, self.groups)
        if not getattr(self, "sampling", True):
            return mu_out  # deterministic forward with posterior mean weights
        var_out = F.conv2d(x_p ** 2, self.weight_sigma ** 2, self.bias_sigma ** 2,
                           self.stride, pad, self.dilation, self.groups)
        return mu_out + torch.sqrt(var_out + 1e-8) * torch.randn_like(mu_out)

    def kl_divergence(self) -> torch.Tensor:
        """MC estimate of KL(q(w|μ,σ) ‖ scale-mixture prior), one sample."""
        w = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_mu)
        b = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_mu)
        log_q = (_log_gaussian(w, self.weight_mu, self.weight_sigma).sum()
                 + _log_gaussian(b, self.bias_mu, self.bias_sigma).sum())
        log_p = (_log_scale_mixture(w, self.prior_sigma1, self.prior_sigma2, self.prior_pi).sum()
                 + _log_scale_mixture(b, self.prior_sigma1, self.prior_sigma2, self.prior_pi).sum())
        return log_q - log_p


# ---------------------------------------------------------------------------
# Bayesian ConvTranspose2d
# ---------------------------------------------------------------------------

class BayesConvTranspose2d(nn.Module):
    """Bayesian ConvTranspose2d (local reparameterization trick, scale-mixture prior)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        prior_sigma1: float = 1.0,
        prior_sigma2: float = 0.002,
        prior_pi: float = 0.5,
        init_rho: float = -3.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else tuple(output_padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups = groups

        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.prior_pi = prior_pi

        # ConvTranspose2d weight shape: (in_channels, out_channels // groups, kH, kW)
        weight_shape = (in_channels, out_channels // groups, *self.kernel_size)
        self.weight_mu = nn.Parameter(torch.empty(weight_shape))
        self.weight_rho = nn.Parameter(torch.full(weight_shape, init_rho))
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_rho = nn.Parameter(torch.full((out_channels,), init_rho))

        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @property
    def weight_sigma(self) -> torch.Tensor:
        return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self) -> torch.Tensor:
        return F.softplus(self.bias_rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu_out = F.conv_transpose2d(x, self.weight_mu, self.bias_mu,
                                    self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        if not getattr(self, "sampling", True):
            return mu_out  # deterministic forward with posterior mean weights
        var_out = F.conv_transpose2d(x ** 2, self.weight_sigma ** 2, self.bias_sigma ** 2,
                                     self.stride, self.padding, self.output_padding,
                                     self.groups, self.dilation)
        return mu_out + torch.sqrt(var_out + 1e-8) * torch.randn_like(mu_out)

    def kl_divergence(self) -> torch.Tensor:
        w = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_mu)
        b = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_mu)
        log_q = (_log_gaussian(w, self.weight_mu, self.weight_sigma).sum()
                 + _log_gaussian(b, self.bias_mu, self.bias_sigma).sum())
        log_p = (_log_scale_mixture(w, self.prior_sigma1, self.prior_sigma2, self.prior_pi).sum()
                 + _log_scale_mixture(b, self.prior_sigma1, self.prior_sigma2, self.prior_pi).sum())
        return log_q - log_p


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def convert_to_bayesian(
    module: nn.Module,
    prior_sigma1: float = 1.0,
    prior_sigma2: float = 0.002,
    prior_pi: float = 0.5,
    init_rho: float = -3.0,
) -> None:
    """Recursively replace Conv2d and ConvTranspose2d with Bayesian equivalents.

    Modifies the module in-place. BatchNorm, Linear (LTAE attention), and
    Conv1d layers are left deterministic.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, BayesConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                padding_mode=child.padding_mode,
                prior_sigma1=prior_sigma1,
                prior_sigma2=prior_sigma2,
                prior_pi=prior_pi,
                init_rho=init_rho,
            ))
        elif isinstance(child, nn.ConvTranspose2d):
            setattr(module, name, BayesConvTranspose2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                output_padding=child.output_padding,
                dilation=child.dilation,
                groups=child.groups,
                prior_sigma1=prior_sigma1,
                prior_sigma2=prior_sigma2,
                prior_pi=prior_pi,
                init_rho=init_rho,
            ))
        else:
            convert_to_bayesian(child, prior_sigma1, prior_sigma2, prior_pi, init_rho)


def set_bayes_sampling(model: nn.Module, enabled: bool) -> None:
    """Toggle weight sampling in all Bayesian layers.

    When disabled, layers use the posterior mean weights (deterministic forward).
    Used for stable validation-loss checkpointing.
    """
    for m in model.modules():
        if isinstance(m, (BayesConv2d, BayesConvTranspose2d)):
            m.sampling = enabled


def compute_total_kl(model: nn.Module) -> torch.Tensor:
    """Sum KL divergences from all Bayesian layers in the model."""
    kl = None
    for m in model.modules():
        if isinstance(m, (BayesConv2d, BayesConvTranspose2d)):
            layer_kl = m.kl_divergence()
            kl = layer_kl if kl is None else kl + layer_kl
    return kl if kl is not None else torch.tensor(0.0)
