"""
Conditional Flow Matching model for wildfire spread prediction.

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
(conditional OT probability paths). Generative segmentation in the spirit of
DDPM-based approaches (e.g. Amit et al., SegDiff), but trained by regressing a
velocity field instead of denoising, and sampled with a plain Euler ODE solver.

Architecture:
  - UTAE backbone (out_conv=[32]) encodes the multitemporal input x into a
    32-channel conditioning feature map (computed once per image).
  - A small time-conditioned convolutional ResNet (velocity net) takes the
    current state y_t (1 channel) concatenated with the conditioning features
    and predicts the velocity v_theta(y_t, t | x).

Training (OT flow matching): y1 = mask in {-1,+1}, y0 ~ N(0,I),
  y_t = (1-t) y0 + t y1,   loss = || v_theta(y_t, t | x) - (y1 - y0) ||^2,
optionally with fire pixels up-weighted by pos_class_weight (class imbalance).

Validation integrates a single ODE trajectory from fixed-seed noise
(deterministic val_loss for checkpointing). At test time, fm_samples
trajectories are integrated in parallel with fm_ode_steps Euler steps; the
resulting segmentation maps are averaged and their variance is reported as
aleatoric uncertainty.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import wandb

from .BaseModel import BaseModel
from .utae_paps_models.utae import UTAE


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1))
        args = t[:, None] * freqs[None, :] * 1000  # t in [0,1] → scale like timesteps
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResBlock(nn.Module):
    """3x3 conv residual block with per-channel scale/shift from the time embedding."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.film = nn.Linear(time_dim, 2 * channels)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(t_emb)[:, :, None, None].chunk(2, dim=1)
        out = self.conv1(self.act(self.norm1(h)))
        out = out * (1 + scale) + shift
        out = self.conv2(self.act(self.norm2(out)))
        return h + out


class VelocityNet(nn.Module):
    """Time-conditioned conv ResNet predicting the flow velocity field."""

    def __init__(self, cond_channels: int, hidden_channels: int = 64,
                 n_blocks: int = 4, time_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.conv_in = nn.Conv2d(1 + cond_channels, hidden_channels,
                                 kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            FiLMResBlock(hidden_channels, time_dim) for _ in range(n_blocks)])
        self.conv_out = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        # Near-zero (but not exactly zero) init: the velocity starts ~0 without
        # blocking gradient flow to the conditioning backbone at step one.
        nn.init.normal_(self.conv_out.weight, std=1e-4)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, y_t: torch.Tensor, cond: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.conv_in(torch.cat([y_t, cond], dim=1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.conv_out(h)


class FlowMatchingLightning(BaseModel):
    """UTAE-conditioned flow matching over next-day fire masks (Lipman et al., 2023)."""

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        fm_samples: int = 16,
        fm_ode_steps: int = 20,
        fm_hidden_channels: int = 64,
        fm_n_blocks: int = 4,
        fm_time_dim: int = 128,
        fm_pos_weight_loss: bool = True,
        fm_sample_chunk: int = 4,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            n_channels: input channels per time step.
            flatten_temporal_dimension: unused (always False for UTAE).
            pos_class_weight: positive class weight; also up-weights fire pixels
                in the flow matching loss if fm_pos_weight_loss is set.
            fm_samples: ODE trajectories sampled at test time.
            fm_ode_steps: Euler integration steps per trajectory.
            fm_hidden_channels: width of the velocity net.
            fm_n_blocks: number of FiLM residual blocks in the velocity net.
            fm_time_dim: dimensionality of the time embedding.
            fm_pos_weight_loss: up-weight fire pixels in the velocity regression
                loss by pos_class_weight (normalized to keep the loss scale).
            fm_sample_chunk: max trajectories integrated in parallel at test
                time (bounds GPU memory on large test images).

        Note: the flow matching training loss is a velocity regression (MSE);
        the loss_function hyperparameter only affects val/test loss logging.
        """
        # use_doy is saved in hparams by BaseModel; drop it if it comes back in
        # through load_from_checkpoint to avoid a duplicate keyword argument.
        kwargs.pop("use_doy", None)
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=False,
            pos_class_weight=pos_class_weight,
            use_doy=True,
            *args,
            **kwargs,
        )
        self.fm_samples = fm_samples
        self.fm_ode_steps = fm_ode_steps
        self.fm_pos_weight_loss = fm_pos_weight_loss
        self.fm_sample_chunk = max(1, fm_sample_chunk)

        # out_conv=[32]: the backbone returns a 32-channel conditioning feature
        # map; the segmentation is generated by integrating the velocity field.
        self.model = UTAE(
            input_dim=n_channels,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=4,
            encoder=False,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
        )

        self.velocity_net = VelocityNet(
            cond_channels=32,
            hidden_channels=fm_hidden_channels,
            n_blocks=fm_n_blocks,
            time_dim=fm_time_dim,
        )

        self._fm_variance_sample = None

    # ------------------------------------------------------------------
    # Forward / sampling
    # ------------------------------------------------------------------

    def _backbone_features(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        return self.model(x, batch_positions=doys, return_att=False)  # [B, 32, H, W]

    def _sample_masks(self, features: torch.Tensor, n_samples: int,
                      generator: torch.Generator = None) -> torch.Tensor:
        """Integrate n_samples Euler ODE trajectories, chunked to bound memory.

        Trajectories are integrated in parallel within chunks of at most
        fm_sample_chunk; test images can be large (batch_size=1 but multi-
        megapixel), so integrating all samples at once can exhaust GPU memory
        on the velocity net's activations.

        Returns probability maps in [0, 1] of shape [n_samples, B, H, W]:
        the terminal states y(1) live near {-1, +1} and are mapped affinely.
        """
        B, C, H, W = features.shape
        dt = 1.0 / self.fm_ode_steps

        chunks = []
        remaining = n_samples
        while remaining > 0:
            S = min(remaining, self.fm_sample_chunk)
            cond = features.unsqueeze(0).expand(S, -1, -1, -1, -1) \
                .reshape(S * B, C, H, W)

            y = torch.empty(S * B, 1, H, W, device=features.device)
            if generator is not None:
                y.normal_(generator=generator)
            else:
                y.normal_()

            for i in range(self.fm_ode_steps):
                t = torch.full((S * B,), i * dt, device=features.device)
                y = y + dt * self.velocity_net(y, cond, t)

            chunks.append(((y + 1) / 2).clamp(0, 1).view(S, B, H, W))
            remaining -= S

        return torch.cat(chunks, dim=0)

    def forward(self, x: torch.Tensor, doys: torch.Tensor = None) -> torch.Tensor:
        """Single trajectory from fixed-seed noise (deterministic, used for val)."""
        features = self._backbone_features(x, doys)
        generator = torch.Generator(device=features.device).manual_seed(0)
        probs = self._sample_masks(features, 1, generator=generator)  # [1, B, H, W]
        return torch.logit(probs[0].clamp(1e-6, 1 - 1e-6)).unsqueeze(1)

    def get_pred_and_gt(self, batch):
        x, y, doys = batch

        if self.trainer.training:
            # Flow matching loss; the returned y_hat is only a byproduct for
            # interface compatibility (train F1 is not meaningful/logged here).
            features = self._backbone_features(x, doys)
            y1 = (2 * y - 1).float().unsqueeze(1)              # masks in {-1, +1}
            y0 = torch.randn_like(y1)
            t = torch.rand(y1.shape[0], device=y1.device)
            t_b = t[:, None, None, None]
            y_t = (1 - t_b) * y0 + t_b * y1
            v_pred = self.velocity_net(y_t, features, t)
            v_target = y1 - y0

            sq_err = (v_pred - v_target) ** 2
            if self.fm_pos_weight_loss and self.hparams.pos_class_weight > 1:
                w = 1 + (self.hparams.pos_class_weight - 1) * y.float().unsqueeze(1)
                self._fm_loss = (w * sq_err).sum() / w.sum()
            else:
                self._fm_loss = sq_err.mean()
            return v_pred.squeeze(1), y

        if not self.trainer.testing or self.fm_samples <= 1:
            # Validation / single-sample test: one fixed-seed trajectory
            y_hat = self(x, doys).squeeze(1)
            return y_hat, y

        # Test: integrate fm_samples trajectories (batched in parallel)
        with torch.no_grad():
            features = self._backbone_features(x, doys)
            probs = self._sample_masks(features, self.fm_samples)  # [S, B, H, W]

        mean_prob = probs.mean(dim=0)           # [B, H, W]
        variance = probs.var(dim=0)             # [B, H, W] — aleatoric uncertainty

        self.log("test_fm_uncertainty_mean", variance.mean(), sync_dist=True)
        self.log("test_fm_uncertainty_max", variance.max(), sync_dist=True,
                 reduce_fx="max")
        self.log("test_fm_uncertainty_std", variance.std(), sync_dist=True)

        self.update_uncertainty_metrics_from_samples(probs, y)

        if self._fm_variance_sample is None:
            self._fm_variance_sample = variance[0].detach().cpu()

        mean_logit = torch.logit(mean_prob.clamp(1e-6, 1 - 1e-6))
        return mean_logit, y

    # ------------------------------------------------------------------
    # Training step (velocity regression loss)
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        _, _ = self.get_pred_and_gt(batch)

        loss = self._fm_loss
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Calibration offset
    # ------------------------------------------------------------------

    def pos_weight_logit_offset(self) -> float:
        # The generative model is not trained with weighted BCE, so the logit
        # shift correction of BaseModel does not apply.
        return 0.0

    # ------------------------------------------------------------------
    # Epoch end — log uncertainty map to wandb
    # ------------------------------------------------------------------

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        if self._fm_variance_sample is None:
            return

        v = self._fm_variance_sample
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)

        for logger in self.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
                logger.experiment.add_image(
                    "test/fm_variance_map", v_norm.unsqueeze(0), self.current_epoch
                )

        wandb.log({"test/fm_variance_map": wandb.Image(v_norm.numpy())})
        self._fm_variance_sample = None
