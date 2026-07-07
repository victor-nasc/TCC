"""
Probabilistic U-Net for wildfire spread prediction.

Reference: Kohl et al., "A Probabilistic U-Net for Segmentation of Ambiguous
Images", NeurIPS 2018.

Architecture: UTAE backbone (out_conv=[32], so it yields a 32-channel feature
map instead of logits) combined with a low-dimensional axis-aligned Gaussian
latent space:

  - prior net      q(z | x)      — conv encoder on the temporally-flattened input
  - posterior net  q(z | x, y)   — same encoder, input additionally contains y
  - f_comb         — 1x1-conv head that fuses the UTAE feature map with a
                     broadcast latent sample z into the final logit map

Training loss (ELBO) = reconstruction_loss(z ~ posterior) + beta * KL(post || prior),
with the KL divided by the pixels per image because the reconstruction loss is
mean-reduced (keeps beta on the scale of Kohl et al.'s summed-CE formulation).

Validation uses the prior mean (deterministic) for a stable val_loss.
At test time, prob_unet_samples latents are drawn from the prior, the resulting
probability maps are averaged, and their variance is reported as aleatoric
uncertainty.
"""

from typing import Any, List

import torch
import torch.nn as nn
import wandb
from torch.distributions import Independent, Normal, kl_divergence

from .BaseModel import BaseModel
from .utae_paps_models.utae import UTAE


class AxisAlignedConvGaussian(nn.Module):
    """Conv encoder mapping an image to an axis-aligned Gaussian over z."""

    def __init__(self, in_channels: int, latent_dim: int, num_filters: List[int]):
        super().__init__()
        layers = []
        prev = in_channels
        for n_filters in num_filters:
            layers += [
                nn.Conv2d(prev, n_filters, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
            ]
            prev = n_filters
        self.encoder = nn.Sequential(*layers)
        self.mu_log_sigma = nn.Conv2d(prev, 2 * latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Independent:
        feats = self.encoder(x).mean(dim=(2, 3), keepdim=True)  # global avg pool
        mu_log_sigma = self.mu_log_sigma(feats).squeeze(-1).squeeze(-1)
        mu, log_sigma = mu_log_sigma.chunk(2, dim=1)
        log_sigma = log_sigma.clamp(-10, 10)  # numerical stability
        return Independent(Normal(mu, torch.exp(log_sigma)), 1)


class Fcomb(nn.Module):
    """Fuse the backbone feature map with a broadcast latent sample via 1x1 convs."""

    def __init__(self, feature_channels: int, latent_dim: int,
                 hidden_channels: int = 32, n_layers: int = 3):
        super().__init__()
        layers = []
        prev = feature_channels + latent_dim
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(prev, hidden_channels, kernel_size=1),
                       nn.ReLU(inplace=True)]
            prev = hidden_channels
        layers.append(nn.Conv2d(prev, 1, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z[:, :, None, None].expand(-1, -1, features.shape[-2], features.shape[-1])
        return self.layers(torch.cat([features, z], dim=1))


class ProbUNetLightning(BaseModel):
    """UTAE backbone + Probabilistic U-Net latent space (Kohl et al., 2018)."""

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        n_leading_observations: int = 5,
        latent_dim: int = 6,
        beta: float = 10.0,
        beta_anneal_epochs: int = 0,
        prob_unet_samples: int = 16,
        prior_filters: List[int] = (32, 64, 128, 192),
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            n_channels: input channels per time step.
            flatten_temporal_dimension: unused (always False for UTAE).
            pos_class_weight: positive class weight for loss.
            n_leading_observations: number of input time steps T. The prior and
                posterior encoders operate on the temporally-flattened input
                (T * n_channels channels), so this must match the data config.
                train.py links it automatically from --data.n_leading_observations.
            latent_dim: dimensionality of the Gaussian latent space (paper: 6).
            beta: weight of the KL(posterior || prior) term in the ELBO.
            beta_anneal_epochs: linearly ramp beta from 0 → beta over this many
                epochs (0 = no annealing).
            prob_unet_samples: number of prior samples at test time.
            prior_filters: channel widths of the prior/posterior encoder stages.
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
        self.latent_dim = latent_dim
        self.beta = beta
        self.beta_anneal_epochs = beta_anneal_epochs
        self.prob_unet_samples = prob_unet_samples

        # out_conv=[32]: the backbone returns a 32-channel feature map; the
        # final logit is produced by f_comb after injecting the latent sample.
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

        flat_channels = n_leading_observations * n_channels
        self.prior_net = AxisAlignedConvGaussian(
            flat_channels, latent_dim, list(prior_filters))
        self.posterior_net = AxisAlignedConvGaussian(
            flat_channels + 1, latent_dim, list(prior_filters))
        self.f_comb = Fcomb(feature_channels=32, latent_dim=latent_dim)

        self._prob_variance_sample = None

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def _backbone_features(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        return self.model(x, batch_positions=doys, return_att=False)  # [B, 32, H, W]

    def forward(self, x: torch.Tensor, doys: torch.Tensor = None) -> torch.Tensor:
        """Deterministic prediction using the prior mean (used for val)."""
        features = self._backbone_features(x, doys)
        prior = self.prior_net(x.flatten(start_dim=1, end_dim=2))
        return self.f_comb(features, prior.mean)

    def get_pred_and_gt(self, batch):
        x, y, doys = batch

        if self.trainer.training:
            # ELBO forward: decode with a posterior sample, stash KL for training_step
            x_flat = x.flatten(start_dim=1, end_dim=2)
            prior = self.prior_net(x_flat)
            posterior = self.posterior_net(
                torch.cat([x_flat, y.unsqueeze(1).float()], dim=1))
            z = posterior.rsample()
            features = self._backbone_features(x, doys)
            y_hat = self.f_comb(features, z).squeeze(1)
            self._kl = kl_divergence(posterior, prior).mean()
            return y_hat, y

        if not self.trainer.testing or self.prob_unet_samples <= 1:
            # Validation / single-sample test: prior mean (deterministic)
            y_hat = self(x, doys).squeeze(1)
            return y_hat, y

        # Test: sample prob_unet_samples latents from the prior; the backbone
        # feature map is computed once and reused across samples.
        with torch.no_grad():
            features = self._backbone_features(x, doys)
            prior = self.prior_net(x.flatten(start_dim=1, end_dim=2))
            samples = torch.stack([
                self.f_comb(features, prior.sample()).squeeze(1)
                for _ in range(self.prob_unet_samples)
            ])  # [S, B, H, W] logits

        probs = torch.sigmoid(samples)          # [S, B, H, W]
        mean_prob = probs.mean(dim=0)           # [B, H, W]
        variance = probs.var(dim=0)             # [B, H, W] — aleatoric uncertainty

        self.log("test_prob_unet_uncertainty_mean", variance.mean(), sync_dist=True)
        self.log("test_prob_unet_uncertainty_max", variance.max(), sync_dist=True,
                 reduce_fx="max")
        self.log("test_prob_unet_uncertainty_std", variance.std(), sync_dist=True)

        self.update_uncertainty_metrics_from_samples(probs, y)

        if self._prob_variance_sample is None:
            self._prob_variance_sample = variance[0].detach().cpu()

        mean_logit = torch.logit(mean_prob.clamp(1e-6, 1 - 1e-6))
        return mean_logit, y

    # ------------------------------------------------------------------
    # Training step (ELBO: reconstruction + beta * KL)
    # ------------------------------------------------------------------

    def _effective_beta(self) -> float:
        if self.beta_anneal_epochs > 0:
            return self.beta * min(1.0, self.current_epoch / self.beta_anneal_epochs)
        return self.beta

    def training_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)

        recon_loss = self.compute_loss(y_hat, y)
        kl = self._kl
        beta = self._effective_beta()
        # Kohl et al. weight the KL against the *summed* per-image CE; our
        # reconstruction loss is the per-pixel mean, so the KL is divided by
        # the pixels per image to keep beta on the paper's scale.
        loss = recon_loss + beta * kl / y[0].numel()

        self.train_f1(y_hat, y)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_recon_loss", recon_loss.item(), on_step=False, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log("train_kl", kl.item(), on_step=False, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log("train_beta", beta, on_step=False, on_epoch=True, logger=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    # ------------------------------------------------------------------
    # Epoch end — log uncertainty map to wandb
    # ------------------------------------------------------------------

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        if self._prob_variance_sample is None:
            return

        v = self._prob_variance_sample
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)

        for logger in self.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
                logger.experiment.add_image(
                    "test/prob_unet_variance_map", v_norm.unsqueeze(0), self.current_epoch
                )

        wandb.log({"test/prob_unet_variance_map": wandb.Image(v_norm.numpy())})
        self._prob_variance_sample = None
