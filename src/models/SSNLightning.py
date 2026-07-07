"""
Stochastic Segmentation Network (SSN) for wildfire spread prediction.

Reference: Monteiro et al., "Stochastic Segmentation Networks: Modelling
Spatially Correlated Aleatoric Uncertainty", NeurIPS 2020.

Architecture: UTAE backbone (out_conv=[32], so it yields a 32-channel feature
map instead of logits) with three 1x1-conv heads that parameterize a low-rank
multivariate Gaussian over the per-pixel logit map:

  logits ~ N(mean, cov_factor @ cov_factor^T + diag(cov_diag))

Unlike the Probabilistic U-Net's global latent, the low-rank covariance models
spatially correlated aleatoric uncertainty: each sample is a coherent
segmentation hypothesis, not per-pixel noise.

Training loss = -log E_z[p(y|z)], estimated with ssn_mc_samples reparameterized
logit samples via logsumexp (Monteiro et al., Eq. 6).

Validation uses the distribution mean (deterministic). At test time,
ssn_samples logit maps are drawn, the resulting probability maps are averaged,
and their variance is reported as aleatoric uncertainty.
"""

import math
from typing import Any

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .BaseModel import BaseModel
from .utae_paps_models.utae import UTAE


class SSNLightning(BaseModel):
    """UTAE backbone + low-rank Gaussian logit distribution (Monteiro et al., 2020)."""

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        ssn_rank: int = 10,
        ssn_mc_samples: int = 20,
        ssn_samples: int = 16,
        ssn_diag_eps: float = 1e-5,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            n_channels: input channels per time step.
            flatten_temporal_dimension: unused (always False for UTAE).
            pos_class_weight: positive class weight for the BCE likelihood.
            ssn_rank: rank of the covariance factor (paper: 10).
            ssn_mc_samples: MC samples for the likelihood estimate during
                training (paper: 20).
            ssn_samples: logit-map samples at test time.
            ssn_diag_eps: minimum diagonal variance for numerical stability.

        Note: the SSN likelihood is inherently per-pixel BCE; the loss_function
        hyperparameter only affects BaseModel's val/test loss logging.
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
        self.ssn_rank = ssn_rank
        self.ssn_mc_samples = ssn_mc_samples
        self.ssn_samples = ssn_samples
        self.ssn_diag_eps = ssn_diag_eps

        # out_conv=[32]: the backbone returns a 32-channel feature map; the
        # logit distribution is parameterized by the three heads below.
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

        self.mean_head = nn.Conv2d(32, 1, kernel_size=1)
        self.log_cov_diag_head = nn.Conv2d(32, 1, kernel_size=1)
        self.cov_factor_head = nn.Conv2d(32, ssn_rank, kernel_size=1)

        # Start close to a deterministic model: small diagonal noise, near-zero
        # low-rank factors. Otherwise the huge initial covariance drowns the
        # mean and training stalls.
        nn.init.zeros_(self.log_cov_diag_head.weight)
        nn.init.constant_(self.log_cov_diag_head.bias, -4.0)
        nn.init.normal_(self.cov_factor_head.weight, std=1e-4)
        nn.init.zeros_(self.cov_factor_head.bias)

        self._ssn_variance_sample = None

    # ------------------------------------------------------------------
    # Distribution construction
    # ------------------------------------------------------------------

    def _logit_distribution(self, x: torch.Tensor, doys: torch.Tensor):
        """Low-rank multivariate Gaussian over the flattened logit map.

        Returns (distribution over [B, H*W], (H, W)).
        """
        feats = self.model(x, batch_positions=doys, return_att=False)  # [B, 32, H, W]
        B, _, H, W = feats.shape

        mean = self.mean_head(feats).view(B, H * W)
        cov_diag = self.log_cov_diag_head(feats).clamp(-10, 10).exp() \
            .view(B, H * W) + self.ssn_diag_eps
        cov_factor = self.cov_factor_head(feats).view(B, self.ssn_rank, H * W) \
            .transpose(1, 2)  # [B, H*W, rank]

        try:
            dist = td.LowRankMultivariateNormal(
                loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
        except (RuntimeError, ValueError):
            # Cholesky of the capacitance matrix can fail early in training;
            # fall back to the diagonal Gaussian (Monteiro et al.'s strategy).
            dist = td.Independent(td.Normal(mean, cov_diag.sqrt()), 1)
        return dist, (H, W)

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, doys: torch.Tensor = None) -> torch.Tensor:
        """Deterministic prediction using the distribution mean (used for val)."""
        dist, (H, W) = self._logit_distribution(x, doys)
        return dist.mean.view(-1, 1, H, W)

    def _mc_likelihood_loss(self, dist, y: torch.Tensor) -> torch.Tensor:
        """-log E_z[p(y|z)] via logsumexp over reparameterized logit samples,
        normalized by the number of pixels to keep the scale of plain BCE."""
        samples = dist.rsample((self.ssn_mc_samples,))  # [S, B, H*W]
        target = y.flatten(start_dim=1).float().unsqueeze(0).expand_as(samples)
        pos_weight = torch.tensor(
            self.hparams.pos_class_weight, device=samples.device)
        nll = F.binary_cross_entropy_with_logits(
            samples, target, pos_weight=pos_weight, reduction="none"
        ).sum(dim=-1)  # [S, B] — summed over pixels (joint log-likelihood)
        loglik = torch.logsumexp(-nll, dim=0) - math.log(self.ssn_mc_samples)
        return -loglik.mean() / y[0].numel()

    def get_pred_and_gt(self, batch):
        x, y, doys = batch

        if self.trainer.training:
            dist, (H, W) = self._logit_distribution(x, doys)
            self._ssn_loss = self._mc_likelihood_loss(dist, y)
            # Mean logits for F1 logging only; the loss is the MC likelihood.
            y_hat = dist.mean.view(-1, H, W)
            return y_hat, y

        if not self.trainer.testing or self.ssn_samples <= 1:
            # Validation / single-sample test: distribution mean (deterministic)
            y_hat = self(x, doys).squeeze(1)
            return y_hat, y

        # Test: sample ssn_samples coherent logit maps from the distribution
        with torch.no_grad():
            dist, (H, W) = self._logit_distribution(x, doys)
            samples = dist.sample((self.ssn_samples,)).view(
                self.ssn_samples, -1, H, W)  # [S, B, H, W] logits

        probs = torch.sigmoid(samples)          # [S, B, H, W]
        mean_prob = probs.mean(dim=0)           # [B, H, W]
        variance = probs.var(dim=0)             # [B, H, W] — aleatoric uncertainty

        self.log("test_ssn_uncertainty_mean", variance.mean(), sync_dist=True)
        self.log("test_ssn_uncertainty_max", variance.max(), sync_dist=True,
                 reduce_fx="max")
        self.log("test_ssn_uncertainty_std", variance.std(), sync_dist=True)

        self.update_uncertainty_metrics_from_samples(probs, y)

        if self._ssn_variance_sample is None:
            self._ssn_variance_sample = variance[0].detach().cpu()

        mean_logit = torch.logit(mean_prob.clamp(1e-6, 1 - 1e-6))
        return mean_logit, y

    # ------------------------------------------------------------------
    # Training step (MC likelihood loss)
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)

        loss = self._ssn_loss

        self.train_f1(y_hat, y)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    # ------------------------------------------------------------------
    # Epoch end — log uncertainty map to wandb
    # ------------------------------------------------------------------

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        if self._ssn_variance_sample is None:
            return

        v = self._ssn_variance_sample
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)

        for logger in self.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
                logger.experiment.add_image(
                    "test/ssn_variance_map", v_norm.unsqueeze(0), self.current_epoch
                )

        wandb.log({"test/ssn_variance_map": wandb.Image(v_norm.numpy())})
        self._ssn_variance_sample = None
