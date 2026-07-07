"""
Bayes by Backpropagation (BbB) model for wildfire spread prediction.

Reference: Blundell et al., "Weight Uncertainty in Neural Networks", ICML 2015.

Architecture: UTAE backbone with all Conv2d / ConvTranspose2d layers replaced
by Bayesian equivalents (BayesConv2d / BayesConvTranspose2d via local
reparameterization trick). The LTAE temporal attention layers (Conv1d, Linear)
remain deterministic.

Training loss = reconstruction_loss + kl_weight * KL_total

At test time, bbb_samples independent forward passes are averaged in
probability space, and their variance is reported as epistemic uncertainty.
"""

from typing import Any

import torch
import wandb

from .BaseModel import BaseModel
from .bayes_layers import compute_total_kl, convert_to_bayesian, set_bayes_sampling
from .utae_paps_models.utae import UTAE


class BBBLightning(BaseModel):
    """UTAE with Bayesian spatial convolutions trained via Bayes by Backpropagation."""

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        bbb_samples: int = 10,
        kl_weight: float = 1.0,
        kl_anneal_epochs: int = 20,
        kl_scale_by_batches: bool = True,
        prior_sigma1: float = 1.0,
        prior_sigma2: float = 0.002,
        prior_pi: float = 0.5,
        init_rho: float = -3.0,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            n_channels: input channels per time step.
            flatten_temporal_dimension: unused (always False for UTAE).
            pos_class_weight: positive class weight for loss.
            bbb_samples: number of stochastic forward passes at test time.
            kl_weight: final KL scaling factor (target after annealing).
                With kl_scale_by_batches=True this is a multiplier on the
                ELBO-correct 1/(num_training_batches * pixels_per_batch)
                scale. 1.0 is the mathematically "pure" ELBO value, but at
                that scale the effective per-batch weight is ~1e-7 and the
                posterior collapses toward deterministic (near-zero
                test-time epistemic variance, worse test_AP). A sweep on
                fold0 found kl_weight=1e4 keeps the posterior meaningfully
                spread (uncertainty scales up, AP improves) without the
                val-threshold instability seen at 1e6 — use that as the
                default and re-tune per-dataset if features/imbalance change.
            kl_anneal_epochs: linearly ramp KL weight from 0 → kl_weight
                over this many epochs (0 = no annealing).
            kl_scale_by_batches: divide the KL term by the number of training
                batches times the pixels per batch (minibatch ELBO, Blundell
                et al. §3.4). The pixel factor is needed because compute_loss
                returns the *mean* per-pixel BCE, not the summed batch NLL;
                without it the KL term outweighs the reconstruction by
                ~batch_size*H*W and the posterior collapses onto the prior.
            prior_sigma1: slab std of the scale-mixture prior.
            prior_sigma2: spike std of the scale-mixture prior.
            prior_pi: mixing weight π for the slab component.
            init_rho: initial ρ for weight_sigma = softplus(ρ) ≈ 0.049.
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
        self.bbb_samples = bbb_samples
        self.kl_weight = kl_weight
        self.kl_anneal_epochs = kl_anneal_epochs
        self.kl_scale_by_batches = kl_scale_by_batches

        self.model = UTAE(
            input_dim=n_channels,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 1],
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

        convert_to_bayesian(
            self.model,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
            prior_pi=prior_pi,
            init_rho=init_rho,
        )

        self._bbb_variance_sample = None

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, doys: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, batch_positions=doys, return_att=False)

    def _effective_kl_weight(self, n_batch_pixels: int) -> float:
        """Linearly anneal KL weight from 0 → kl_weight over kl_anneal_epochs,
        optionally scaled by 1/(num_training_batches * pixels_per_batch).

        The reconstruction loss is the mean per-pixel BCE, i.e. the summed
        batch NLL divided by n_batch_pixels. The minibatch ELBO (Blundell
        et al. §3.4) weights the KL by 1/num_training_batches against the
        *summed* NLL, so against the mean-reduced loss the KL must also be
        divided by n_batch_pixels to stay on the same scale.
        """
        w = self.kl_weight
        if self.kl_anneal_epochs > 0:
            w *= min(1.0, self.current_epoch / self.kl_anneal_epochs)
        if self.kl_scale_by_batches:
            n_batches = self.trainer.num_training_batches
            if n_batches and n_batches != float("inf"):
                w /= n_batches * n_batch_pixels
        return w

    def get_pred_and_gt(self, batch):
        x, y, doys = batch

        if not self.trainer.testing or self.bbb_samples <= 1:
            # Single stochastic forward pass (train / val / single-sample test)
            y_hat = self(x, doys).squeeze(1)
            return y_hat, y

        # Test: ensemble of bbb_samples stochastic forward passes
        with torch.no_grad():
            samples = torch.stack([
                self.model(x, batch_positions=doys).squeeze(1)
                for _ in range(self.bbb_samples)
            ])  # [S, B, H, W] logits

        probs = torch.sigmoid(samples)          # [S, B, H, W]
        mean_prob = probs.mean(dim=0)           # [B, H, W]
        variance = probs.var(dim=0)             # [B, H, W] — epistemic uncertainty

        self.log("test_bbb_uncertainty_mean", variance.mean(), sync_dist=True)
        self.log("test_bbb_uncertainty_max", variance.max(), sync_dist=True,
                 reduce_fx="max")
        self.log("test_bbb_uncertainty_std", variance.std(), sync_dist=True)

        self.update_uncertainty_metrics_from_samples(probs, y)

        if self._bbb_variance_sample is None:
            self._bbb_variance_sample = variance[0].detach().cpu()

        mean_logit = torch.logit(mean_prob.clamp(1e-6, 1 - 1e-6))
        return mean_logit, y

    # ------------------------------------------------------------------
    # Training step (adds KL term to reconstruction loss)
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)

        recon_loss = self.compute_loss(y_hat, y)
        kl = compute_total_kl(self.model)
        w = self._effective_kl_weight(y.numel())
        loss = recon_loss + w * kl

        self.train_f1(y_hat, y)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_recon_loss", recon_loss.item(), on_step=False, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log("train_kl", kl.item(), on_step=False, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log("train_kl_weight", w, on_step=False, on_epoch=True, logger=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    # ------------------------------------------------------------------
    # Validation with posterior mean weights (stable val_loss for checkpointing)
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        set_bayes_sampling(self.model, False)

    def on_validation_epoch_end(self) -> None:
        set_bayes_sampling(self.model, True)

    # ------------------------------------------------------------------
    # Epoch end — log uncertainty map to wandb
    # ------------------------------------------------------------------

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        if self._bbb_variance_sample is None:
            return

        v = self._bbb_variance_sample
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)

        for logger in self.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
                logger.experiment.add_image(
                    "test/bbb_variance_map", v_norm.unsqueeze(0), self.current_epoch
                )

        wandb.log({"test/bbb_variance_map": wandb.Image(v_norm.numpy())})
        self._bbb_variance_sample = None
