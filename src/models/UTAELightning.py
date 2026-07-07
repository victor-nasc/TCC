from typing import Any

import torch
import torch.nn as nn
import wandb

from .BaseModel import BaseModel
from .utae_paps_models.utae import UTAE


def _enable_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


class UTAELightning(BaseModel):
    """_summary_ U-Net architecture with temporal attention in the bottleneck and skip connections.
    """
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        mc_dropout_samples: int = 0,
        mc_dropout_rate: float = 0.0,
        *args: Any,
        **kwargs: Any
    ):
        # use_doy is saved in hparams by BaseModel; drop it if it comes back in
        # through load_from_checkpoint to avoid a duplicate keyword argument.
        kwargs.pop("use_doy", None)
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=True, # UTAE uses the day of the year as an input feature
            *args,
            **kwargs
        )

        self.mc_dropout_samples = mc_dropout_samples
        self._mc_variance_sample = None

        self.model = UTAE(
            input_dim=n_channels,
            dropout=mc_dropout_rate,
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

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        return self.model(x, batch_positions=doys, return_att=False)

    def get_pred_and_gt(self, batch):
        if self.mc_dropout_samples == 0 or not self.trainer.testing:
            return super().get_pred_and_gt(batch)

        x, y, doys = batch

        _enable_dropout(self.model)

        with torch.no_grad():
            samples = torch.stack([
                self.model(x, batch_positions=doys).squeeze(1)
                for _ in range(self.mc_dropout_samples)
            ])  # [S, B, H, W] — raw logits per sample

        self.model.eval()  # restore dropout layers to eval mode

        # Average in probability space (principled), return as logits for
        # compatibility with BaseModel's loss and torchmetrics calls.
        probs = torch.sigmoid(samples)                # [S, B, H, W]
        mean_prob = probs.mean(dim=0)                 # [B, H, W]
        variance = probs.var(dim=0)                   # [B, H, W] — epistemic uncertainty

        self.log("test_mc_uncertainty_mean", variance.mean(), sync_dist=True)
        self.log("test_mc_uncertainty_max", variance.max(), sync_dist=True)
        self.log("test_mc_uncertainty_std", variance.std(), sync_dist=True)

        self.update_uncertainty_metrics_from_samples(probs, y)

        # Keep one example map for image logging at epoch end
        if self._mc_variance_sample is None:
            self._mc_variance_sample = variance[0].detach().cpu()

        mean_logit = torch.logit(mean_prob.clamp(1e-6, 1 - 1e-6))
        return mean_logit, y

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        if self._mc_variance_sample is None:
            return

        v = self._mc_variance_sample  # [H, W]
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)  # [0, 1] for display

        # TensorBoard: add_image expects [C, H, W]
        for logger in self.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "add_image"):
                logger.experiment.add_image(
                    "test/mc_variance_map", v_norm.unsqueeze(0), self.current_epoch
                )

        # WandB (consistent with BaseModel.on_test_epoch_end)
        wandb.log({"test/mc_variance_map": wandb.Image(v_norm.numpy())})

        self._mc_variance_sample = None
