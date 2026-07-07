from typing import Any, List

import torch
import wandb

from .BaseModel import BaseModel
from .UTAELightning import UTAELightning


class DeepEnsembleLightning(BaseModel):
    """Ensemble of independently trained UTAELightning models.

    At test time, averages sigmoid probabilities across all members.
    Training is a no-op (members must be pre-trained).
    """

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        ensemble_ckpt_paths: List[str],
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=True,
            *args,
            **kwargs,
        )
        self.ensemble_ckpt_paths = ensemble_ckpt_paths
        # Loaded lazily on first test step so device is known
        self._members: List[UTAELightning] = []
        self._variance_sample = None

    def _load_members(self):
        # This module has no own parameters, so self.device stays "cpu";
        # use the trainer's root device instead.
        device = self.trainer.strategy.root_device if self.trainer else self.device
        for path in self.ensemble_ckpt_paths:
            m = UTAELightning.load_from_checkpoint(path, map_location=device)
            m.to(device)
            m.eval()
            self._members.append(m)

    def forward(self, x, doys=None):
        # Required by BaseModel but never called during ensemble test
        raise NotImplementedError("Call get_pred_and_gt directly.")

    def get_pred_and_gt(self, batch):
        if not self.trainer.testing:
            raise RuntimeError("DeepEnsembleLightning is test-only.")

        if not self._members:
            self._load_members()

        x, y, doys = batch

        with torch.no_grad():
            probs = torch.stack([
                torch.sigmoid(m(x, doys).squeeze(1))
                for m in self._members
            ])  # [N, B, H, W]

        mean_prob = probs.mean(dim=0)       # [B, H, W]
        variance = probs.var(dim=0)         # [B, H, W] — epistemic uncertainty

        self.log("test_ensemble_uncertainty_mean", variance.mean(), sync_dist=True)
        self.log("test_ensemble_uncertainty_max", variance.max(), sync_dist=True)

        self.update_uncertainty_metrics_from_samples(probs, y)

        if self._variance_sample is None:
            self._variance_sample = variance[0].detach().cpu()

        mean_logit = torch.logit(mean_prob.clamp(1e-6, 1 - 1e-6))
        return mean_logit, y

    def on_test_epoch_end(self):
        super().on_test_epoch_end()

        if self._variance_sample is None:
            return

        v = self._variance_sample
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)
        wandb.log({"test/ensemble_variance_map": wandb.Image(v_norm.numpy())})
        self._variance_sample = None
