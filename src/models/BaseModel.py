import math
import os
import time
from abc import ABC
from typing import Any, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
import matplotlib.pyplot as plt
from segmentation_models_pytorch.losses import (DiceLoss, JaccardLoss,
                                                LovaszLoss)
from torchvision.ops import sigmoid_focal_loss


class AverageCalibrationError(torchmetrics.Metric):
    """ACE: mean |accuracy - confidence| across probability bins.

    With adaptive=True, bins are equal-mass (each bin holds the same number of
    pixels; Nixon et al. 2019) instead of equal-width. Under extreme class
    imbalance nearly all pixels fall into the lowest-confidence bin, so
    equal-width binning mostly measures that one bin. Equal-mass binning needs
    a global sort, so it keeps every pixel (list state); it is only used on
    the fire-buffer subset, which is small enough for that to be cheap.

    Equal-width binning is streamed instead: per-bin running sums are bounded
    state (n_bins scalars), so update() is O(batch) and compute() is O(1) —
    unlike accumulating every test-set pixel (millions of images' worth) and
    re-binning once at epoch end, which was both a GPU-memory and a runtime
    problem over the full test set.
    """

    def __init__(self, n_bins: int = 10, adaptive: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.adaptive = adaptive
        if adaptive:
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("targets", default=[], dist_reduce_fx="cat")
        else:
            self.add_state("bin_count", default=torch.zeros(n_bins), dist_reduce_fx="sum")
            self.add_state("bin_pred_sum", default=torch.zeros(n_bins), dist_reduce_fx="sum")
            self.add_state("bin_target_sum", default=torch.zeros(n_bins), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.adaptive:
            self.preds.append(preds)
            self.targets.append(targets)
            return
        preds = preds.detach().float().flatten()
        targets = targets.detach().float().flatten()
        idx = preds.mul(self.n_bins).long().clamp_(0, self.n_bins - 1)
        self.bin_count += torch.bincount(idx, minlength=self.n_bins).to(self.bin_count.dtype)
        self.bin_pred_sum += torch.bincount(idx, weights=preds, minlength=self.n_bins)
        self.bin_target_sum += torch.bincount(idx, weights=targets, minlength=self.n_bins)

    def compute(self) -> torch.Tensor:
        if self.adaptive:
            preds = torch.cat(self.preds)
            targets = torch.cat(self.targets).float()
            total = torch.tensor(0.0, device=preds.device)
            chunks = [c for c in torch.chunk(torch.argsort(preds), self.n_bins) if c.numel() > 0]
            for c in chunks:
                total += (targets[c].mean() - preds[c].mean()).abs()
            return total / max(len(chunks), 1)
        mask = self.bin_count > 0
        if not mask.any():
            return torch.tensor(0.0, device=self.bin_count.device)
        conf = self.bin_pred_sum[mask] / self.bin_count[mask]
        acc = self.bin_target_sum[mask] / self.bin_count[mask]
        return (acc - conf).abs().mean()


class BinaryCalibrationErrorFast(torchmetrics.Metric):
    """Streaming equal-width-bin ECE (L1 norm, samples weighted by bin mass),
    matching torchmetrics.classification.BinaryCalibrationError's formula.

    torchmetrics' own implementation keeps every pixel ever seen and re-bins
    the whole test set at compute() time, which OOMs and is slow at this
    dataset's per-image resolution. Bounded per-bin running sums make update()
    O(batch) and compute() O(n_bins) instead.
    """

    def __init__(self, n_bins: int = 10, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.add_state("bin_count", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("bin_pred_sum", default=torch.zeros(n_bins), dist_reduce_fx="sum")
        self.add_state("bin_target_sum", default=torch.zeros(n_bins), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.detach().float().flatten()
        targets = targets.detach().float().flatten()
        idx = preds.mul(self.n_bins).long().clamp_(0, self.n_bins - 1)
        self.bin_count += torch.bincount(idx, minlength=self.n_bins).to(self.bin_count.dtype)
        self.bin_pred_sum += torch.bincount(idx, weights=preds, minlength=self.n_bins)
        self.bin_target_sum += torch.bincount(idx, weights=targets, minlength=self.n_bins)

    def _bin_stats(self):
        mask = self.bin_count > 0
        conf = self.bin_pred_sum[mask] / self.bin_count[mask]
        acc = self.bin_target_sum[mask] / self.bin_count[mask]
        weight = self.bin_count[mask] / self.bin_count.sum()
        return conf, acc, weight

    def compute(self) -> torch.Tensor:
        if self.bin_count.sum() == 0:
            return torch.tensor(0.0, device=self.bin_count.device)
        conf, acc, weight = self._bin_stats()
        return (weight * (acc - conf).abs()).sum()

    def plot(self, val=None, ax=None):
        conf, acc, _ = self._bin_stats()
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        ax.plot(conf.cpu().numpy(), acc.cpu().numpy(), "o-", label="model")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.legend()
        return fig, ax


class BaseModel(pl.LightningModule, ABC):
    """_summary_ Base model class for all models in this project. Implements the training, validation and test steps, 
    as well as the loss function. 

    """
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: Literal["BCE", "Focal", "Lovasz", "Jaccard", "Dice"],
        use_doy: bool = False,
        required_img_size: Optional[Tuple[int, int]] = None,
        *args: Any,
        **kwargs: Any
    ):
        """_summary_ 

        Args:
            n_channels (int): _description_ Number of feature channels in the input data. Usually means number of features per time step, 
            except for U-Net which flattens the temporal dimension and uses this parameter as the total number of features. 
            flatten_temporal_dimension (bool): _description_ Whether to flatten the temporal dimension of the input data.
            pos_class_weight (float): _description_ Weight of the positive class in the loss function (only used for BCE and Focal loss).
            loss_function (Literal[&#39;BCE&#39;, &#39;Focal&#39;, &#39;Lovasz&#39;, &#39;Jaccard&#39;, &#39;Dice&#39;]): _description_ Which loss function to use. 
            use_doy (bool, optional): _description_. Whether to use the doy of year (doy) as an additional input feature. Defaults to False.
            required_img_size (Optional[Tuple[int,int]], optional): _description_. Defaults to None. 
            When using a model that requires a specific image size, this parameter can be used to indicate it. We assume models require square images, 
            so this parameter indicates the side length. If set, the forward method will perform repeated inference on crops of the 
            image, and aggregate the results. This also works for non-square images. 
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if required_img_size is not None:
            self.hparams.required_img_size = torch.Size(
                required_img_size, device=self.device
            )

        # Normalize class weights by assuming that the negative class has weight 1
        if self.hparams.loss_function == "Focal" and self.hparams.pos_class_weight > 1:
            self.hparams.pos_class_weight /= 1 + self.hparams.pos_class_weight

        self.loss = self.get_loss()

        self.train_f1 = torchmetrics.F1Score("binary")
        self.val_f1 = self.train_f1.clone()
        self.test_f1 = self.train_f1.clone()

        # Bounded thresholds keep memory *and* compute constant over the epoch:
        # a binned AP/AUROC costs O(thresholds) instead of O(all test pixels),
        # which otherwise means storing and sorting every pixel from 3300+
        # full-resolution images (GPU OOM, and very slow even on CPU).
        self.val_avg_precision = torchmetrics.AveragePrecision("binary")
        self.test_avg_precision = torchmetrics.AveragePrecision("binary")
        self.test_precision = torchmetrics.Precision("binary")
        self.test_recall = torchmetrics.Recall("binary")
        self.test_iou = torchmetrics.JaccardIndex("binary")
        self.conf_mat = torchmetrics.ConfusionMatrix("binary")

        # Rank-based metrics (AUROC, AP/AUPR) are invariant to the monotonic
        # pos_class_weight logit shift, so raw logits are used directly.
        self.test_auroc = torchmetrics.AUROC("binary", thresholds=100)
        self.test_roc_curve = torchmetrics.ROC("binary", thresholds=100)

        self.test_brier = torchmetrics.MeanSquaredError()
        self.test_ace = AverageCalibrationError(n_bins=10)
        self.test_ece = BinaryCalibrationErrorFast(n_bins=10)

        # Same calibration metrics after removing the systematic logit shift
        # introduced by training with pos_class_weight-weighted BCE. Without this
        # correction, ACE/Brier/NLL mostly measure the loss weighting, not the
        # model's calibration.
        self.test_brier_cal = torchmetrics.MeanSquaredError()
        self.test_ace_cal = AverageCalibrationError(n_bins=10)
        self.test_ece_cal = BinaryCalibrationErrorFast(n_bins=10)

        # Calibration restricted to the fire-relevant region: pixels within
        # _fire_buffer_radius_px of yesterday's active-fire mask or the target.
        # Whole-image calibration is dominated by trivially-easy background
        # (>99% of pixels), so miscalibration only shows up in this zone.
        self.test_brier_fire = torchmetrics.MeanSquaredError()
        self.test_ace_fire = AverageCalibrationError(n_bins=10, adaptive=True, compute_on_cpu=True)
        self.test_ece_fire = BinaryCalibrationErrorFast(n_bins=10)
        self.test_nll_fire = torchmetrics.MeanMetric()
        self._fire_buffer_radius_px = 16  # ~6 km at 375 m/pixel
        self._fire_calib_pixels = []  # (prob, target) reservoir for the reliability diagram
        self._fire_calib_max_pixels = 2_000_000

        # Operating threshold selected on the validation set (max F1 on the val
        # PR curve, in calibrated-probability space — see validation_step). A
        # buffer, so it is saved in the checkpoint and available when testing
        # from a checkpoint. Checkpoints written before the calibrated-threshold
        # fix store a raw-space threshold; re-run with --do_validate=True before
        # --do_test to refresh it. The *_valthr test metrics use it;
        # test_f1_best/test_iou_best pick the threshold on test data and are
        # therefore oracle upper bounds, not honest operating points.
        self.val_pr_curve = torchmetrics.PrecisionRecallCurve("binary", thresholds=100)
        self.register_buffer("val_best_threshold", torch.tensor(0.5))
        self.test_f1_valthr = torchmetrics.F1Score("binary")
        self.test_iou_valthr = torchmetrics.JaccardIndex("binary")
        self.test_precision_valthr = torchmetrics.Precision("binary")
        self.test_recall_valthr = torchmetrics.Recall("binary")

        # Pixel reservoir for uncertainty-quality metrics (AUSE, uncertainty-error
        # correlation). Filled by UQ subclasses via update_uncertainty_metrics().
        self._unc_pixels = []
        self._unc_pixels_per_image = 5000
        self._unc_max_pixels = 2_000_000

        # Plot PR curve at the end of training. Use fixed number of threshold to avoid the plot becoming 800MB+.
        self.test_pr_curve = torchmetrics.PrecisionRecallCurve("binary", thresholds=100)

        # Computational cost: CUDA-synchronized wall time of get_pred_and_gt
        # per image, averaged over the test epoch. For UQ methods this includes
        # all stochastic forward passes (MC samples, ensemble members, ODE
        # steps), so it reflects the method's true inference cost.
        self.test_time_per_image = torchmetrics.MeanMetric()

        # A few (input fire mask, calibrated prediction, uncertainty, ground
        # truth) test examples, logged as images at the end of the test epoch.
        # The uncertainty panel is present only for UQ models; subclasses set
        # _last_unc_map via the update_uncertainty_metrics* calls.
        self._pred_samples = []
        self._max_pred_samples = 8
        self._last_unc_map = None

    def forward(self, x, doys=None):
        # If doys are used, the model needs to re-implement the forward method
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        return self.model(x)

    def get_pred_and_gt(self, batch):
        """_summary_ Unbatch the data and perform inference on each sample.

        Args:
            batch (_type_): _description_ Either a tuple of (x, y) or (x, y, doys).

        Raises:
            ValueError: _description_ If the batch size is not 1 and the model requires repeated inference on crops of the image. 
            This is the case for ConvLSTM, when predicting on the test set. During training, it uses random crops of the required size,
            so larger batch sizes can be used. 

        Returns:
            _type_: _description_ Prediction and ground truth for each sample in the batch.
        """

        # UTAE and TSViT use an additional doy feature as input. 
        if self.hparams.use_doy:
            x, y, doys = batch
        else:
            x, y = batch
            doys = None

        # If the model requires a certain fixed size, perform repeated inference on crops of the image,
        # and aggregate the results. When we reach the last row or column, which might not be divisible by
        # the required size, we align the crop window with the right/bottom edge of the image. This means 
        # that there is some amount of overlap between the last two crops in each row/column. We handle this
        # by simply overwriting the existing predictions with the new ones. 

        if self.hparams.required_img_size is not None:
            B, T, C, H, W = x.shape

            if x.shape[-2:] != self.hparams.required_img_size:
                if B != 1:
                    raise ValueError(
                        "Not implemented: repeated cropping for batch size > 1."
                    )
                # Use crops of size H_rq x W_rq
                H_req, W_req = self.hparams.required_img_size

                n_H = math.ceil(H / H_req)
                n_W = math.ceil(W / W_req)

                # Aggregate predictions in this tensor
                agg_output = torch.zeros(B, H, W, device=self.device)

                for i in range(n_H):
                    for j in range(n_W):
                        
                        # If we reach the bottom edge of the image, align the crop window with the bottom edge of the image
                        if i == n_H - 1:
                            H1 = H - H_req
                            H2 = H
                        else:
                            H1 = i * H_req
                            H2 = (i + 1) * H_req
                        # If we reach the right edge of the image, align the crop window with the right edge of the image
                        if j == n_W - 1:
                            W1 = W - W_req
                            W2 = W
                        else:
                            W1 = j * W_req
                            W2 = (j + 1) * W_req

                        x_crop = x[:, :, :, H1:H2, W1:W2]

                        agg_output[:, H1:H2, W1:W2] = self(x_crop, doys).squeeze(1)

                y_hat = agg_output
                return y_hat, y

        y_hat = self(x, doys).squeeze(1)

        return y_hat, y

    def training_step(self, batch, batch_idx):
        """_summary_ Compute predictions and loss for the given batch. Log training loss and F1 score.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_hat, y = self.get_pred_and_gt(batch)

        loss = self.compute_loss(y_hat, y)
        f1 = self.train_f1(y_hat, y)
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """_summary_ Compute predictions and loss for the given batch. Log validation loss and F1 score.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_hat, y = self.get_pred_and_gt(batch)

        loss = self.compute_loss(y_hat, y)
        f1 = self.val_f1(y_hat, y)
        # val_AP uses a bounded threshold grid (thresholds=100), so it needs the
        # same calibration shift as the PR curve below to keep the grid usable.
        self.val_avg_precision.update(y_hat - self.pos_weight_logit_offset(), y)
        # PR curve on calibrated logits: with pos_class_weight-weighted BCE the
        # raw probabilities are compressed into [0.99, 1), where the uniform
        # 100-point threshold grid has ~1 sample and the best-F1 search rails
        # against the last grid point. Removing the logit shift moves the
        # operating region back to mid-range, so val_best_threshold lives in
        # calibrated-probability space (applied to calibrated probs at test).
        self.val_pr_curve.update(y_hat - self.pos_weight_logit_offset(), y)
        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_AP",
            self.val_avg_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Select the operating threshold (max F1) on the validation PR curve.
        Stored in the val_best_threshold buffer, so the checkpoint of the best
        epoch carries the matching threshold into testing.
        """
        precision, recall, thresholds = self.val_pr_curve.compute()
        self.val_pr_curve.reset()
        p, r = precision[:-1], recall[:-1]  # last point has no threshold
        f1_scores = 2 * p * r / (p + r + 1e-12)
        best_idx = torch.argmax(f1_scores)
        if f1_scores[best_idx] > 0:
            self.val_best_threshold.fill_(thresholds[best_idx])
        self.log("val_f1_best", f1_scores[best_idx].item(), sync_dist=True)
        self.log("val_best_threshold", self.val_best_threshold.item(), sync_dist=True)

    def on_test_epoch_start(self) -> None:
        """Log static model-cost stats: parameter count and in-memory size
        (parameters + buffers; for Deep Ensembles this spans all members).
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_bytes = sum(p.numel() * p.element_size() for p in self.parameters()) \
            + sum(b.numel() * b.element_size() for b in self.buffers())
        self.log("test_params_M", n_params / 1e6, sync_dist=True)
        self.log("test_params_trainable_M", n_trainable / 1e6, sync_dist=True)
        self.log("test_model_size_MB", size_bytes / 2**20, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """_summary_ Compute predictions and loss for the given batch. Log test loss, F1, AP, precision, recall, IoU and confusion matrix.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Compute-cost accounting around the full prediction call, so UQ
        # sampling loops are included. FLOPs are measured once, on the first
        # image, via the profiler (torch 2.0 has no flop_counter module);
        # profiling overhead would skew the wall clock, so that batch is
        # excluded from timing. Later batches get CUDA-synchronized timing.
        batch_size = batch[0].shape[0]
        if batch_idx == 0:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True, with_flops=True,
            ) as prof:
                y_hat, y = self.get_pred_and_gt(batch)
            flops = sum(e.flops for e in prof.key_averages() if e.flops)
            self.log("test_gflops_per_image", flops / 1e9 / batch_size, sync_dist=True)
        else:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            y_hat, y = self.get_pred_and_gt(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.test_time_per_image.update((time.perf_counter() - t0) / batch_size)

        loss = self.compute_loss(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_avg_precision(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_iou(y_hat, y)
        self.conf_mat.update(y_hat, y)
        self.test_auroc.update(y_hat.flatten(), y.flatten())
        self.test_roc_curve.update(y_hat.flatten(), y.flatten())

        probs = torch.sigmoid(y_hat)
        self.test_brier.update(probs.flatten(), y.float().flatten())
        self.test_ace.update(probs.flatten(), y.float().flatten())
        self.test_ece.update(probs.flatten(), y.flatten())

        nll = F.binary_cross_entropy_with_logits(y_hat, y.float(), reduction='mean')

        # Weight-corrected calibration: undo the log(pos_class_weight) logit shift
        offset = self.pos_weight_logit_offset()
        y_hat_cal = y_hat - offset
        probs_cal = torch.sigmoid(y_hat_cal)
        self.test_brier_cal.update(probs_cal.flatten(), y.float().flatten())
        self.test_ace_cal.update(probs_cal.flatten(), y.float().flatten())
        self.test_ece_cal.update(probs_cal.flatten(), y.flatten())
        nll_cal = F.binary_cross_entropy_with_logits(y_hat_cal, y.float(), reduction='mean')

        # Calibrated logits, like the val PR curve: keeps the bounded threshold
        # grid usable (see validation_step) and test_f1_best comparable to
        # val_f1_best / test_f1_valthr.
        self.test_pr_curve.update(y_hat_cal, y)

        # Operating-point metrics at the threshold selected on the val set.
        # The threshold was chosen on calibrated probabilities (val PR curve on
        # calibrated logits), so it is applied to calibrated probs here too.
        hard_pred = (probs_cal >= self.val_best_threshold.item()).long()
        self.test_f1_valthr.update(hard_pred, y)
        self.test_iou_valthr.update(hard_pred, y)
        self.test_precision_valthr.update(hard_pred, y)
        self.test_recall_valthr.update(hard_pred, y)

        # Calibration restricted to the fire-relevant region: dilation of
        # (yesterday's active fire ∪ ground truth). The last input channel of
        # the last time step is the binary active fire mask.
        x = batch[0]
        af_prev = (x[:, -1, -1] if x.dim() == 5 else x[:, -1]) > 0
        fire_zone = (af_prev | (y > 0)).float().unsqueeze(1)
        radius = self._fire_buffer_radius_px
        fire_zone = F.max_pool2d(
            fire_zone, kernel_size=2 * radius + 1, stride=1, padding=radius
        ).squeeze(1) > 0
        if fire_zone.any():
            fire_probs = probs_cal[fire_zone]
            fire_y = y[fire_zone].float()
            self.test_brier_fire.update(fire_probs, fire_y)
            self.test_ace_fire.update(fire_probs, fire_y)
            self.test_ece_fire.update(fire_probs, fire_y)
            nll_fire = F.binary_cross_entropy_with_logits(
                y_hat_cal[fire_zone], fire_y, reduction='mean')
            self.test_nll_fire.update(nll_fire, weight=fire_zone.sum())
            k = min(self._unc_pixels_per_image, fire_probs.numel())
            idx = torch.randint(0, fire_probs.numel(), (k,), device=fire_probs.device)
            self._fire_calib_pixels.append(
                torch.stack([fire_probs[idx], fire_y[idx]], dim=1).cpu())

        # Keep the samples with the largest ground-truth fires for image
        # logging at the end of the test epoch. The last channel of the last
        # time step is the binary active fire mask.
        fire_size = y[0].sum().item()
        if fire_size > 0:
            x_af = x[0, -1, -1] if x.dim() == 5 else x[0, -1]
            unc = self._last_unc_map[0].detach().cpu() if self._last_unc_map is not None else None
            self._pred_samples.append(
                (fire_size, x_af.detach().cpu(), probs_cal[0].detach().cpu(),
                 hard_pred[0].detach().cpu(), unc, y[0].detach().cpu()))
            self._pred_samples.sort(key=lambda s: s[0], reverse=True)
            del self._pred_samples[self._max_pred_samples:]
        # Uncertainty maps are set per batch by the UQ subclass (inside
        # get_pred_and_gt); clear so a later batch cannot reuse a stale map.
        self._last_unc_map = None

        self.log("test_loss", loss.item(), sync_dist=True)
        self.log("test_nll", nll.item(), sync_dist=True)
        self.log("test_nll_cal", nll_cal.item(), sync_dist=True)
        self.log_dict(
            {
                "test_f1": self.test_f1,
                "test_AP": self.test_avg_precision,
                "test_AUPR": self.test_avg_precision,
                "test_AUROC": self.test_auroc,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
                "test_iou": self.test_iou,
                "test_brier": self.test_brier,
                "test_ace": self.test_ace,
                "test_ece": self.test_ece,
                "test_brier_cal": self.test_brier_cal,
                "test_ace_cal": self.test_ace_cal,
                "test_ece_cal": self.test_ece_cal,
                "test_f1_valthr": self.test_f1_valthr,
                "test_iou_valthr": self.test_iou_valthr,
                "test_precision_valthr": self.test_precision_valthr,
                "test_recall_valthr": self.test_recall_valthr,
                "test_time_per_image_s": self.test_time_per_image,
            }
        )
        return loss

    def pos_weight_logit_offset(self) -> float:
        """Systematic logit shift caused by training BCE with pos_weight > 1.

        A model trained with weighted BCE converges to logit(p) + log(w) instead
        of logit(p). Subtracting log(w) recovers (approximately) calibrated
        probabilities for calibration metrics.
        """
        if self.hparams.loss_function == "BCE" and self.hparams.pos_class_weight > 1:
            return math.log(self.hparams.pos_class_weight)
        return 0.0

    def _calibrated_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Undo the pos_class_weight logit shift on probabilities."""
        offset = self.pos_weight_logit_offset()
        return torch.sigmoid(torch.logit(probs.detach().float().clamp(1e-6, 1 - 1e-6)) - offset)

    @staticmethod
    def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(1e-6, 1 - 1e-6)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    def update_uncertainty_metrics(self, probs: torch.Tensor, uncertainty: torch.Tensor, y: torch.Tensor):
        """Legacy single-map path: stores the given uncertainty as the 'total'
        component, without an aleatoric/epistemic decomposition. Prefer
        update_uncertainty_metrics_from_samples when per-sample probabilities
        are available.
        """
        self._last_unc_map = uncertainty.detach()
        p = self._calibrated_probs(probs).flatten()
        u = uncertainty.detach().flatten().float()
        nan = torch.full_like(u, float("nan"))
        self._store_uncertainty_pixels(p, y, u, nan, nan)

    def update_uncertainty_metrics_from_samples(self, sample_probs: torch.Tensor, y: torch.Tensor):
        """Entropy-based uncertainty decomposition from per-sample probabilities
        (Depeweg et al. 2018; Kendall & Gal 2017):

            total     = H[ E_s p_s ]   (predictive entropy)
            aleatoric = E_s H[ p_s ]   (expected entropy)
            epistemic = total - aleatoric   (mutual information / BALD)

        For sampling-based epistemic methods (MC Dropout, ensembles, BBB) the
        MI term is model uncertainty. For latent-variable aleatoric methods
        (Prob U-Net, SSN, flow matching) it measures inter-sample disagreement
        of the learned output distribution; interpret accordingly.

        Args:
            sample_probs: [S, B, H, W] fire probabilities, one per sample.
            y: [B, H, W] ground truth.
        """
        p_s = self._calibrated_probs(sample_probs)
        p = p_s.mean(dim=0)
        total = self._binary_entropy(p)
        aleatoric = self._binary_entropy(p_s).mean(dim=0)
        epistemic = (total - aleatoric).clamp(min=0)
        self._last_unc_map = total.detach()
        self._store_uncertainty_pixels(
            p.flatten(), y, total.flatten(), aleatoric.flatten(), epistemic.flatten())

    def _store_uncertainty_pixels(self, p, y, u_total, u_aleatoric, u_epistemic):
        """Subsample pixels into the reservoir used for uncertainty-quality
        metrics (AUSE, Spearman, misclassification AUROC, risk-coverage) at
        test epoch end. Columns: [prob, target, total, aleatoric, epistemic].
        """
        t = y.detach().flatten().float()
        k = min(self._unc_pixels_per_image, p.numel())
        idx = torch.randint(0, p.numel(), (k,), device=p.device)
        self._unc_pixels.append(torch.stack(
            [p[idx], t[idx], u_total[idx], u_aleatoric[idx], u_epistemic[idx]], dim=1).cpu())

    def _log_figure(self, name: str, fig) -> None:
        """Log a matplotlib figure to wandb, or save it as a local PNG when
        wandb is disabled (WANDB_MODE=disabled), so plots aren't silently
        dropped when logging locally to ./lightning_logs."""
        if wandb.run is not None and not getattr(wandb.run, "disabled", False):
            wandb.log({name: wandb.Image(fig)})
        else:
            out_dir = os.path.join(
                getattr(self.trainer, "default_root_dir", None) or ".", "test_plots")
            os.makedirs(out_dir, exist_ok=True)
            safe_name = name.replace("/", "-").replace(" ", "_")
            fig.savefig(os.path.join(out_dir, f"{safe_name}.png"), dpi=150, bbox_inches="tight")

    @staticmethod
    def _spearman_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        def ranks(x):
            return torch.argsort(torch.argsort(x)).float()
        ra, rb = ranks(a), ranks(b)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        return (ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12)

    def _compute_uncertainty_quality(self):
        """Uncertainty-quality metrics from the pixel reservoir:

        - AUSE (Brier-based) + Spearman uncertainty-error correlation, per
          uncertainty component (total / aleatoric / epistemic when available).
          Sparsification curve: mean per-pixel Brier error of the pixels that
          remain after removing the fraction f most-uncertain pixels; the
          oracle removes by true error. AUSE = area between the curves
          (normalized by the full-set mean error); 0 = perfect error ranking.
        - Misclassification-detection AUROC: total uncertainty as a score for
          "pixel is wrong at the val-selected threshold".
        - Risk-coverage curve + AURC (selective prediction): Brier risk of the
          retained pixels when abstaining on the most-uncertain fraction.
        """
        data = torch.cat(self._unc_pixels)
        self._unc_pixels = []
        if data.shape[0] > self._unc_max_pixels:
            data = data[torch.randperm(data.shape[0])[:self._unc_max_pixels]]
        p, t = data[:, 0], data[:, 1]
        err = (p - t) ** 2

        n = err.numel()
        base_err = err.mean() + 1e-12

        def sparsification_curve(order):
            # suffix means: mean error of pixels kept after removing the k most-uncertain
            e = err[order]
            suffix = torch.flip(torch.cumsum(torch.flip(e, [0]), 0), [0])
            counts = torch.arange(n, 0, -1, dtype=torch.float32)
            fracs = (torch.linspace(0, 0.99, 100) * n).long().clamp(max=n - 1)
            return suffix[fracs] / counts[fracs] / base_err

        curve_oracle = sparsification_curve(torch.argsort(err, descending=True))

        fracs = torch.linspace(0, 0.99, 100)
        fig, ax = plt.subplots()
        for name, suffix, u in [("total", "", data[:, 2]),
                                ("aleatoric", "_aleatoric", data[:, 3]),
                                ("epistemic", "_epistemic", data[:, 4])]:
            if torch.isnan(u).all():
                continue  # legacy path stores only the total component
            spearman = self._spearman_corr(u, err)
            curve = sparsification_curve(torch.argsort(u, descending=True))
            ause = (curve - curve_oracle).mean()
            self.log(f"test_ause{suffix}", ause.item(), sync_dist=True)
            self.log(f"test_unc_err_spearman{suffix}", spearman.item(), sync_dist=True)
            ax.plot(fracs.numpy(), curve.numpy(),
                    label=f"{name} (AUSE={ause.item():.4f})")
        ax.plot(fracs.numpy(), curve_oracle.numpy(), "k--", label="oracle (by error)")
        ax.set_xlabel("Fraction of most-uncertain pixels removed")
        ax.set_ylabel("Relative Brier error of remaining pixels")
        ax.set_title("Sparsification per uncertainty component")
        ax.legend()
        self._log_figure("Test sparsification curve", fig)
        plt.close(fig)

        u_total = data[:, 2]

        # Misclassification-detection AUROC (rank-sum estimate). The val
        # threshold and the reservoir probs both live on the calibrated scale.
        thr = self.val_best_threshold.detach().cpu()
        wrong = ((p >= thr).float() != t).float()
        n_pos = wrong.sum()
        if 0 < n_pos < n:
            ranks = torch.argsort(torch.argsort(u_total)).float() + 1
            n_neg = n - n_pos
            auroc = (ranks[wrong.bool()].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            self.log("test_misclf_auroc", auroc.item(), sync_dist=True)

        # Risk-coverage: mean Brier error of the c most-certain fraction
        def risk_coverage(order):
            csum = torch.cumsum(err[order], 0)
            ks = (torch.linspace(0.01, 1.0, 100) * n).long().clamp(1, n)
            return csum[ks - 1] / ks.float()

        coverages = torch.linspace(0.01, 1.0, 100)
        risks = risk_coverage(torch.argsort(u_total))
        risks_oracle = risk_coverage(torch.argsort(err))
        aurc = risks.mean()
        self.log("test_aurc", aurc.item(), sync_dist=True)

        fig, ax = plt.subplots()
        ax.plot(coverages.numpy(), risks.numpy(), label=f"by uncertainty (AURC={aurc.item():.5f})")
        ax.plot(coverages.numpy(), risks_oracle.numpy(), "k--", label="oracle (by error)")
        ax.set_xlabel("Coverage (fraction of pixels retained)")
        ax.set_ylabel("Brier risk of retained pixels")
        ax.set_title("Risk-coverage curve")
        ax.legend()
        self._log_figure("Test risk-coverage curve", fig)
        plt.close(fig)

    def on_test_epoch_end(self) -> None:
        """_summary_ Log the test PR curve and confusion matrix after predicting all test samples.
        """
        conf_mat = self.conf_mat.compute().cpu().numpy()
        if wandb.run is not None and not getattr(wandb.run, "disabled", False):
            wandb_table = wandb.Table(
                data=conf_mat, columns=["PredictedBackground", "PredictedFire"]
            )
            wandb.log({"Test confusion matrix": wandb_table})
        else:
            fig, ax = self.conf_mat.plot()
            self._log_figure("Test confusion matrix", fig)
            plt.close(fig)

        # F1/IoU at the best threshold from the PR curve. The fixed 0.5 threshold
        # is a near-degenerate operating point when training with pos_class_weight,
        # so test_f1/test_iou drastically understate segmentation quality.
        precision, recall, thresholds = self.test_pr_curve.compute()
        p, r = precision[:-1], recall[:-1]  # last point has no threshold
        f1_scores = 2 * p * r / (p + r + 1e-12)
        best_idx = torch.argmax(f1_scores)
        best_p, best_r = p[best_idx], r[best_idx]
        best_iou = best_p * best_r / (best_p + best_r - best_p * best_r + 1e-12)
        self.log("test_f1_best", f1_scores[best_idx].item(), sync_dist=True)
        self.log("test_iou_best", best_iou.item(), sync_dist=True)
        self.log("test_f1_best_threshold", thresholds[best_idx].item(), sync_dist=True)

        fig, ax = self.test_pr_curve.plot(score=True)
        self._log_figure("Test PR Curve", fig)
        plt.close(fig)

        fig, ax = self.test_roc_curve.plot(score=True)
        self._log_figure("Test ROC Curve", fig)
        plt.close(fig)

        fig, ax = self.test_ece.plot()
        self._log_figure("Test calibration diagram (ECE)", fig)
        plt.close(fig)

        fig, ax = self.test_ece_cal.plot()
        self._log_figure("Test calibration diagram, calibrated (ECE)", fig)
        plt.close(fig)

        # Fire-buffer calibration metrics (skipped if no test image had any
        # fire pixels, in which case the metrics were never updated)
        if self._fire_calib_pixels:
            self.log("test_brier_fire", self.test_brier_fire.compute().item(), sync_dist=True)
            self.log("test_ace_fire", self.test_ace_fire.compute().item(), sync_dist=True)
            self.log("test_ece_fire", self.test_ece_fire.compute().item(), sync_dist=True)
            self.log("test_nll_fire", self.test_nll_fire.compute().item(), sync_dist=True)
            self.test_brier_fire.reset()
            self.test_ace_fire.reset()
            self.test_ece_fire.reset()
            self.test_nll_fire.reset()
            self._log_reliability_diagram()

        # Uncertainty-quality metrics, if a UQ subclass accumulated pixels
        if self._unc_pixels:
            self._compute_uncertainty_quality()

        if self._pred_samples:
            self._log_prediction_samples()

    def _log_reliability_diagram(self):
        """Reliability diagram over the fire-buffer region, equal-mass bins."""
        data = torch.cat(self._fire_calib_pixels)
        self._fire_calib_pixels = []
        if data.shape[0] > self._fire_calib_max_pixels:
            data = data[torch.randperm(data.shape[0])[:self._fire_calib_max_pixels]]
        p, t = data[:, 0], data[:, 1]
        confs, freqs = [], []
        for c in torch.chunk(torch.argsort(p), 10):
            if c.numel() > 0:
                confs.append(p[c].mean().item())
                freqs.append(t[c].mean().item())
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        ax.plot(confs, freqs, "o-", label="model")
        ax.set_xlabel("Mean predicted fire probability (bin)")
        ax.set_ylabel("Observed fire frequency (bin)")
        ax.set_title("Reliability diagram, fire-buffer region (equal-mass bins)")
        ax.legend()
        self._log_figure("Test reliability diagram (fire region)", fig)
        plt.close(fig)

    def _log_prediction_samples(self):
        """Log (input fire mask, calibrated prediction, thresholded prediction
        [, uncertainty], ground truth) panels for the test samples with the
        largest ground-truth fires. Panels are cropped to the fire region, and
        the probability panel uses a per-image color scale because calibrated
        fire probabilities live near the (tiny) class base rate. The
        uncertainty panel (total predictive entropy) appears only for UQ
        models."""
        images = []
        margin = 32
        for i, (fire_size, x_af, prob, pred_mask, unc, y) in enumerate(self._pred_samples):
            # Crop to the bounding box of (input fire ∪ ground truth) + margin;
            # fires are tiny relative to full test images.
            rows, cols = torch.where((x_af > 0) | (y > 0))
            r0 = max(int(rows.min()) - margin, 0)
            r1 = min(int(rows.max()) + margin + 1, y.shape[0])
            c0 = max(int(cols.min()) - margin, 0)
            c1 = min(int(cols.max()) + margin + 1, y.shape[1])
            crop = (slice(r0, r1), slice(c0, c1))
            x_af, prob, pred_mask, y = x_af[crop], prob[crop], pred_mask[crop], y[crop]
            unc = unc[crop] if unc is not None else None

            n_panels = 4 if unc is None else 5
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
            axes[0].imshow(x_af.numpy(), cmap="gray", interpolation="nearest")
            axes[0].set_title("Active fire (last input day)")
            vmax = max(prob.max().item(), 1e-3)
            im = axes[1].imshow(prob.numpy(), cmap="viridis", vmin=0, vmax=vmax,
                                interpolation="nearest")
            axes[1].set_title("Predicted fire probability (cal.)")
            fig.colorbar(im, ax=axes[1], fraction=0.046)
            axes[2].imshow(pred_mask.numpy(), cmap="gray", vmin=0, vmax=1,
                           interpolation="nearest")
            axes[2].set_title("Prediction @ val threshold")
            if unc is not None:
                im = axes[3].imshow(unc.numpy(), cmap="magma", interpolation="nearest")
                axes[3].set_title("Uncertainty (total)")
                fig.colorbar(im, ax=axes[3], fraction=0.046)
            axes[-1].imshow(y.numpy(), cmap="gray", vmin=0, vmax=1,
                            interpolation="nearest")
            axes[-1].set_title("Ground truth (next day)")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            if wandb.run is not None and not getattr(wandb.run, "disabled", False):
                images.append(wandb.Image(
                    fig, caption=f"test sample {i} ({int(fire_size)} fire px)"))
            else:
                self._log_figure(f"Test prediction samples/sample_{i}", fig)
            plt.close(fig)
        if images:
            wandb.log({"Test prediction samples": images})
        self._pred_samples = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x_af = x[:, :, -1, :, :]
        y_hat = self(x).squeeze(1)
        return x_af, y, y_hat

    def get_loss(self):
        if self.hparams.loss_function == "BCE":
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor(
                    [self.hparams.pos_class_weight], device=self.device
                )
            )
        elif self.hparams.loss_function == "Focal":
            return sigmoid_focal_loss
        elif self.hparams.loss_function == "Lovasz":
            return LovaszLoss(mode="binary")
        elif self.hparams.loss_function == "Jaccard":
            return JaccardLoss(mode="binary")
        elif self.hparams.loss_function == "Dice":
            return DiceLoss(mode="binary")

    def compute_loss(self, y_hat, y):
        if self.hparams.loss_function == "Focal":
            return self.loss(
                y_hat,
                y.float(),
                alpha=1 - self.hparams.pos_class_weight,
                gamma=2,
                reduction="mean",
            )
        else:
            return self.loss(y_hat, y.float())
