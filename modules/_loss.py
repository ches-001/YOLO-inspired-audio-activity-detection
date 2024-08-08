import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from dataset import AudioDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Optional, Dict, List
 

class AudioDetectionLoss(nn.Module):
    def __init__(
        self, 
        anchors_dict: Dict[str, List[float]],
        num_classes: int,
        anchor_t: float=4.0,
        sample_duration: float=60,
        box_w: float=1.0,
        conf_w: float=1.0,
        class_w: float=1.0,
        multi_label: bool=False,
        class_weights: Optional[torch.Tensor]=None,
        label_smoothing: float=0,
    ):
        super(AudioDetectionLoss, self).__init__()
        self.anchors_dict = anchors_dict
        self.num_classes = num_classes
        self.anchor_t = anchor_t
        self.sample_duration = sample_duration
        self.box_w = box_w
        self.conf_w = conf_w
        self.class_w = class_w
        self.multi_label = multi_label
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(
        self, preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:        
        metrics_dict = {}
        sm_preds, md_preds, lg_preds = preds
        (sm_lbox, sm_lconf, sm_lcls), sm_metrics_dict = self.loss_fn(sm_preds, targets, anchors=self.anchors_dict["sm"])
        (md_lbox, md_lconf, md_lcls), md_metrics_dict = self.loss_fn(md_preds, targets, anchors=self.anchors_dict["md"])
        (lg_lbox, lg_lconf, lg_lcls), lg_metrics_dict = self.loss_fn(lg_preds, targets, anchors=self.anchors_dict["lg"])

        lbox = sm_lbox + md_lbox + lg_lbox
        lconf = (sm_lconf * 4.0) + (md_lconf * 1.0) + (lg_lconf * 0.4)
        lcls = sm_lcls + md_lcls + lg_lcls

        loss = ((self.box_w * lbox) + (self.conf_w * lconf) + (self.class_w * lcls)) * preds[-1].shape[0]
        metrics_df = pd.DataFrame([sm_metrics_dict, md_metrics_dict, lg_metrics_dict])

        metrics_dict["aggregate_loss"] = loss.item()
        metrics_dict["mean_ciou"] = metrics_df["mean_ciou"].mean()
        metrics_dict["conf_loss"] = metrics_df["conf_loss"].mean()
        metrics_dict["class_loss"] = metrics_df["class_loss"].mean()
        metrics_dict["accuracy"] = metrics_df["accuracy"].mean()
        metrics_dict["f1"] = metrics_df["f1"].mean()
        metrics_dict["precision"] = metrics_df["precision"].mean()
        metrics_dict["recall"] = metrics_df["recall"].mean()
        return loss, metrics_dict


    def loss_fn(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            anchors: List[float],
        ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, float]]:
        
        _device = preds.device
        t_conf = torch.zeros(preds.shape[:-1], device=_device, dtype=preds.dtype)
        indices, t_classes, t_cw = AudioDataset.build_target_by_scale(
            targets, preds.shape[1], anchors, anchor_threshold=self.anchor_t, sample_duration=self.sample_duration
        )
        batch_idx, grid_idx, anchor_idx = indices
        match_preds = preds[batch_idx, grid_idx, anchor_idx]
        p_cls_proba = match_preds[:, 1:1+self.num_classes]
        p_cw = match_preds[:, -2:]
        ciou = AudioDetectionLoss.compute_ciou(p_cw, t_cw)
        
        # bbox loss
        ciou_loss = (1 - ciou).mean()

        # conf loss
        t_conf[batch_idx, grid_idx, anchor_idx] = ciou.detach()
        p_conf = preds[..., 0]
        conf_loss = F.binary_cross_entropy_with_logits(p_conf, t_conf, reduce="mean")

        # class loss
        if not self.multi_label:
            class_loss = F.cross_entropy(p_cls_proba, t_classes, weight=self.class_weights, reduction="mean")
        else:
            cn = 0.5 * self.label_smoothing
            cp = 1.0 - cn
            t_cls_proba = torch.full_like(p_cls_proba, cn)
            t_cls_proba[range(batch_idx.shape[0]), t_classes] = cp
            class_loss = F.binary_cross_entropy_with_logits(p_cls_proba, t_cls_proba, reduce="mean")

        # accuracy, precision, recall
        if t_classes.shape[0]:
            pred_labels = p_cls_proba.detach().argmax(dim=-1).cpu().numpy()
            target_labels = t_classes.cpu().numpy()
            accuracy = accuracy_score(target_labels, pred_labels)
            f1 = f1_score(target_labels, pred_labels, average="macro")
            precision = precision_score(target_labels, pred_labels, average="macro")    
            recall = recall_score(target_labels, pred_labels, average="macro")
        else:
             accuracy, f1, precision, recall = [torch.nan] * 4

        # aggregate losses
        handle_nan = lambda val : val if val == val else torch.tensor(0.0, device=_device)
        losses = (handle_nan(ciou_loss), handle_nan(conf_loss), handle_nan(class_loss))
        metrics_dict = {}
        metrics_dict["mean_ciou"] = ciou.mean().item()
        metrics_dict["conf_loss"] = conf_loss.item()
        metrics_dict["class_loss"] = class_loss.item()
        metrics_dict["accuracy"] = accuracy
        metrics_dict["f1"] = f1
        metrics_dict["precision"] = precision
        metrics_dict["recall"] = recall
        return losses, metrics_dict


    @staticmethod
    def compute_ciou(preds_cw: torch.Tensor, targets_cw: torch.Tensor, e: float=1e-15, _h: float=10.0) -> torch.Tensor:
        assert (preds_cw.ndim == targets_cw.ndim + 1) or (preds_cw.ndim == targets_cw.ndim)
        if targets_cw.ndim != preds_cw.ndim:
                targets_cw = targets_cw.unsqueeze(dim=-2)
        pred_c = preds_cw[..., :1]
        pred_w = preds_cw[..., -1:]
        pred_h = torch.ones_like(pred_w) * _h
        pred_x1 = pred_c - (pred_w / 2)
        pred_y1 = torch.zeros_like(pred_x1)
        pred_x2 = pred_c + (pred_w / 2)
        pred_y2 = pred_h

        target_c = targets_cw[..., :1]
        target_w = targets_cw[..., -1:]
        target_h = torch.ones_like(target_w) * _h
        target_x1 = target_c - (target_w / 2)
        target_y1 = torch.zeros_like(target_x1)
        target_x2 = target_c + (target_w / 2)
        target_y2 = target_h

        intersection_w = (torch.min(pred_x2, target_x2) - torch.max(pred_x1, target_x1)).clip(min=0)
        intersection_h = (torch.min(pred_y2, target_y2) - torch.max(pred_y1, target_y1)).clip(min=0)
        intersection = intersection_w * intersection_h
        union = (pred_w * pred_h) + (target_w * target_h) - intersection
        iou = intersection / (union + e)

        cw = (torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1))
        ch = (torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1))
        c2 = cw.pow(2) + ch.pow(2) + e
        v = (4 / (torch.pi**2)) * (torch.arctan(target_w / target_h) - torch.arctan(pred_w / pred_h)).pow(2)
        rho2 = (pred_c - target_c).pow(2) + (pred_h/2 - target_h/2).pow(2)
        with torch.no_grad():
            a = v / ((1 + e) - iou) + v
        ciou = iou - ((rho2/c2) + (a * v))
        return ciou.squeeze(-1).clip(min=0)