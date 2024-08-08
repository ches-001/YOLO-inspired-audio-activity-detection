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
        ciou_w: float=1.0,
        conf_w: float=1.0,
        class_w: float=1.0,
        sm_w: float=1.0,
        md_w: float=1.0,
        lg_w: float=1.0,
        class_weights: Optional[torch.Tensor]=None,
        label_smoothing: float=0,
        iou_confidence: bool=False,
    ):
        super(AudioDetectionLoss, self).__init__()
        self.anchors_dict = anchors_dict
        self.num_classes = num_classes
        self.anchor_t = anchor_t
        self.sample_duration = sample_duration
        self.sm_w = sm_w
        self.md_w = md_w
        self.lg_w = lg_w
        self.ciou_w = ciou_w
        self.conf_w = conf_w
        self.class_w = class_w
        self.label_smoothing = label_smoothing
        self.iou_confidence = iou_confidence
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
            self, 
            preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
            targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        metrics_dict = {}
        sm_preds, md_preds, lg_preds = preds
        sm_loss, sm_metrics_dict = self.loss_fn(sm_preds, targets, anchors=self.anchors_dict["sm"])
        md_loss, md_metrics_dict = self.loss_fn(md_preds, targets, anchors=self.anchors_dict["md"])
        lg_loss, lg_metrics_dict = self.loss_fn(lg_preds, targets, anchors=self.anchors_dict["lg"])
        loss = (self.sm_w * sm_loss) + (self.md_w * md_loss) + (self.lg_w * lg_loss)
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
            e: float=1e-15
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        _device = preds.device
        t_conf = torch.zeros(preds.shape[:-1], device=_device, dtype=preds.dtype)
        indices, t_classes, t_cw = AudioDataset.build_target_by_scale(
            targets, preds.shape[1], anchors, anchor_threshold=self.anchor_t, sample_duration=self.sample_duration
        )
        batch_idx, grid_idx, anchor_idx = indices
        match_preds = preds[batch_idx, grid_idx, anchor_idx]
        p_classes = match_preds[:, 1:1+self.num_classes]
        p_cw = match_preds[:, -2:]
        ciou = AudioDetectionLoss.compute_ciou(p_cw, t_cw)
        
        # bbox loss
        ciou_loss = (1 - ciou).mean()

        # conf loss
        if self.iou_confidence:
            t_conf[batch_idx, grid_idx, anchor_idx] = ciou.detach()
        else:
            t_conf[batch_idx, grid_idx, anchor_idx] = 1
        p_conf = preds[..., 0]
        pos_weight = torch.tensor([t_conf[t_conf == 0].shape[0] / (t_conf[t_conf > 0].shape[0] + e)], device=_device)
        conf_loss = F.binary_cross_entropy_with_logits(p_conf, t_conf, pos_weight=pos_weight)

        # class loss
        class_loss = self.cls_loss_fn(p_classes, t_classes)

        # accuracy, precision, recall
        if t_classes.shape[0]:
            pred_labels = p_classes.detach().argmax(dim=-1).cpu().numpy()
            target_labels = t_classes.cpu().numpy()
            accuracy = accuracy_score(target_labels, pred_labels)
            f1 = f1_score(target_labels, pred_labels, average="macro")
            precision = precision_score(target_labels, pred_labels, average="macro")    
            recall = recall_score(target_labels, pred_labels, average="macro")
        else:
             accuracy, f1, precision, recall = [torch.nan] * 4

        # aggregate losses
        handle_nan = lambda val, w : (w * val) if val == val else torch.tensor(0.0, device=_device)
        loss = (
            handle_nan(ciou_loss, self.ciou_w) +
            handle_nan(conf_loss, self.conf_w) + 
            handle_nan(class_loss, self.class_w)
        )
        metrics_dict = {}
        metrics_dict["mean_ciou"] = 1 - ciou_loss.item()
        metrics_dict["conf_loss"] = conf_loss.item()
        metrics_dict["class_loss"] = class_loss.item()
        metrics_dict["accuracy"] = accuracy
        metrics_dict["f1"] = f1
        metrics_dict["precision"] = precision
        metrics_dict["recall"] = recall
        return loss, metrics_dict


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