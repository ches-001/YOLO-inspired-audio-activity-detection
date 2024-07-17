import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Optional, Dict

class AudioDetectionLoss(nn.Module):
    def __init__(
        self, 
        ciou_loss_w: float=1.0,
        conf_loss_w: float=1.0,
        class_loss_w: float=1.0,
        sm_loss_w: float=.0,
        md_loss_w: float=1.0,
        lg_loss_w: float=1.0,
        class_weights: Optional[torch.Tensor]=None,
        label_smoothing: float=0,
        iou_confidence: bool=False,
        ignore_conf_threshold: float=0.5,
        scale_t: Optional[int]=None,
        ignore_index: int=-100
    ):
        super(AudioDetectionLoss, self).__init__()
        self.sm_loss_w = sm_loss_w
        self.md_loss_w = md_loss_w
        self.lg_loss_w = lg_loss_w
        self.ciou_loss_w = ciou_loss_w
        self.conf_loss_w = conf_loss_w
        self.class_loss_w = class_loss_w
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.ignore_conf_threshold = ignore_conf_threshold
        self.iou_confidence = iou_confidence
        self.scale_t = scale_t if scale_t else 1
        self.ignore_index = ignore_index

        self.ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean", ignore_index=self.ignore_index)

    def forward(
            self, 
            preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
            targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        metrics_dict = {}
        sm_preds, md_preds, lg_preds = preds
        sm_targets, md_targets, lg_targets = targets
        sm_loss, sm_metrics_dict = self.loss_fn(sm_preds, sm_targets)
        md_loss, md_metrics_dict = self.loss_fn(md_preds, md_targets)
        lg_loss, lg_metrics_dict = self.loss_fn(lg_preds, lg_targets)
        loss = (self.sm_loss_w * sm_loss) + (self.md_loss_w * md_loss) + (self.lg_loss_w * lg_loss)
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


    def loss_fn(self, preds: torch.Tensor, targets: torch.Tensor, e: float=1e-15) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert preds.ndim == targets.ndim + 1
        _device = preds.device
        # sm_pred: (S x 3 x (3 + C))     sm_target: (S, 4)
        ious_per_anchors = AudioDetectionLoss.compute_ciou(preds[..., -2:], targets[..., -2:], e=e)
        ious_max, best_anchoridx = torch.max(ious_per_anchors, dim=-1)

        iou_sub_mask = torch.ones_like(ious_per_anchors, dtype=torch.bool).scatter(2, best_anchoridx.unsqueeze(-1), False)
        noobj_pred_mask = torch.ones_like(preds[..., 0], dtype=torch.bool).scatter(2, best_anchoridx.unsqueeze(-1), False)

        # target confidence scores (1 if object, 0 if no object)
        valid_confidence = targets[..., :1]
        _index = best_anchoridx.unsqueeze(-1).unsqueeze(-1).expand(*preds.shape[0:2], -1, preds.shape[-1])
        best_preds = torch.gather(preds, dim=2, index=_index)
        best_preds = best_preds.squeeze(dim=2)

        # segment center and duration loss
        ciou_loss = 1 - ious_max.unsqueeze(-1)[valid_confidence == 1].mean()

        # confidence loss
        iou_sub = ious_per_anchors[iou_sub_mask].reshape(*ious_per_anchors.shape[:2], 2)
        noobj_pred_confidence = preds[noobj_pred_mask].reshape(*preds.shape[:2], 2, -1)[..., 0]
        noobj_target_confidence = torch.zeros_like(noobj_pred_confidence, device=_device)
        obj_pred_confidence = best_preds[..., :1]
        obj_target_confidence = valid_confidence
        # target confidence can either be 1 where an audio segment is present and 0s where
        # no audio segment is, or (1 x IoU) where an audio segment is present and (0 x IoU)
        # where no audio segment is. For the latter, the model would essentally aim to predict
        # Its IoU to the target as its confidence, this can also theoretically make for a good
        # regularisation technique, akin to label smoothening.
        if self.iou_confidence:
            obj_target_confidence = obj_target_confidence * ious_max.unsqueeze(-1)
        pos_mask = obj_target_confidence > 0
        noobj_comp_mask = iou_sub <= self.ignore_conf_threshold
        pos_weight = (
             obj_target_confidence[torch.bitwise_not(pos_mask)].shape[0] /
            (obj_target_confidence[pos_mask].shape[0] + e)
        )
        pos_weight = torch.tensor(pos_weight, device=_device)
        noobj_conf_loss = F.binary_cross_entropy_with_logits(
            noobj_pred_confidence[noobj_comp_mask], noobj_target_confidence[noobj_comp_mask], reduction="none"
        )
        obj_conf_loss = F.binary_cross_entropy_with_logits(
            obj_pred_confidence, obj_target_confidence, pos_weight=pos_weight, reduction="none"
        )
        conf_loss = torch.cat([noobj_conf_loss, obj_conf_loss.reshape(-1)]).mean()

        # class loss
        best_pred_proba = best_preds[..., 1:-2]
        target_classes = targets[..., 1:-2]  
        target_classes_idx = target_classes.argmax(dim=-1)
        target_classes_idx[targets[..., 1:-2].sum(dim=-1) == 0] = self.ignore_index
        target_classes_idx = target_classes_idx.reshape(-1)
        best_pred_proba = best_pred_proba.reshape(-1, best_pred_proba.shape[-1])
        class_loss = self.ce_loss_fn(best_pred_proba, target_classes_idx)
        
        # accuracy, precision, recall
        if target_classes.max() > 0:
            _mask = target_classes_idx != self.ignore_index
            pred_labels = best_pred_proba.detach().argmax(dim=-1)[_mask].cpu().numpy()
            target_labels = target_classes_idx[_mask].cpu().numpy()
            accuracy = accuracy_score(target_labels, pred_labels)
            f1 = f1_score(target_labels, pred_labels, average="macro")
            precision = precision_score(target_labels, pred_labels, average="macro")    
            recall = recall_score(target_labels, pred_labels, average="macro")
        else:
             accuracy, f1, precision, recall = [torch.nan] * 4

        # aggregate losses
        loss = (
             (self.conf_loss_w * conf_loss) + 
             (self.ciou_loss_w * (ciou_loss) if ciou_loss == ciou_loss else torch.tensor(0.0, device=_device)) + 
             (self.class_loss_w * (class_loss) if class_loss == class_loss else torch.tensor(0.0, device=_device))
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
        # preds_cw: (S x 3 x 2)     targets_cw: (S x 2) | (S x 1 x 2)
        assert (preds_cw.ndim == targets_cw.ndim + 1) or (preds_cw.ndim == targets_cw.ndim)
        _device = preds_cw.device
        if targets_cw.ndim != preds_cw.ndim:
                targets_cw = targets_cw.unsqueeze(dim=-2)

        pred_c = preds_cw[..., :1]
        pred_w = preds_cw[..., -1:]
        pred_h = torch.ones_like(pred_w, device=_device) * _h
        pred_x1 = pred_c - (pred_w / 2)
        pred_y1 = torch.zeros_like(pred_x1, device=_device)
        pred_x2 = pred_c + (pred_w / 2)
        pred_y2 = pred_h

        target_c = targets_cw[..., :1]
        target_w = targets_cw[..., -1:]
        target_h = torch.ones_like(target_w, device=_device) * _h
        target_x1 = target_c - (target_w / 2)
        target_y1 = torch.zeros_like(target_x1, device=_device)
        target_x2 = target_c + (target_w / 2)
        target_y2 = target_h

        intersection_w = (torch.min(pred_x2, target_x2) - torch.max(pred_x1, target_x1))
        intersection_h = (torch.min(pred_y2, target_y2) - torch.max(pred_y1, target_y1))
        intersection = intersection_w * intersection_h
        union = (pred_w * pred_h) + (target_w * target_h) - intersection
        iou = intersection / (union + e)

        cw = (torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1))
        ch = (torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1))
        c2 = cw.pow(2) + ch.pow(2) + e
        v = (4 / (torch.pi**2)) * (torch.arctan(target_w / target_h) - torch.arctan(pred_w / pred_h)).pow(2)
        rho2 = (pred_c - target_c).pow(2) + (pred_h/2 - target_h/2).pow(2)
        a = v / ((1 + e) - iou) + v
        ciou = iou - ((rho2/c2) + (a * v))
        return ciou.squeeze(-1)