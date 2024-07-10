import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Optional, Dict

class AudioDetectionLoss(nn.Module):
    def __init__(
        self, 
        segment_loss_w: float=1.0,
        obj_loss_w: float=1.0,
        noobj_loss_w: float=0.1,
        class_loss_w: float=1.0,
        ignore_index: int=-100, 
        class_weights: Optional[torch.Tensor]=None,
        scale_t: Optional[int]=None,
    ):
        assert isinstance(ignore_index, int) and ignore_index < 0, "ignore_index must be an integer less than zero"
        super(AudioDetectionLoss, self).__init__()
        self.segment_loss_w = segment_loss_w
        self.obj_loss_w = obj_loss_w
        self.noobj_loss_w = noobj_loss_w
        self.class_loss_w = class_loss_w
        self.ignore_index = ignore_index
        self.bce_loss_fn = nn.BCELoss(reduction="none")
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index, reduction="mean"
        )
        self.scale_t = scale_t if scale_t else 1

    def forward(
            self, 
            preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
            targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        sm_preds, md_preds, lg_preds = preds
        sm_targets, md_targets, lg_targets = targets
        
        sm_loss, sm_loss_dict = self.loss_fn(sm_preds, sm_targets)
        md_loss, md_loss_dict = self.loss_fn(md_preds, md_targets)
        lg_loss, lg_loss_dict = self.loss_fn(lg_preds, lg_targets)
        loss = (sm_loss + md_loss + lg_loss) / 3

        loss_dict = {}
        loss_dict["aggregate_loss"] = loss.item()
        loss_dict["segment_loss"] = (sm_loss_dict["segment_loss"] + md_loss_dict["segment_loss"] + lg_loss_dict["segment_loss"]) / 3
        loss_dict["obj_loss"] = (sm_loss_dict["obj_loss"] + md_loss_dict["obj_loss"] + lg_loss_dict["obj_loss"]) / 3
        loss_dict["noobj_loss"] = (sm_loss_dict["noobj_loss"] + md_loss_dict["noobj_loss"] + lg_loss_dict["noobj_loss"]) / 3
        loss_dict["class_loss"] = (sm_loss_dict["class_loss"] + md_loss_dict["class_loss"] + lg_loss_dict["class_loss"]) / 3
        loss_dict["accuracy"] = (sm_loss_dict["accuracy"] + md_loss_dict["accuracy"] + lg_loss_dict["accuracy"]) / 3
        loss_dict["f1"] = (sm_loss_dict["f1"] + md_loss_dict["f1"] + lg_loss_dict["f1"]) / 3
        loss_dict["precision"] = (sm_loss_dict["precision"] + md_loss_dict["precision"] + lg_loss_dict["precision"]) / 3
        loss_dict["recall"] = (sm_loss_dict["recall"] + md_loss_dict["recall"] + lg_loss_dict["recall"]) / 3
        return loss, loss_dict


    def loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert preds.ndim == targets.ndim + 1
        # sm_pred: (S x 3 x (3 + C))     sm_target: (S, 4)
        ious_per_anchors = AudioDetectionLoss.compute_iou(preds[..., -2:], targets[..., -2:])
        ious_max, best_anchoridx = torch.max(ious_per_anchors, dim=-1)
        best_anchoridx = best_anchoridx.unsqueeze(-1).unsqueeze(-1)
        if preds.ndim == 4:
             best_anchoridx = best_anchoridx.expand(*preds.shape[0:2], -1, preds.shape[-1])
             best_preds = torch.gather(preds, dim=2, index=best_anchoridx)
             best_preds = best_preds.squeeze(dim=2)
        else:
             best_anchoridx = best_anchoridx.expand(preds.shape[0], -1, preds.shape[-1])
             best_preds = torch.gather(preds, dim=1, index=best_anchoridx)
             best_preds = best_preds.squeeze(dim=1)
        
        # target objectness scores (1 if object, 0 if no object)
        target_objectness = targets[..., :1]

        # segment center and duration loss
        pred_segments = best_preds[..., -2:] / self.scale_t
        target_segments = targets[..., -2:] / self.scale_t
        segment_loss = torch.nn.functional.mse_loss(
             (target_objectness*pred_segments), (target_objectness*target_segments), reduction="none"
        )
        segment_loss = segment_loss.sum(dim=(1, 2)).mean()

        # confidence / objectness loss
        bce_loss = self.bce_loss_fn(best_preds[..., :1], targets[..., :1])
        obj_loss = target_objectness * bce_loss
        noobj_loss = (1 - target_objectness) * bce_loss
        obj_loss = obj_loss.sum(dim=(1, 2)).mean()
        noobj_loss = noobj_loss.sum(dim=(1, 2)).mean()

        # class loss
        pred_probs = best_preds[..., 1:-2].flatten(0, -2)
        target_classes = targets[..., 1:2].clone()
        target_classes[target_objectness == 0] = self.ignore_index
        target_classes = target_classes.flatten(0, -1).to(device=targets.device, dtype=torch.int64)
        class_loss = self.ce_loss_fn(pred_probs, target_classes)
        if class_loss != class_loss: # if class_loss is NaN:
             class_loss = torch.tensor(0.0, dtype=pred_probs.dtype, device=pred_probs.device)

        # accuracy, precision, recall
        pred_labels = pred_probs[target_classes != self.ignore_index].detach().argmax(dim=-1).cpu().numpy()
        target_labels = target_classes[target_classes != self.ignore_index].cpu().numpy()
        if pred_labels.shape[0] != 0:
            accuracy = accuracy_score(target_labels, pred_labels)
            f1 = f1_score(target_labels, pred_labels, average="macro")
            precision = precision_score(target_labels, pred_labels, average="macro")    
            recall = recall_score(target_labels, pred_labels, average="macro")
        else:
             accuracy, f1, precision, recall = [0] * 4

        # aggregate losses
        loss = (
             (self.segment_loss_w * segment_loss) + 
             (self.obj_loss_w * obj_loss) + 
             (self.noobj_loss_w * noobj_loss) + 
             (self.class_loss_w * class_loss)
        )
        loss_dict = {}
        loss_dict["segment_loss"] = segment_loss.item()
        loss_dict["obj_loss"] = obj_loss.item()
        loss_dict["noobj_loss"] = noobj_loss.item()
        loss_dict["class_loss"] = class_loss.item()
        loss_dict["accuracy"] = accuracy
        loss_dict["f1"] = f1
        loss_dict["precision"] = precision
        loss_dict["recall"] = recall  
        return loss, loss_dict

    @staticmethod
    def compute_iou(preds_cw: torch.Tensor, targets_cw: torch.Tensor, e: float=1e-15) -> torch.Tensor:
        # preds_cw: (S x 3 x 2)     targets_cw: (S x 2) | (S x 3 x 2)
        assert (preds_cw.ndim == targets_cw.ndim + 1) or (preds_cw.ndim == targets_cw.ndim)
        if targets_cw.ndim != preds_cw.ndim:
                targets_cw = targets_cw.unsqueeze(dim=-2)
        preds_w, targets_w = preds_cw[..., 1], targets_cw[..., 1]
        preds_c, targets_c = preds_cw[..., 0], targets_cw[..., 0]
        preds_x1, targets_x1 = preds_c - (preds_w / 2), targets_c - (targets_w / 2)
        preds_x2, targets_x2 = preds_c + (preds_w / 2), targets_c + (targets_w / 2)
        intersection = torch.min(preds_x2, targets_x2) - torch.max(preds_x1, targets_x1)
        intersection = torch.clip(intersection, min=0)
        union = torch.max(preds_x2, targets_x2) - torch.min(preds_x1, targets_x1)
        union = union + e
        iou = intersection / union
        return iou