import os
import tqdm
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import *


class AudioDataset(Dataset):
    def __init__(
            self, 
            audios_path: str, 
            annotations: Dict[str, Any], 
            anchors_dict: Dict[str, Iterable[float]],
            sample_duration: int=60,
            sample_rate: int=22_050,
            extension: str="wav",
        ):
        self.audios_path = audios_path
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate
        self.extension = extension
        self.orig_annotations = annotations
        self.label_counts = {}
        self.label2idx = {}
        self._samples = []
        self.label_counts = {}
        self.anchors_dict = anchors_dict
        audio_filenames = [i.replace(f".{extension}", "") for i in os.listdir(self.audios_path)]
        annotations = {k:v for k, v in self.orig_annotations.items() if k in audio_filenames}
        self._index_samples(annotations)


    def __len__(self) -> int:
        return len(self._samples)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[idx]
        filename = sample["filename"]
        sample = sample["sample"]
        sample_times = sample[:, :2].astype(float)
        sample_classes = sample[:, -1]
        filepath = os.path.join(self.audios_path, f"{filename}.{self.extension}")
        audio_start, audio_end = sample_times[0][0], sample_times[-1][1]
        audio_start = float(audio_start)
        audio_end = float(audio_end)
        audio_tensor, _ = torchaudio.load(
            filepath,
            frame_offset=int(audio_start * self.sample_rate),
            num_frames=int((audio_end - audio_start) * self.sample_rate),
            backend="soundfile"
        )
        if audio_tensor.shape[-1] > (self.sample_duration * self.sample_rate):
            raise Exception(
                f"audio sample is more than {self.sample_duration}, ensure that "
                f"the specified sample rate value ({self.sample_rate}) is correct"
            )
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.shape[0] != 1:
            audio_tensor = audio_tensor.mean(dim=0).unsqueeze(0)

        sample_classes = torch.from_numpy(
            np.vectorize(lambda label : self.label2idx[label])(sample_classes)
        ).to(torch.int64) 
        # sample segment length instead of sample segment end (aspar YOLO convention)
        sample_times[:, 1] = (sample_times[:, 1] - sample_times[:, 0])
        # segment center instead of segment start (aspar YOLO convention)
        sample_times[:, 0] = sample_times[:, 0] + (sample_times[:, 1] / 2)
        sample_times = torch.from_numpy(sample_times).to(dtype=torch.float32)
        sample_labels = torch.cat((sample_classes[:, None], sample_times), dim=-1)

        # pad audio file if audio file duration not up to `sample_duration`
        max_num_samples = self.sample_duration * self.sample_rate
        if audio_tensor.shape[-1] < max_num_samples:
            pad = torch.zeros(
                (audio_tensor.shape[0], max_num_samples - audio_tensor.shape[-1]),
                dtype=audio_tensor.dtype
            )
            audio_tensor = torch.cat((audio_tensor, pad), dim=-1)
            _pad_duration = (audio_start + self.sample_duration) - audio_end
            _pad_center = audio_end + (_pad_duration / 2)
            pad_label = torch.tensor([torch.zeros(0), _pad_center, _pad_duration]).unsqueeze(dim=0)
            sample_labels = torch.cat((sample_labels, pad_label), dim=0)

        targets = torch.zeros((sample_labels.shape[0], sample_labels.shape[1]+1), dtype=sample_labels.dtype)
        targets[:, 1:] = sample_labels
        return audio_tensor, targets


    def get_class_weights(self, device: Optional[str]=None) -> torch.Tensor:
        if not device: device = "cpu"
        label_weights = list(self.label_counts.values())
        label_weights = torch.tensor(label_weights, dtype=torch.float32, device=device)
        label_weights = label_weights.sum() / (label_weights.shape[0] * label_weights)
        return label_weights


    def _index_samples(self, annotations: Dict[str, Any]):
        self._samples = []
        unique_classes = []
        class_counts = {}
        for filename in tqdm.tqdm(annotations.keys()):
            annotation = annotations[filename]
            segment_keys = sorted(list(annotation.keys()))
            sample = []
            for key in segment_keys:
                _class = annotation[key]["class"]
                if _class not in unique_classes:
                    unique_classes.append(_class)
                
                if _class not in class_counts:
                    class_counts[_class] = 1
                else:
                    class_counts[_class] += 1
                sample.append([annotation[key]["start"], annotation[key]["end"], _class])

            sample = np.asarray(sample)
            sample = {"filename": filename, "sample":sample}
            self._samples.append(sample)
        
        unique_classes = sorted(unique_classes)
        self.label2idx = {label:i for i, label in enumerate(unique_classes)}
        self.label_counts = {k:class_counts[k] for k in unique_classes}


    @staticmethod
    def collate_fn(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_signals, targets = zip(*batch)
        for i, target in enumerate(targets):
            target[:, 0] = i
        audio_signals = torch.stack(audio_signals, dim=0)
        targets = torch.cat(targets, dim=0)
        return audio_signals, targets
    
    @staticmethod
    def build_target_by_scale(
            targets: torch.Tensor, 
            fmap_shape: Union[int, torch.Size],
            anchors: Union[List[float], torch.Tensor],
            anchor_threshold: float=4.0,
            sample_duration: float=60,
        ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        # targets; (batch_idx, cls, center, duration)
        g = 0.5
        # match segments to anchors
        if not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(anchors)
        if isinstance(fmap_shape, torch.Size):
            fmap_shape = fmap_shape[0]

        _device = targets.device
        num_anchors = anchors.shape[0]
        num_targets = targets.shape[0]
        targets = targets.unsqueeze(dim=0).tile(num_anchors, 1, 1)
        anchor_idx = torch.arange(num_anchors).unsqueeze(dim=-1).tile(1, num_targets).to(_device)
        targets = torch.cat([targets, anchor_idx[..., None]], dim=-1)
        # compute the ratio between the segment duration and the anchors, segments that are
        # (anchor_threshold x) more or (anchor_threshold x) less than the corresponding achors
        # are filtered, while the rest are discarded
        r = targets[..., -2:-1] / anchors[:, None, None]
        targets = targets[torch.max(r, 1/r).max(dim=-1).values < anchor_threshold]

        # this variable scales the segment width / duration down within the range of 0 to 1
        # then multiplies by the fmap_shape (number of grids cells for a given scale)
        # by doing so, we can determine what cell a given audio segment's center lies in
        grid_c = (targets[..., -3:-2] / sample_duration) * fmap_shape
        grid_i = fmap_shape - grid_c
        c_mask = ((grid_c % 1 < g) & (grid_c > 1)).T
        i_mask = ((grid_i % 1 < g) & (grid_i > 1)).T
        mask = torch.cat([torch.ones_like(c_mask), c_mask, i_mask])
        targets = targets.repeat(mask.shape[0], 1, 1)[mask]

        batch_idx = targets[:, 0].long()
        anchor_idx = targets[:, -1].long()
        classes = targets[:, 1].long()
        cw = targets[:, 2:-1]
        gid_c = ((cw[:, 0] / sample_duration) * fmap_shape).long().clip(min=0, max=fmap_shape)
        anchors = anchors[anchor_idx]
        indices = [batch_idx, gid_c, anchor_idx]
        return indices, classes, cw
