import os
import glob
import tqdm
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from typing import *


class AudioDataset(Dataset):
    def __init__(
            self, 
            audios_path: str, 
            annotations: Dict[str, Any], 
            anchors_dict: Dict[str, Iterable[float]],
            sample_duration: int=60,
            num_sm_segments: int=60,
            sample_rate: int=22_050,
            extension: str="wav",
            ignore_index: int=-100,
        ):
        self.audios_path = audios_path
        self.sample_duration = sample_duration
        self.num_sm_segments = num_sm_segments
        self.sample_rate = sample_rate
        self.extension = extension
        self.orig_annotations = annotations
        self.label_counts = {}
        self.label2idx = {}
        self._samples = []
        self.label_counts = {}
        self.anchors_dict = anchors_dict
        self.ignore_index = ignore_index
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
        sample_labels = torch.cat((sample_classes.unsqueeze(dim=-1), sample_times), dim=-1)

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
            pad_label = torch.tensor([self.ignore_index, _pad_center, _pad_duration]).unsqueeze(dim=0)
            sample_labels = torch.cat((sample_labels, pad_label), dim=0)

        # format labels to 3D tensors of different scales
        sm_bsegments, md_bsegments, lg_bsegments = self._get_segments_by_scale(sample_labels)
        targets = {}
        targets["sm"] = sm_bsegments
        targets["md"] = md_bsegments
        targets["lg"] = lg_bsegments
        return audio_tensor, targets
    

    def _get_segments_by_scale(self, segments: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sm_anchors = torch.Tensor(self.anchors_dict["sm"])
        md_anchors = torch.Tensor(self.anchors_dict["md"])
        lg_anchors = torch.Tensor(self.anchors_dict["lg"])
        anchors = torch.cat((sm_anchors, md_anchors, lg_anchors)).unsqueeze(-1)
        cdist = torch.cdist(segments[:, -1, None], anchors)
        segment_anchors_idx = torch.argmin(cdist, dim=-1)
        num_anchors_per_scale = len(self.anchors_dict["sm"])
        scales = torch.ceil((segment_anchors_idx + 1) / num_anchors_per_scale) - 1
        sm_segments = segments[scales == 0]
        md_segments = segments[scales == 1]
        lg_segments = segments[scales == 2]
        num_sm_segments = self.num_sm_segments
        num_md_segments = num_sm_segments // 2
        num_lg_segments = num_md_segments // 2
        # generate bounding segments (similar to bounding boxes in YOLO)
        # first index of last dimension corresponds to confidence score
        # second index of last dimension corresponds to label
        # third index of last dimension corresponds to center of segment
        # last index of last dimension corresponds to duration / width of segment
        sm_bsegments = torch.zeros((num_sm_segments, 4), dtype=torch.float32)
        md_bsegments = torch.zeros((num_md_segments, 4), dtype=torch.float32)
        lg_bsegments = torch.zeros((num_lg_segments, 4), dtype=torch.float32)
        if sm_segments.numel() > 0:
            num_cells = self.sample_duration / num_sm_segments
            sm_cellidx = torch.ceil((sm_segments[:, 1] / num_cells) - 1).to(dtype=torch.int64)
            sm_bsegments[sm_cellidx, 0] = 1
            sm_bsegments[sm_cellidx, 1:] = sm_segments
        if md_segments.numel() > 0:
            num_cells = self.sample_duration / num_md_segments
            md_cellidx = torch.ceil((md_segments[:, 1] / num_cells) - 1).to(dtype=torch.int64)
            md_bsegments[md_cellidx, 0] = 1
            md_bsegments[md_cellidx, 1:] = md_segments
        if lg_segments.numel() > 0:
            num_cells = self.sample_duration / num_lg_segments
            lg_cellidx = torch.ceil((lg_segments[:, 1] / num_cells) - 1).to(dtype=torch.int64)
            lg_bsegments[lg_cellidx, 0] = 1
            lg_bsegments[lg_cellidx, 1:] = lg_segments
        return sm_bsegments, md_bsegments, lg_bsegments


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
            for key in sorted(list(annotation.keys())):
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

