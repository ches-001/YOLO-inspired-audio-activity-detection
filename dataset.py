import os
import json
import tqdm
import logging
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from typing import *

logger = logging.getLogger(__name__)


class AudioConcatDataset(ConcatDataset):
    datasets: List["AudioDataset"]

    def __init__(self, *args, **kwargs):
        super(AudioConcatDataset, self).__init__(*args, **kwargs)
        self.class2idx, self.class_counts = self.__combine_class_counts()


    def __add__(self, other: Union["AudioDataset", "AudioConcatDataset"]) -> "AudioConcatDataset":
        if not isinstance(other, (AudioDataset, AudioConcatDataset)):
            raise ValueError(f"cannot add / concat {self.__class__.__name__} and {type(other)} together")
        return AudioConcatDataset([self, other])
    

    def __combine_class_counts(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        class2idx = {}
        class_counts = {}
        for dataset in self.datasets:
            classnames = list(dataset.class_counts.keys())
            for _cn in classnames:
                if _cn not in class2idx.keys():
                    class2idx[_cn] = dataset.class2idx[_cn]
                    class_counts[_cn] = dataset.class_counts[_cn]
                else:
                    class2idx[_cn] += dataset.class2idx[_cn]
                    class_counts[_cn] += dataset.class_counts[_cn]

        unique_classes = sorted(list(class2idx.keys()))
        class2idx = {label:i for i, label in enumerate(unique_classes)}
        class_counts = {k:class_counts[k] for k in unique_classes}

        for dataset in self.datasets:
            dataset.class2idx = class2idx

        return class2idx, class_counts


    def get_class_weights(self, device: Optional[str]=None) -> torch.Tensor:
        if not device: device = "cpu"
        label_weights = list(self.class_counts.values())
        label_weights = torch.tensor(label_weights, dtype=torch.float32, device=device)
        label_weights = label_weights.sum() / (label_weights.shape[0] * label_weights)
        return label_weights
    

    @classmethod
    def make_combo_dataset(cls, audio_paths: List[str], annotations_list: List[Dict[str, Any]], **kwargs):
        datasets = None
        for audio_path, annotations in zip(audio_paths, annotations_list):
            if datasets is None:
                datasets = AudioDataset(audio_path, annotations, **kwargs)
            else:
                datasets += AudioDataset(audio_path, annotations, **kwargs)
        return datasets


class AudioDataset(Dataset):
    def __init__(
            self, 
            audios_path: str, 
            annotations: Dict[str, Any], 
            sample_duration: int=60,
            sample_rate: int=22_050,
            extension: str="wav",
            ignore_index: int=-100,
        ):
        self.audios_path = audios_path
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate
        self.extension = extension
        self.ignore_index = ignore_index
        audio_filenames = [i.replace(f".{extension}", "") for i in os.listdir(self.audios_path)]
        annotations = {k:v for k, v in annotations.items() if k in audio_filenames}

        if not AudioDataset.is_grouped_annotations(annotations):
            self._samples, self.class2idx, self.class_counts = self._index_samples(annotations)
        else:
            self._samples, self.class2idx, self.class_counts = self._index_grouped_samples(annotations)
        self.ignore_index = ignore_index


    def __len__(self) -> int:
        return len(self._samples)
    

    def __add__(self, other: "AudioDataset") -> AudioConcatDataset:
        return AudioConcatDataset([self, other])


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[idx]
        filename = sample["filename"]
        gmin = 0
        if "group_minmax" in sample.keys():
            gmin, _ = sample["group_minmax"]
        sample = sample["sample"]

        sample_times = sample[:, :2].astype(float)
        sample_classes = sample[:, -1]
        filepath = os.path.join(self.audios_path, f"{filename}.{self.extension}")
        audio_start, audio_end = sample_times[0][0], sample_times[-1][1]
        audio_tensor, _ = torchaudio.load(
            filepath,
            frame_offset=int(audio_start * self.sample_rate),
            num_frames=int((audio_end - audio_start) * self.sample_rate),
            backend="soundfile"
        )
        # scale by gmain (applicable to grouped annotations to ensure start and ends are within 
        # range of 0 and sample duration), inotherwords, treating each group as its own seperate
        # audio file with duration equal to sample duration (default to 60secs). if annotations
        # not grouped, then gmin = 0.
        audio_start, audio_end = audio_start - gmin, audio_end - gmin
        sample_times -= gmin

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
            np.vectorize(lambda label : self.class2idx[label])(sample_classes)
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
            zeros = torch.zeros(1).fill_(self.ignore_index)
            pad_label = torch.tensor([zeros, _pad_center, _pad_duration], dtype=sample_labels.dtype).unsqueeze(dim=0)
            sample_labels = torch.cat((sample_labels, pad_label), dim=0)

        targets = torch.zeros((sample_labels.shape[0], sample_labels.shape[1]+1), dtype=sample_labels.dtype)
        targets[:, 1:] = sample_labels
        return audio_tensor, targets


    def get_class_weights(self, device: Optional[str]=None) -> torch.Tensor:
        if not device: device = "cpu"
        label_weights = list(self.class_counts.values())
        label_weights = torch.tensor(label_weights, dtype=torch.float32, device=device)
        label_weights = label_weights.sum() / (label_weights.shape[0] * label_weights)
        return label_weights


    def _index_samples(self, annotations: Dict[str, Any]) -> Tuple[List[Any], Dict[str, int], Dict[str, int]]:
        _samples = []
        unique_classes = []
        class_counts = {}
        for filename in tqdm.tqdm(annotations.keys()):
            annotation = annotations[filename]
            segment_keys = sorted(list(annotation.keys()))
            # if len(segment_keys) == 1:
            #     continue
            file_duration = annotation[segment_keys[-1]]["end"] - annotation[segment_keys[0]]["start"]
            if file_duration > self.sample_duration:
                logger.warn(
                    f"duration of {filename} is more than {self.sample_duration} "
                    "and will not be included in the processed dataset"
                )
                continue
            sample = []
            for key in segment_keys:
                _class = annotation[key]["class"]
                _class = _class.strip().replace(" ", "-")
                if _class not in unique_classes:
                    unique_classes.append(_class)
                
                if _class not in class_counts:
                    class_counts[_class] = 1
                else:
                    class_counts[_class] += 1
                sample.append([annotation[key]["start"], annotation[key]["end"], _class])

            sample = np.asarray(sample)
            sample = {"filename": filename, "sample":sample}
            _samples.append(sample)
        
        unique_classes = sorted(unique_classes)
        class2idx = {label:i for i, label in enumerate(unique_classes)}
        class_counts = {k:class_counts[k] for k in unique_classes}
        return _samples, class2idx, class_counts

    
    def _index_grouped_samples(self, annotations: Dict[str, Any]) -> Tuple[List[Any], Dict[str, int], Dict[str, int]]:
        _samples = []
        unique_classes = []
        class_counts = {}
        for filename in tqdm.tqdm(annotations.keys()):
            groups = annotations[filename]
            group_keys = sorted(list(groups.keys()), key=lambda k : int(k.split("-")[-1]))
            gmin, gmax = 0, self.sample_duration
            for group in group_keys:
                annotation = groups[group]
                segment_keys = sorted(list(annotation.keys()))
                # if len(segment_keys) == 1:
                #     continue
                group_duration = annotation[segment_keys[-1]]["end"] - annotation[segment_keys[0]]["start"]
                if group_duration > self.sample_duration:
                    logger.warn(
                        f"the duration of {group} of {filename} is more than {self.sample_duration} "
                        "and will not be included in the processed dataset"
                    )
                    continue
                sample = []
                for key in segment_keys:
                    _class = annotation[key]["class"]
                    _class = _class.strip().replace(" ", "-")
                    if _class not in unique_classes:
                        unique_classes.append(_class)
                    
                    if _class not in class_counts:
                        class_counts[_class] = 1
                    else:
                        class_counts[_class] += 1
                    sample.append([annotation[key]["start"], annotation[key]["end"], _class])

                sample = np.asarray(sample)
                sample = {"filename": filename, "group_minmax": np.asarray([gmin, gmax]), "sample": sample}
                _samples.append(sample)
                gmin, gmax = gmax, gmax+self.sample_duration
        
        unique_classes = sorted(unique_classes)
        class2idx = {label:i for i, label in enumerate(unique_classes)}
        class_counts = {k:class_counts[k] for k in unique_classes}
        return _samples, class2idx, class_counts
    

    @staticmethod
    def save_label_map(class2idx_map: Dict[str, int], _dir: str):
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        idx2class_map = {v:k for k, v in class2idx_map.items()}
        with open(os.path.join(_dir, "class_map.json"), "w") as f:
            json.dump(idx2class_map, f)
        f.close()


    @staticmethod    
    def is_grouped_annotations(annotations: Dict[str, Any]) -> bool:
        filenames = list(annotations.keys())
        groups_or_segments = annotations[filenames[0]]
        ks = list(groups_or_segments.keys())
        return ks[0].startswith("group")


    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
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
            edge_threshold: float=0.5
        ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        # targets; (batch_idx, cls, center, duration)
        # match segments to anchors
        if not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(anchors)
        if isinstance(fmap_shape, torch.Size):
            fmap_shape = fmap_shape[0]

        _device = targets.device
        anchors = anchors.to(_device)
        num_anchors = anchors.shape[0]
        num_targets = targets.shape[0]
        targets = targets.unsqueeze(dim=0).tile(num_anchors, 1, 1)
        anchor_idx = torch.arange(num_anchors, device=_device).unsqueeze(dim=-1).tile(1, num_targets)
        targets = torch.cat([targets, anchor_idx[..., None]], dim=-1)
        
        # compute the ratio between the segment duration and the anchors, segments that are
        # (anchor_threshold x) more or (anchor_threshold x) less than the corresponding achors
        # are filtered, while the rest are discarded
        r = targets[..., -2:-1] / anchors[:, None, None]
        targets = targets[torch.max(r, 1/r).squeeze(dim=-1) < anchor_threshold]

        # this line scales the segment width / duration down within the range of 0 to 1
        # then multiplies by the fmap_shape (number of grids cells for a given scale)
        # by doing so, we can determine what cell a given audio segment's center lies in
        grid_c = (targets[..., -3:-2] / sample_duration) * fmap_shape

        # this line computes the grid of segment centers in reverse order, imagine it as
        # a method where the grid is flipped left to right to get the reverse locations
        grid_i = fmap_shape - grid_c

        # next we look for segments, whose corresponding centers are close to the edge of 
        # its grid cell (either leftwards or rightwards) while simultanously not being at
        # the edge of the image / spectrogram (in the first grid or in the last grid)
        c_mask = ((grid_c % 1 < edge_threshold) & (grid_c > 1)).T
        i_mask = ((grid_i % 1 < edge_threshold) & (grid_i > 1)).T

        # we formulate a mask that includes all the selected targets as well as all the targets
        # that satisfy the conditions of the c_mask and i_mask (Note that there will mostly always
        # be repititions of segments in the final targets)
        mask = torch.cat([torch.ones_like(c_mask), c_mask, i_mask])
        targets = targets.repeat(mask.shape[0], 1, 1)[mask]

        # based on the masks generated above, we formulate offsets. The idea here is that for a certain
        # segment whose center falls close to the left edge of a grid cell, there is also a chance that 
        # the preceding grid cell can be capable of predicting that segment. Similarly, if the center
        # falls close to the right edge of a grid cell, there is also a good chance that the superseding
        # grid cell can also be capable of predicting that segment. Hence the offsets are made such that
        # when such segments are found, we match them against the predictions at its grid cell, as well as
        # the preceding and superseding grid cells if applicable.
        # Eg: suppose we have a segment centered at 40.89 sec with duration of 10 sec in a 60secs signal, 
        # with a grid map of 120 cells, this segment will fall in the cell: floor((40.89 / 60) * 120) = 81
        # note that (40.89 / 60) * 120) = 81.78, the decimal part (0.78) is greater than our edge_threshold
        # of 0.5, hence it implies that this segment is close to the right edge of its corresponding grid cell
        # (81) hence we include cell 81 and cell 82 in the index of cells as cells that are capable of 
        # predicting this segment.
        # The offsets are defined such 0 corresponds to the offset of the original grids, indicating no changes
        # to them, -1 indicates a leftward shift to the preceding grid cell if segment's center is close to the
        # left edge of its grid, and 1 indicates a rightward shift to next grid cell if segment's center is 
        # close to the right edge of its grid.
        offset = torch.tensor([[0], [-1], [1]], device=_device, dtype=targets.dtype) * edge_threshold
        offset = (torch.zeros_like(grid_c)[None] + offset[:, None])[mask]

        batch_idx = targets[:, 0].long()
        anchor_idx = targets[:, -1].long()
        classes = targets[:, 1].long()
        cw = targets[:, 2:-1]
        grid_idx = ((cw[:, 0] / sample_duration) * fmap_shape) + offset.squeeze(dim=-1)
        grid_idx = grid_idx.long().clip(min=0, max=fmap_shape-1)
        anchors = anchors[anchor_idx]
        indices = [batch_idx, grid_idx, anchor_idx]
        return indices, classes, cw
