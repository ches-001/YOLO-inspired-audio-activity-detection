import logging
import os
import tqdm
import glob
import json
import asyncio
import argparse
import warnings
import torch
import torchaudio
import torchvision
import numpy as np
import pandas as pd
from datetime import timedelta
from modules import AudioDetectionNetwork
from train import load_config
from typing import *

warnings.filterwarnings(action="ignore")
logger = logging.getLogger(__name__)

ANNOTATIONS_DIR = "dataset/annotations"
SAVED_MODEL_DIR = "saved_model"


def load_model_weights(model: AudioDetectionNetwork, model_path: str, device: str="cpu"):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"path: {model_path} does not exist")
    state_dict = torch.load(model_path, map_location=device)
    model_weights = state_dict["network_params"]
    model.to(device)
    model.init_zeros_taper_window(model_weights["taper_window"])
    model.load_state_dict(model_weights)
    model.inference()


def get_annotation_label_map(annotations_path: str) -> Dict[int, str]:
    with open(annotations_path, "r") as f:
        annotations_dict = json.load(f)
    f.close()
    annotations = annotations_dict["annotations"]
    annotations = annotations[list(annotations.keys())[0]]

    unique_classes = []
    for filename in tqdm.tqdm(annotations.keys()):
        annotation = annotations[filename]
        segment_keys = sorted(list(annotation.keys()))
        for key in segment_keys:
            _class = annotation[key]["class"]
            if _class not in unique_classes:
                unique_classes.append(_class)
    unique_classes = sorted(unique_classes)
    label_map = {i:label for i, label in enumerate(unique_classes)}
    return label_map


def process_model_outputs(
        outputs: torch.Tensor, 
        iou_threshold: float=0.1, 
        conf_threshold: float=0.65,
        sample_duration: float=60,
        return_start_end: bool=True,
        _h: int=10,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:

    if outputs.ndim != 3:
        outputs = outputs.unsqueeze(0)
    assert outputs.ndim == 3, "input is expected to have 2 or 3 dimensions"
    _device, _dtype = outputs.device, outputs.dtype
    cw = outputs[..., -2:]
    x1 = x2 = cw[..., :1] - (cw[..., -1:] / 2)
    x2 = cw[..., :1] + (cw[..., -1:] / 2)
    y1 = torch.zeros_like(x1, device=_device, dtype=_dtype)
    y2 = torch.zeros_like(x2, device=_device, dtype=_dtype) + _h
    coords = torch.cat([x1, y1, x2, y2], dim=-1).clip(min=0, max=sample_duration).squeeze(0)
    objectness = outputs[..., :1].sigmoid()
    class_scores = torch.nn.functional.softmax(outputs[..., 1:-2], dim=-1)
    class_scores = torch.gather(class_scores, dim=-1, index=class_scores.argmax(dim=-1, keepdim=True))
    confidence = class_scores * objectness
    batch_size = outputs.shape[0]
    num_preds_per_batch = outputs.shape[1]
    num_boxes = (batch_size * num_preds_per_batch)
    batch_idx = torch.zeros(num_boxes, device=_device, dtype=torch.int64)
    _idx = 0
    for i in range(0, num_boxes, num_preds_per_batch):
        batch_idx[i:i+num_preds_per_batch] = _idx
        _idx += 1
    flattened_coords = coords.flatten(0, -2)
    flattened_confidence = confidence.reshape(-1)
    keep = torchvision.ops.batched_nms(
        flattened_coords, 
        flattened_confidence.squeeze(), 
        idxs=batch_idx, 
        iou_threshold=iou_threshold
    )
    outputs = outputs.flatten(0, -2)
    kept_batch_idx = batch_idx[keep]
    kept_confidence = flattened_confidence[keep]
    kept_outputs = outputs[keep]
    valid_mask = kept_confidence > conf_threshold
    valid_batch_idx = kept_batch_idx[valid_mask]
    valid_confidence = kept_confidence[valid_mask]
    valid_outputs = kept_outputs[valid_mask]
    valid_outputs = torch.cat([valid_confidence.unsqueeze(-1), valid_outputs], dim=-1)
    
    batch_idx_list = []
    segments_list = []
    for i in torch.unique(valid_batch_idx).sort().values:
        segments = valid_outputs[valid_batch_idx == i]
        sort_idx =  segments[:, -2].argsort()
        segments = segments[sort_idx]
        segments[..., -2] = segments[..., -2] #+ (i * sample_duration)
        segments_list.append(segments)
        batch_idx_list.append(valid_batch_idx[valid_batch_idx == i])
    segments = torch.cat(segments_list, dim=0)
    batch_idxs = torch.cat(batch_idx_list, dim=0)
    if return_start_end:
        w = segments[..., -1]
        segments[..., -2] = segments[..., -2] - (w / 2)
        segments[..., -1] = segments[..., -2] + w
        segments[..., -2:] = segments[..., -2:].clip(min=0, max=sample_duration)

    class_labels = segments[..., 2:-2].argmax(dim=-1, keepdim=True)
    segments = torch.cat([segments[..., :2], class_labels, segments[..., -2:]], dim=-1)
    return segments, batch_idxs


def evaluate_audio(
        model: AudioDetectionNetwork, 
        audio_filepath: str, 
        output_dir: str,
        model_sample_rate: int, 
        sample_duration: float, 
        batch_size: int,
        id2label_map: Dict[int, str],
        device: str="cpu",
        iou_threshold: float=0.1, 
        conf_threshold: float=0.65,
    ) -> Dict[str, Dict[str, Dict[str, Union[float, str]]]]:

    batch_start = 0
    batch_end = batch_size * sample_duration
    segments_list, batch_idxs_list = [], []
    _, og_sample_rate = torchaudio.load(audio_filepath, frame_offset=0, num_frames=10_000)
    sample_size = int(sample_duration * og_sample_rate)
    while True:
        batch_audio_tensor, _ = torchaudio.load(
            audio_filepath, 
            frame_offset=int(batch_start * og_sample_rate), 
            num_frames=int((batch_end - batch_start) * og_sample_rate),
            backend="soundfile"
        )
        if batch_audio_tensor.shape[-1] == 0: break
        if og_sample_rate != model_sample_rate:
            model.resampler = torchaudio.transforms.Resample(
                orig_freq=og_sample_rate, 
                new_freq=model_sample_rate
            )
        batch_audio_tensor = batch_audio_tensor.squeeze(0)

        if batch_audio_tensor.ndim == 2 and batch_audio_tensor.shape[0] > 1:
            batch_audio_tensor = batch_audio_tensor[0].squeeze(0)

        nbatch = batch_size
        if batch_audio_tensor.shape[0] % sample_duration != 0:
            nbatch = np.ceil(batch_audio_tensor.shape[0] / sample_size).astype(int)
            pad_size =  (nbatch * sample_size) - batch_audio_tensor.shape[0]
            _zeros = torch.zeros((pad_size, ), dtype=batch_audio_tensor.dtype)
            batch_audio_tensor = torch.cat([batch_audio_tensor, _zeros], dim=0)

        batch_audio_tensor = batch_audio_tensor.reshape(-1, 1, sample_size)
        batch_audio_tensor = batch_audio_tensor.to(device)

        with torch.no_grad():
            output: torch.Tensor = model.forward(batch_audio_tensor.to(device), combine_scales=True)
        num_class_outputs =  output.shape[-1] - 3

        if num_class_outputs not in [2, 3, 6]:
            raise RuntimeError(f"model output is expected to be 2, 3 or 6, got {output.shape[-1]}")
        segments, batch_idxs = process_model_outputs(
            output, 
            iou_threshold=iou_threshold, 
            conf_threshold=conf_threshold, 
            sample_duration=sample_duration, 
            return_start_end=True
        )
        if len(batch_idxs_list) > 0:
            batch_idxs += batch_idxs_list[-1][-1]
        segments_list.append(segments)
        batch_idxs_list.append(batch_idxs)
        batch_start = batch_end
        batch_end += batch_size * sample_duration
    
    segments = torch.cat(segments_list, dim=0)
    batch_idxs = torch.cat(batch_idxs_list, dim=0)
    segments[..., -2:] = segments[..., -2:] + (batch_idxs.unsqueeze(-1) * sample_duration)

    rle_results = []
    for i in range(0, segments.shape[0]):
        _segment = segments[i]
        start = timedelta(seconds=_segment[-2].item())
        end = timedelta(seconds=_segment[-1].item())
        class_idx = int(_segment[2].item())
        if len(rle_results) == 0 or rle_results[-1]["class"] != id2label_map[class_idx]:
            rle_results.append({"start": start, "end": end, "class": id2label_map[class_idx]})
            continue
        rle_results[-1]["end"] = end

    filename = "".join(audio_filepath.split((os.sep if os.sep in audio_filepath else "/"))[-1].split(".")[:-1])
    df = pd.DataFrame(rle_results)
    df.to_csv(os.path.join(output_dir, f"{filename}_results.csv"), index=False)


async def async_evaluate_audio(semaphore: asyncio.Semaphore, **kwargs):
    loop = asyncio.get_running_loop()
    async with semaphore:
        return await loop.run_in_executor(None, lambda : evaluate_audio(**kwargs))


async def evaluate_dir(
        model: AudioDetectionNetwork, 
        audio_dir: str, 
        output_dir: str,
        extension: str="wav", 
        num_concurrency: int=10, 
        **kwargs
    ):
    audio_filepaths = glob.glob(os.path.join(audio_dir, f"*.{extension}"))
    semaphore = asyncio.Semaphore(num_concurrency)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tasks = [
        asyncio.ensure_future(
            async_evaluate_audio(semaphore, model=model, audio_filepath=path, output_dir=output_dir, **kwargs)
        ) for path in audio_filepaths
    ]
    [await task for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]


if __name__ == "__main__": 
    config = load_config()

    model_sample_rate = config["new_sample_rate"]
    sample_duration = config["sample_duration"]
    batch_size = config["train_config"]["batch_size"]
    device = config["train_config"]["device"] if torch.cuda.is_available() else "cpu"
    model_path = f"resnet_BasicBlock_3_4_6_3/MD_mapping/{AudioDetectionNetwork.__name__}.pth.tar"
    audio_dir = "dataset/eval"
    extension = "wav"
    output_dir = "model_predictions"
    num_concurrency = 10
    iou_threshold = 0.1
    conf_threshold = 0.65

    parser = argparse.ArgumentParser(description=f"Audio model inference")

    parser.add_argument(
        "--batch_size", type=int, default=batch_size, metavar="", 
        help=f"number of segments batch to process at a time for a given audio file (default = {batch_size})"
    )
    parser.add_argument(
        "--device", type=str, default=device, choices=["cpu", "cuda"], metavar="", 
        help=f"device to run on (cuda or cpu) (default = {device})"
    )
    parser.add_argument(
        "--audio_filepath", type=str, default="", metavar="", 
        help=f"single audio file to run inference on (default = '')"
    )
    parser.add_argument(
        "--audio_dir", type=str, default=audio_dir, metavar="", 
        help=f"directory of audio files to run inference on (default = {audio_dir})"
    )
    parser.add_argument(
        "--extension", type=str, default=extension, metavar="", 
        help=f"audio files extension (wav, mp3, etc...) (default = {extension})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=output_dir, metavar="", 
        help=f"directory to store JSON model predictions (default = {output_dir})"
    )
    parser.add_argument("--model_path", type=str, default=model_path, metavar="", 
        help=f"Path to pretrained model weights (default = {model_path})"
    )
    parser.add_argument(
        "--num_concurrency", type=int, default=num_concurrency, metavar="", 
        help=f"Number of files to process at a time (default = {num_concurrency})"
    )
    parser.add_argument(
        "--iou_threshold", type=int, default=iou_threshold, metavar="", 
        help=f"IoU thresold for Non-max suppression (default = {iou_threshold})"
    )
    parser.add_argument(
        "--conf_threshold", type=int, default=conf_threshold, metavar="", 
        help=(f"inference confidence thresold: any audio segment with class confidence equal to or"
              f"below this thresold is discarded (default = {conf_threshold})")
    )

    args = parser.parse_args()
    audio_dir = args.audio_dir
    _split = args.model_path.split("/")
    annotation_filename = _split[-2]
    subdir = _split[0]
    annotations_path = os.path.join(ANNOTATIONS_DIR, f"{annotation_filename}.json")
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(f"{annotation_filename}.json file does not exist in {ANNOTATIONS_DIR}")
    id2label_map = get_annotation_label_map(annotations_path)
    num_classes = len(id2label_map)

    model = AudioDetectionNetwork(num_classes, config=config)
    load_model_weights(model, model_path=os.path.join(SAVED_MODEL_DIR, args.model_path), device=args.device)
    output_dir = os.path.join(args.output_dir, subdir, annotation_filename)
    kwargs = dict(
        model_sample_rate=model_sample_rate, 
        sample_duration=sample_duration, 
        batch_size=args.batch_size, 
        id2label_map=id2label_map,
        device=args.device,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold
    )
    if args.audio_filepath:
        if not os.path.isfile(args.audio_filepath):
            raise FileNotFoundError(f"{args.audio_file} not found")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        reesult = evaluate_audio(model, args.audio_filepath, output_dir, **kwargs)
    else:
        if not os.path.isdir(audio_dir):
            raise OSError(f"directory {audio_dir} not found")
        extension = extension.replace(".", "")
        asyncio.run(evaluate_dir(model, audio_dir, output_dir, args.extension, args.num_concurrency, **kwargs))