import warnings
warnings.filterwarnings(action="ignore")
import logging
import os
import glob
import yaml
import json
import random
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from dataset import AudioDataset, AudioConcatDataset
from modules import AudioDetectionNetwork, AudioDetectionLoss
from smoothener import EMAParamsSmoothener
from pipeline import TrainerPipeline
from typing import *

SEED = 42
CONFIG_PATH = "config/config.yaml"
NUM_WORKERS = os.cpu_count()
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    f.close()
    return config

def load_annotations(data_path: str, annotator: str) -> Dict[str, Any]:
    path = os.path.join(data_path, "annotations", "annotation.json")
    with open(path, "r") as f:
        data = json.load(f)
    f.close()
    return data["annotations"][annotator]

def make_dataset(
        path: Union[str, List[str]], 
        annotations: Union[Dict[str, Any], List[Dict[str, Any]]], 
        config: Dict[str, Any]) -> Union[AudioDataset, AudioConcatDataset]:
    
    kwargs = dict(
        sample_duration=config["sample_duration"],
        sample_rate=config["sample_rate"],
        extension=config["audio_extension"],
    )
    if isinstance(path, str) and isinstance(annotations, dict):
        dataset = AudioDataset(path, annotations, **kwargs)
    elif isinstance(path, list) and isinstance(annotations, list):
        dataset = AudioConcatDataset.make_combo_dataset(path, annotations, **kwargs)
    else:
        raise Exception("expects path and annotations to be str and dict or list of str and list of dict")
    return dataset

def make_dataloader(dataset: AudioDataset, config: Dict[str, Any]) -> DataLoader:
    kwargs = dict(
        num_workers=NUM_WORKERS, 
        batch_size=config["train_config"]["batch_size"], 
        shuffle=config["train_config"]["shuffle_samples"]
    )
    return DataLoader(dataset, collate_fn=AudioDataset.collate_fn, **kwargs)

def make_model(config: Dict[str, Any], num_classes: int) -> AudioDetectionNetwork:
    model = AudioDetectionNetwork(num_classes=num_classes, config=config)
    model.train()
    return model

def make_loss_fn(config: Dict[str, Any], num_classes: int, class_weights: torch.Tensor) -> AudioDetectionLoss:
    return AudioDetectionLoss(
        anchors_dict=config["anchors"],
        num_classes=num_classes,
        sample_duration=config["sample_duration"],
        class_weights=class_weights,
        **config["train_config"]["loss_config"]
    )

def make_optimizer(model: AudioDetectionNetwork, config: Dict[str, Any]) -> torch.optim.Optimizer:
    config = config.copy()
    optimizer_name = config["train_config"]["optimizer_config"].pop("name")
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), **config["train_config"]["optimizer_config"]
    )
    return optimizer

def make_lr_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler:
    config = config.copy()
    scheduler_name = config["train_config"]["lr_scheduler_config"].pop("name")
    lr_scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
        optimizer, **config["train_config"]["lr_scheduler_config"]
    )
    return lr_scheduler

def run(config: Dict[str, Any]):
    device = config["train_config"]["device"] if torch.cuda.is_available() else "cpu"
    data_path: str = config["train_config"]["dataset_path"]
    split_data_paths = data_path.split(";")
    annotator = config["train_config"]["annotator"]

    if (not data_path.endswith("*")) and len(split_data_paths) == 1:
        train_data_path = os.path.join(data_path, "train")
        eval_data_path = os.path.join(data_path, "eval")
        annotations = load_annotations(data_path, annotator)
        dataset_name = data_path.split("/")[-1]
        train_dataset = make_dataset(train_data_path, annotations, config)
        eval_dataset = make_dataset(eval_data_path, annotations, config)

    elif data_path.endswith("*") or len(split_data_paths) > 1:
        annotations_list, train_data_paths, eval_data_paths = [], [], []
        _dnames = []
        if len(split_data_paths) > 1:
            data_paths = split_data_paths
        else:
            data_paths = glob.glob(data_path)
        for i, path in enumerate(data_paths):
            if not os.path.exists(path):
                raise OSError(f"path {path} not found")
            annotations_list.append(load_annotations(path, annotator))
            train_data_paths.append(os.path.join(path, "train"))
            eval_data_paths.append(os.path.join(path, "eval"))
            _dnames.append(path.split(os.sep)[-1])
        dataset_name = "-".join(sorted(_dnames))
        train_dataset = make_dataset(train_data_paths, annotations_list, config)
        eval_dataset = make_dataset(eval_data_paths, annotations_list, config)

    else:
        raise Exception(f"Invalid data path {data_path}")
    
    model_path = os.path.join(config["train_config"]["model_path"], dataset_name)
    metrics_path = os.path.join(config["train_config"]["metrics_path"], dataset_name)
    class_map_path = os.path.join(config["train_config"]["class_map_path"], dataset_name)
    AudioDataset.save_label_map(train_dataset.class2idx, class_map_path)
    train_dataloader = make_dataloader(train_dataset, config)
    eval_dataloader = make_dataloader(eval_dataset, config)

    num_classes = len(train_dataset.class2idx)
    model = make_model(config, num_classes=num_classes)
    model.to(device)
    loss_fn = make_loss_fn(config, num_classes=num_classes, class_weights=train_dataset.get_class_weights(device=device))
    optimizer = make_optimizer(model, config)
    lr_scheduler = None
    if config["train_config"]["use_lr_scheduler"]:
        lr_scheduler = make_lr_scheduler(optimizer, config)

    backbone = config["backbone"]
    block_layers = config["block_layers"]
    blocks_str = "_".join(map(lambda x : str(x), block_layers))
    if backbone == "custom":
        model_path = os.path.join(model_path, f"{backbone}_{blocks_str}")
        metrics_path = os.path.join(metrics_path, f"{backbone}_{blocks_str}")
    elif backbone == "resnet":
        block = config["resnet_config"]["block"]
        model_path = os.path.join(model_path, f"{backbone}_{block}_{blocks_str}")
        metrics_path = os.path.join(metrics_path, f"{backbone}_{block}_{blocks_str}")

    use_ema = config["train_config"]["use_ema"]
    ema_smoothener = None
    if use_ema:
        ema_smoothener = EMAParamsSmoothener(model, **config["train_config"]["ema_config"])
    trainer_pipeline = TrainerPipeline(
        model, 
        loss_fn, 
        optimizer, 
        model_path=model_path, 
        metrics_path=metrics_path, 
        device=device,
        ema_smoothener=ema_smoothener
    )
    verbose = config["train_config"]["verbose"]
    epochs = config["train_config"]["epochs"]

     # training loop
    best_loss = np.inf
    for epoch in range(0, epochs):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{current_time}]: Epoch {epoch}")
        train_metrics = trainer_pipeline.train(train_dataloader, verbose=verbose)
        eval_metrics = trainer_pipeline.evaluate(eval_dataloader, verbose=verbose)
        eval_loss = eval_metrics["aggregate_loss"]
        if eval_loss < best_loss:
            trainer_pipeline.save_model()
            best_loss = eval_loss
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}] Model saved at epoch: {epoch+1} loss: {best_loss}")
        if lr_scheduler:
            lr_scheduler.step()
    trainer_pipeline.metrics_to_csv()
    trainer_pipeline.save_metrics_plots(figsize=(25, 10))


if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # torch.autograd.set_detect_anomaly(True)
    LOG_FORMAT="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    LOG_DATE_FORMAT="%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    config = load_config()
    run(config)