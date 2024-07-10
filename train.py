import os
import yaml
import json
import random
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from dataset import AudioDataset
from modules import AudioDetectionNetwork, AudioDetectionLoss
from pipeline import TrainerPipeline
from typing import *

SEED = 42
CONFIG_PATH = "config/config.yaml"
TRAIN_DATA_PATH = "dataset/train"
EVAL_DATA_PATH = "dataset/eval"
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

def load_annotations(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    f.close()
    return data

def make_dataset(path: str, config: Dict[str, Any], annotations: Dict[str, Any]) -> AudioDataset:
    num_sm_segments = config["sample_duration"] * config["new_sample_rate"]
    num_sm_segments = (num_sm_segments / config["melspectrogram_config"]["n_fft"]) / 8
    num_sm_segments = int(num_sm_segments)
    kwargs = dict(
        annotations=annotations, 
        anchors_dict=config["anchors"], 
        sample_duration=config["sample_duration"],
        num_sm_segments=num_sm_segments,
        sample_rate=config["sample_rate"],
        extension=config["audio_extension"],
        ignore_index=config["train_config"]["ignore_index"]
    )
    return AudioDataset(path, **kwargs)

def make_dataloader(dataset: AudioDataset, config: Dict[str, Any]) -> DataLoader:
    kwargs = dict(
        num_workers=NUM_WORKERS, 
        batch_size=config["train_config"]["batch_size"], 
        shuffle=config["train_config"]["shuffle_samples"]
    )
    return DataLoader(dataset, **kwargs)

def make_model(config: Dict[str, Any], num_classes: int) -> AudioDetectionNetwork:
    model = AudioDetectionNetwork(num_classes=num_classes, config=config)
    model.train()
    return model

def make_loss_fn(config: Dict[str, Any], class_weights: torch.Tensor) -> AudioDetectionLoss:
    scale_t = None
    if config["train_config"]["scale_segments"]:
        scale_t = config["sample_duration"]
    loss_fn = AudioDetectionLoss(
        class_weights=class_weights,
        scale_t=scale_t,
        ignore_index=config["train_config"]["ignore_index"],
        **config["train_config"]["loss_weights"]
    )
    return loss_fn

def make_optimizer(model: AudioDetectionNetwork, config: Dict[str, Any]) -> torch.optim.Optimizer:
    return getattr(torch.optim, config["train_config"]["optimizer"])(
        model.parameters(), 
        lr=config["train_config"]["lr"], 
        weight_decay=config["train_config"]["lr"]
    )

def make_lr_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_name = config["train_config"]["lr_scheduler_config"].pop("name")
    lr_scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
        optimizer, **config["train_config"]["lr_scheduler_config"]
    )
    return lr_scheduler

def run():
    config = load_config()
    device = config["train_config"]["device"] if torch.cuda.is_available() else "cpu"
    annotations = load_annotations(config["train_config"]["annotation_file_path"])
    annotator = config["train_config"]["annotator"]

    train_dataset = make_dataset(TRAIN_DATA_PATH, config, annotations=annotations["annotations"][annotator])
    eval_dataset = make_dataset(EVAL_DATA_PATH, config, annotations=annotations["annotations"][annotator])
    train_dataloader = make_dataloader(train_dataset, config)
    eval_dataloader = make_dataloader(eval_dataset, config)

    model = make_model(config, num_classes=len(train_dataset.label2idx))
    model.to(device)
    loss_fn = make_loss_fn(config, class_weights=train_dataset.get_class_weights(device=device))
    optimizer = make_optimizer(model, config)
    lr_scheduler = None
    if config["train_config"]["use_lr_scheduler"]:
        lr_scheduler = make_lr_scheduler(optimizer, config)

    annotation_filename = config["train_config"]["annotation_file_path"].split("/")[-1].replace(".json", "")
    trainer_pipeline = TrainerPipeline(
        model, 
        loss_fn, 
        optimizer, 
        model_path=config["train_config"]["model_path"], 
        metrics_path=config["train_config"]["metrics_path"], 
        annotation_filename=annotation_filename,
        device=device
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
    # torch.autograd.set_detect_anomaly(True)
    run()