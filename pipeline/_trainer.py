import os
import tqdm
import math
import torch
import pandas as pd
from modules import AudioDetectionNetwork, AudioDetectionLoss
from smoothener import EMAParamsSmoothener
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from typing import *


class TrainerPipeline:
    def __init__(
        self, 
        model: AudioDetectionNetwork, 
        loss_fn: AudioDetectionLoss, 
        optimizer: torch.optim.Optimizer,
        model_path: str,
        metrics_path: str,
        annotation_filename: str,
        device: str="cpu",
        ema_smoothener: Optional[EMAParamsSmoothener]=None
    ):
        self.model = model
        self.model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device =  device
        self.annotation_filename = annotation_filename
        self.ema_smoothener = ema_smoothener
        self.model_path =  os.path.join(model_path, self.annotation_filename)
        self.metrics_path = os.path.join(metrics_path, self.annotation_filename)
        self.saved_model_path = os.path.join(self.model_path, f"{self.model.__class__.__name__}.pth.tar")

        # collect metrics in this list of dicts
        self._train_metrics: List[Dict[str, float]] = []
        self._eval_metrics: List[Dict[str, float]] = []

    def save_model(self):
        if not os.path.isdir(self.model_path): os.makedirs(self.model_path, exist_ok=True)
        model_state_dict = (
            self.model.state_dict() if not (self.ema_smoothener) else self.ema_smoothener.get_ema_state_dict()
        )
        state_dicts = {
            "network_params": model_state_dict,
            "optimizer_params":self.optimizer.state_dict(),
        }
        return torch.save(state_dicts, self.saved_model_path)
    
    def load_model(self):
        if not os.path.exists(self.saved_model_path):
            raise OSError(f"model is yet to be saved in path: {self.saved_model_path}")
        saved_params = torch.load(self.saved_model_path, map_location=self.device)
        return self.model.load_state_dict(saved_params["network_params"])
        
    def save_metrics_plots(self, figsize: Tuple[float, float]=(15, 60)):
        self.__save_metrics_plots("train", figsize)
        self.__save_metrics_plots("eval", figsize)

    def __save_metrics_plots(self, mode: str, figsize: Tuple[float, float]=(15, 60)):        
        valid_modes = self._valid_modes
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        df = pd.DataFrame(getattr(self, f"_{mode}_metrics"))
        fig, axs = plt.subplots(len(df.columns), 1, figsize=figsize)
        
        for i, col in enumerate(df.columns):
            label = col.replace("_", " ").title()
            axs[i].plot(df[col].to_numpy())
            axs[1].grid(visible=True)
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel(label)
            axs[i].set_title(f"[{mode.title()}] {label} vs Epoch", fontsize=24)
            axs[i].tick_params(axis='x', which='major', labelsize=20)

        if os.path.isdir(self.metrics_path): os.makedirs(self.metrics_path, exist_ok=True)
        fig.savefig(os.path.join(self.metrics_path, f"{mode}_metrics_plot.jpg"))
        fig.clear()
        plt.close(fig)
    
    def train(self, dataloader: DataLoader, verbose: bool=False) -> Dict[str, float]:
        return self.__feed(dataloader, "train", verbose)
    
    def evaluate(self, dataloader: DataLoader, verbose: bool=False) -> Dict[str, float]:        
        with torch.no_grad():
            return self.__feed(dataloader, "eval", verbose)

    def __feed(self, dataloader: DataLoader, mode: str, verbose: bool=False) -> Dict[str, float]:
        if mode not in self._valid_modes:
            raise ValueError(f"Invalid mode {mode} expected either one of {self._valid_modes}")
        getattr(self.model, mode)()
        metrics = {}
        total = math.ceil(len(dataloader.dataset) / dataloader.batch_size)

        for audio_tensor, targets in tqdm.tqdm(dataloader, total=total):
            audio_tensor: torch.Tensor = audio_tensor.to(self.device)
            targets: torch.Tensor = targets.to(self.device)
            if self.ema_smoothener and mode == "eval":
                sm_preds, md_preds, lg_preds = self.ema_smoothener.ema_model(audio_tensor)
            else:
                sm_preds, md_preds, lg_preds = self.model(audio_tensor)
            batch_loss: torch.Tensor
            batch_loss, batch_metrics = self.loss_fn((sm_preds, md_preds, lg_preds), targets)
            if mode == "train":
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.ema_smoothener:
                    self.ema_smoothener.update()
                
            for key in batch_metrics.keys(): 
                if key not in metrics.keys(): metrics[key] = (batch_metrics[key] / total)
                else: metrics[key] += (batch_metrics[key] / total)
                
        getattr(self, f"_{mode}_metrics").append(metrics)
        if verbose:
            log = "[" + mode.title() + "]: " + "\t".join([f"{k.replace('_', ' ')}: {v :.4f}" for k, v in metrics.items()])
            print(log)
        return metrics
    
    def metrics_to_csv(self):
        if not os.path.isdir(self.metrics_path): os.makedirs(self.metrics_path, exist_ok=True)
        pd.DataFrame(self._train_metrics).to_csv(os.path.join(self.metrics_path, "train_metrics.csv"), index=False)
        pd.DataFrame(self._eval_metrics).to_csv(os.path.join(self.metrics_path, "eval_metrics.csv"), index=False)

    @property
    def _valid_modes(self) -> Iterable[str]:
        return ["train", "eval"]