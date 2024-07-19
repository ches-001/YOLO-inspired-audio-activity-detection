import math
import torch
from copy import deepcopy
from modules import AudioDetectionNetwork
from typing import Dict, Any

class EMAParamsSmoothener:
    def __init__(self, model: AudioDetectionNetwork, momentum: float=0.002, num_updates: int=0, N: int=2_000):
        self.model = model
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.num_updates = num_updates
        # this expression ensures that the momentum begins at 1.0 and gradually decays
        # to `momentum` value to enable initial epochs
        self.momentum_ = lambda n : 1 - ((1 - momentum) * (1 - math.exp(-n / N)))

        for ema_param in self.ema_model.parameters():
            ema_param.requires_grad_(False)
    
    def update(self):
        momentum = self.momentum_(self.num_updates)
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_((1 - momentum)).add_(param.data, alpha=momentum)
        self.num_updates += 1

    def get_ema_state_dict(self) -> Dict[str, Any]:
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.ema_model.load_state_dict(state_dict)
    