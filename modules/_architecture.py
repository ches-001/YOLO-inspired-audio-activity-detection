import yaml
import torch
import torchaudio
import torch.nn as nn
from ._backbone import CustomBackBone, ResNetBackBone
from ._common import RepVGGBlock, MultiScaleFmapModule
from typing import *


class AudioDetectionNetwork(nn.Module):
    def __init__(self, num_classes: int, config: Union[str, Dict[str, Any]]="config/config.yaml"):
        super(AudioDetectionNetwork, self).__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)
            f.close()
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError(f"config is expected to be str or dict type got {type(config)}")
        
        self.num_classes = num_classes
        self.out_channels = self.config["num_anchors"]*(3 + num_classes)

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.config["sample_rate"], 
            new_freq=self.config["new_sample_rate"]
        )
        self.power_to_db_tfmr = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.melspectogram_tfmr = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config["new_sample_rate"], 
            **self.config["melspectrogram_config"]
        )
        self.mfcc_tfmr = torchaudio.transforms.MFCC(
            sample_rate=self.config["new_sample_rate"], 
            **self.config["mfcc_config"]
        )
        self.register_buffer("taper_window", torch.empty(0), persistent=True)
        _train_anchors = self.config["train_anchors"]
        _sample_duration = self.config["sample_duration"]
        self.sm_anchors = nn.Parameter(
            torch.FloatTensor(self.config["anchors"]["sm"]) / _sample_duration, 
            requires_grad=_train_anchors
        )
        self.md_anchors = nn.Parameter(
            torch.FloatTensor(self.config["anchors"]["md"]) / _sample_duration, 
            requires_grad=_train_anchors
        )
        self.lg_anchors = nn.Parameter(
            torch.FloatTensor(self.config["anchors"]["lg"]) / _sample_duration, 
            requires_grad=_train_anchors
        )

        if self.config["backbone"] == "custom":
            self.feature_extractor = CustomBackBone(
                2, 
                dropout=self.config["dropout"], 
                block_layers=self.config["block_layers"]
            )
        elif self.config["backbone"] == "resnet":
            self.feature_extractor = ResNetBackBone(
                2, 
                dropout=self.config["dropout"], 
                block_layers=self.config["block_layers"],
                **self.config["resnet_config"]
            )
        else:
            raise Exception("Unkown backbone type")
        self.multiscale_module = MultiScaleFmapModule(
            self.feature_extractor.fmap1_ch,
            self.feature_extractor.fmap2_ch,
            self.feature_extractor.fmap3_ch,
            self.feature_extractor.fmap4_ch,
            out_channels=self.out_channels
        )
        self.apply(self.xavier_init_weights)

    def forward(
            self, 
            x: torch.Tensor, 
            combine_scales: bool=False,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # resample signal (input (x) size: (N, channels, n_time))
        x = self.resampler(x)

        # taper the ends of the input signal:
        if self.config["taper_input"]:
            if self.taper_window.numel() == 0:
                self.taper_window = getattr(torch, f"{self.config['taper_window']}_window")(
                    x.shape[-1], 
                    periodic=False, 
                    device=x.device
                )
            x = x * self.taper_window[None, None, :].tile(1, x.shape[1], 1)
            
        # mel_spectrogram size: (N, channels, n_freq, n_time)
        # mfcc size: (N, channels, n_freq, n_time)
        mel_spectrogram = self.melspectogram_tfmr(x)
        mfcc = self.mfcc_tfmr(x)
        mel_spectrogram = self.power_to_db_tfmr(mel_spectrogram)
        mfcc = self.power_to_db_tfmr(mfcc)

        if self.config["scale_input"]:
            mel_spectrogram = AudioDetectionNetwork.scale_input(mel_spectrogram)
            mfcc = AudioDetectionNetwork.scale_input(mfcc)
            
        # x_spectral size: (N, 2*channels, n_freq, n_time)
        x_spectral = torch.cat((mel_spectrogram, mfcc), dim=1)
        fmaps = self.feature_extractor(x_spectral)
        sm_scale, md_scale, lg_scale = self.multiscale_module(*fmaps)

        # process predictions at different scales
        sm_preds = self.get_scale_pred(
            sm_scale, self.sm_anchors*self.config["sample_duration"], input_size=x.shape[-1], spectral_size=x_spectral.shape[-1]
        )
        md_preds = self.get_scale_pred(
            md_scale, self.md_anchors*self.config["sample_duration"], input_size=x.shape[-1], spectral_size=x_spectral.shape[-1]
        )
        lg_preds = self.get_scale_pred(
            lg_scale, self.lg_anchors*self.config["sample_duration"], input_size=x.shape[-1], spectral_size=x_spectral.shape[-1]
        )

        if not combine_scales:
            return sm_preds, md_preds, lg_preds
        batch_size = x.shape[0]
        sm_preds = sm_preds.reshape(batch_size, -1, self.num_classes+3)
        md_preds = md_preds.reshape(batch_size, -1, self.num_classes+3)
        lg_preds = lg_preds.reshape(batch_size, -1, self.num_classes+3)
        preds = torch.cat((sm_preds, md_preds, lg_preds), dim=1).flatten(start_dim=1, end_dim=-2)
        return preds

    def get_scale_pred(self, scale_pred: torch.Tensor, anchors: torch.Tensor, input_size: int, spectral_size: int):
        batch_size, grid_size, _ = scale_pred.shape
        num_anchors = anchors.shape[0]
        scale_pred = scale_pred.reshape(batch_size, grid_size, num_anchors, -1)

        # first index corresponds to objectness of each box
        objectness = scale_pred[..., :1]

        # next `num_class` indexes correspond to class probabilities of each bbox
        class_proba = scale_pred[..., 1:1+self.num_classes]

        # second to last index corresponds to center of segment along the temporal axis
        stride = spectral_size // grid_size
        center_scaler = spectral_size / (input_size / self.config["new_sample_rate"])
        sm_1dgrid = self.get_1dgrid(grid_size, device=scale_pred.device).unsqueeze(-1)
        centers = (scale_pred[..., -2:-1].sigmoid() * 2 - 0.5) + sm_1dgrid
        centers = (centers * stride) / center_scaler

        # last index of last dimension corresponds to segment duration / width
        widths = (scale_pred[..., -1:].sigmoid() * 2).pow(2) * anchors.unsqueeze(-1)

        centers = centers.clip(min=0, max=self.config["sample_duration"])
        widths = widths.clip(min=0, max=self.config["sample_duration"])
        pred = torch.cat((objectness, class_proba, centers, widths), dim=-1)
        return pred

    def get_1dgrid(self, w: int, device: Union[str, torch.device, int]) -> torch.Tensor:
        xcoords = torch.arange(0, w)
        return xcoords[:, None].to(device=device)

    def init_zeros_taper_window(self, taper_window: torch.Tensor):
        self.taper_window = torch.zeros_like(taper_window)

    def xavier_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    def inference(self):
        self.eval()
        def toggle_inference_mode(m: nn.Module):
            if isinstance(m, RepVGGBlock):
                if (
                    isinstance(m.identity, (nn.BatchNorm2d, nn.Identity)) and 
                    isinstance(m.conv1x1.norm, nn.BatchNorm2d) and 
                    isinstance(m.conv3x3.norm, nn.BatchNorm2d)
                ): m.toggle_inference_mode()
        self.apply(toggle_inference_mode)
    
    @staticmethod
    def scale_input(x: torch.Tensor, e: float=1e-5) -> torch.Tensor:
        # _max = x.max(dim=-1).values.max(dim=-1).values.max(dim=-1).values[:, None, None, None]
        # _min = x.min(dim=-1).values.min(dim=-1).values.min(dim=-1).values[:, None, None, None]
        # return (x - _min) / ((_max - _min) + e)
        mu = x.mean(dim=(-2, -1))[:, :, None, None]
        std = x.std(dim=(-2, -1))[:, :, None, None]
        return (x - mu) / (std + e)