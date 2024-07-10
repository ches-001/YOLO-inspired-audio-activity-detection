import yaml
import torch
import torchaudio
import torch.nn as nn
from ._backbone import BackBone
from ._common import MultiScaleFmapModule
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

        self.sm_anchors = nn.Parameter(torch.FloatTensor(self.config["anchors"]["sm"]), requires_grad=True)
        self.md_anchors = nn.Parameter(torch.FloatTensor(self.config["anchors"]["md"]), requires_grad=True)
        self.lg_anchors = nn.Parameter(torch.FloatTensor(self.config["anchors"]["lg"]), requires_grad=True)

        self.feature_extractor = BackBone(2, dropout=self.config["dropout"])
        self.multiscale_module = MultiScaleFmapModule(
            self.feature_extractor.block1.out_channels, 
            self.feature_extractor.block2.out_channels, 
            self.feature_extractor.block3.out_channels, 
            self.feature_extractor.block4.out_channels, 
            out_channels=self.out_channels
        )
        self.apply(self.xavier_init_weights)

    def forward(
            self, 
            x: torch.Tensor, 
            combine_scales: bool=False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # resample signal (input (x) size: (N, channels, n_time))
        x = self.resampler(x)

        # taper signal:
        if self.config["taper_input"]:
            if self.taper_window.numel() == 0:
                self.taper_window = getattr(torch, f"{self.config['taper_window']}_window")(
                    x.shape[-1], 
                    periodic=False, 
                    device=x.device
                )
            x = x * self.taper_window[None, None, :].tile(1, x.shape[1], 1)
            
        # extra mel-spectral features (mel-spectrogram and MFCC)
        mel_spectrogram = self.melspectogram_tfmr(x)              # size: (N, channels, n_freq, n_time)
        mfcc = self.mfcc_tfmr(x)                                  # size: (N, channels, n_freq, n_time)
        mel_spectrogram = self.power_to_db_tfmr(mel_spectrogram)
        mfcc = self.power_to_db_tfmr(mfcc)

        if self.config["normalize"]:
            mel_spectrogram = AudioDetectionNetwork.normalize(mel_spectrogram)
            mfcc = AudioDetectionNetwork.normalize(mfcc)
            
        # spectro-temporal input goes to 2d conv network
        # The spectral input is created by concatinating the mel-spectrogram with the MFCC
        # to create a multi-channel spectro-temporal image
        x_spectral = torch.cat((mel_spectrogram, mfcc), dim=1)    # size: (N, 2*channels, n_freq, n_time)
        fmaps = self.feature_extractor(x_spectral)
        sm_scale, md_scale, lg_scale = self.multiscale_module(*fmaps)

        batch_size, num_sm_segments, _ = sm_scale.shape
        _, num_md_segments, _ = md_scale.shape
        _, num_lg_segments, _ = lg_scale.shape
        num_sm_anchors = self.sm_anchors.shape[0]
        num_md_anchors = self.md_anchors.shape[0]
        num_lg_anchors = self.lg_anchors.shape[0]

        sm_scale = sm_scale.reshape(batch_size, num_sm_segments, num_sm_anchors, -1)
        md_scale = md_scale.reshape(batch_size, num_md_segments, num_md_anchors, -1)
        lg_scale = lg_scale.reshape(batch_size, num_lg_segments, num_lg_anchors, -1)

        # first index corresponds to objectness of each box
        sm_objectness = sm_scale[..., :1].sigmoid()
        md_objectness = md_scale[..., :1].sigmoid()
        lg_objectness = lg_scale[..., :1].sigmoid()

        # next `num_class` indexes correspond to class probabilities of each bbox
        sm_class_proba = nn.functional.softmax(sm_scale[..., 1:1+self.num_classes], dim=-1)
        md_class_proba = nn.functional.softmax(md_scale[..., 1:1+self.num_classes], dim=-1)
        lg_class_proba = nn.functional.softmax(lg_scale[..., 1:1+self.num_classes], dim=-1)
        
        # second to last index corresponds to center of segment along the temporal axis
        sm_stride = x_spectral.shape[-1] // sm_scale.shape[1]
        md_stride = x_spectral.shape[-1] // md_scale.shape[1]
        lg_stride = x_spectral.shape[-1] // lg_scale.shape[1]
        center_scaler = x_spectral.shape[-1] / (x.shape[-1] / self.config["new_sample_rate"])
        sm_1dgrid = self.get_segment_coords(num_sm_segments, device=x.device).unsqueeze(-1)
        md_1dgrid = self.get_segment_coords(num_md_segments, device=x.device).unsqueeze(-1)
        lg_1dgrid = self.get_segment_coords(num_lg_segments, device=x.device).unsqueeze(-1)
        sm_center = ((torch.sigmoid(sm_scale[..., -2:-1]) + sm_1dgrid) * sm_stride) / center_scaler
        md_center = ((torch.sigmoid(md_scale[..., -2:-1]) + md_1dgrid) * md_stride) / center_scaler
        lg_center = ((torch.sigmoid(lg_scale[..., -2:-1]) + lg_1dgrid) * lg_stride) / center_scaler

        # last index of last dimension corresponds to segment duration / width
        sm_width = torch.exp(sm_scale[..., -1:]) * self.sm_anchors.unsqueeze(-1)
        md_width = torch.exp(md_scale[..., -1:]) * self.md_anchors.unsqueeze(-1)
        lg_width = torch.exp(lg_scale[..., -1:]) * self.lg_anchors.unsqueeze(-1)

        sm_preds = torch.cat((sm_objectness, sm_class_proba, sm_center, sm_width), dim=-1)
        md_preds = torch.cat((md_objectness, md_class_proba, md_center, md_width), dim=-1)
        lg_preds = torch.cat((lg_objectness, lg_class_proba, lg_center, lg_width), dim=-1)
        if not combine_scales:
            return sm_preds, md_preds, lg_preds
        sm_preds = sm_preds.reshape(batch_size, -1, self.num_classes+3)
        md_preds = md_preds.reshape(batch_size, -1, self.num_classes+3)
        lg_preds = lg_preds.reshape(batch_size, -1, self.num_classes+3)
        preds = torch.cat((sm_preds, md_preds, lg_preds), dim=-1)
        return preds

    def get_segment_coords(self, w: int, device: Union[str, torch.device, int]) -> torch.Tensor:
        xcoords = torch.arange(0, w)
        return xcoords[:, None].to(device=device)

    def init_zeros_taper_window(self, taper_window: torch.Tensor):
        self.taper_window = torch.zeros_like(taper_window)

    def xavier_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)
    
    @staticmethod
    def normalize(x: torch.Tensor, e: float=1e-8) -> torch.Tensor:
        mu = x.mean(dim=(-2, -1))[:, :, None, None]
        std = x.std(dim=(-2, -1))[:, :, None, None]
        return (x - mu) / (std + e)