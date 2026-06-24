# network_anbloss.py
"""
network.py 의 DiffAE 에 z_sem(semantic_enc 출력) 위 보조 ANB 회귀 헤드(anb_head)를
추가한 버전. 기존 network.py 는 수정하지 않는다.

forward(..., return_anb=True) 를 주면 (out, style_emb, anb_pred) 를 반환하고,
기본값(False)에서는 (out, style_emb) 만 반환해 기존 DiffAE 와 동일하게 동작한다.
"""
import yaml
import torch
import torch.nn as nn

from diffae import util
from pathlib import Path
from guided_diffusion.script_util import create_model, create_encoder


class DiffAEANB(torch.nn.Module):
    def __init__(self, opt, log, noise_levels, use_fp16=False, cfg_dir="cfg/", anb_hidden=256):
        """
        Args:
            opt: Argument dictionary.
            log: Logging function.
            noise_levels (List): Noise level list according to the number of timesteps.
                shape = (opt.interval, )
                dtype = list
            use_fp16: If specified, use float point 16 bits for faster sampling.
            cfg_dir: The directory location of config yaml file.
            anb_hidden: anb_head 의 hidden layer 크기.
        """
        super(DiffAEANB, self).__init__()

        # initialize model
        cfg_yaml = Path(cfg_dir) / f"{opt.image_size}_model.yml"
        with cfg_yaml.open('r') as fp:
            kwargs = yaml.safe_load(fp)
        unet_kwargs = kwargs["model"]["network"]["unet"]
        enc_kwargs  = kwargs["model"]["network"]["encoder"]
        unet_kwargs["use_fp16"]        = use_fp16
        enc_kwargs["encoder_use_fp16"] = use_fp16

        # channel size = sbae-xs:128, sbae-s:192, sbae-m:256, sbae-l:320, sbae-xl:384
        self.diffusion_model = create_model(**unet_kwargs)
        self.semantic_enc    = create_encoder(**enc_kwargs)
        log.info(f"[Net] Initialized network from {cfg_yaml}! Size={util.count_parameters(self.diffusion_model)}!")
        log.info(f"[Enc] Initialized network from {cfg_yaml}! Size={util.count_parameters(self.semantic_enc)}!")

        # ---------------- 보조 ANB 회귀 헤드 ----------------
        # z_sem(semantic_enc 출력) 전체(512차원)를 입력으로 받아 ANB 각도 1개를 예측.
        z_sem_dim = enc_kwargs["out_channels"]
        self.anb_head = nn.Sequential(
            nn.Linear(z_sem_dim, anb_hidden),
            nn.SiLU(),
            nn.Linear(anb_hidden, 1),
        )
        log.info(f"[ANBHead] Initialized auxiliary ANB regression head: {z_sem_dim} -> {anb_hidden} -> 1")

        self.diffusion_model.eval()
        self.semantic_enc.eval()
        self.noise_levels = noise_levels

    def forward(self, xt, steps, x0=None, generated_style=None, return_anb=False):
            """
            Args:
                xt (torch.tensor): A tensor of x at time step t.
                    shape = (B, C, H, W)
                steps (torch.tensor): A tensor of time steps.
                    shape = (B, )
                x0 (torch.tensor): A tensor of original image.
                    shape = (B, C, H, W)
                generated_style (torch.tensor): A tensor of z_sem.
                    shape = (B, emb_dim)
                return_anb (bool): True 면 anb_head 예측값도 같이 반환.

            Returns:
                out (torch.tensor): A tensor of output. shape = (B, C, H, W)
                style_emb (torch.tensor): z_sem. shape = (B, emb_dim)
                anb_pred (torch.tensor, optional): shape = (B, 1). return_anb=True 일 때만.
            """
            t = self.noise_levels[steps].detach()
            assert t.dim() == 1 and t.shape[0] == xt.shape[0]
            if generated_style is None:
                style_emb = self.semantic_enc(x0)   # detach 제거 → encoder까지 gradient 전파
            else:
                style_emb = generated_style
            out = self.diffusion_model(xt, t, style_emb)

            if return_anb:
                anb_pred = self.anb_head(style_emb)
                return out, style_emb, anb_pred
            return out, style_emb
