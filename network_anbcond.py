# network_anbcond.py
"""
network.py 의 DiffAE 에 ANB(전후방 골격성 각도)를 입력 condition 으로 주입한 버전.
기존 network.py / network_anbloss.py 는 수정하지 않는다.

anbloss 방식과의 차이:
  - anbloss : z_sem 위 회귀헤드(anb_head)로 ANB 를 "예측"(출력)하는 멀티태스크.
  - anbcond : ANB 값을 작은 MLP(anb_embed)로 임베딩해 z_sem 에 더해(condition 주입),
              diffusion 모델이 ANB 를 조건으로 받아 복원/생성하게 한다.

ANB 임베딩(anb_embed)은 마지막 Linear 를 zero-init 으로 두어, 학습 초기에는
cond_emb == z_sem 이 되어 기존 DiffAE 와 동일하게 시작하고 점진적으로 ANB 조건을
학습하도록 한다.

forward(..., anb=tensor) 로 ANB(정규화된 값, shape=(B,) 또는 (B,1))를 주면
cond_emb = z_sem + anb_embed(anb) 를 diffusion 모델에 넣는다.
anb=None 이면 cond_emb = z_sem 으로 기존 DiffAE 와 동일하게 동작한다.

반환값은 기존 DiffAE 와 동일하게 (out, style_emb) 이다.
(gaussian_diffusion 의 p_mean_variance 가 `model_output, _ = model(...)` 로
 언패킹하므로 시그니처를 맞춰야 한다.)
"""
import yaml
import torch
import torch.nn as nn

from diffae import util
from pathlib import Path
from guided_diffusion.script_util import create_model, create_encoder


class DiffAEANBCond(torch.nn.Module):
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
            anb_hidden: anb_embed 의 hidden layer 크기.
        """
        super(DiffAEANBCond, self).__init__()

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

        # ---------------- ANB condition 임베딩 ----------------
        # ANB 각도(스칼라 1개)를 z_sem 차원(512)으로 임베딩해 z_sem 에 더한다.
        z_sem_dim = enc_kwargs["out_channels"]
        self.anb_embed = nn.Sequential(
            nn.Linear(1, anb_hidden),
            nn.SiLU(),
            nn.Linear(anb_hidden, z_sem_dim),
        )
        # 마지막 Linear zero-init: 학습 초기에 cond_emb == z_sem 으로 시작.
        nn.init.zeros_(self.anb_embed[-1].weight)
        nn.init.zeros_(self.anb_embed[-1].bias)
        log.info(f"[ANBEmbed] Initialized ANB conditioning embed: 1 -> {anb_hidden} -> {z_sem_dim} (last layer zero-init)")

        self.diffusion_model.eval()
        self.semantic_enc.eval()
        self.noise_levels = noise_levels

    def forward(self, xt, steps, x0=None, generated_style=None, anb=None):
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
                anb (torch.tensor): 정규화된 ANB 값. shape = (B,) 또는 (B, 1).
                    None 이면 condition 없이 기존 DiffAE 와 동일하게 동작.

            Returns:
                out (torch.tensor): A tensor of output. shape = (B, C, H, W)
                style_emb (torch.tensor): z_sem. shape = (B, emb_dim)
            """
            t = self.noise_levels[steps].detach()
            assert t.dim() == 1 and t.shape[0] == xt.shape[0]
            if generated_style is None:
                style_emb = self.semantic_enc(x0)   # detach 제거 → encoder까지 gradient 전파
            else:
                style_emb = generated_style

            if anb is not None:
                anb = anb.view(-1, 1).to(style_emb.dtype)
                anb_emb = self.anb_embed(anb)
                cond_emb = style_emb + anb_emb
            else:
                cond_emb = style_emb

            out = self.diffusion_model(xt, t, cond_emb)
            return out, style_emb
