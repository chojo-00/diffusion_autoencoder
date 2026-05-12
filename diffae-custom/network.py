import yaml
import torch
import torch.nn as nn

from diffae import util
from pathlib import Path
from guided_diffusion.script_util import create_model, create_encoder

from ipdb import set_trace as debug

#===================================================================================================#
#                                    First Stage: DiffAE Network                                    #
#===================================================================================================#

class DiffAE(torch.nn.Module):
    def __init__(self, opt, log, noise_levels, use_fp16=False, cfg_dir="cfg/"):
        """
        Args:
            opt: Argument dictionary.
            log: Logging function.
            noise_levels (List): Noise level list according to the number of timesteps.
                shape = (opt.interval, )
                dtype = list
            use_fp16: If specified, use float point 16 bits for faster sampling.
            cfg_dir: The directory location of config yaml file.
        """
        super(DiffAE, self).__init__()

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

        self.diffusion_model.eval()
        self.semantic_enc.eval()
        self.noise_levels = noise_levels

    def forward(self, xt, steps, x0=None, generated_style=None):
        """
        Args:
            xt (torch.tensor): A tensor of x at time step t.
                shape = (B, C, H, W)
                dtype = torch.float32
            steps (torch.tensor): A tensor of time steps.
                shape = (B, )
                dtype = torch.float32
            x0 (torch.tensor): A tensor of original image.
                shape = (B, C, H, W)
                dtype = torch.float32
            genetrated_style (torch.tensor): A tensor of z_sem.
                shape = (B, emb_dim)
                dtype = torch.float32

        Returns:
            out (torch.tensor): A tensor of output.
                shape = (B, C, H, W)
                dtype = torch.float32
        """
        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == xt.shape[0]
        style_emb = self.semantic_enc(x0).detach() if generated_style is None else generated_style
        out = self.diffusion_model(xt, t, style_emb)
        return out, style_emb
    
#===================================================================================================#
#                                 Second Stage: Latent DDIM Network                                 #
#===================================================================================================#

"""It will be released soon..."""


#===================================================================================================#
#                                       Classification Network                                      #
#===================================================================================================#

class Classifier(torch.nn.Module):
    def __init__(self, opt, log, pretrained_network, latent_dim=512, num_classes=3, freeze_encoder=False, cfg_dir="cfg/"):
        """
        Args:
            opt: Argument dictionary.
            log: Logging function.
            pretrained_network: Pretrained encoder in diffusion autoencoder.
            latent_dim: The dimension of output from pretrained network.
            num_classes: The number of classes.
            freeze_encoder: If true, no gradient with pretrained encoder when training.
            cfg_dir: The directory location of config yaml file.
        """
        super(Classifier, self).__init__()
        
        if pretrained_network is None:
            # initialize model
            cfg_yaml = Path(cfg_dir) / f"{opt.image_size}_model.yml"
            with cfg_yaml.open('r') as fp:
                kwargs = yaml.safe_load(fp)
            enc_kwargs  = kwargs["model"]["network"]["encoder"]
            self.encoder = create_encoder(**enc_kwargs)
        else:
            self.encoder = pretrained_network

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(latent_dim, num_classes)

        self.encoder.eval()
        self.classifier.eval()

    def forward(self, x):
        """
        Args:
            x (torch.tensor): A tensor of x.
                shape = (B, C, H, W)
                dtype = torch.float32

        Returns:
            logits (torch.tensor): A tensor of logits about classes.
                shape = (B, N)
                dtype = torch.float32
        """
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            z = self.encoder(x)
        logits = self.classifier(z)
        return logits
