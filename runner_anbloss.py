# diffae/runner_anbloss.py
"""
diffae/runner.py 의 Runner 에 ANB 보조 회귀 loss 를 추가한 RunnerANB.
기존 diffae/runner.py 는 수정하지 않는다.

loss = recon_loss + opt.lambda_anb * anb_loss
  - recon_loss: 기존과 동일한 diffusion noise-prediction MSE.
  - anb_loss  : z_sem(style_emb) 위 anb_head 예측과 정규화된 ANB 타깃의 SmoothL1.
                ANB 매칭 실패(결측) 샘플은 anb_mask 로 제외하고 평균낸다.
"""
import os
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

import distributed_util as dist_util

from . import util
from network_anbloss import DiffAEANB
from guided_diffusion.script_util import create_gaussian_diffusion


def build_optimizer_sched(opt, net, log):
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class RunnerANB(object):
    def __init__(self, opt, log, save_opt=True):
        super(RunnerANB, self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        self.diffusion = create_gaussian_diffusion(steps=opt.interval, noise_schedule=opt.schedule_name)
        log.info(f"[Diffusion] Built latent diffusion: steps={opt.interval}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = DiffAEANB(opt, log, noise_levels=noise_levels, use_fp16=opt.use_fp16, anb_hidden=opt.anb_hidden)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    def sample_batch(self, opt, loader):
        img, _, anb, anb_mask, _ = next(loader)
        x0 = img.detach().to(torch.float32)
        anb = anb.detach().to(torch.float32)
        anb_mask = anb_mask.detach().to(torch.float32)
        return x0, anb, anb_mask

    def train(self, opt, train_dataset):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device], find_unused_parameters=True)
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        net.train()

        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.start_itr, opt.num_itr + 1):
            optimizer.zero_grad()

            recon_loss_val = 0.0
            anb_loss_val = 0.0
            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, anb, anb_mask = self.sample_batch(opt, train_loader)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                label = torch.randn_like(x0)
                xt = self.diffusion.q_sample(x0, step, noise=label).to(opt.device)

                x0       = x0.to(opt.device)
                label    = label.to(opt.device)
                anb      = anb.to(opt.device)
                anb_mask = anb_mask.to(opt.device)

                pred, style_emb, anb_pred = net(xt, step, x0=x0, generated_style=None, return_anb=True)
                assert x0.shape == xt.shape == label.shape == pred.shape

                recon_loss = F.mse_loss(pred, label)

                anb_pred = anb_pred.squeeze(-1)
                n_valid = anb_mask.sum().clamp(min=1.0)
                anb_loss = (F.smooth_l1_loss(anb_pred, anb, reduction="none") * anb_mask).sum() / n_valid

                loss = recon_loss + opt.lambda_anb * anb_loss
                loss.backward()

                recon_loss_val = recon_loss.item()
                anb_loss_val = anb_loss.item()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            total_loss_val = recon_loss_val + opt.lambda_anb * anb_loss_val
            log.info("train_it {}/{} | lr:{} | recon:{} | anb:{} | total:{}".format(
                it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(recon_loss_val),
                "{:+.4f}".format(anb_loss_val),
                "{:+.4f}".format(total_loss_val),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss/recon', recon_loss_val)
                self.writer.add_scalar(it, 'loss/anb', anb_loss_val)
                self.writer.add_scalar(it, 'loss/total', total_loss_val)

            if it % 5000 == 0 or it == opt.num_itr:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                        "anb_mean": train_dataset.anb_mean,
                        "anb_std": train_dataset.anb_std,
                    }, opt.ckpt_path / f"{it:07}.pt")
                    log.info(f"Saved latest(it={it}) checkpoint to {opt.ckpt_path}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it == 500 or it % 5000 == 0: # 0, 0.5k, 5k, 10k, 15k
                net.eval()
                self.evaluation(opt, it, x0, opt.ckpt_path)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x0, batch, nfe=50):
        diffusion = create_gaussian_diffusion(steps=opt.interval, noise_schedule=opt.schedule_name, timestep_respacing=f"ddim{nfe}")
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, nfe={nfe}!")

        x0 = x0.to(opt.device)
        image = x0.detach().clone()
        noise = diffusion.ddim_reverse_sample_loop(self.net, x0, image, clip_denoised=opt.clip_denoise, progress=True)
        img_recon = diffusion.ddim_sample_loop(self.net, x0, (batch, opt.in_channels, opt.image_size, opt.image_size), noise=noise, clip_denoised=opt.clip_denoise, progress=True)

        return img_recon

    @torch.no_grad()
    def evaluation(self, opt, it, x0, ckpt_path):
        image_path = ckpt_path / 'valid_images'
        os.makedirs(image_path, exist_ok=True)

        log = self.log
        log.info(f"========== DDPM Sampling started: iter={it} ==========")

        img = x0.to(opt.device).detach().clone()
        batch, *xdim = img.shape
        img_recon = self.ddpm_sampling(opt, x0, batch)
        assert img_recon.shape == (batch, *xdim)

        log.info("Collecting tensors ...")
        img       = all_cat_cpu(opt, log, img)
        img_recon = all_cat_cpu(opt, log, img_recon)

        log.info(f"Generated recon images: size={img_recon.shape}")

        # temp img png save
        log.info("Logging images ...")

        # GPU 0번에서만 저장 실행 (중복 쓰기 방지)
        if opt.global_rank == 0:
            for idx in range(img.shape[0]):
                tu.save_image((img[idx]+1)/2, os.path.join(image_path, f'{it:07}_real_{idx}.png'), value_range=(0, 1))
                tu.save_image((img_recon[idx]+1)/2, os.path.join(image_path, f'{it:07}_recon_{idx}.png'), value_range=(0, 1))

        log.info(f"========== DDPM Sampling finished: iter={it} ==========")
        torch.cuda.empty_cache()
