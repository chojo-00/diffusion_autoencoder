import os
import cv2
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam, lr_scheduler

from tensorboardX import SummaryWriter
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import TSNE

from torch_ema import ExponentialMovingAverage

from diffae import ckpt_util
from dataset.dataset import CLASS_LIST
from network import Classifier
from diffae.runner import Runner as Diffae_Runner
from . import utils, eval_metric

RESULT_DIR = Path("results")

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        self.log = log
        self.num_classes = opt.num_classes
        self.class_list = CLASS_LIST
        self.auc_list = ['1', '2', '3']
        
        # network.
        freeze_encoder = False if opt.transfer_mode == "finetune" else True
        if opt.pretrained:
            # load pretrained encoder.
            diffae_ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / "diffae" / opt.diffae_ckpt)
            diffae_runner = Diffae_Runner(diffae_ckpt_opt, log, save_opt=False)
            pretrained_network = diffae_runner.net.semantic_enc
            log.info(f"[Net] Loaded pretrained network from {diffae_ckpt_opt.load}!")
        else:
            # use scratch network.
            pretrained_network = None
            log.info("[Net] Using scratch network!")
        self.net = Classifier(opt, log, pretrained_network=pretrained_network, latent_dim=512,
                              num_classes=self.num_classes, freeze_encoder=freeze_encoder)
        self.net.to(opt.device)
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        log.info(f"Built network: {self.net}!")
        
        # optimizer.
        self.optimizer = Adam(self.net.parameters(), lr=opt.lr, weight_decay=opt.l2_norm)
            
        # scheduler.
        # self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        # Network load.
        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            pretrained_dict = checkpoint['net']
            pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
            if isinstance(self.net, nn.DataParallel):
                self.net.module.load_state_dict(pretrained_dict)
            else:
                self.net.load_state_dict(pretrained_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")

    def train(self, opt, log, train_dataset, val_dataset):
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                num_workers=opt.num_workers, pin_memory=True,
                                                drop_last=True, shuffle=True)
        validloader = torch.utils.data.DataLoader(val_dataset, batch_size = opt.batch_size,
                                                num_workers=opt.num_workers, pin_memory=True,
                                                drop_last=True, shuffle=True)

        # Loss.
        criterion = nn.CrossEntropyLoss().cuda()
        best_loss = 100.

        # probability caculation from logit.
        softmax = torch.nn.Softmax(dim=1)

        # loss and accuracy list.
        train_losses_list = []
        train_accuracies_list = []
        val_losses_list = []
        val_accuracies_list = []

        log.info(f'[LR] Learning rate: {opt.lr} | [WD] Weight decay: {opt.l2_norm}')
        
        writer = SummaryWriter(opt.log_dir)
        
        for epoch in range(opt.resume_epoch, opt.num_epoch):
            log.info("=======================================================")
            log.info("                      Train phase                      ")
            log.info("=======================================================")

            batch_time = utils.AverageMeter('Time', ':6.3f')
            losses = utils.AverageMeter('Loss', ':.4f')

            progress = utils.ProgressMeter(
                log,
                len(trainloader),
                [batch_time, losses],
                prefix='Epoch: [{}]'.format(epoch)
            )

            running_loss = 0
            correct = 0
            total = 0
            
            overall_gts = []
            overall_logits = []
            
            end = time.time()
            
            self.net.train()
            for idx, (imgs, labels, _) in enumerate(iter(trainloader)):
                imgs = imgs.to(torch.float32).to(opt.device)
                labels = labels.to(opt.device)
                logits = self.net(imgs)
                
                loss = criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.update(loss.item(), imgs[0].size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                
                # For evaluation
                running_loss += loss.item()

                ## AUC
                probs = softmax(logits)
                overall_logits += probs.cpu().detach().numpy().tolist()
                overall_gts += labels.cpu().detach().numpy().tolist()
                
                ## ACCURACY
                total += labels.size(0)
                prediction = torch.argmax(logits, dim=1)
                correct += torch.sum(prediction == labels.data).item()
                
                progress.display(idx)
                if (idx % opt.print_freq == 0) & (idx != 0):
                    print()
                    writer.add_scalar('train loss', running_loss/idx, (epoch*len(trainloader))+idx)
                    writer.add_scalar('train acc', correct/idx, (epoch*len(trainloader))+idx)
                
            AUROCs = eval_metric.compute_AUCs(overall_gts, overall_logits, self.num_classes)
            train_loss = running_loss / len(train_dataset)
            train_acc = correct / total
            train_losses_list.append(train_loss)
            train_accuracies_list.append(train_acc)

            log.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train AUC={AUROCs:.4f}')

            log.info("=======================================================")
            log.info("                      Valid phase                      ")
            log.info("=======================================================")

            val_batch_time = utils.AverageMeter('Time', ':6.3f')
            val_losses = utils.AverageMeter('Loss', ':.4f')

            progress = utils.ProgressMeter(
                log,
                len(validloader),
                [val_batch_time, val_losses],
                prefix='Epoch: [{}]'.format(epoch)
            )

            running_loss = 0
            correct = 0
            total = 0
            
            overall_gts = []
            overall_logits = []
            
            end = time.time()

            self.net.eval()
            with torch.no_grad():
                for idx, (imgs, labels, _) in enumerate(iter(validloader)):
                    imgs = imgs.to(torch.float32).to(opt.device)
                    labels = labels.to(opt.device)
                    logits = self.net(imgs)

                    loss = criterion(logits, labels)

                    val_losses.update(loss.item(), imgs[0].size(0))
                    val_batch_time.update(time.time() - end)
                    end = time.time()

                    # For evaluation
                    running_loss += loss.item()

                    ## AUC
                    probs = softmax(logits)
                    overall_logits += probs.cpu().detach().numpy().tolist()
                    overall_gts += labels.cpu().detach().numpy().tolist()

                    ## ACCURACY
                    total += labels.size(0)
                    prediction = torch.argmax(logits, dim=1)
                    correct += torch.sum(prediction == labels.data).item()
                    
                    progress.display(idx)
                    if (idx % opt.print_freq == 0) & (idx != 0):
                        print()
                        writer.add_scalar('valid_loss', running_loss/idx, (epoch*len(validloader))+idx)
                        writer.add_scalar('valid_acc', correct/idx, (epoch*len(validloader))+idx)

                AUROCs = eval_metric.compute_AUCs(overall_gts, overall_logits, self.num_classes)
                val_loss = running_loss / len(val_dataset)
                val_acc = correct / total
                val_losses_list.append(val_loss)
                val_accuracies_list.append(val_acc)

                log.info(f'Valid Loss: {val_loss:.4f}, Valid Accuracy: {val_acc:.4f}, AUC={AUROCs:.4f}')
            writer.close()

            self.net.train()

            if epoch % opt.save_epoch == 0 or epoch == opt.num_epoch - 1:
                log.info("=======================================================")
                log.info("                      Model  save                      ")
                log.info("=======================================================")

                save_name = f'{epoch}.pth.tar'
                save_name = os.path.join(opt.ckpt_path, save_name)
                utils.save_model(save_name, epoch, self.net, self.optimizer) # scheduler temporally remove
            
            if val_loss < best_loss:
                torch.save({'state_dict':self.net.state_dict()}, os.path.join(opt.ckpt_path, f'best_model.pth.tar'))
                log.info(f'best_loss: {val_loss:.4f} & epoch: {epoch}')
                best_loss = val_loss

        # loss and accarcy plot save        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(opt.num_epoch), train_losses_list, label='Train Loss')
        plt.plot(range(opt.num_epoch), val_losses_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Over Epochs')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(opt.num_epoch), train_accuracies_list, label='Train Accuracy')
        plt.plot(range(opt.num_epoch), val_accuracies_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Over Epochs')
    
        plt.savefig(opt.ckpt_path / "loss_and_accuracy.png", dpi=300)
        plt.close()
        
    @torch.no_grad()
    def test(self, opt, log, test_loader):
        log.info("=======================================================")
        log.info("                      Test  phase                      ")
        log.info("=======================================================")

        testloader = torch.utils.data.DataLoader(test_loader, batch_size=1,
                                                  num_workers=opt.num_workers, pin_memory=True,
                                                  shuffle=False, drop_last=False)

        overall_gts = []
        overall_probs = []
        overall_preds = []
  
        correct = 0
        miscorrect = []
        total = 0

        self.net.eval()
        for idx, (imgs, labels, fnames) in enumerate(tqdm(iter(testloader))):
            imgs = imgs.to(torch.float32).to(opt.device)
            labels = labels.to(opt.device)
            
            logits = self.net(imgs)

            ## For evaluation
            # AUROCs
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(logits)
            overall_probs += probs.cpu().detach().numpy().tolist()
            overall_gts += labels.cpu().detach().numpy().tolist()
            
            # ACCURACY
            total += labels.size(0)
            prediction = torch.argmax(logits, dim=1)
            overall_preds += prediction.cpu().detach().numpy().tolist()
            correct += torch.sum(prediction == labels.data).item()

            # Miscorrection sample
            if prediction != labels.data:
                pred_class_name = CLASS_LIST[prediction.cpu().detach().item()]
                miscorrect.append([fnames, pred_class_name])
    
        # AUROCs = compute_AUCs(overall_gts, overall_probs, self.num_classes)
        eval_metric.plot_multiclass_roc(opt, log, overall_gts, overall_probs, self.num_classes)
                
        # log.info(f'TEST AUC={AUROCs:.4f}')
        # log.info(f'TEST ACC={accuracy:.4f}')
        # log.info(f'Miscorrected_file_path: {miscorrect}')

        conf_matrix = eval_metric.save_confusion_matrix(opt, log, overall_gts, overall_preds)
        eval_metric.show_metrics(opt, log, conf_matrix)
    
    @torch.no_grad()
    def t_sne(self, opt, log, test_loader):
        log.info("=======================================================")
        log.info("                    T-SNE plot save                    ")
        log.info("=======================================================")

        testloader = torch.utils.data.DataLoader(test_loader, batch_size=1,
                                                  num_workers=opt.num_workers, pin_memory=True,
                                                  shuffle=False, drop_last=False)
        n_samples = len(testloader)

        # create save folder
        t_sne_fn = utils.get_t_sne_fn(opt)
        log.info(f"T-SNE plot will be saved to {t_sne_fn}!")

        # feature list create
        images = []
        labels = []
        latent_features = []
        num = 0

        self.net.eval()
        for idx, out in enumerate(testloader):
            x0, label, _ = utils.compute_batch(out)
            x0 = x0.to(opt.device) # (B, 1, 256, 256)

            z_sem = self.net(x0).detach().clone().cpu()

            images.append(x0) # (B, 256, 256)
            labels.append(label) # (B, 1)
            latent_features.append(z_sem) # (B, 512)

            # [-1,1]
            # gathered_latent_features = collect_all_subset(latent_features, log)
            # latent_features.append(gathered_latent_features)

            num += len(z_sem)
            log.info(f"Collected {num} latent featrues!")
            # dist.barrier()

        images = torch.cat(images, 0)
        # labels = torch.cat(labels, 0).numpy()
        labels = np.concatenate(labels, 0)
        feats = torch.cat(latent_features, axis=0)[:n_samples]

        log.info(f"Feature extract complete! Collect latent_features={feats.shape}")

        # Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(feats.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.title("Overall Feature Distribution")
        plt.xlabel("Feature Value")
        plt.ylabel("Count")
        plt.savefig(t_sne_fn / f"hists_{opt.load_itr}.png", dpi=300)

        # 2D T-SNE
        # set T-SNE parameter
        n_components = 2
        perplexity = 30
        save_name = f'Ceph_perplexity{perplexity}_seed{opt.seed}_2d'

        # latents = latent_features.reshape(latent_features.shape[0], -1)
        # labels = [i // len(latent_features) for i in range(int(len(latent_features)/3))] \
        #     + [(i // len(latent_features)) + 1 for i in range(int(len(latent_features)/3))] \
        #     + [(i // len(latent_features)) + 2 for i in range(int(len(latent_features)/3))]
        log.info(f"features size: {feats.numpy().shape}")
        log.info(f"labels size: {len(labels)}")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=opt.seed)
        tsne_result = tsne.fit_transform(feats.data)
        log.info(f"T-SNE result shape: {tsne_result.shape}")

        # T-SNE dataframe create
        tsne_df = pd.DataFrame(columns = ['x-tsne', 'y-tsne', 'label'])
        tsne_df['x-tsne'] = tsne_result[:, 0]
        tsne_df['y-tsne'] = tsne_result[:, 1]
        tsne_df['label']  = labels

        # split dataframe according to label
        tsne_df_0 = tsne_df[tsne_df['label'] == 0]
        tsne_df_1 = tsne_df[tsne_df['label'] == 1]
        tsne_df_2 = tsne_df[tsne_df['label'] == 2]

        # 2D scatter plot
        plt.figure(figsize=(6,6))

        plt.scatter(tsne_df_0['x-tsne'], tsne_df_0['y-tsne'], s = 5, color = 'red', label = 'Class1')
        plt.scatter(tsne_df_1['x-tsne'], tsne_df_1['y-tsne'], s = 5, color = 'blue', label = 'Class2')
        plt.scatter(tsne_df_2['x-tsne'], tsne_df_2['y-tsne'], s = 5, color = 'olive', label = 'Class3')

        plt.xlabel('component 0')
        plt.ylabel('component 1')
        plt.legend()
        plt.savefig(t_sne_fn / f"{save_name}.png", dpi=300)
        log.info('2D T-SNE plot save!')

        # 2D image T-SNE plot
        plot_size = 30000
        max_image_size = 256
        offset = max_image_size // 2
        image_centers_area_size = plot_size - 2 * offset

        tx = tsne_result[:, 0]
        ty = tsne_result[:, 1]

        tx = utils.scale_to_01_range(tx)
        ty = utils.scale_to_01_range(ty)

        tsne_plot = np.ones(shape=(plot_size, plot_size), dtype=np.uint8) * 255

        plt.figure(figsize=(50,50))

        for image, x, y in tqdm(zip(images, tx, ty), total=len(images)):      
            # draw a rectangle with a color corresponding to the image class
            # image = draw_rectangle_by_class(image, label, palette)

            image = cv2.resize(image.cpu().numpy().squeeze(), (max_image_size, max_image_size), interpolation = cv2.INTER_AREA)
            image = ((image+1)/2*255).astype(np.uint8)
            
            # compute the coordinates of the image on the scaled plot visualization
            tl_x, tl_y, br_x, br_y = utils.compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

            # put the image to its TSNE coordinates using numpy subarray indices
            tsne_plot[tl_y:br_y, tl_x:br_x] = image

        plt.imshow(tsne_plot, cmap='gray')
        plt.savefig(t_sne_fn / f"{save_name}_image_emb.png", dpi=600)
        plt.close('all')
        log.info('2D T-SNE image embedding plot save!')
