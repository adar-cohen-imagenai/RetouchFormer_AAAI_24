import os
import glob
import logging
import importlib
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from core.loss import AdversarialLoss, VGGLoss
from core.dataset import UnpairFaceDataset, FaceRetouchingDataset
import wandb
import lpips
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

class EnhancedTrainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.psnr = 0
        self.ssim = 0
        self.maxiteration = 0
        
        # Load datasets
        self.train_dataset = FaceRetouchingDataset(path = config['train_data_loader']['dataroot'],
                                                   resolution=config['train_data_loader']['size'],
                                                   data_type="train", data_percentage=config['train_data_loader']['percentage'])
        self.test_dataset = FaceRetouchingDataset(path = config['train_data_loader']['dataroot'],
                                                  resolution=config['train_data_loader']['size'],
                                                  data_type="test", data_percentage=1)
        self.unpair_dataset = UnpairFaceDataset(path = config['train_data_loader']['dataroot'], 
                                                    resolution=config['train_data_loader']['size'], 
                                                    data_type="train", return_gt=True, data_percentage=0)
        
        print(f"📊 Dataset loaded:")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Unpaired samples: {len(self.unpair_dataset)}")
        print(f"  Network: {self.config['model']['net']}")
        
        # Setup wandb or tensorboard
        if config['trainer']['use_wandb']==1:
            wandb.init(project="retouching", name=self.config['model']['net'] + "_subset")
            self.wandb = True       
        else:
            self.wandb = False
        
        self.train_sampler = None
        self.train_args = config['trainer']
        
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        # Setup data loaders
        batch_size = max(1, self.train_args['batch_size'] // config.get('world_size', 1))
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        self.unpair_loader = DataLoader(self.unpair_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.train_args['num_workers'])
        
        self.eval_txt = config['eval_txt']
        
        # Set loss functions
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.SmoothL1Loss().to(self.config['device'])
        self.lpips_loss = lpips.LPIPS(net='alex', lpips=False).to(self.config['device'])
        self.vgg_loss = VGGLoss(self.config['device'])
        
        # Setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.InpaintGenerator()
        self.netG = self.netG.to(self.config['device'])
    
        if not self.config['model']['no_dis']:
            self.netD = net.Discriminator(
                in_channels=3,
                use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.to(self.config['device'])

        # Setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD,
                                device_ids=[self.config['local_rank']],
                                output_device=self.config['local_rank'],
                                broadcast_buffers=True,
                                find_unused_parameters=False)

        # Enhanced tensorboard setup with separate train/val writers
        self.train_writer = None
        self.val_writer = None
        self.summary = {}
        
        if self.config['global_rank'] == 0 or (not config['distributed']):
            tensorboard_dir = os.path.join(config['save_dir'], 'tensorboard')
            self.train_writer = SummaryWriter(os.path.join(tensorboard_dir, 'train'))
            self.val_writer = SummaryWriter(os.path.join(tensorboard_dir, 'val'))
            print(f"📈 Tensorboard logs: {tensorboard_dir}")
            print(f"  Train writer: {os.path.join(tensorboard_dir, 'train')}")
            print(f"  Val writer: {os.path.join(tensorboard_dir, 'val')}")

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        maskG_params = []
        for name, param in self.netG.named_parameters():
            if not param.requires_grad:
                continue
            elif 'mask_generator' in name:
                maskG_params.append(param)
            else:
                backbone_params.append(param)

        optim_params = [{'params': backbone_params,'lr': self.config['trainer']['lr']}]
        self.optimG = torch.optim.Adam(optim_params, betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))

        opt_maskG_params =  [{'params':maskG_params, 'lr': 4.5e-6}]
        self.optim_maskG = torch.optim.Adam(opt_maskG_params, betas=(0.5, 0.9))

        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(self.netD.parameters(), lr=self.config['trainer']['lr'], betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))        

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(self.optimG, milestones=scheduler_opt['milestones'],
                                            gamma=scheduler_opt['gamma'])
            if not self.config['model']['no_dis']:
                self.scheD = MultiStepRestartLR(self.optimD, milestones=scheduler_opt['milestones'],
                                                gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
            if not self.config['model']['no_dis']:
                self.scheD = CosineAnnealingRestartLR(
                    self.optimD,
                    periods=scheduler_opt['periods'],
                    restart_weights=scheduler_opt['restart_weights'])
            self.sche_maskG = torch.optim.lr_scheduler.MultiStepLR(self.optim_maskG, milestones=[1000, 1500], gamma=0.1, verbose=True)
        elif scheduler_type == "ExponentialLR":
            self.scheG = ExponentialLR(self.optimG, gamma=0.7)
            if not self.config['model']['no_dis']:
                self.scheD = ExponentialLR(self.optimD, gamma=0.7)
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        if not self.config['model']['no_dis']:
            self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.scheG.get_lr()[0]

    def add_train_summary(self, name, val):
        """Add training tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if self.train_writer is not None and self.iteration % self.train_args['log_freq'] == 0:
            self.train_writer.add_scalar(name, self.summary[name] / self.train_args['log_freq'], self.iteration)
            self.summary[name] = 0

    def add_val_summary(self, name, val):
        """Add validation tensorboard summary."""
        if self.val_writer is not None:
            self.val_writer.add_scalar(name, val, self.iteration)

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path, f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path, f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path, f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'✅ Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis']:
                dataD = torch.load(dis_path, map_location=self.config['device'])
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])
            self.scheG.load_state_dict(data_opt['scheG'])
            self.optim_maskG.load_state_dict(data_opt['optim_maskG'])
            self.sche_maskG.load_state_dict(data_opt['sche_maskG'])
            if not self.config['model']['no_dis']:
                self.optimD.load_state_dict(data_opt['optimD'])
                self.scheD.load_state_dict(data_opt['scheD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']
            self.test(self.iteration, lr = self.get_lr())
        else:
            if self.config['global_rank'] == 0:
                print('⚠️  Warning: No trained model found. Starting with initialized model.')

    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'], f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'], f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'], f'opt_{it:06d}.pth')
            print(f'💾 Saving model to {gen_path}...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                torch.save(netD.state_dict(), dis_path)
                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'optimG': self.optimG.state_dict(),
                    'optim_maskG': self.optim_maskG.state_dict(),
                    'optimD': self.optimD.state_dict(),
                    'scheG': self.scheG.state_dict(),
                    'sche_maskG': self.sche_maskG.state_dict(),
                    'scheD': self.scheD.state_dict()
                }, opt_path)
            else:
                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'optimG': self.optimG.state_dict(),
                    'scheG': self.scheG.state_dict()
                }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")
    
    def test(self, iteration, lr):
        """Enhanced validation with tensorboard logging"""
        self.netG.eval()
        cnt = 0
        PSNR = 0
        SSIM = 0
        LPIPS = 0
        device = self.config['device']
        loss_fn = lpips.LPIPS(net='alex').to(device)
        
        print(f"🧪 Running validation at iteration {iteration}...")
        
        for batch in tqdm(self.test_loader, desc="Validation"):
            if len(batch) == 3:
                name, source_tensor, target_tensor = batch
            else:
                source_tensor, target_tensor = batch
                name = f"test_{cnt:04d}"
            with torch.no_grad(): 
                pred_img, _ = self.netG(source_tensor.to(device))
                lpips_loss = loss_fn(pred_img, target_tensor.to(device)).mean()
                s_img = pred_img[0].cpu().numpy()
                t_img = target_tensor[0].numpy()
                psnr = compare_psnr(t_img, s_img, data_range=2.0)  # Range from -1 to 1
                ssim = compare_ssim(t_img, s_img, channel_axis=0, data_range=2.0)
                PSNR += psnr
                SSIM += ssim
                LPIPS += lpips_loss
                cnt += 1
        
        PSNR /= cnt
        SSIM /= cnt
        LPIPS /= cnt
        
        # Log to tensorboard
        self.add_val_summary('PSNR', PSNR)
        self.add_val_summary('SSIM', SSIM)
        self.add_val_summary('LPIPS', LPIPS.item())
        self.add_val_summary('Learning_Rate', lr)
        
        print(f"📊 Validation Results - Iter {iteration}: PSNR: {PSNR:.4f}, SSIM: {SSIM:.4f}, LPIPS: {LPIPS:.4f}")
        
        if self.wandb:
            wandb.log({"val_PSNR": PSNR.item(), "val_SSIM": SSIM.item(), "val_LPIPS": LPIPS.item()})
        
        with open(self.eval_txt, 'a') as f:
            f.writelines(f"lr: {lr}; {iteration}: PSNR: {PSNR}; SSIM: {SSIM}; LPIPS: {LPIPS}\n")  
        f.close()
        
        self.netG.train()
        return PSNR, SSIM, LPIPS

    def train(self):
        """Enhanced training entry with better logging"""
        print("🚀 Starting RetouchFormer training on subset dataset...")
        print(f"📋 Training Configuration:")
        print(f"  Total iterations: {self.train_args['iterations']}")
        print(f"  Batch size: {self.train_args['batch_size']}")
        print(f"  Learning rate: {self.train_args['lr']}")
        print(f"  Log frequency: {self.train_args['log_freq']}")
        print(f"  Validation frequency: {self.train_args['val_freq']}")
        print(f"  Save frequency: {self.train_args['save_freq']}")
        
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')
        
        # Initialize optimizers
        self.optimG.zero_grad()
        self.optimG.step()
        self.optim_maskG.zero_grad()
        self.optim_maskG.step()
        if not self.config['model']['no_dis']:
            self.optimD.zero_grad()
            self.optimD.step()
        
        # Initial validation
        if self.iteration == 0:
            self.test(self.iteration, lr=self.get_lr())
        
        while True:
            self.epoch += 1      
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)
            self._train_epoch(pbar)    
            if self.iteration > self.train_args['iterations']:
                break
        
        print('\n✅ Training completed!')
        # Final validation
        self.test(self.iteration, lr=self.get_lr())
    
    def attention_loss(self, model, lq_paired_imgs, hq_paired_imgs, lq_unpaired_imgs, hq_unpaired_imgs):
        """Simplified attention loss for subset training"""
        # Simplified version - focusing on main reconstruction loss
        # Original has complex multi-stage training that might be overkill for subset
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    def unpair(self, unpair_tensor):
        """Unpaired loss"""
        if self.config['model']['no_dis']:
            return torch.tensor(0.0)
        unpair_loss = 0
        unpair_pred = self.netD(unpair_tensor)
        _, attention = self.netG(unpair_tensor)
        atten_acc = torch.zeros_like(unpair_pred)
        for atten in attention:
            atten_acc += atten.view(unpair_pred.shape)
        unpair_loss += (F.softplus(unpair_pred * atten_acc)).mean()
        self.optimD.zero_grad()
        unpair_loss.backward()
        self.optimD.step()
        return unpair_loss

    def pair(self, source_tensor, target_tensor):
        """Paired training loss"""
        b, c, h, w = source_tensor.size()
        pred_imgs, _ = self.netG(source_tensor)
        pred_imgs = pred_imgs.view(b, c, h, w)

        gen_loss = 0
        dis_loss = 0
        
        if not self.config['model']['no_dis']:
            # Discriminator training
            real_clip = self.netD(target_tensor)
            fake_clip = self.netD(pred_imgs.detach())
            dis_real_loss = self.adversarial_loss(real_clip, True, True)
            dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            self.add_train_summary('loss/dis_real', dis_real_loss.item())
            self.add_train_summary('loss/dis_fake', dis_fake_loss.item())
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            # Generator adversarial loss
            gen_clip = self.netD(pred_imgs)
            gan_loss = self.adversarial_loss(gen_clip, True, False)
            gan_loss = gan_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_loss
            self.add_train_summary('loss/gan_loss', gan_loss.item())

        # Generator L1 loss
        valid_loss = self.l1_loss(pred_imgs, target_tensor)
        valid_loss = valid_loss * self.config['losses']['valid_weight']
        gen_loss += valid_loss
        self.add_train_summary('loss/l1_loss', valid_loss.item())
            
        # VGG loss
        vgg_loss = self.vgg_loss(pred_imgs, target_tensor)
        vgg_loss = vgg_loss * self.config['losses']['vgg_weight']
        gen_loss += vgg_loss
        self.add_train_summary('loss/vgg_loss', vgg_loss.item())
        
        self.optimG.zero_grad()
        gen_loss.backward()
        self.optimG.step()            
        return dis_loss, valid_loss, vgg_loss

    def _train_epoch(self, pbar):
        """Enhanced training epoch with better logging"""
        device = self.config['device']
        
        # Handle different dataset sizes
        min_len = min(len(self.train_loader), len(self.unpair_loader))
        train_iter = iter(self.train_loader)
        unpair_iter = iter(self.unpair_loader)
        
        for _ in range(min_len):
            self.iteration += 1
            
            try:
                source_tensor, target_tensor = next(train_iter)
                unpair_tensor_lq, unpair_tensor_hq = next(unpair_iter)
            except StopIteration:
                break
            
            source_tensor, target_tensor = source_tensor.to(device), target_tensor.to(device)
            unpair_tensor_lq, unpair_tensor_hq = unpair_tensor_lq.to(device), unpair_tensor_hq.to(device)
            
            # Simplified training for subset - focus on main reconstruction
            stage_1_loss, stage_2_loss, stage_3_loss, stage_4_loss = self.attention_loss(
                None, source_tensor, target_tensor, unpair_tensor_lq, unpair_tensor_hq)
            dis_loss, valid_loss, vgg_loss = self.pair(source_tensor, target_tensor)
            unpair_loss = self.unpair(unpair_tensor_lq)
            
            # Update learning rate less frequently for subset
            if self.iteration % 500 == 0:
                self.update_learning_rate()
            
            # Console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                lr = self.get_lr()
                pbar.set_description((f"d: {dis_loss.item():.3f}; "
                                      f"l1: {valid_loss.item():.3f}; "
                                      f"vgg: {vgg_loss.item():.3f}; "
                                      f"unpair: {unpair_loss.item():.6f}; "
                                      f"lr: {lr:.6f}"))
                
                # Log learning rate
                self.add_train_summary('Learning_Rate', lr)
                
                if self.wandb:
                    wandb.log({
                        "train_dis": dis_loss.item(),
                        "train_l1": valid_loss.item(),
                        "train_vgg": vgg_loss.item(),
                        "train_unpair": unpair_loss.item(),
                        "lr": lr
                    })
            
            # Validation - more frequent for subset
            if self.iteration % self.train_args['val_freq'] == 0:
                self.test(self.iteration, lr=self.get_lr())
            
            # Saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))
            
            if self.iteration > self.train_args['iterations']:
                break 