"""
@File    : train.py
@Time    : 2021/10/4 21:39
@Author  : Makoto
@Email   : yucheng.zhang@tum.de
@Software: PyCharm
"""

import sys

from os.path import join
from torch.utils.data import DataLoader
from datasets import *
from loss import *
from models import *

if __name__ == '__main__':

    # --------------------------------------------------------
    #  Initialize Parameters, including Generator/Dis./Dataset
    # --------------------------------------------------------

    cfg = get_cfg()

    os.makedirs("saved_models/%s" % cfg.train_dataset_name, exist_ok=True)

    # Loss weight of L1 pixel-wise loss between translated image and real image

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, cfg.img_height // 2 ** 4, cfg.img_width // 2 ** 4)

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Initialize:
    # 1.Generator
    generator = GeneratorUNet()
    generator = generator.cuda()
    generator.apply(weights_init_normal)
    # 2.Discriminator
    discriminator = Discriminator()
    discriminator = discriminator.cuda()
    discriminator.apply(weights_init_normal)
    # 3.Loss
    criterion_GAN = torch.nn.MSELoss()
    criterion_GAN.cuda()
    # criterion_pixelwise = torch.nn.L1Loss()
    # criterion_pixelwise.cuda()
    criterion_ms_ssim = MS_SSIM_L1_LOSS()
    criterion_ms_ssim.cuda()
    # 4.Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    # 5.Dataloaders
    dataloader = DataLoader(
        ImageDataset(cfg.train_image_path_ooF),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.n_cpu,
    )
    valdataloader = DataLoader(
        ValDataset(cfg.val_image_path_ooF),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.n_cpu
    )


    # ----------
    #  Training
    # ----------

    def train(a, b):
        """
        a: NOT IN focus, tensor 3*256*256
        b:   IN   focus, tensor 3*256*256
        """
        real_A = a.unsqueeze(0)
        real_B = b.unsqueeze(0)
        real_A = Variable(real_A.type(Tensor))
        real_B = Variable(real_B.type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_ms_ssim(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + cfg.lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Compute SSIM PSNR MSE and log
        # --------------


        # --------------
        #  Log Progress
        # --------------

        # Print log
        print(
            "\r[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [D loss: {:.2f}] [G loss: {:.2f}, pixel: {:.2f}, adv: {:.2f}] ["
            "blurred image: {}] [ground truth image: {}] [Sub-image: {:d}/{:d}] "
                .format(epoch,
                        cfg.n_epoch,
                        i + 1,
                        len(dataloader),
                        loss_D,
                        loss_G,
                        loss_pixel,
                        loss_GAN,
                        a_name,
                        b_name,
                        sub,
                        len(real_a)),
            flush=True
        )
        return loss_G.item(), loss_D.item()


    def val(a, b):
        """
        a: NOT IN focus, tensor 3*256*256
        b:   IN   focus, tensor 3*256*256
        """
        real_A = a.unsqueeze(0)
        real_B = b.unsqueeze(0)
        real_A = Variable(real_A.type(Tensor))
        real_B = Variable(real_B.type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Validation of Generator
        # ------------------

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_ms_ssim(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + cfg.lambda_pixel * loss_pixel

        # ---------------------
        # Validation of Discriminator
        # ---------------------

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        # --------------
        #  Log Progress
        # --------------

        # Print log
        print(
            'Validation: G loss: %.2f , D loss: %.2f' % (loss_G, loss_D),
            flush=True
        )
        return loss_G.item(), loss_D.item()


    # Train every epoch
    for epoch in range(1, cfg.n_epoch + 1):
        # Train every batch
        for i, batch in enumerate(dataloader):
            aG = list()
            aD = list()
            for a_name in batch:
                # open and cut a\b imgs
                b_name = a_name[:-8] + a_name[-4:]
                a_img = Image.open(join(cfg.train_image_path_ooF, a_name))
                b_img = Image.open(join(cfg.train_image_path_inF, b_name))
                real_a = cutimage(a_img)
                real_b = cutimage(b_img)
                # train for every subimgs
                for sub in range(len(real_a)):
                    loss_G, loss_D = train(real_a[sub], real_b[sub])
                    aG.append(loss_G)
                    aD.append(loss_D)
                avgG = sum(aG) / len(aG)
                avgD = sum(aD) / len(aD)
                with open('train_loss.txt', 'a') as f:
                    f.write('Train Epoch: %d , Batch: %d ,loss_G: %f , loss_D: %f \n' % (epoch, i, avgG, avgD))

        # Validation
        for i, batch in enumerate(valdataloader):
            aG = list()
            aD = list()
            for a_name in batch:
                b_name = a_name[:-8] + a_name[-4:]
                a_img = Image.open(join(cfg.val_image_path_ooF, a_name))
                b_img = Image.open(join(cfg.val_image_path_inF, b_name))
                real_a = cutimage(a_img)
                real_b = cutimage(b_img)
                # validation for every subimgs
                for sub in range(len(real_a)):
                    lossG, lossD = val(real_a[sub], real_b[sub])
                    aG.append(lossG)
                    aD.append(lossD)
                avgG = sum(aG) / len(aG)
                avgD = sum(aD) / len(aD)
                with open('val_loss.txt', 'a') as f:
                    f.write('Validation Epoch: %d , Batch: %d ,loss_G: %f , loss_D: %f \n' % (epoch, i, avgG, avgD))

        # Save model checkpoints
        if cfg.checkpoint_interval != -1 and epoch % cfg.checkpoint_interval == 0:
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (
                cfg.train_dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (
                cfg.train_dataset_name, epoch))
            print()
            print('model saved!')
        else:
            print()
            print('warning! model not saved here!')
