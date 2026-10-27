import os
import torch
import torch.nn as nn

class UNetSkipConnectionBlock(nn.Module):
    """Defines the U-Net submodule with skip connection."""
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, use_dropout=False):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Pix2PixGenerator(nn.Module):
    """U-Net Generator for Pix2Pix."""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, use_dropout=True):
        super(Pix2PixGenerator, self).__init__()
        # Construct 8-layer U-Net for 256x256 image input
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for _ in range(3):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block)
        self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, input_sketch):
        return self.model(input_sketch)


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator for Pix2Pix."""
    def __init__(self, input_nc=6, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input_pair):
        return self.model(input_pair)


class Pix2PixModel:
    """Wrapper class managing Pix2Pix training and inference."""
    def __init__(self, device="cuda", lr=0.0002, beta1=0.5, lambda_l1=100.0):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.netG = Pix2PixGenerator(input_nc=3, output_nc=3).to(self.device)
        self.netD = NLayerDiscriminator(input_nc=6).to(self.device)
        
        self.criterionGAN = nn.BCEWithLogitsLoss().to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.lambda_l1 = lambda_l1

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))

    def train_step(self, sketch, real_photo, scaler=None):
        sketch = sketch.to(self.device)
        real_photo = real_photo.to(self.device)

        # ------------------
        #  Train Generator
        # ------------------
        self.optimizer_G.zero_grad()
        if scaler is not None and self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                fake_photo = self.netG(sketch)
                fake_pair = torch.cat((sketch, fake_photo), 1)
                pred_fake = self.netD(fake_pair)
                loss_G_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_L1 = self.criterionL1(fake_photo, real_photo) * self.lambda_l1
                loss_G = loss_G_GAN + loss_G_L1
            scaler.scale(loss_G).backward()
            scaler.step(self.optimizer_G)
        else:
            fake_photo = self.netG(sketch)
            fake_pair = torch.cat((sketch, fake_photo), 1)
            pred_fake = self.netD(fake_pair)
            loss_G_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = self.criterionL1(fake_photo, real_photo) * self.lambda_l1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            self.optimizer_G.step()

        # ----------------------
        #  Train Discriminator
        # ----------------------
        self.optimizer_D.zero_grad()
        real_pair = torch.cat((sketch, real_photo), 1)
        fake_pair = torch.cat((sketch, fake_photo.detach()), 1)
        
        if scaler is not None and self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred_real = self.netD(real_pair)
                loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
                pred_fake = self.netD(fake_pair)
                loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = (loss_D_real + loss_D_fake) * 0.5
            scaler.scale(loss_D).backward()
            scaler.step(self.optimizer_D)
            scaler.update()
        else:
            pred_real = self.netD(real_pair)
            loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
            pred_fake = self.netD(fake_pair)
            loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            self.optimizer_D.step()

        return {"loss_G": loss_G.item(), "loss_D": loss_D.item(), "loss_L1": loss_G_L1.item()}

    def generate(self, sketch_tensor):
        self.netG.eval()
        with torch.no_grad():
            sketch_tensor = sketch_tensor.to(self.device)
            return self.netG(sketch_tensor)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "netG": self.netG.state_dict(),
            "netD": self.netD.state_dict()
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.netG.load_state_dict(checkpoint["netG"])
        self.netD.load_state_dict(checkpoint["netD"])
