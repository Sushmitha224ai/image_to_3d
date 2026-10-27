import os
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    """Residual Block with Reflection Padding."""
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    """ResNet-based Bottleneck Generator."""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        super(ResNetGenerator, self).__init__()
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # ResNet Bottleneck Blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input_sketch):
        return self.model(input_sketch)


class LSGANDiscriminator(nn.Module):
    """Patch Discriminator for Least Squares GAN."""
    def __init__(self, input_nc=6, ndf=64):
        super(LSGANDiscriminator, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True))
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, True))
        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, True))
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, 1, 4, stride=1, padding=1))

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        out = self.layer4(f3)
        return out, [f1, f2, f3]


class ResNetLSGANModel:
    """Wrapper class managing ResNet Generator + LSGAN training & inference."""
    def __init__(self, device="cuda", lr=0.0002, lambda_fm=10.0):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.netG = ResNetGenerator(input_nc=3, output_nc=3, n_blocks=6).to(self.device)
        self.netD = LSGANDiscriminator(input_nc=6).to(self.device)
        
        self.criterionMSE = nn.MSELoss().to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.lambda_fm = lambda_fm

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))

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
                pred_fake, feats_fake = self.netD(fake_pair)
                real_pair = torch.cat((sketch, real_photo), 1)
                _, feats_real = self.netD(real_pair)

                # LSGAN Adversarial Loss (Target = 1)
                loss_G_adv = self.criterionMSE(pred_fake, torch.ones_like(pred_fake))
                
                # Feature Matching Loss
                loss_fm = sum(self.criterionL1(ff, fr.detach()) for ff, fr in zip(feats_fake, feats_real))
                loss_l1 = self.criterionL1(fake_photo, real_photo)

                loss_G = loss_G_adv + self.lambda_fm * loss_fm + 50.0 * loss_l1

            scaler.scale(loss_G).backward()
            scaler.step(self.optimizer_G)
        else:
            fake_photo = self.netG(sketch)
            fake_pair = torch.cat((sketch, fake_photo), 1)
            pred_fake, feats_fake = self.netD(fake_pair)
            real_pair = torch.cat((sketch, real_photo), 1)
            _, feats_real = self.netD(real_pair)

            loss_G_adv = self.criterionMSE(pred_fake, torch.ones_like(pred_fake))
            loss_fm = sum(self.criterionL1(ff, fr.detach()) for ff, fr in zip(feats_fake, feats_real))
            loss_l1 = self.criterionL1(fake_photo, real_photo)

            loss_G = loss_G_adv + self.lambda_fm * loss_fm + 50.0 * loss_l1
            loss_G.backward()
            self.optimizer_G.step()

        # ----------------------
        #  Train Discriminator
        # ----------------------
        self.optimizer_D.zero_grad()
        fake_pair = torch.cat((sketch, fake_photo.detach()), 1)

        if scaler is not None and self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred_real, _ = self.netD(real_pair)
                loss_D_real = self.criterionMSE(pred_real, torch.ones_like(pred_real))
                pred_fake, _ = self.netD(fake_pair)
                loss_D_fake = self.criterionMSE(pred_fake, torch.zeros_like(pred_fake))
                loss_D = (loss_D_real + loss_D_fake) * 0.5
            scaler.scale(loss_D).backward()
            scaler.step(self.optimizer_D)
            scaler.update()
        else:
            pred_real, _ = self.netD(real_pair)
            loss_D_real = self.criterionMSE(pred_real, torch.ones_like(pred_real))
            pred_fake, _ = self.netD(fake_pair)
            loss_D_fake = self.criterionMSE(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            self.optimizer_D.step()

        return {"loss_G": loss_G.item(), "loss_D": loss_D.item(), "loss_fm": loss_fm.item()}

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
