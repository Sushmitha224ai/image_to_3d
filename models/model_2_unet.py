import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual Block with Instance Normalization."""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class DeepUNetGenerator(nn.Module):
    """Deep U-Net Architecture with Residual Bottlenecks."""
    def __init__(self, in_channels=3, out_channels=3, init_features=64):
        super(DeepUNetGenerator, self).__init__()
        
        features = init_features
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            ResBlock(features * 8),
            ResBlock(features * 8),
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True)
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(features * 8, features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True)
        )

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True)
        )

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        return self.decoder1(dec1)


class FeatureExtractor(nn.Module):
    """Multi-scale feature extractor for perceptual loss calculation."""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True))

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]


class PerceptualUNetModel:
    """Wrapper class managing Deep U-Net training and inference."""
    def __init__(self, device="cuda", lr=0.0003, perceptual_weight=1.0, l1_weight=1.0):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.netG = DeepUNetGenerator(in_channels=3, out_channels=3).to(self.device)
        self.feature_extractor = FeatureExtractor().to(self.device).eval()
        
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr)

    def train_step(self, sketch, real_photo, scaler=None):
        sketch = sketch.to(self.device)
        real_photo = real_photo.to(self.device)

        self.optimizer_G.zero_grad()

        if scaler is not None and self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                fake_photo = self.netG(sketch)
                loss_l1 = self.criterionL1(fake_photo, real_photo)
                
                # Perceptual feature loss
                feat_fake = self.feature_extractor(fake_photo)
                feat_real = self.feature_extractor(real_photo)
                loss_perceptual = sum(F.mse_loss(ff, fr) for ff, fr in zip(feat_fake, feat_real))

                loss_total = self.l1_weight * loss_l1 + self.perceptual_weight * loss_perceptual

            scaler.scale(loss_total).backward()
            scaler.step(self.optimizer_G)
            scaler.update()
        else:
            fake_photo = self.netG(sketch)
            loss_l1 = self.criterionL1(fake_photo, real_photo)
            
            feat_fake = self.feature_extractor(fake_photo)
            feat_real = self.feature_extractor(real_photo)
            loss_perceptual = sum(F.mse_loss(ff, fr) for ff, fr in zip(feat_fake, feat_real))

            loss_total = self.l1_weight * loss_l1 + self.perceptual_weight * loss_perceptual
            loss_total.backward()
            self.optimizer_G.step()

        return {"loss_total": loss_total.item(), "loss_l1": loss_l1.item(), "loss_perceptual": loss_perceptual.item()}

    def generate(self, sketch_tensor):
        self.netG.eval()
        with torch.no_grad():
            sketch_tensor = sketch_tensor.to(self.device)
            return self.netG(sketch_tensor)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.netG.state_dict(), filepath)

    def load(self, filepath):
        self.netG.load_state_dict(torch.load(filepath, map_location=self.device))
