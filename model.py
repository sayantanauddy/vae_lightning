import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.models as models

import pytorch_lightning as pl

# Residual down sampling block for the encoder
# Average pooling is used to perform the downsampling
class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(Res_down, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x

    
# Residual up sampling block for the decoder
# Nearest neighbour is used to perform the upsampling
class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(Res_up, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor=scale, mode="nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x
    
# Encoder block
# Built for a 64x64x3 image and will result in a latent vector of size Z x 1 x 1 
# As the network is fully convolutional it will work for other larger images sized 2^n the latent
# feature map size will just no longer be 1 - aka Z x H x W
class Encoder(nn.Module):
    def __init__(self, channels, ch=64, z=512):
        super(Encoder, self).__init__()
        self.conv1 = Res_down(channels, ch)  #64
        self.conv2 = Res_down(ch, 2*ch)  #32
        self.conv3 = Res_down(2*ch, 4*ch)  #16
        self.conv4 = Res_down(4*ch, 8*ch)  #8
        self.conv5 = Res_down(8*ch, 8*ch)  #4
        self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)  #2
        self.conv_logvar = nn.Conv2d(8*ch, z, 2, 2)  #2

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x, Train = True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if Train:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            x = self.sample(mu, logvar)
        else:
            x = self.conv_mu(x)
            mu = None
            logvar = None
        return x, mu, logvar
    
#Decoder block
#Built to be a mirror of the encoder block
class Decoder(nn.Module):
    def __init__(self, channels, ch=64, z=512):
        super(Decoder, self).__init__()
        self.conv1 = Res_up(z, ch*8)
        self.conv2 = Res_up(ch*8, ch*8)
        self.conv3 = Res_up(ch*8, ch*4)
        self.conv4 = Res_up(ch*4, ch*2)
        self.conv5 = Res_up(ch*2, ch)
        self.conv6 = Res_up(ch, ch//2)
        self.conv7 = nn.Conv2d(ch//2, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x 


# create an empty layer that will simply record the feature map passed to it.
class GetFeatures(nn.Module):
    def __init__(self):
        super(GetFeatures, self).__init__()
        self.features = None

    def forward(self, x):
        self.features = x
        return x


class VAELightningModule(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VAELightningModule")
        parser.add_argument('--channel_in', type=int, default=3)
        parser.add_argument('--z', type=int, default=512)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--perceptual_net_name', type=str, default='vgg19')
        parser.add_argument('--perceptual_depth', type=int, default=7)
        parser.add_argument('--kl_weight', type=float, default=0.01)
        parser.add_argument('--perceptual_weight', type=float, default=1.0)
        parser.add_argument('--recon_weight', type=float, default=1.0)
        return parent_parser

    def __init__(self, args):
        super().__init__()

        self.channel_in = args.channel_in
        self.z = args.z
        self.lr = args.lr
        self.perceptual_net_name = args.perceptual_net_name
        if self.perceptual_net_name == 'vgg19':
            perceptual_net = models.vgg19(pretrained=True)
            perceptual_net = perceptual_net.eval()
            for p in perceptual_net.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError(f"Unknown model {args.perceptual_net_name}")
        self.perceptual_depth = args.perceptual_depth
        self.kl_weight = args.kl_weight
        self.perceptual_weight = args.perceptual_weight
        self.recon_weight = args.recon_weight

        # Save the hyperparameters
        self.save_hyperparameters()

        self.encoder = Encoder(self.channel_in, z=self.z)
        self.decoder = Decoder(self.channel_in, z=self.z)

        # Create a perceptual feature extractor using the frozen
        # pretrained network
        layers = []
        for i in range(self.perceptual_depth):
            layers.append(perceptual_net.features[i])
            if isinstance(perceptual_net.features[i], nn.ReLU):
                layers.append(GetFeatures())
        self.perceptual_feature_extractor = nn.Sequential(*layers)

    def get_perceptual_loss(self, img, recon_data):
        """
        #this function calculates the L2 loss (MSE) on the feature maps copied by the layers_deep
        #between the reconstructed image and the origional
        """
        img_cat = torch.cat((img, torch.sigmoid(recon_data)), 0)
        out = self.perceptual_feature_extractor(img_cat)
        loss = 0
        for i in range(len(self.perceptual_feature_extractor)):
            if isinstance(self.perceptual_feature_extractor[i], GetFeatures):
                loss += (self.perceptual_feature_extractor[i].features[:(img.shape[0])] - self.perceptual_feature_extractor[i].features[(img.shape[0]):]).pow(2).mean()
        return loss/(i+1)

    def get_recon_loss(self, recon, x,):
        return F.binary_cross_entropy_with_logits(recon, x)

    def get_kl_loss(self, mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        encoding, mu, logvar = self.encoder(x, Train=False)
        recon = self.decoder(encoding)
        return recon, mu, logvar

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, _ = batch
        
        encoding, mu, logvar = self.encoder(x, Train=True)
        recon = self.decoder(encoding)
        
        recon_loss = self.get_recon_loss(recon, x)
        kl_loss = self.get_kl_loss(mu, logvar)
        perceptual_loss = self.get_perceptual_loss(x, recon)

        loss = self.recon_weight*recon_loss + self.kl_weight*kl_loss + self.perceptual_weight*perceptual_loss

        # Log the training loss
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_perceptual_loss', perceptual_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        
        encoding, mu, logvar = self.encoder(x, Train=True)
        recon = self.decoder(encoding)
        
        recon_loss = self.get_recon_loss(recon, x)
        kl_loss = self.get_kl_loss(mu, logvar)
        perceptual_loss = self.get_perceptual_loss(x, recon)

        loss = self.recon_weight*recon_loss + self.kl_weight*kl_loss + self.perceptual_weight*perceptual_loss

        # Log the training loss
        self.log('val_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_perceptual_loss', perceptual_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
