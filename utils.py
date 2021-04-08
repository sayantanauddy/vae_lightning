import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from model import VAELightningModule
from data import CelebADataset


def generate_samples(checkpoint_path, z=512, num_samples=16, save_dir="plots"):

    print("Starting image generation")

    model = VAELightningModule.load_from_checkpoint(checkpoint_path)
    model = model.eval()

    # Sample from a standard normal distribution
    decoder_input = torch.normal(mean=0.0, std=1.0, size=(num_samples, z,1,1))
    # Use sigmoid to restrict pixels to 0,1 range
    recon = torch.sigmoid(model.decoder(decoder_input))

    fig, ax = plt.subplots(1, num_samples, figsize=(num_samples,2), sharey=True)
    recon_np = recon.permute(0,2,3,1).detach().numpy()
    for i in range(num_samples):
        ax[i].imshow(recon_np[i])

    ax[0].set_ylabel("Generated")

    plt.savefig(os.path.join(save_dir, "generated.png"), bbox_inches="tight")

    print("Done!")


def reconstruct_samples(checkpoint_path, val_transform, num_samples=16, save_dir="plots"):

    print("Starting image reconstruction")

    model = VAELightningModule.load_from_checkpoint(checkpoint_path)
    model = model.eval()

    celebA_valset = CelebADataset(root="dataset",
                                  split='valid',
                                  target_type="attr",
                                  download=False,
                                  transform=val_transform,
                                  target_transform=None)

    val_loader = DataLoader(celebA_valset, batch_size=num_samples, shuffle=False, drop_last=False, num_workers=8)

    fig, ax = plt.subplots(2, num_samples, figsize=(num_samples,3), sharey=True)
    for x,y in val_loader:
        encoding, _, _ = model.encoder(x, Train=False)

        # Use sigmoid to restrict pixels to 0,1 range
        recon = torch.sigmoid(model.decoder(encoding))

        # Permute the tensor to make it (N,H,W,C) and convert to numpy
        x_np = x.permute(0,2,3,1).detach().numpy()
        recon_np = recon.permute(0,2,3,1).detach().numpy()
        for i in range(num_samples):
            ax[0][i].imshow(x_np[i])
            ax[1][i].imshow(recon_np[i])
        break

    ax[0][0].set_ylabel("Original")
    ax[1][0].set_ylabel("Reconstructed")

    plt.savefig(os.path.join(save_dir, "reconstructed.png"), bbox_inches="tight")

    print("Done!")
