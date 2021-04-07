import os
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import torch
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

from data import CelebADataModule
from model import VAELightningModule

def main(args):

    # exp_name is set from the name of this file (without extension)
    exp_name = os.path.basename(Path(__file__).stem)

    # version is set from the datetime
    version = datetime.now().strftime("%Y%m%d%H%M%S")

    # Print all hyperparameters to the screen
    print(f"Starting exp_name: {exp_name} version: {version}")
    print('Hyperparameters:')
    for k,v in vars(args).items():
        print(f"{k}: {v} {type(v)}")

    # Reproducability
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the transforms
    image_size = (64,64)
    train_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()]) 
    val_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()]) 
    target_transform = None

    # Initialize the DataModule
    celebA_dm = CelebADataModule(data_dir=args.data_dir,
                                 target_type=args.target_type,
                                 train_transform=train_transform,
                                 val_transform=val_transform,
                                 target_transform=target_transform,
                                 download=args.download,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(args.log_dir, exp_name, version, "checkpoints"),
        filename='best-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,  # Saves the best model
        save_last=True,  # Saves the latest model to resume training
        mode='min',
    )

    # Create a logger to explicitly set the path
    logger = TensorBoardLogger(save_dir=args.log_dir, name=exp_name, version=version)

    # Training
    # Check if training needs to be restarted from a previous checkpoint
    if args.resume_checkpoint is not None:
        # Hyperaparameters, model and training state are loaded from the checkpoint
        model = VAELightningModule()
        # Need to specify the following args for the Trainer:
        # default_root_dir, max_epochs
        trainer = Trainer(default_root_dir=args.log_dir,
                          max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          resume_from_checkpoint=resume_checkpoint,
                          logger=logger,
                          gpus=args.gpus)
    else:
        # If no checkpoint is provided, start training from scratch
        model = VAELightningModule(args)
        # Need to specify the following args for the Trainer:
        # default_root_dir, max_epochs
        trainer = Trainer(default_root_dir=args.log_dir,
                          max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          logger=logger,
                          gpus=args.gpus)
        
    trainer.fit(model, celebA_dm)


if __name__ == "__main__":

    # See here for more on handling hyperparams
    # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    parser = ArgumentParser()

    # Program level args
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--gpus', type=int, default=-1)  # Use all GPUs by default

    # Data level args
    # The below PR, when completed, would enable a cleaner way of doing this
    # https://github.com/PyTorchLightning/pytorch-lightning/pull/3792
    # Data level hyperparameters are not automatically saved
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default="dataset")
    parser.add_argument('--target_type', type=str, default="attr")
    parser.add_argument('--download', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)

    # Add model specific args
    parser = VAELightningModule.add_model_specific_args(parser)

    # It is also possible to do this, but we are not doing this atm
    # parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)