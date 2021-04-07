# Variational Auto Encoders with PyTorch Lightning

This repository contains code for creating and training a variational auto encoder using [PyTorch Lightning](https://pytorch-lightning.readthedocs.io). The VAE being trained here is a Res-Net Style VAE with an adjustable perception loss using a pre-trained vgg19. The code for the core VAE architecture is from [this excellent repository](https://github.com/LukeDitria/CNN-VAE).

New stuff:
- Use PyTorch Lightning for training the VAE
- An alternative Dataset class for the CelebA dataset that downloads the data from Kaggle. The version of this dataset provided in `torchvision.datasets` [(link)](https://pytorch.org/vision/stable/datasets.html#celeba) does not currently work as expected. Read more about the issue [here](https://github.com/pytorch/vision/issues/2262). The `CelebADataset` Dataset class provided in this repository is adapted from the `torchvision.datasets.CelebA` class.

## Training

```python
python train_vae_perceptual.py --seed 100 --batch_size 32 --download True --epochs 20 --lr 0.0001
```
Use `python train_vae_perceptual.py --help` to see all available flags.

This is a WIP and the code and documentation will be updated.
