# UNet: semantic segmentation with PyTorch


The input images and target masks should be in the `data/imgs`, `data/masks`, `data/imgval`, `data/masksval` folders respectively.

## Tensorboard
You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`
