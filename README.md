# UNet: semantic segmentation with PyTorch


The input images and target masks should be in the `data/imgs`, `data/masks`, `data/imgval`, `data/masksval` folders respectively.

Data
https://vanderbilt.box.com/s/ljspeh9kcqgnkcpty11dkvky9gpg9a7r

## Tensorboard
You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`
