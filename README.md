# Art DCGAN in Keras
WIP of Robbie Barrat's Art DCGAN in Keras.

# Inspiration
Since Torch is an utter nightmare to install on Windows, here's a Keras implementation of Barrat's Art DCGAN

The code was originally [Jason Brownlee's CIFAR10 GAN](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/) before I butchered it

The topology of the Discriminator and Generator are from [Barrat's Art DCGAN](https://github.com/robbiebarrat/art-DCGAN)

Model based on [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) (2015 preprint, Radford, et al.)

# Usage
Currently, images MUST be RGB and 128x128

python keras-art-dcgan.py -dataset=datasetname

Image data should all be in /data/datasetname


# To do
This is very heavily a WIP!

Wishlist:
- Let user define network hyperparameters
- Dynamic image input size
- Allow images of only one colour channel
- Figure out a better topology for faster learning
- Figure out a better topology to prevent failure cases (eg. losses hitting 0 and training ceasing)

# Have fun!
I'd love to see what you generate using this, please post your synthesised images!

Cheers,
Jordan J. Bird
http://jordanjamesbird.com/
