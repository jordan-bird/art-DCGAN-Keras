# Art DCGAN in Keras
WIP of Robbie Barrat's Art DCGAN in Keras.

## Inspiration
I didn't know how GANs worked, and I wanted to know how GANs worked... oh, [*and that AI painting sold for $432,500*](https://www.christies.com/features/A-collaboration-between-two-artists-one-human-one-a-machine-9332-1.aspx).

Since Torch is an utter nightmare to install on Windows, here's a Keras implementation of Barrat's Art DCGAN

The code was originally [Jason Brownlee's CIFAR10 GAN](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/) before I butchered it

The topology of the Discriminator and Generator are from [Barrat's Art DCGAN](https://github.com/robbiebarrat/art-DCGAN)

Model based on [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) (2015 preprint, Radford, et al.)

## Usage
Currently, images MUST be RGB and 128x128

python keras-art-dcgan.py -dataset=datasetname

Image data should all be in /data/datasetname

Four examples are generated every time an epoch is finished

## Requirements
- Keras
- Tensorflow (or Theano if you switch it to channels last)
- NumPy
- Pillow
- progressbar


## To do
This is very heavily a WIP!

Wishlist:
- Saving and reloading of weights
- Let user define network hyperparameters
- Dynamic image input size
- Allow images of only one colour channel
- Figure out a better topology for faster learning
- Figure out a better topology to prevent failure cases (eg. losses hitting 0 and training ceasing)
- Properly implement and test Barrat's topology - currently I am using a GTX980Ti which doesn't have enough memory for the amount of filters he uses in the two networks. I've watered it down slightly as to have fewer parameters

## Issues
Sometimes, depending on data, losses can hit 0 and training will cease. 

I'm not sure (yet) how to fix this if it's the Generator, but if the Discriminator is too strong then it can help to add Leaky ReLu or Dropout between the CNN layers.

Datasets should be relatively varied, but not so much as to throw the model off-genre (unless that's what you're going for!)

## Have fun!
I'd love to see what you generate using this, please post your synthesised images!

Cheers,

Jordan J. Bird

http://jordanjamesbird.com/
