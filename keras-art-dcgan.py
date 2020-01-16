import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from matplotlib import pyplot
import argparse
import os, os.path
import progressbar
from PIL import Image
import time

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', help='Folder name of the dataset', required=True, type=str)
parser.add_argument('-load_model', help='h5 file to load', required=False, type=str, default='NONE')
parser.add_argument('-save_model', help='weight to be saved', required=False, type=str, default='NONE')
args = parser.parse_args()

# Best practice initialiser for GANS
initialWeights = RandomNormal(mean=0.0, stddev=0.02, seed=None)

def define_discriminator(in_shape=(128,128,3)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape, kernel_initializer=initialWeights))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', kernel_initializer=initialWeights))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', kernel_initializer=initialWeights))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', kernel_initializer=initialWeights))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', kernel_initializer=initialWeights))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def define_generator(latent_dim):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	#model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	# upsample to 8x8
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu', kernel_initializer=initialWeights))
	#model.add(LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu', kernel_initializer=initialWeights))
	#model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu', kernel_initializer=initialWeights))
	#model.add(LeakyReLU(alpha=0.2))
	# upsample to 64x64
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu', kernel_initializer=initialWeights))
	#model.add(LeakyReLU(alpha=0.2))
	# upsample to 128x128
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu', kernel_initializer=initialWeights))
	#model.add(LeakyReLU(alpha=0.2))
	# output layer
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model

# Stick them both together to form the overall GAN model (Generator > Discriminator > Classification)
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	
	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=2):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i])
	# save plot to file
	filename = str(int(round(time.time() * 1000))) + '.png'
	pyplot.savefig("./output/samples/" + filename)
	pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	if(args.save_model != 'NONE'):
		g_model.save("./output/weights/" + args.save_model + '-g.h5')
		d_model.save("./output/weights/" + args.save_model + '-d.h5')

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1000, n_batch=64):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
		if (i+1) % 1 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)
		


def loadImages(location):
	print("")
	numImages = len([name for name in os.listdir(location) if os.path.isfile(os.path.join(location, name))])
	print("Loading " + str(numImages) + " images from: " + location)
	
	X = np.zeros((numImages, 128, 128, 3)) # change as per image size
	print ("Loading Images...")
	
	with progressbar.ProgressBar(max_value=numImages) as bar:
		i = 0
		for file in os.listdir(location):
			filename = os.fsdecode(file)
			toLoad = os.path.join(location, filename)
			#print(toLoad)
			X[i] = np.array(Image.open(toLoad),dtype=np.uint8)
			#X[i] = np.array(Image.open(url).resize((128,128), Image.LANCZOS),dtype=np.uint8)
			i+=1
			#print(i)
			bar.update(i)	
	
	X = X.astype('float32')
	X = (X - 127.5) / 127.5
	return X

imageFiles = "./data/" + args.dataset


args.load_model


# size of the latent space
latent_dim = 100

#load weights 
if(args.load_model != 'NONE'):
	print("Loading weights for: " + args.load_model)
	g_model = load_model("./output/weights/" + args.load_model + '-g.h5')
	d_model = load_model("./output/weights/" + args.load_model + '-d.h5')
	print("Success!")
else:
	print("Creating new models")
	# create the discriminator
	d_model = define_discriminator()
	# create the generator
	g_model = define_generator(latent_dim)
	print("Success!")

# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = loadImages(imageFiles)
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
