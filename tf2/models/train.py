import tensorflow as tf
import dolhasz
import sys
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from resnet import identity_block, conv_block, upconv_block
from axialnet import AxialDecoderBlock, AxialEncoderBlock


def make_axial_unet(start_ch=16, groups=2, n_blocks=3, n_layers=4, ksize=64, dense=False):
	def make_ds(target=128):
		return tf.keras.Sequential(
			[tf.keras.layers.Conv2D(target, (1,1), strides=2, padding="same"),
			tf.keras.layers.BatchNormalization()]
		)

	# Build root	
	inpt = tf.keras.layers.Input(shape=(256,256,3))
	x = tf.keras.layers.Conv2D(int(start_ch), (7,7), strides=2, padding="same")(inpt)
	x = tf.keras.layers.BatchNormalization()(x)
	xo = tf.keras.layers.Activation('relu')(x)
	xp = tf.keras.layers.MaxPool2D()(xo)
	x = xp

	# Build encoder
	skips = []
	for layer_idx in range(n_layers):
		x = AxialEncoderBlock(start_ch, start_ch*2, stride=2, groups=groups, base_width=start_ch, kernel_size=ksize, downsample=make_ds(start_ch*2))(x)
		start_ch *= 2
		ksize //= 2
		for b in range(n_blocks-1):
			x = AxialEncoderBlock(start_ch, start_ch, stride=1, groups=groups, base_width=start_ch, kernel_size=ksize)(x)
			if b == (n_blocks-2):
				skips.append(x)

	if dense:
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1024)(x)
		x = tf.keras.layers.Reshape((1,1,1024))(x)

	# Build decoder
	for l in range(n_layers):
		start_ch //= 2
		x = AxialDecoderBlock(start_ch, start_ch//2, stride=1, groups=groups, base_width=start_ch, kernel_size=ksize)([x, skips[n_layers-l-1]])
		ksize *= 2

	x = tf.keras.layers.Add()([x, xp])
	x = tf.keras.layers.UpSampling2D()(x)
	x = tf.keras.layers.Conv2D(start_ch, (4,4), padding="same")(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Add()([x, xo])
	x = tf.keras.layers.UpSampling2D()(x)
	x = tf.keras.layers.Conv2D(3, (4,4), padding="same")(x)

	return tf.keras.Model(inpt, x)
	

def make_unet(start_ch=16):

	inpt = tf.keras.layers.Input(shape=(256,256,3))
	x = tf.keras.layers.Conv2D(int(start_ch), (7,7), strides=2, padding="same")(inpt)
	x = tf.keras.layers.BatchNormalization()(x)
	xo = tf.keras.layers.Activation('relu')(x)
	xp = tf.keras.layers.MaxPool2D()(xo)
	x = xp
	
	x = conv_block(x, 4, start_ch, start_ch*4)
	x = identity_block(x, 4, start_ch, start_ch*4)
	x1 = identity_block(x, 4, start_ch, start_ch*4)

	x = conv_block(x1, 4, start_ch*2, start_ch*8)
	x = identity_block(x, 4, start_ch*2, start_ch*8)
	x = identity_block(x, 4, start_ch*2, start_ch*8)
	x2 = identity_block(x, 4, start_ch*2, start_ch*8)

	x = conv_block(x2, 4, start_ch*4, start_ch*16)
	x = identity_block(x, 4, start_ch*4, start_ch*16)
	x = identity_block(x, 4, start_ch*4, start_ch*16)
	x = identity_block(x, 4, start_ch*4, start_ch*16)
	x = identity_block(x, 4, start_ch*4, start_ch*16)
	x3 = identity_block(x, 4, start_ch*4, start_ch*16)

	x = conv_block(x3, 4, start_ch*8, start_ch*32)
	x = identity_block(x, 4, start_ch*8, start_ch*32)
	x = identity_block(x, 4, start_ch*8, start_ch*32)

	x = AxialDecoderBlock(start_ch*32, start_ch*16, stride=1, groups=4, base_width=64, kernel_size=8)([x, x3])
	x = AxialDecoderBlock(start_ch*16, start_ch*8, stride=1, groups=4, base_width=64, kernel_size=16)([x, x2])
	x = AxialDecoderBlock(start_ch*8, start_ch*4, stride=1, groups=4, base_width=64, kernel_size=32)([x, x1])
	x = AxialDecoderBlock(start_ch*4, start_ch*2, stride=1, groups=4, base_width=64, kernel_size=64)([x, xp])
	x = AxialDecoderBlock(start_ch*2, start_ch, stride=1, groups=4, base_width=64, kernel_size=128)([x, xo])

	x = tf.keras.layers.UpSampling2D()(x)
	x = tf.keras.layers.Conv2D(3, (4,4), padding="same")(x)
	return tf.keras.models.Model(inpt, x)


def train(args=None):
	logpath = os.path.join('logs', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
	os.mkdir(logpath)
	batch_size = 6
	epochs = 50
	train_gen = dolhasz.data_opt.iHarmonyGenerator(dataset='HFlickr', epochs=epochs, batch_size=batch_size, augment=True).no_masks()
	val_gen = dolhasz.data_opt.iHarmonyGenerator(dataset='HFlickr', epochs=epochs, batch_size=batch_size, training=False).no_masks()
	strategy = tf.distribute.MirroredStrategy()
	callbacks = [
		# tf.keras.callbacks.ModelCheckpoint(os.path.join(logpath, 'cp.ckpt'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True),
		tf.keras.callbacks.TensorBoard(log_dir=logpath),
		tf.keras.callbacks.ReduceLROnPlateau(patience=10)
		]
	with strategy.scope():
		model = make_axial_unet(
			start_ch=32, 
			groups=1, 
			n_blocks=2, 
			n_layers=3, 
			ksize=64, 
			dense=False
		)
		# model = make_unet()
		model.summary()
		opt = tf.keras.optimizers.Adam(lr=0.0001)
		model.compile(opt, 'mse', metrics=['mse', 'mae'])
	model.fit(
		x=train_gen,
		epochs=epochs,
		steps_per_epoch=tf.data.experimental.cardinality(train_gen).numpy()//epochs,
		callbacks=callbacks,
		validation_data=val_gen,
		validation_steps=tf.data.experimental.cardinality(val_gen).numpy()//epochs,
		max_queue_size=512,
		workers=16,
		use_multiprocessing=False,
		shuffle=True
   	)


def mse_scaled(y_true, y_pred):
	error = y_true-y_pred
	mse = tf.reduce_mean(tf.math.square(error), axis=[1,2,3])
	mse =  mse / (tf.math.reduce_sum(error, axis=[1,2,3]) + 0.00000001)
	return mse


def test(path):
	batch_size = 1
	epochs = 1
	val_gen = dolhasz.data_opt.iHarmonyGenerator(dataset='all', epochs=epochs, batch_size=batch_size, training=False).no_masks()
	# model = tf.keras.models.load_model(path, compile=False)
	model = AxialUnet(scale=2)
	model.build((batch_size,256,256,3))
	model.load_weights(path).expect_partial()
	model.compile('adam', 'mse')
	# model.evaluate(val_gen)

	for batch in val_gen:
		x, y = batch
		p = model.predict(x)
		f, ax = plt.subplots(1,3)
		ax[0].imshow(np.array(x).reshape(256,256,3))
		ax[1].imshow(np.array(y).reshape(256,256,3))
		ax[2].imshow(np.array(p).reshape(256,256,3))
		plt.show()

if __name__ == "__main__":

	if len(sys.argv) > 1:
		test(sys.argv[1])
	else:
		train()