import tensorflow as tf
import dolhasz
import sys
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

class AxialAttention(tf.keras.layers.Layer):
	def __init__(self, in_ch, out_ch, groups=8, kernel_size=56,
				 stride=1, bias=False, width=False):

		assert (in_ch % groups == 0) and (out_ch % groups == 0)
		super(AxialAttention, self).__init__()

		self.in_ch = in_ch
		self.out_ch = out_ch
		self.groups = groups
		self.group_ch = out_ch // groups
		self.kernel_size = kernel_size
		self.stride = stride
		self.bias = bias
		self.width = width

		# Multi-head self attention
		self.q_transform = tf.keras.layers.Conv2D(out_ch // 2, (1, 1), 
						   strides=(1, 1), padding='valid', use_bias=False, 
						   kernel_initializer=tf.keras.initializers.RandomNormal(0, tf.math.sqrt(1. / self.in_ch * self.group_ch)))
		self.k_transform = tf.keras.layers.Conv2D(out_ch // 2, (1, 1), 
						   strides=(1, 1), padding='valid', use_bias=False,
						   kernel_initializer=tf.keras.initializers.RandomNormal(0, tf.math.sqrt(1. / self.in_ch)))
		self.v_transform = tf.keras.layers.Conv2D(out_ch, (1, 1), 
						   strides=(1, 1), padding='valid', use_bias=False,
						   kernel_initializer=tf.keras.initializers.RandomNormal(0, tf.math.sqrt(1. / self.in_ch)))

		self.bn_q = tf.keras.layers.BatchNormalization()
		self.bn_k = tf.keras.layers.BatchNormalization()
		self.bn_v = tf.keras.layers.BatchNormalization()

		self.bn_qk = tf.keras.layers.BatchNormalization()
		self.bn_qr = tf.keras.layers.BatchNormalization()
		self.bn_kr = tf.keras.layers.BatchNormalization()

		self.bn_sv = tf.keras.layers.BatchNormalization()
		self.bn_sve = tf.keras.layers.BatchNormalization()

		# Positional embedding
		self.q_relative = tf.Variable(
			tf.random.normal((int(kernel_size * 2 - 1), 1, self.group_ch // 2)),
			trainable=True, name='q_rel'
		)

		self.k_relative = tf.Variable(
			tf.random.normal((int(kernel_size * 2 - 1), 1, self.group_ch // 2)),
			trainable=True, name='k_rel'
		)

		self.v_relative = tf.Variable(
			tf.random.normal((int(kernel_size * 2 - 1), 1, self.group_ch)),
			trainable=True, name='v_rel'
		)

		if stride > 1:
			self.pooling = tf.keras.layers.AveragePooling2D(
				pool_size=(stride, stride), 
				strides=(stride, stride)
			)
		else:
			self.pooling = None

		self.reset_parameters()

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'in_ch' : self.in_ch,
			'out_ch' : self.out_ch,
			'groups' : self.groups,
			'group_ch' : self.group_ch,
			'kernel_size' : self.kernel_size,
			'stride' : self.stride,
			'bias' : self.bias,
			'width' : self.width,
			'q_transform' : self.q_transform,
			'k_transform' : self.k_transform,
			'v_transform' : self.v_transform,
			'bn_q' : self.bn_q,
			'bn_k' : self.bn_k,
			'bn_v' : self.bn_v,
			'bn_qk' : self.bn_qk,
			'bn_qr' : self.bn_qr,
			'bn_kr' : self.bn_kr,
			'bn_sv' : self.bn_sv,
			'bn_sve' : self.bn_sve,
			'q_relative' : self.q_relative,
			'k_relative' : self.k_relative,
			'v_relative' : self.v_relative,
			'pooling' : self.pooling
		})
		return config

	def call(self, x, training=False):
		if self.width:
			x = tf.transpose(x, perm=[0,2,1,3])
		N, H, W, C = x.get_shape().as_list()
		N = tf.shape(x)[0]

		# x = tf.transpose(x, (0, 2, 3, 1))
		# Transformations
		q = self.q_transform(x)
		q = self.bn_q(q, training=training)
		k = self.k_transform(x)
		k = self.bn_k(k, training=training)
		v = self.v_transform(x)
		v = self.bn_v(v, training=training)

		# Calculate position embedding
		q_embedding = []
		k_embedding = []
		v_embedding = []
		for i in range(self.kernel_size):
			q_embedding.append(self.q_relative[self.kernel_size - 1 - i: self.kernel_size * 2 - 1 - i, :])
			k_embedding.append(self.k_relative[self.kernel_size - 1 - i: self.kernel_size * 2 - 1 - i, :])
			v_embedding.append(self.v_relative[self.kernel_size - 1 - i: self.kernel_size * 2 - 1 - i, :])
		q_embedding = tf.concat(q_embedding, axis=1)
		k_embedding = tf.concat(k_embedding, axis=1)
		v_embedding = tf.concat(v_embedding, axis=1)
		# 'ij, ij -> i
		new_q = tf.reshape(q, (N, H, W, self.groups, self.group_ch // 2))
		qr = tf.einsum('biwgc, ijc->bijwg', 
				new_q, 
				q_embedding
			)
		qr = tf.reshape(self.bn_qr(tf.reshape(qr, (N, -1, W, self.groups)), training=training), (N, H, H, W, self.groups))

		kr = tf.einsum('biwgc, ijc->bijwg', 
				tf.reshape(k, (N, H, W, self.groups, self.group_ch // 2)), 
				k_embedding
			)
		kr = tf.reshape(self.bn_kr(tf.reshape(kr, (N, -1, W, self.groups)), training=training), (N, H, H, W, self.groups))
		kr = tf.transpose(kr, (0, 2, 1, 3, 4))

		# Blocks of axial attention
		q = tf.reshape(q, (N, H, W, self.groups, self.group_ch // 2))
		k = tf.reshape(k, (N, H, W, self.groups, self.group_ch // 2))

		# (q, k)
		qk = tf.einsum('biwgc, bjwgc->bijwg', q, k)
		qk = tf.reshape(self.bn_qk(tf.reshape(qk, (N,-1, W, self.groups)), training=training), (N, H, H, W, self.groups))
	
		# (N, groups, H, H, W)
		similarity = tf.math.softmax(qk + qr + kr, axis=-2)
		sv = tf.einsum('bijwg, bjwgc->biwgc', similarity, tf.reshape(v, (N, H, W, self.groups, self.group_ch)))
		sve = tf.einsum('bijwg, jic->biwgc', similarity, v_embedding)
		output = self.bn_sv(tf.reshape(sv, (N, H, W, self.groups*self.group_ch)), training=training) + self.bn_sve(tf.reshape(sve, (N, H, W, self.groups*self.group_ch)), training=training)

		if self.width:
			output = tf.transpose(output, perm=[0,2,1,3])

		if self.stride > 1:
			output = self.pooling(output)

		return output

	def reset_parameters(self):
		n = self.out_ch // 2
		init = tf.keras.initializers.RandomNormal(0, tf.math.sqrt(1. / n))
		self.q_relative.assign(init(self.q_relative.shape))
		self.k_relative.assign(init(self.k_relative.shape))
		n = self.out_ch
		self.v_relative.assign(init(self.v_relative.shape))


class AxialEncoderBlock(tf.keras.layers.Layer):
	expansion = 1
	def __init__(self, in_ch, out_ch, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, kernel_size=56):
		super().__init__()
		width = int(in_ch * (base_width / 64.))
		self.conv_down = tf.keras.layers.Conv2D(width, (1,1))
		self.bn1 = tf.keras.layers.BatchNormalization()

		self.height_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
		self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)

		self.conv_up = tf.keras.layers.Conv2D(out_ch, (1,1))
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.relu1 = tf.keras.layers.Activation('relu')
		self.relu2 = tf.keras.layers.Activation('relu')
		self.relu3 = tf.keras.layers.Activation('relu')
		self.downsample = downsample
		self.stride = stride

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'conv_down' : self.conv_down,
			'bn1' : self.bn1,
			'height_block' : self.height_block,
			'width_block' : self.width_block,
			'conv_up' : self.conv_up,
			'bn2' : self.bn2,
			'relu1' : self.relu1,
			'relu2' : self.relu2,
			'relu3' : self.relu3,
			'downsample' : self.downsample,
			'stride' : self.stride
		})
		return config

	def call(self, x, training=True):
		identity = x
		out = self.conv_down(x)
		out = self.bn1(out, training=training)
		out = self.relu1(out)
		out = self.height_block(out, training=training)
		out = self.width_block(out, training=training)
		out = self.relu2(out)
		out = self.conv_up(out)
		out = self.bn2(out, training=training)
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu3(out)

		return out


class AxialDecoderBlock(tf.keras.layers.Layer):
	expansion = 1
	def __init__(self, in_ch, out_ch, stride=1, upsample=None, downsample=None, groups=1,
				 base_width=64, dilation=1, kernel_size=56, skip=None):
		super().__init__()
		width = int(in_ch * (base_width / 64.))
		self.conv_down = tf.keras.layers.Conv2D(width, (1,1))
		self.bn1 = tf.keras.layers.BatchNormalization()

		self.height_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
		self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)

		self.conv_up = tf.keras.layers.Conv2D(out_ch, (1,1))
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.relu1 = tf.keras.layers.Activation('relu')
		self.relu2 = tf.keras.layers.Activation('relu')
		self.relu3 = tf.keras.layers.Activation('relu')
		self.upconv = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(out_ch, (1,1)),
			tf.keras.layers.UpSampling2D()
		])
		self.stride = stride
		self.skip_conv = tf.keras.layers.Conv2D(out_ch, (1,1))


	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'conv_down' : self.conv_down,
			'bn1' : self.bn1,
			'height_block' : self.height_block,
			'width_block' : self.width_block,
			'conv_up' : self.conv_up,
			'bn2' : self.bn2,
			'relu1' : self.relu1,
			'relu2' : self.relu2,
			'relu3' : self.relu3,
			'upconv' : self.upconv,
			'stride' : self.stride,
			'skip_conv' : self.skip_conv
		})
		return config

	def call(self, x, training=True): # TODO: if input is list of tensors, then unpack skip connection and pass to attention blocks
		skip = None
		if isinstance(x, list):
			x, skip = x
		identity = x
		out = self.conv_down(x)
		out = tf.keras.layers.UpSampling2D()(out)
		out = self.bn1(out, training=training)
		out = self.relu1(out)
		out = self.height_block(out, training=training)
		out = self.width_block(out, training=training)
		out = self.relu2(out)
		out = self.conv_up(out)
		out = self.bn2(out, training=training)
		out += self.upconv(identity)
		if skip is not None: # TODO: This should probably change
			skip = self.skip_conv(skip)
			out += skip
		out = self.relu3(out)

		return out


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
		for b in range(n_blocks-1):
			x = AxialEncoderBlock(start_ch, start_ch, stride=1, groups=groups, base_width=start_ch, kernel_size=ksize)(x)
			if b == 0:
				skips.append(x)
		x = AxialEncoderBlock(start_ch, start_ch*2, stride=2, groups=groups, base_width=start_ch, kernel_size=ksize, downsample=make_ds(start_ch*2))(x)
		start_ch *= 2
		ksize //= 2

	if dense:
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1024)(x)
		x = tf.keras.layers.Reshape((1,1,1024))(x)

	# Build decoder
	for l in range(n_layers):
		ksize *= 2
		x = AxialDecoderBlock(start_ch, start_ch//2, stride=1, groups=groups, base_width=start_ch, kernel_size=ksize)([x, skips[n_layers-l-1]])
		start_ch //= 2

	x = tf.keras.layers.Add()([x, xp])
	x = tf.keras.layers.UpSampling2D()(x)
	x = tf.keras.layers.Conv2D(start_ch, (4,4), padding="same")(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Add()([x, xo])
	x = tf.keras.layers.UpSampling2D()(x)
	x = tf.keras.layers.Conv2D(3, (4,4), padding="same")(x)

	return tf.keras.Model(inpt, x)


class AxialUnet(tf.keras.Model):
	def __init__(self, in_ch=64, groups=8, base_width=64, scale=1):
		super(AxialUnet, self).__init__()
		self.dilation = None
		self.in_ch = int(in_ch / scale)
		self.base_width = int(base_width / scale)
		self.groups = groups
		self.conv_1 = tf.keras.layers.Conv2D(int(64/scale), (7,7), strides=2, padding="same")
		self.bn_1 = tf.keras.layers.BatchNormalization()
		self.relu = tf.keras.layers.Activation('relu')
		self.mp = tf.keras.layers.MaxPool2D()

		self.layer1 = self._make_layer(AxialEncoderBlock, int(128/scale), 3, kernel_size=64)
		self.layer2 = self._make_layer(AxialEncoderBlock, int(256/scale), 4, stride=2, kernel_size=64,
									   dilate=False)
		self.layer3 = self._make_layer(AxialEncoderBlock, int(512/scale), 6, stride=2, kernel_size=32,
									   dilate=False)
		self.layer4 = self._make_layer(AxialEncoderBlock, int(1024/scale), 3, stride=2, kernel_size=16,
									   dilate=False)
		self.in_ch = int(512/scale)
		self.layer5 = self._make_layer(AxialDecoderBlock, int(512/scale), 1, stride=1, kernel_size=16)

		# self.mid_dense = tf.keras.layers.Dense(1024)
		# self.reshape = tf.keras.layers.Reshape((1,1,1024))

		self.in_ch = int(256/scale)
		self.layer6 = self._make_layer(AxialDecoderBlock, int(256/scale), 1, stride=1, kernel_size=32,
									   dilate=False)
		self.in_ch = int(128/scale)
		self.layer7 = self._make_layer(AxialDecoderBlock, int(128/scale), 1, stride=1, kernel_size=64,
									   dilate=False)
		self.in_ch = int(64/scale)
		self.layer8 = self._make_layer(AxialDecoderBlock, int(64/scale), 1, stride=1, kernel_size=128,
									   dilate=False)
		self.upsample = tf.keras.layers.UpSampling2D()
		self.final = tf.keras.layers.Conv2D(3, (3,3), padding="same")

	def call(self, x, training=False):
		x = self.conv_1(x)
		x = self.bn_1(x, training=training)
		xmp = self.relu(x)
		x = self.mp(xmp)
		x1 = self.layer1(x, training=training)
		x2 = self.layer2(x1, training=training)
		x3 = self.layer3(x2, training=training)
		x = self.layer4(x3, training=training)
		x = self.layer5([x, x3], training=training)
		x = self.layer6([x, x2], training=training)
		x = self.layer7([x, x1], training=training)
		x = self.layer8([x, xmp], training=training)
		x = self.upsample(x)
		x = self.final(x)
		# x = self.aa4(x)
		return x

	def _make_layer(self, block, ch, blocks, kernel_size=56, stride=1, dilate=False):
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if (stride != 1 or self.in_ch != ch * block.expansion):
			downsample = tf.keras.Sequential(
				[tf.keras.layers.Conv2D(ch * block.expansion, (1,1), strides=stride, padding="same"),
				tf.keras.layers.BatchNormalization()]
			)

		layers = []
		layers.append(block(self.in_ch, ch, stride=stride, downsample=downsample, groups=self.groups,
							base_width=self.base_width, dilation=previous_dilation, 
							kernel_size=kernel_size))
		print(dict(in_ch=self.in_ch, out_ch=ch, stride=stride, downsample=downsample, groups=self.groups,
							base_width=self.base_width, dilation=previous_dilation, 
							kernel_size=kernel_size))
		self.in_ch = ch * block.expansion
		if stride != 1:# and not isinstance(block, AxialDecoderBlock):
			kernel_size = kernel_size // 2

		for _ in range(1, blocks):
			layers.append(block(self.in_ch, ch, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								kernel_size=kernel_size))
			print(dict(in_ch=self.in_ch, out_ch=ch, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								kernel_size=kernel_size))

		return tf.keras.models.Sequential(layers)


def train(args=None):
	logpath = os.path.join('logs', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
	os.mkdir(logpath)
	batch_size = 12
	epochs = 50
	train_gen = dolhasz.data_opt.iHarmonyGenerator(dataset='HFlickr', epochs=epochs, batch_size=batch_size).no_masks()
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
	# count = tf.math.count_nonzero(error, [1,2,3])
	# area = tf.cast(count, 'float32') /  (y_true.shape[1]*y_true.shape[2]*y_true.shape[3])
	mse = tf.reduce_mean(tf.math.square(error), axis=[1,2,3]) / (tf.math.reduce_sum(error, axis=[1,2,3])+0.00000001)
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
		# print(x.shape)
		# print(y.shape)
		p = model.predict(x)
		print(y)
		f, ax = plt.subplots(1,3)
		ax[0].imshow(np.array(x).reshape(256,256,3))
		ax[1].imshow(np.array(y).reshape(256,256,3))
		ax[2].imshow(np.array(p).reshape(256,256,3))
		plt.show()

if __name__ == "__main__":

	# make_axial_unet(start_ch=16, groups=2, n_blocks=2, n_layers=4, ksize=64, dense=False).summary()

	if len(sys.argv) > 1:
		test(sys.argv[1])
	else:
		train()


	

#	model.load_weights('/tmp/BEST_MODEL.tf')
