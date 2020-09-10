import tensorflow as tf

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
		self.k_transform = tf.keras.layers.Conv2D(out_ch // 2, (1,1), 
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

		self.reset_parameters()

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
		x = tf.transpose(x, (0, 3, 1, 2))

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
		output = self.bn_sv(tf.reshape(sv, (N, H, W, -1)), training=training) + self.bn_sve(tf.reshape(sve, (N, H, W, -1)), training=training)

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
		self.relu = tf.keras.layers.Activation('relu')
		self.downsample = downsample
		self.stride = stride

	def call(self, x, training=False):
		identity = x
		out = self.conv_down(x)
		out = self.bn1(out, training=training)
		out = self.relu(out)
		out = self.height_block(out, training=training)
		out = self.width_block(out, training=training)
		out = self.relu(out)
		out = self.conv_up(out)
		out = self.bn2(out, training=training)
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)

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
		self.relu = tf.keras.layers.Activation('relu')
		self.upconv = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(in_ch, (1,1)),
			tf.keras.layers.UpSampling2D()
		])
		self.stride = stride
		self.skip = skip

	def call(self, x, training=False): # TODO: if input is list of tensors, then unpack skip connection and pass to attention blocks
		identity = x
		out = self.conv_down(x)
		out = tf.keras.layers.UpSampling2D()(out)
		out = self.bn1(out, training=training)
		out = self.relu(out)
		out = self.height_block(out)
		out = self.width_block(out)
		out = self.relu(out)
		out = self.conv_up(out)
		out = self.bn2(out, training=training)
		out += self.upconv(identity)
		if self.skip is not None: # TODO: This should probably change
			out += self.skip
		out = self.relu(out)

		return out


class AxialUnet(tf.keras.Model):
	def __init__(self, in_ch=64, groups=8, base_width=64):
		super(AxialUnet, self).__init__()
		self.dilation = 1
		self.in_ch = in_ch
		self.base_width = base_width
		self.groups = groups
		self.conv_1 = tf.keras.layers.Conv2D(64, (7,7), strides=2, padding="same")
		self.bn_1 = tf.keras.layers.BatchNormalization()
		self.relu = tf.keras.layers.Activation('relu')
		self.mp = tf.keras.layers.MaxPool2D()

		self.layer1 = self._make_layer(AxialEncoderBlock, int(128), 1, kernel_size=64)
		self.layer2 = self._make_layer(AxialEncoderBlock, int(256), 1, stride=2, kernel_size=64,
									   dilate=False)
		self.layer3 = self._make_layer(AxialEncoderBlock, int(512), 1, stride=2, kernel_size=32,
									   dilate=False)
		self.layer4 = self._make_layer(AxialEncoderBlock, int(1024), 1, stride=2, kernel_size=16,
									   dilate=False)
		self.in_ch = 512
		self.layer5 = self._make_layer(AxialDecoderBlock, int(512), 1, kernel_size=16)
		self.in_ch = 256
		self.layer6 = self._make_layer(AxialDecoderBlock, int(256), 1, stride=1, kernel_size=32,
									   dilate=False)
		self.in_ch = 128
		self.layer7 = self._make_layer(AxialDecoderBlock, int(128), 1, stride=1, kernel_size=64,
									   dilate=False)
		self.in_ch = 64
		self.layer8 = self._make_layer(AxialDecoderBlock, int(64), 1, stride=1, kernel_size=128,
									   dilate=False)
		self.upsample = tf.keras.layers.UpSampling2D()
		self.final = tf.keras.layers.Conv2D(3, (3,3), padding="same")

	def call(self, x, training=False):
		x = self.conv_1(x)
		x = self.bn_1(x, training=training)
		x = self.relu(x)
		x = self.mp(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x = self.layer7(x)
		x = self.layer8(x)
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
		if stride != 1 or self.in_ch != ch * block.expansion:
			downsample = tf.keras.Sequential(
				[tf.keras.layers.Conv2D(ch * block.expansion, (1,1), strides=stride, padding="same"),
				tf.keras.layers.BatchNormalization()]
			)

		layers = []
		layers.append(block(self.in_ch, ch, stride=stride, downsample=downsample, groups=self.groups,
							base_width=self.base_width, dilation=previous_dilation, 
							kernel_size=kernel_size))
		self.in_ch = ch * block.expansion
		if stride != 1:
			kernel_size = kernel_size // 2

		for _ in range(1, blocks):
			layers.append(block(self.in_ch, ch, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								kernel_size=kernel_size))

		return tf.keras.models.Sequential(layers)



if __name__ == "__main__":
	import dolhasz
	
	batch_size = 6*3
	epochs = 50

	train_gen = dolhasz.data_opt.iHarmonyGenerator(epochs=epochs, batch_size=batch_size).no_masks()
	val_gen = dolhasz.data_opt.iHarmonyGenerator(epochs=epochs, batch_size=batch_size, training=False).no_masks()
	strategy = tf.distribute.MirroredStrategy()
	callbacks = [tf.keras.callbacks.ModelCheckpoint('/tmp/BEST_MODEL.tf', monitor='val_loss', verbose=0, save_best_only=True)]
	with strategy.scope():
		model = AxialUnet()
		model.build((batch_size,256,256,3))
		model.summary()
		model.compile('adam', 'mse')
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
#	model.load_weights('/tmp/BEST_MODEL.tf')
