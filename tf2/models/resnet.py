import tensorflow as tf

def identity_block(x, f, in_ch, out_ch):
	x_skip = x
	x = tf.keras.layers.Conv2D(in_ch, (1,1), strides=(1,1), padding="valid")(x)
	x = tf.keras.layers.BatchNormalization(axis=-1)(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(in_ch, (f,f), strides=(1,1), padding="same")(x)
	x = tf.keras.layers.BatchNormalization(axis=-1)(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(out_ch, (1,1), strides=(1,1), padding="valid")(x)
	x = tf.keras.layers.BatchNormalization(axis=-1)(x)
	x = tf.keras.layers.Add()([x, x_skip])
	x = tf.keras.layers.Activation('relu')(x)
	return x


def conv_block(x, f, in_ch, out_ch, stride=2):
	x_skip = x
	x = tf.keras.layers.Conv2D(in_ch, (1,1), strides=(stride,stride), padding="valid")(x)
	x = tf.keras.layers.BatchNormalization(axis=-1)(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(in_ch, (f,f), strides=(1,1), padding="same")(x)
	x = tf.keras.layers.BatchNormalization(axis=-1)(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(out_ch, (1,1), strides=(1,1), padding="valid")(x)
	x = tf.keras.layers.BatchNormalization(axis=-1)(x)
	x_skip = tf.keras.layers.Conv2D(out_ch, (1,1), strides=(stride,stride), padding="valid")(x_skip)
	x_skip = tf.keras.layers.BatchNormalization(axis=-1)(x_skip)
	x = tf.keras.layers.Add()([x, x_skip])
	x = tf.keras.layers.Activation('relu')(x)
	return x


def upconv_block(x, f, in_ch, out_ch):
	x = tf.keras.layers.UpSampling2D()(x)
	x_skip = x
	x = tf.keras.layers.Conv2D(in_ch, (1,1), strides=(1,1), padding="valid")(x)
	x = tf.keras.layers.BatchNormalization(axis=-1)(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(in_ch, (f,f), strides=(1,1), padding="same")(x)
	x = tf.keras.layers.BatchNormalization(axis=-1)(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(out_ch, (1,1), strides=(1,1), padding="valid")(x)
	x = tf.keras.layers.BatchNormalization(axis=-1)(x)
	x_skip = tf.keras.layers.Conv2D(out_ch, (1,1), strides=(1,1), padding="valid")(x_skip)
	x_skip = tf.keras.layers.BatchNormalization(axis=-1)(x_skip)
	x = tf.keras.layers.Add()([x, x_skip])
	x = tf.keras.layers.Activation('relu')(x)
	return x