import tensorflow as tf
from axialnet import AxialDecoderBlock, AxialEncoderBlock
from resnet import upconv_block, identity_block


def baseline_resnet():

    skip_names = ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu')

    inpt = tf.keras.layers.Input(shape=(256,256,3), name="main_input")
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', 
        input_tensor=inpt, pooling=None
    )

    for layer in backbone.layers:
        layer.trainable = False

    skips = [backbone.get_layer(s).output for s in skip_names]

    x = upconv_block(backbone.output, 4, 2048, 1024)
    x = tf.keras.layers.Add()([x, skips[0]])
    x = upconv_block(x, 4, 1024, 512)
    x = tf.keras.layers.Add()([x, skips[1]])
    x = upconv_block(x, 4, 512, 256)
    x = tf.keras.layers.Add()([x, skips[2]])
    x = upconv_block(x, 4, 256, 128)
    x = tf.keras.layers.Conv2D(64, (1,1))(x)
    x = tf.keras.layers.Add()([x, skips[3]])
    x = upconv_block(x, 4, 128, 64)
    # x = tf.keras.layers.Add()([x, skips[4]])
    x = tf.keras.layers.Conv2D(3, (4,4), padding="same")(x)
    model = tf.keras.Model(inpt, x)
    return model


def baseline_resnet_dense():
    skip_names = ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu')

    inpt = tf.keras.layers.Input(shape=(256,256,3), name="main_input")
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', 
        input_tensor=inpt, pooling='avg'
    )

    for layer in backbone.layers:
        layer.trainable = False

    skips = [backbone.get_layer(s).output for s in skip_names]

    x = tf.keras.layers.Flatten()(backbone.output)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Reshape((1,1,1024))(x)
    x = upconv_block(x, 4, 1024, 512)
    x = upconv_block(x, 4, 512, 256)
    x = upconv_block(x, 4, 256, 256)
    x = upconv_block(x, 4, 256, 256, skips[0])
    x = upconv_block(x, 4, 256, 256, skips[1])
    x = upconv_block(x, 4, 256, 128, skips[2])
    x = upconv_block(x, 4, 128, 64)
    x = tf.keras.layers.Conv2D(64, (1,1))(x)
    x = tf.keras.layers.Add()([x, skips[3]])
    x = upconv_block(x, 4, 128, 64)
    x = tf.keras.layers.Conv2D(3, (4,4), padding="same")(x)
    model = tf.keras.Model(inpt, x)
    model.summary()
    return model


def resnet_axial_decoder():
    skip_names = ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu')

    inp = tf.keras.layers.Input(shape=(256,256,3))
    resnet50 = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', 
        input_tensor=inp, pooling=None
    )
    for layer in resnet50.layers:
        layer.trainable = False

    skips = [resnet50.get_layer(s).output for s in skip_names]

    x = AxialDecoderBlock(2048, 1024, stride=1, upsample=True, downsample=None, groups=16,
				 base_width=64, kernel_size=16)([resnet50.output, skips[0]])
    x = AxialDecoderBlock(1024, 512, stride=1, upsample=True, downsample=None, groups=16,
				 base_width=64, kernel_size=32)([x, skips[1]])
    x = AxialDecoderBlock(512, 256, stride=1, upsample=True, downsample=None, groups=16,
				 base_width=64, kernel_size=64)([x, skips[2]])
    x = upconv_block(x, 4, 256, 128, skips[3])
    x = upconv_block(x, 4, 128, 64)
    x = tf.keras.layers.Conv2D(3, (4,4), padding="same")(x)
    return tf.keras.Model(inp, x)


if __name__ == "__main__":
    model = baseline_resnet_dense()
    model.summary()