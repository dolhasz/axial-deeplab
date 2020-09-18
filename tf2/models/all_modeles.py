import tensorflow as tf
from axialnet import AxialDecoderBlock, AxialEncoderBlock
from resnet import upconv_block, identity_block


def baseline_resnet():

    skip_names = ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu')

    resnet50 = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', 
        input_shape=(256,256,3), pooling=None
    )
    for layer in resnet50.layers:
        layer.trainable = False

    skips = [resnet50.get_layer(s).output for s in skip_names]

    inp = tf.keras.layers.Input(shape=(256,256,3))
    x = resnet50(inp)

    x = upconv_block(x, 4, 2048, 1024)
    x = upconv_block(x, 4, 1024, 512)
    x = upconv_block(x, 4, 512, 256)
    x = upconv_block(x, 4, 256, 128)
    x = upconv_block(x, 4, 128, 64)
    x = tf.keras.layers.Conv2D(3, (4,4), padding="same")(x)

    return tf.keras.Model(inp, x)

if __name__ == "__main__":
    baseline_resnet().summary()