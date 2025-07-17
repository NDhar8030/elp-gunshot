import tensorflow as tf
from keras import layers, Model, Input

def create_model(input_shape):
    def conv_block(inputs, filters, kernel_size, strides):
        """Standard Conv2D + BN + ReLU6 block."""
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        return x

    def depthwise_conv_block(inputs, pointwise_filters, depth_multiplier=1, kernel_size=(3,3), strides=(1,1)):
        """Depthwise Separable Conv2D block: Depthwise + Pointwise + BN + ReLU6."""
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', depth_multiplier=depth_multiplier, use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.Conv2D(pointwise_filters, (1,1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        return x
    
    inputs = Input(shape=(input_shape))
    print(inputs.shape)
    #inputs = layers.Lambda(lambda x: tf.reshape(inputs.shape[2],inputs.shape[3]))

    expanded_inputs = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)
    print(expanded_inputs.shape)
    x = conv_block(expanded_inputs, 32, (5,5), (2,2))
    x = depthwise_conv_block(x, 32, 1, (3,3), (1,1))
    x = depthwise_conv_block(x, 64, 1, (3,3), (2,2))
    x = depthwise_conv_block(x, 128, 1, (3,3), (2,2))
    x = depthwise_conv_block(x, 256, 1, (3,3), (2,2))
    x = depthwise_conv_block(x, 512, 1, (3,3), (2,2))

    x = layers.GlobalAveragePooling2D(keepdims=True)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model