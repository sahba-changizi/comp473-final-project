from keras.layers import Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, add, Dense, Input, \
    GlobalAveragePooling2D
from keras.models import Model


def block_depthwise(id, input, thread):
    blockName = 'block #'.format(id)
    result = DepthwiseConv2D(3, strides=(thread, thread), padding='same', use_bias=False,
                             name=blockName + 'DepthwiseConv2D')(input)
    result = BatchNormalization(name=blockName + 'DepthwiseBatchNormalization')(result)
    result = ReLU(6, name=blockName + 'DepthWiseReLU')(result)
    return result


def block_expansion(id, convolutions, input, size):
    blockName = 'block #'.format(id)
    total_filters = size * convolutions
    result = Conv2D(total_filters, 1, padding='same', use_bias=False, name=blockName + 'expansion')(input)
    result = BatchNormalization(name=blockName + 'expansionBatchNormalization')(result)
    result = ReLU(6, name=blockName + 'expansionRelu')(result)
    return result


def block_projection(id, input, out_channels):
    blockName = 'block #'.format(id)
    result = Conv2D(out_channels, kernel_size=1, padding='same', use_bias=False, name=blockName + 'projection')(
        input)
    result = BatchNormalization(name=blockName + 'projectionBatchNormalization')(result)
    return result


def block_bottleneck(id, convolutions, input, size, thread, result):
    mediate = block_expansion(id, convolutions, input, size)
    mediate = block_depthwise(id, mediate, thread)
    mediate = block_projection(id, mediate, result)
    if mediate.shape[-1] == input.shape[-1]:
        mediate = add([input, mediate])
    return mediate


def mobile_net_customized_object_classifier(input_image=(224, 224, 3)):
    input = Input(input_image)
    result = Conv2D(32, kernel_size=3, strides=(2, 2), padding='same', use_bias=False)(input)
    result = BatchNormalization(name='batchNormalization')(result)
    result = ReLU(6, name='relu')(result)
    result = block_depthwise(1, result, 1)
    result = block_projection(1, result, 16)
    result = block_bottleneck(2, result.shape[-1], result, 6, 2, 24)
    result = block_bottleneck(3, result.shape[-1], result, 6, 1, 24)
    result = block_bottleneck(4, result.shape[-1], result, 6, 2, 32)
    result = block_bottleneck(5, result.shape[-1], result, 6, 1, 32)
    result = block_bottleneck(6, result.shape[-1], result, 6, 1, 32)
    result = block_bottleneck(7, result.shape[-1], result, 6, 2, 64)
    result = block_bottleneck(8, result.shape[-1], result, 6, 1, 64)
    result = block_bottleneck(9, result.shape[-1], result, 6, 1, 64)
    result = block_bottleneck(10, result.shape[-1], result, 6, 1, 64)
    result = block_bottleneck(11, result.shape[-1], result, 6, 1, 96)
    result = block_bottleneck(12, result.shape[-1], result, 6, 1, 96)
    result = block_bottleneck(13, result.shape[-1], result, 6, 1, 96)
    result = block_bottleneck(14, result.shape[-1], result, 6, 2, 160)
    result = block_bottleneck(15, result.shape[-1], result, 6, 1, 160)
    result = block_bottleneck(16, result.shape[-1], result, 6, 1, 160)
    result = block_bottleneck(17, result.shape[-1], result, 6, 1, 320)
    result = Conv2D(1280, kernel_size=1, padding='same', use_bias=False, name='finalConvolution')(result)
    result = BatchNormalization(name='finalBatchNormalization')(result)
    result = ReLU(6, name='finalRelu')(result)
    result = GlobalAveragePooling2D(name='averagePooling2D')(result)
    output = Dense(3, activation='softmax')(result)
    model = Model(input, output)
    return model


input_shape = (224, 224, 3)
model = mobile_net_customized_object_classifier(input_shape)
model.summary()
