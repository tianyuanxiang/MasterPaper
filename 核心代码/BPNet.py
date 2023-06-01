import keras.utils
from keras.activations import sigmoid
from keras.backend import categorical_crossentropy
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
import numpy as np
import os

def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits


    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    # 负样本之和
    count_neg = tf.reduce_sum(1. - y_true)
    # 正样本之和
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    # pos_weight：正样本中使用的系数
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def positive_num(y_true, y_pred):
    one = tf.ones_like(y_true)
    zero = tf.zeros_like(y_true)
    y_pred = tf.where(y_pred < 0.5, x=zero, y=one)
    return tf.count_nonzero(y_pred)

def focal_loss(alpha, gamma):
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        # y_true 是个一阶向量, 下式按照加号分为左右两部分
        # 注意到 y_true的取值只能是 0或者1 (假设二分类问题)，可以视为“掩码”
        # 加号左边的 y_true*alpha 表示将 y_true中等于1的槽位置为标量 alpha
        # 加号右边的 (ones-y_true)*(1-alpha) 则是将等于0的槽位置为 1-alpha
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
        # 类似上面，y_true仍然视为 0/1 掩码
        # 第1部分 `y_true*y_pred` 表示 将 y_true中为1的槽位置为 y_pred对应槽位的值
        # 第2部分 `(ones-y_true)*(ones-y_pred)` 表示 将 y_true中为0的槽位置为 (1-y_pred)对应槽位的值
        # 第3部分 K.epsilon() 避免后面 log(0) 溢出
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed


def SWF_Conv(inputs, in_cannel ,stride, kernel_size):
    result = inputs
    # 多尺度
    r = kernel_size
    filters = in_cannel
    # (r:第一行, r：最后一行), (r：第一列, 0：最后一列))
    P_L = ZeroPadding2D(((r, r), (r, 0)))(result)
    P_R = ZeroPadding2D(((r, r), (0, r)))(result)
    P_U = ZeroPadding2D(((r, 0), (r, r)))(result)
    P_D = ZeroPadding2D(((0, r), (r, r)))(result)


    P_NW = ZeroPadding2D(((r, 0), (r, 0)))(result)
    P_NE = ZeroPadding2D(((r, 0), (0, r)))(result)
    P_SW = ZeroPadding2D(((0, r), (r, 0)))(result)
    P_SE = ZeroPadding2D(((0, r), (0, r)))(result)

    # 八种方式卷积 + relu
    C_L = Conv2D(filters, (2 * r + 1, r + 1), strides=stride, activation='relu')(P_L)
    C_R = Conv2D(filters, (2 * r + 1, r + 1), strides=stride, activation='relu')(P_R)
    C_U = Conv2D(filters, (r + 1, 2 * r + 1), strides=stride, activation='relu')(P_U)
    C_D = Conv2D(filters, (r + 1, 2 * r + 1), strides=stride, activation='relu')(P_D)
    C_NW = Conv2D(filters, (r + 1, r + 1), strides=stride, activation='relu')(P_NW)
    C_NE = Conv2D(filters, (r + 1, r + 1), strides=stride, activation='relu')(P_NE)
    C_SW = Conv2D(filters, (r + 1, r + 1), strides=stride, activation='relu')(P_SW)
    C_SE = Conv2D(filters, (r + 1, r + 1), strides=stride, activation='relu')(P_SE)

    # 拼接
    x = Concatenate()([C_L, C_R, C_U, C_D, C_NW, C_NE, C_SW, C_SE])




    # #         # 8 * w -> 4 * 8 * w -> 8 * w
    # d = 4 * 8 * w
    # expan = Conv2D(d, 1, activation='relu')(x)
    # dotsum = Lambda(tf.reduce_sum, arguments={'axis': (3), 'keepdims': True})(expan)
    # return dotsum

    # x = Conv2D(in_cannel, 1, strides=1, activation='relu')(x)
    return x

def SceneEmbedding(x):
    batch, row, col, channel = x.shape.as_list()
    squeeze = GlobalAveragePooling2D()(x)
    squeeze = Reshape((1, 1, channel))(squeeze)

    squeeze = Conv2D(channel, 1, activation='sigmoid', padding = 'same')(squeeze)
    Emb = Multiply()([squeeze, x])

    max = MaxPooling2D(pool_size=(2, 2))(Emb)
    avg = AvgPool2D(pool_size=(2, 2))(Emb)
    max = Conv2D(channel, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(max))
    avg = Conv2D(channel, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(avg))
    Edge_Feature = Subtract()([x, avg])
    fusion = Concatenate()([Edge_Feature, max])
    fusion = Conv2D(channel, 1, activation='relu', padding='same')(fusion)
    fusion = BatchNormalization()(fusion)
    out = Add()([fusion, x])
    return out

def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if out_filters >= 512:
        x = Dropout(0.4)(x)
    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])


    x = Activation('relu')(x)
    return x


def BoundaryGuidanceModule(N):
    
    # SWF = SWF_Conv(N, 32, stride=1, kernel_size=1)
    x1 = Conv2D(32, 3, padding='same')(N)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    B1 = Conv2D(32, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(x1)
    B1 = BatchNormalization()(B1)
    B1 = Activation('relu')(B1)
    return B1

def DFP(input_size, pretrained_weights = False, Falg_summary=True , model_summary=False):
    inputs = Input(input_size)

    # ==================================== 编码器 ==============================================
    # c0：256,256,64
    conv0_1 = Conv2D(64, 7, strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    conv0_1 = BatchNormalization(axis=3)(conv0_1)
    conv0_1 = Activation('relu')(conv0_1)
    BG1 = BoundaryGuidanceModule(conv0_1)

    # c1：128,128,128
    conv0_2 = MaxPooling2D(pool_size=(2, 2))(conv0_1)
    conv1_1 = bottleneck_Block(conv0_2, 128, strides=(1, 1), with_conv_shortcut=True)
    conv1_2 = bottleneck_Block(conv1_1, 128)
    conv1_3 = bottleneck_Block(conv1_2, 128)
    BG2 = BoundaryGuidanceModule(conv1_3)

    # c2：64,64,256
    conv2_1 = bottleneck_Block(conv1_3, 256, strides=(2, 2), with_conv_shortcut=True)
    conv2_2 = bottleneck_Block(conv2_1, 256)
    conv2_3 = bottleneck_Block(conv2_2, 256)
    conv2_4 = bottleneck_Block(conv2_3, 256)
    BG3 = BoundaryGuidanceModule(conv2_4)

    # c4：32,32,512
    conv3_1 = bottleneck_Block(conv2_4, 512, strides=(2, 2), with_conv_shortcut=True)
    conv3_2 = bottleneck_Block(conv3_1, 512)
    conv3_3 = bottleneck_Block(conv3_2, 512)
    conv3_4 = bottleneck_Block(conv3_3, 512)
    conv3_5 = bottleneck_Block(conv3_4, 512)
    conv3_6 = bottleneck_Block(conv3_5, 512)
    conv3_7 = bottleneck_Block(conv3_6, 512)
    conv3_8 = bottleneck_Block(conv3_7, 512)
    conv3_9 = bottleneck_Block(conv3_8, 512)
    conv3_10 = bottleneck_Block(conv3_9, 512)
    conv3_11 = bottleneck_Block(conv3_10, 512)
    conv3_12 = bottleneck_Block(conv3_11, 512)
    conv3_13 = bottleneck_Block(conv3_12, 512)
    conv3_14 = bottleneck_Block(conv3_13, 512)
    conv3_15 = bottleneck_Block(conv3_14, 512)
    conv3_16 = bottleneck_Block(conv3_15, 512)
    conv3_17 = bottleneck_Block(conv3_16, 512)
    conv3_18 = bottleneck_Block(conv3_17, 512)
    conv3_19 = bottleneck_Block(conv3_18, 512)
    conv3_20 = bottleneck_Block(conv3_19, 512)
    conv3_21 = bottleneck_Block(conv3_20, 512)
    conv3_22 = bottleneck_Block(conv3_21, 512)
    conv3_23 = bottleneck_Block(conv3_22, 512)

    # c5：16,16,1024
    conv4_1 = bottleneck_Block(conv3_23, 1024, strides=(2, 2), with_conv_shortcut=True)
    conv4_2 = bottleneck_Block(conv4_1, 1024)
    conv4_3 = bottleneck_Block(conv4_2, 1024)
    # 上采样，用于拼接
    BG2_2 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(BG2))
    BG3_3 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(4, 4))(BG3))
    BG_Sum_C = Concatenate()([BG1, BG2_2, BG3_3])

    BG_Sum_C = Conv2D(32, 3, strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(BG_Sum_C)
    BG_Sum_C = BatchNormalization(axis=3)(BG_Sum_C)
    BG_Sum_C = Activation('relu')(BG_Sum_C)

    BG_Sum = Conv2D(2, 3, padding='same', activation='relu', use_bias=False, kernel_initializer='he_normal')(BG_Sum_C)
    BG_Sum = Conv2D(1, 1, activation='sigmoid', name='boundary_guidance')(BG_Sum)

    # ==================================== 解码器 ==============================================
    p1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4_3))
    p1 = concatenate([conv3_23, p1])

    p2 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(p1))
    p2 = concatenate([conv2_4, p2])

    p3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(p2))
    p3 = concatenate([conv1_3, p3])

    p4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(p3))
    p4 = concatenate([conv0_1, p4])

    # 对象增强模块
    # 由深到浅
    Z1 = Conv2D(512, 1, strides=1, padding='same')(p1)
    Z1 = BatchNormalization(axis=3)(Z1)
    Z1 = Activation('relu')(Z1)
    # R1 = SceneEmbedding(Z1)

    f_p1 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(Z1))
    Z2 = Conv2D(256, 1, strides=1, padding='same')(p2)
    Z2 = BatchNormalization(axis=3)(Z2)
    Z2 = Activation('relu')(Z2)
    # R2 = SceneEmbedding(Z2)
    R2 = Concatenate()([f_p1, Z2])

    f_p2 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(R2))
    Z3 = Conv2D(128, 1, strides=1, padding='same')(p3)
    Z3 = BatchNormalization(axis=3)(Z3)
    Z3 = Activation('relu')(Z3)
    # R3 = SceneEmbedding(Z3)
    R3 = Concatenate()([f_p2, Z3])

    f_p3 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(R3))
    Z4 = Conv2D(64, 1, strides=1, padding='same')(p4)
    Z4 = BatchNormalization(axis=3)(Z4)
    Z4 = Activation('relu')(Z4)
    R4 = SceneEmbedding(Z4)
    R4 = Concatenate()([R4, f_p3])

    f_all = Concatenate()([R4, BG_Sum_C])
    Finish = Conv2D(16, 1, padding='same', activation='relu', use_bias=False, kernel_initializer='he_normal')(f_all)
    Finish = Conv2D(1, 1, activation='sigmoid', name='sealand_seg')(Finish)

    model = Model(inputs=inputs, outputs=[Finish, BG_Sum])

    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'sealand_seg': 'binary_crossentropy',
                        'boundary_guidance': focal_loss(0.9,2.)
                        },
                  loss_weights={
                      'sealand_seg': 1.0,
                      'boundary_guidance': 1.2,
                  },
                  metrics={'sealand_seg': ['accuracy'],
                           'boundary_guidance': positive_num}
                           )

    if Falg_summary:
        model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    model = DFP(input_size=(256, 256, 4), Falg_summary=True)
