# -*- coding: utf-8 -*-
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
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
def lovasz_softmax(labels, probas, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """

    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels


def _SE(inputs, reduction=16):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    out_dim = K.int_shape(inputs)[channel_axis]  # 计算输入特征图的通道数
    temp_dim = max(out_dim // reduction, reduction)

    squeeze = GlobalAvgPool2D()(inputs)
    if channel_axis == -1:
        excitation = Reshape((1, 1, out_dim))(squeeze)
    else:
        excitation = Reshape((out_dim, 1, 1))(squeeze)
    excitation = Conv2D(temp_dim, 1, 1, activation='relu')(excitation)
    excitation = Conv2D(out_dim, 1, 1, activation='sigmoid')(excitation)

    return excitation
# kernel_size:3; groups:1
# kernel_size:5; groups:4
# kernel_size:7; groups:8
# kernel_size:9; groups:16
def group_conv2(x, filters, kernel_size, stride, groups, padding='same'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]  # 计算输入特征图的通道数
    nb_ig = in_channels // groups  # 对输入特征图通道进行分组
    nb_og = filters // groups  # 对输出特征图通道进行分组
    assert in_channels % groups == 0
    assert filters % groups == 0
    # assert filters > groups
    gc_list = []
    for i in range(groups):
        if channel_axis == -1:
            x_group = Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
        x_group = ZeroPadding2D(padding=padding, data_format=None)(x_group)
        gc_list.append(Conv2D(filters=nb_og, kernel_size=kernel_size, strides=stride, use_bias=False)(x_group))
    return Concatenate(axis=channel_axis)(gc_list) if groups != 1 else gc_list[0]

def Fusion(inputs, out_channel, conv_kernels, stride = 1, conv_groups=[1, 4, 8, 16]):
    conv_1 = group_conv2(inputs, out_channel // 4, kernel_size=3, padding=conv_kernels[0]//2,
                         stride=stride, groups=conv_groups[0])

    conv_2 = group_conv2(inputs, out_channel // 4, kernel_size=5, padding=conv_kernels[1]//2,
                         stride=stride, groups=conv_groups[1])

    conv_3 = group_conv2(inputs, out_channel // 4, kernel_size=7, padding=conv_kernels[2]//2,
                         stride=stride, groups=conv_groups[2])

    conv_4 = group_conv2(inputs, out_channel // 4, kernel_size=9, padding=conv_kernels[3]//2,
                         stride=stride, groups=conv_groups[3])
    feats = Concatenate()([conv_1, conv_2, conv_3, conv_4])
    feats = Dropout(0.2)(feats)
    # feats = Reshape((in_dim[1], in_dim[2], split_channel, split_num))(feats)
    return feats
def EPSA2(inputs, out_channel, conv_kernels, stride = 1, conv_groups=[1, 4, 8, 16]):
    in_dim = K.int_shape(inputs)
    split_num = len(conv_kernels)
    split_channel = out_channel // len(conv_kernels)
    conv_1 = group_conv2(inputs, out_channel//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
    conv_2 = group_conv2(inputs, out_channel // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                  stride=stride, groups=conv_groups[1])
    conv_3 = group_conv2(inputs, out_channel // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                  stride=stride, groups=conv_groups[2])
    conv_4 = group_conv2(inputs, out_channel // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                  stride=stride, groups=conv_groups[3])
    # 多尺度拼接
    feats = Concatenate()([conv_1, conv_2, conv_3, conv_4])
    feats = Reshape((in_dim[1], in_dim[2], split_channel, split_num))(feats)

    x1_se = _SE(conv_1)
    x2_se = _SE(conv_2)
    x3_se = _SE(conv_3)
    x4_se = _SE(conv_4)

    merge_se = Concatenate()([x1_se, x2_se, x3_se, x4_se])
    merge_se = Reshape((1, 1, split_channel, split_num))(merge_se)

    merge_se = Activation('softmax')(merge_se)
    feats_weight = Multiply()([merge_se, feats])

    feats_weight = Reshape((in_dim[1], in_dim[2], out_channel))(feats_weight)

    return feats_weight
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

# x, 3, [64, 64, 256], stage=2, block='b'
def identity_block(input_tensor, kernel_size, filters, stage, block, flag):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 换的就是你
    # x = EPSA2(x, filters2, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# EPSA(in_channel, out_channel, conv_kernels=[3, 5, 7, 9], stride = 1, conv_groups=[1, 4, 8, 16]):
def conv_block(input_tensor, kernel_size, filters, flag, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    # x = EPSA2(x, filters2, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)


    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x
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
    if out_filters >= 1024:
        x = Dropout(0.4)(x)
    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])


    x = Activation('relu')(x)
    return x
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

def SceneEmbedding(x):
    squeeze = GlobalAveragePooling2D()(x)
    squeeze = Reshape((1, 1, 256))(squeeze)
    squeeze = Conv2D(256, 1, activation='relu', padding = 'same')(squeeze)
    squeeze = Activation('sigmoid')(squeeze)
    Emb = Multiply()([squeeze, x])


    max = MaxPooling2D(pool_size=(2, 2))(Emb)
    avg = AvgPool2D(pool_size=(2, 2))(Emb)
    max = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(max))
    avg = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(avg))
    Edge_Feature = Subtract()([x, avg])
    fusion = Concatenate()([Edge_Feature, max])
    fusion = Conv2D(256, 1, activation='relu', padding='same')(fusion)
    fusion = BatchNormalization()(fusion)
    out = Add()([fusion, x])

    return out

def positive_num(y_true, y_pred):
    one = tf.ones_like(y_true)
    zero = tf.zeros_like(y_true)
    y_pred = tf.where(y_pred < 0.5, x=zero, y=one)
    return tf.count_nonzero(y_pred)

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """

    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss

    return focal_loss

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

def SUM(n):
    grad_mag_components = n ** 2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)
    return grad_mag_square


def MyCanny(sea_super):
    grad_components = tf.image.sobel_edges(sea_super)
    grad_mag_components = grad_components ** 2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)
    return grad_mag_square

def my_upsampling(x, img_w, img_h, x2):
    """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""
    return tf.image.resize_images(x, (img_w * x2, img_h * x2), 0)

def FarSeg(input_size, num_class, pretrained_weights = False, Falg_summary=True , model_summary=False):
    inputs = Input(input_size)

    x1 = SWF_Conv(inputs, 32, stride=1, kernel_size=1)

    x1 = Conv2D(32, 3, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x = Conv2D(32, 3, padding='same')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Concatenate()([x1, x])

    # conv2_x  1/4
    conv2_1 = bottleneck_Block(x, 256, strides=(4, 4), with_conv_shortcut=True)
    conv2_2 = bottleneck_Block(conv2_1, 256)
    conv2_3 = bottleneck_Block(conv2_2, 256)

    # conv3_x  1/8
    conv3_1 = bottleneck_Block(conv2_3, 512, strides=(2, 2), with_conv_shortcut=True)
    conv3_2 = bottleneck_Block(conv3_1, 512)
    conv3_3 = bottleneck_Block(conv3_2, 512)
    conv3_4 = bottleneck_Block(conv3_3, 512)

    # conv4_x  1/16
    conv4_1 = bottleneck_Block(conv3_4, 1024, strides=(2, 2), with_conv_shortcut=True)
    conv4_2 = bottleneck_Block(conv4_1, 1024)
    conv4_3 = bottleneck_Block(conv4_2, 1024)
    conv4_4 = bottleneck_Block(conv4_3, 1024)
    conv4_5 = bottleneck_Block(conv4_4, 1024)
    conv4_6 = bottleneck_Block(conv4_5, 1024)
    conv4_7 = bottleneck_Block(conv4_6, 1024)
    conv4_8 = bottleneck_Block(conv4_7, 1024)
    conv4_9 = bottleneck_Block(conv4_8, 1024)
    conv4_10 = bottleneck_Block(conv4_9, 1024)
    conv4_11 = bottleneck_Block(conv4_10, 1024)
    conv4_12 = bottleneck_Block(conv4_11, 1024)
    conv4_13 = bottleneck_Block(conv4_12, 1024)
    conv4_14 = bottleneck_Block(conv4_13, 1024)
    conv4_15 = bottleneck_Block(conv4_14, 1024)
    conv4_16 = bottleneck_Block(conv4_15, 1024)
    conv4_17 = bottleneck_Block(conv4_16, 1024)
    conv4_18 = bottleneck_Block(conv4_17, 1024)
    conv4_19 = bottleneck_Block(conv4_18, 1024)
    conv4_20 = bottleneck_Block(conv4_19, 1024)
    conv4_21 = bottleneck_Block(conv4_20, 1024)
    conv4_22 = bottleneck_Block(conv4_21, 1024)
    conv4_23 = bottleneck_Block(conv4_22, 1024)

    # conv5_x  1/32
    conv5_1 = bottleneck_Block(conv4_23, 2048, strides=(2, 2), with_conv_shortcut=True)
    conv5_2 = bottleneck_Block(conv5_1, 2048)
    conv5_3 = bottleneck_Block(conv5_2, 2048)

    # 1,1,n向量是256维的
# ====================================解码器 1 ====================================================
#     # # 8,8,256
    multi5 = Conv2D(2048, 2, padding='same',  kernel_initializer='he_normal')(conv5_3)


# ====================================解码器 2 ====================================================
    multi4 = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(multi5))
    multi4 = concatenate([conv4_23, multi4])


# ====================================解码器 3 ====================================================
    multi3 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(multi4))
    # c3 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    multi3 = concatenate([conv3_4, multi3])


# ====================================解码器 4 ====================================================
    multi2 = Conv2D(256, 2, activation='relu', padding='same', name = 'out_go1', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(multi3))
    multi2 = concatenate([conv2_3, multi2])

# ========================================================================================
    Z1 = Conv2D(256, 1, strides=1, padding='same')(multi5)
    Z1 = BatchNormalization(axis=3)(Z1)
    Z1 = Activation('relu')(Z1)
    # R1 = SceneEmbedding(Z1)

    Z2 = Conv2D(256, 1, strides=1, padding='same')(multi4)
    Z2 = BatchNormalization(axis=3)(Z2)
    Z2 = Activation('relu')(Z2)
    # R2 = SceneEmbedding(Z2)
    #
    Z3 = Conv2D(256, 1, strides=1, padding='same')(multi3)
    Z3 = BatchNormalization(axis=3)(Z3)
    Z3 = Activation('relu')(Z3)
    # R3 = SceneEmbedding(Z3)
    #
    Z4 = Conv2D(256, 1, strides=1, padding='same')(multi2)
    Z4 = BatchNormalization(axis=3)(Z4)
    Z4 = Activation('relu')(Z4)
    # R4 = SceneEmbedding(Z4)

    f_p4 = Conv2DTranspose(32, 2, strides=(8, 8), activation='relu', padding='same', kernel_initializer='he_normal')(Z1)

    f_p3 = Conv2DTranspose(32, 2, strides=(4, 4), activation='relu', padding='same', kernel_initializer='he_normal')(Z2)

    f_p2 = Conv2DTranspose(32, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(Z3)

    f_p1 = Conv2DTranspose(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(Z4)
# ======================================================================================================================
    multi_end = Concatenate()([f_p1, f_p2, f_p3, f_p4])
    multi_end = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(4, 4))(multi_end))

# ======================================================================================================================

    # 损失函数1，监督多分类
    class_out = Conv2D(num_class, 1, activation='softmax', name='class_out')(multi_end)

    # 损失函数3， 监督水边线
    # 取出第6维
    sea_super = Lambda(lambda x: x[:, :, :, 6])(class_out)
    b, row, col = sea_super.get_shape()
    sea_super = Reshape((row, col, 1))(sea_super)

    waterline_out1 = Lambda(MyCanny)(sea_super)
    waterline_out2 = Lambda(MyCanny)(sea_super)
    waterline_out3 = Lambda(MyCanny)(sea_super)
    waterline_out4 = Lambda(MyCanny)(sea_super)
    waterline_out5 = Lambda(MyCanny)(sea_super)
    waterline_out6 = Lambda(MyCanny)(sea_super)
    waterline_out7 = Lambda(MyCanny)(sea_super)
    waterline_out = Concatenate()([waterline_out1, waterline_out2, waterline_out3, waterline_out4, waterline_out5, waterline_out6, waterline_out7])
    seaLand_out = Conv2D(1, 1, activation='sigmoid', name='seaLand_out')(sea_super)


    # 融合策略
    fusion = Add()([waterline_out, class_out])
    fusion = Conv2D(64, 1, strides=1, activation='relu', padding='same')(fusion)

    fusion = Fusion(fusion, 64, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])
    out_line_type = Conv2D(num_class, 1, activation='softmax', name='out_line_type')(fusion)

    model = Model(inputs=inputs, outputs=[class_out, seaLand_out, out_line_type])
    # model = Model(inputs=inputs, outputs=class_out)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'class_out': lovasz_softmax, 'seaLand_out': 'binary_crossentropy',
                        'out_line_type': categorical_focal_loss(2., 0.25)},
                  metrics={'class_out': ['accuracy'],
                            'seaLand_out': ['accuracy'],
                           'out_line_type': ['accuracy']})
    # keras.utils.plot_model(model, to_file=r"E:\model_data\Ablation_Study\Ablation_Again\model1.pdf", show_shapes=True, show_layer_names=True,rankdir='LR')
    if Falg_summary:
        model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    model = FarSeg(num_class=7, input_size=(256, 256, 4), Falg_summary=True)
