from __future__ import division
import numpy as np
import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
input_shape=np.zeros((1,34,34,3))
residual_shape=np.zeros((1,10,10,3))
ROW_AXIS=1
COL_AXIS=2
stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))#round 四舍五入到整数





shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],#这个变形操作复杂一点.论文上是用一个矩阵来做线性变换,这里面是用卷积压缩
                   kernel_size=(1, 1),#第一个filters保证了channel的维度不变.
                   strides=(stride_width, stride_height),#下面证明为什么kernal_size(1,1) strides取这个数值时候会输出residual_shape
                   padding="valid",                         #虽然具体画图比较显然,严格证明感觉还是吃力.
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(0.0001))(input)
model = shortcut((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])                
model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger])
                          
                          
                          
                          
                          
                          
                          
                          
                          