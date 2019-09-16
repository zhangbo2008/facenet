# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn





a=tf.random_normal([10,20,30])

a = tf.unstack(a, 20, 1)

print(len(a))
print(a[0].shape)
print(a[0])



