'''
注意只要跟reuse有关的代码都需要重启kernal来运行
'''
import tensorflow as tf



with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
assert v1 is v