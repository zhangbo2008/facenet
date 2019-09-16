import tensorflow as tf;  
import numpy as np;  
import matplotlib.pyplot as plt;  
with tf.variable_scope("foo"): 
    a1 = tf.get_variable(name='a1', shape=[2,3], initializer=tf.random_normal_initializer(mean=0, stddev=1))
    a2 = tf.get_variable(name='a2', shape=[1], initializer=tf.constant_initializer(1))
with tf.variable_scope("foo", reuse=True): 
    a3 = tf.get_variable(name='a1', shape=[2,3])
 
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print (sess.run(a1))
	print (sess.run(a2))
	print (sess.run(a3))
