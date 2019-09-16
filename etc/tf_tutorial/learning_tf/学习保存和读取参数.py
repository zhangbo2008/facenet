import os
import tensorflow as tf



with tf.name_scope('hidden1') as scope:#这里面命名是用于tensorboard的.修饰的是name这个变量连接符用/
    
    v1 = tf.Variable(2222222)
    
    v2 = tf.Variable(675)  #可以起一样的名字,name一样也行.只不过后面的覆盖前面的.

 





with tf.Session() as sess:
  saver = tf.train.Saver({"my_v34324232": v2})#这个起名随便起,把变量存到v2中.读取时候还是v2.
  sess.run(tf.initialize_all_variables())

  
  #saver.save(sess,"./tmp/model.ckpt")
  saver.restore(sess, "./tmp/model.ckpt")
  print(v1.eval())
  print(v2.eval())
  print('over')












'''



# Create some variables.
v1 = tf.Variable(0, name="v1")
v2 = tf.Variable(0, name="v2")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "./tmp/model.ckpt")
  print(v1.eval())
  print(v2.eval())
  print( "Model restored.")
  # Do some work with the model
'''


'''


# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore only 'v2' using the name "my_v2"
saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.

'''





