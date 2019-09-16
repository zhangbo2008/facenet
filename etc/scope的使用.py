'''
https://www.cnblogs.com/MY0213/p/9208503.html
'''
import tensorflow as tf

'''
[1]表示变量的shape
'''

with tf.variable_scope("foo",reuse=False):#如果变量v不存在就必须reuse写False,或者不写reuse默认也是False
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):#如果变量存在就必须写True
    v1 = tf.get_variable("v", [1])
assert v1 is v




with tf.variable_scope("foo2"):
    v = tf.get_variable("v", [1])
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 is v

'''
多级嵌套variable_scope的reuse开关
'''
with tf.variable_scope("root"):
    # At start, the scope is not reusing.
    assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo"):
        # Opened a sub-scope, still not reusing.
        assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo", reuse=True):
        # Explicitly opened a reusing scope.
        assert tf.get_variable_scope().reuse == True
        with tf.variable_scope("bar"):
            # Now sub-scope inherits the reuse flag.
            assert tf.get_variable_scope().reuse == True
    # Exited the reusing scope, back to a non-reusing one.
    assert tf.get_variable_scope().reuse == False














