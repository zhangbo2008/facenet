# -*- coding: utf-8 -*-

"""
这段代码用tensorflow来处理向量,这些向量是从Word2Vec这个工具得到的.
嵌入是通过一层神经元来实现.one-hot模型.


最后根据向量的相似度来计算那些词是eval_words中每一个词的近义词.






 Word2Vec.
Implement Word2Vec algorithm to compute vector representations of words.
This example is using a small chunk of Wikipedia articles to train from.
References:
    - Mikolov, Tomas et al. "Efficient Estimation of Word Representations
    in Vector Space.", 2013.
Links:
    - [Word2Vec] https://arxiv.org/pdf/1301.3781.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import collections
import os
import random
import urllib
import zipfile

import numpy as np
import tensorflow as tf

# Training Parameters
learning_rate = 0.1
batch_size = 128
num_steps = 30000
display_step = 10000
eval_step = 10000

# Evaluation Parameters
eval_words = [b'going', b'hardware', b'american', b'britain',b'five', b'of' ]

# Word2Vec Parameters
embedding_size = 200 # Dimension of the embedding vector
max_vocabulary_size = 50000 # Total number of different words in the vocabulary
min_occurrence = 10 # Remove all words that does not appears at least n times
skip_window = 3 # How many words to consider left and right
num_skips = 2 # How many times to reuse an input to generate a label
num_sampled = 64 # Number of negative examples to sample


# Download a small chunk of Wikipedia articles collection


#下面这段是作为下载数据用的
url = 'http://mattmahoney.net/dc/text8.zip'
data_path = 'text8.zip'
if not os.path.exists(data_path):#如果不存在就下载
    print("Downloading the dataset... (It may take some time)")
    filename, _ = urllib.request.urlretrieve(url, data_path)
    print("Done!")
# Unzip the dataset file. Text has already been processed
with zipfile.ZipFile(data_path) as f:#否则就直接调用文件即可.
    text_words = f.read(f.namelist()[0]).lower().split()
    
    
    
print(text_words[0])
print(text_words[10])
print(text_words[0])
print(text_words[0])
print(text_words[0])
print(type(text_words[0]))
'''
从这里就能看出来字典里面数据类型是bytes的,而我们eval_words里面是string
这在python3里面已经没法自动转化了.所以只需要在eval_words前面加上b就可以啦!

但是仍然运行很慢
'''

# Build the dictionary and replace rare words with UNK token
count = [('UNK', -1)]
# Retrieve the most common words
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))
#上面得到text_words里面出现次数最多的5万个单词放到count里面,写的很精简啊.用了collection这个库包,轻松实现字典的频率排序.
print(count)
print(len(count))

#下面这段把频率低的都扔了.这里是低于10
# Remove samples with less than 'min_occurrence' occurrences
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        # The collection is ordered, so stop when 'min_occurrence' is reached
        break
    
    
    
    
# Compute the vocabulary size
vocabulary_size = len(count)
# Assign an id to each word
word2id = dict()

#下面用1到len(count)来进行编码,存成字典.
for i, (word, _)in enumerate(count):
    word2id[word] = i

data = list()
unk_count = 0
for word in text_words:
    # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
    '''
    Python 字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值。这里面就是返回0这个值.
    '''
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
#data是所有单词的编码组成的list
    
    #更新count这个数组
count[0] = ('UNK', unk_count)
#因为value里面的元素互相都不同,所以直接zip就能得到反字典.
id2word = dict(zip(word2id.values(), word2id.keys()))


print(word2id)
print(id2word)
print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print("Most common words:", count[:10])










#%%

'''
下面有错误,需要修改
'''
data_index = 0
# Generate training batch for the skip-gram model
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # get window size (words left and right + current one)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
#    print(buffer)
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
#        print(context_words)
        words_to_use = random.sample(context_words, num_skips)
        #words_to_use随机找2个需要的数0到6之间
        
        
        #这里面的操作是,每6个数据互相进行shuffle然后给batch赋值.做到数据shuffle的作用.
        #可能因为数据量太大一起shuffle太慢,所以他做的是局部shuffle
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        #下面行:如果最外层的global计数器读到了最后,那么就让buffer从头再拼接一段.
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            #因为是限制最大值的双端队列所以append的时候自动会剔除最后一个值.
#            print(buffer)
#            print(len(buffer))
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


#test
batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
#print(batch_x, batch_y )
#raise

# Input data
X = tf.placeholder(tf.int32, shape=[None])
# Input label
Y = tf.placeholder(tf.int32, shape=[None, 1])

# Ensure the following ops & var are assigned on CPU
# (some ops are not compatible on GPU)
with tf.device('/cpu:0'):#设置设备为第一个cpu
    # Create the embedding variable (each row represent a word embedding vector)
    embedding = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    
    '''
    这里面术语叫:word embedding vector
    表示的是向量化,把字典里面每一个字都映射为一个向量.这里面embedding_size设置为200
    把这个映射到向量就可以了.就可以计算夹角的余弦值来作为2个词汇之间的相差程度.
    每一个单词变成一个200维的向量.
    
    
    One-hot型的矩阵相乘，可以简化为查表操作，这大大降低了运算量。
    
    tf.nn.embedding_lookup()就是根据input_ids中的id，寻找embeddings中的第id行。比如input_ids=[1,3,5]，则找出embeddings中第1，3，5行，加成一个tensor返回。

embedding_lookup不是简单的查表，id对应的向量是可以训练的，训练参数个数应该是 category num*embedding size，也就是说lookup是一种全连接层。


这个.py文件整体说明:
    https://www.jianshu.com/p/d44ce1e3ec2f
    
    '''
    
    
    
    # Lookup the corresponding embedding vectors for each sample in X
    '''
    embedding是一个单词对应的向量.X是batch_x
    '''
    X_embed = tf.nn.embedding_lookup(embedding, X)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch
loss_op = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=Y,
                   inputs=X_embed,
                   num_sampled=num_sampled,
                   num_classes=vocabulary_size))

# Define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluation
# Compute the cosine similarity between input data embedding and every embedding vectors
X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))
embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
cosine_sim_op = tf.matmul(X_embed_norm, embedding_norm, transpose_b=True)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Testing data
    x_test = np.array([word2id[w] for w in eval_words])
    print(x_test)
    
    average_loss = 0
    for step in range(1, num_steps + 1):
        # Get a new batch of data
        batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
#        print(batch_x,00000000000000000000000000)
        # Run training op
        _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        average_loss += loss

        if step % display_step == 0 or step == 1:
            if step > 1:
                average_loss /= display_step
#            print("Step " + str(step) + ", Average Loss= " + \
#                  "{:.4f}".format(average_loss))
            average_loss = 0

        # Evaluation
        if step % eval_step == 0 or step == 1:
#            print("Evaluation...")
            sim = sess.run(cosine_sim_op, feed_dict={X: x_test})
            for i in range(len(eval_words)):
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = '"%s" nearest neighbors:' % eval_words[i]
                for k in range(top_k):
                    log_str = '%s %s,' % (log_str, id2word[nearest[k]])
                print(log_str)