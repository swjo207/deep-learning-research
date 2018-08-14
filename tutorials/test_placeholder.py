import tensorflow as tf
from tensorflow.contrib.framework import nest

with tf.Session() as sess:
	p = tf.placeholder('float32', [2, 2])
	print(p, type(p))
	q = tf.reduce_sum(p)
	b = sess.run(q, feed_dict = {p: [[1,1], [2,2]]})
	print(b, type(b))

