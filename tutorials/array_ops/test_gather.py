import tensorflow as tf

with tf.Session() as sess:
	a = tf.gather([[1,2],[3,4]], [1, 0, 1])
	print(a.eval())

