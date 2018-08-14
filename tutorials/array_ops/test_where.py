import tensorflow as tf

with tf.Session() as sess:
	print('*where')
	a = tf.where([True, False], [1,2], [3,4])
	print(a.eval())

