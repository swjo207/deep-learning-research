import tensorflow as tf

with tf.Session() as sess:
	print('*onehot')
	a = tf.one_hot([2, 0, 1], 4)
	print(a.eval())

	print('*seq mask')
	a = tf.sequence_mask([2, 0, 1], 4)
	print(a.eval())
