import tensorflow as tf

with tf.Session() as sess:
	a = tf.constant([[1, 2], [3, 4]], dtype='float32')
	print(a, type(a), a.dtype, a.shape, a.eval())

	b = tf.convert_to_tensor([[1, 2], [3, 4]], dtype='float32')
	print(b, type(b), b.dtype, b.shape, b.eval())
