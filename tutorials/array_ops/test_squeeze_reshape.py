import tensorflow as tf

with tf.Session() as sess:
	print('*squeeze')
	a = tf.constant([[1, 2]])
	b = tf.squeeze(a)
	print(a.shape, b.shape, b.eval())

	a = tf.constant([[1], [2]])
	b = tf.squeeze(a)
	print(a.shape, b.shape, b.eval())

	a = tf.constant([[[1, 2]], [[3, 4]], [[5, 6]]])
	b = tf.squeeze(a)
	print(a.shape, b.shape, b.eval())

	print('*expand_dims')
	a = tf.constant([[1,2],[3,4]])
	b = tf.expand_dims(a, 1)
	print(a.shape, b.shape, b.eval())

	a = tf.constant([1])
	b = tf.expand_dims(a, -1)
	print(a.shape, b.shape, b.eval())

	print('*reshape')
	a = tf.constant([[[1, 2]], [[3, 4]], [[5, 6]]])
	b = tf.reshape(a, [3, 2])
	print(a.shape, b.shape, b.eval())

	a = tf.constant([[[1, 2]], [[3, 4]], [[5, 6]]])
	b = tf.reshape(a, [2, -1])
	print(a.shape, b.shape, b.eval())

	print('*shape')
	a = tf.constant([[[1, 2]], [[3, 4]], [[5, 6]]])
	b = tf.shape(a)
	print(a.shape, b.shape, b.eval())

