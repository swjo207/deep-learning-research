import tensorflow as tf

with tf.Session() as sess:
	a = tf.constant('a-bc-d')
	print(a, type(a), a.eval())
	
	b = tf.substr(a, 2, 2)
	print(b, type(b), b.eval())

	b = tf.string_split([a], r'-')
	print(b.eval())

	b = tf.string_join(['a', 'b'], r'-')
	print(b.eval())

	print((tf.equal(b, 'a-b')).eval())
