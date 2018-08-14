import tensorflow as tf
from tensorflow.contrib.framework import nest

with tf.Session() as sess:
	s1 = tf.data.Dataset.from_tensor_slices([[1,2,3], [4,5,6]])
	s2 = tf.data.Dataset.from_tensor_slices([[1,1,1], [0,0,0]])
	ds = tf.data.Dataset.zip((s1, s2))
	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = sess.run(a)
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

