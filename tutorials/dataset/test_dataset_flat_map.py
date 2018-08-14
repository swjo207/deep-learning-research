import tensorflow as tf
from tensorflow.contrib.framework import nest

with tf.Session() as sess:
	ds = tf.data.Dataset.from_tensor_slices([[1,2,3], [4,5,6]])
	ds = ds.flat_map(lambda e:
		tf.data.Dataset.range(tf.cast(tf.reduce_sum(e), 'int64')))
	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = sess.run(a)
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

