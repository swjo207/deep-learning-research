import tensorflow as tf
from tensorflow.contrib.framework import nest

def print_dataset(ds):
	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = sess.run(a)
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

with tf.Session() as sess:
	ds = tf.data.Dataset.from_tensor_slices([[1,1], [2,1], [3,1], [4,0], [5,0]])
	def get_key(e): return tf.to_int64(e[1])
	def reduce_subset(k, ds): print(ds); return ds.repeat(2)
	ds = tf.contrib.data.group_by_window(get_key, reduce_subset, window_size = 2)(ds)
	print_dataset(ds)


