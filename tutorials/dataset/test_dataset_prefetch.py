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
	ds = tf.data.Dataset.range(10)
	print_dataset(ds.prefetch(5))
