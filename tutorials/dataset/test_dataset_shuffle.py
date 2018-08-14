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
	print('*shuffle')
	ds = tf.data.Dataset.from_tensor_slices(tf.eye(3))
	ds = ds.shuffle(3)
	print_dataset(ds)

	print('*shuffle and repeat')
	ds = tf.data.Dataset.from_tensor_slices(tf.eye(3))
	ds = ds.shuffle(3).repeat(2)
	print_dataset(ds)
