import tensorflow as tf
from tensorflow.contrib.framework import nest

def print_iter(iter):
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = sess.run(a)
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

def print_dataset(ds):
	print_iter(ds.make_one_shot_iterator())

with tf.Session() as sess:
	ds = tf.data.Dataset.range(10)
	ds1 = ds.take(7)
	ds2 = ds.skip(7)

	iter = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)

	ds1_init = iter.make_initializer(ds1)
	ds2_init = iter.make_initializer(ds2)

	print('*ds1')
	sess.run(ds1_init)
	print_iter(iter)

	print('*ds2')
	sess.run(ds2_init)
	print_iter(iter)
