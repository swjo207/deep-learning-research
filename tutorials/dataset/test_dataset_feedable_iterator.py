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
	ds1 = ds.take(7)
	ds2 = ds.skip(7)

	p = tf.placeholder(tf.string, shape=[]) # scalar
	iter = tf.data.Iterator.from_string_handle(p, ds1.output_types, ds1.output_shapes)
	iter_get_next = iter.get_next()

	ds1_iter = ds1.make_one_shot_iterator()
	ds2_iter = ds2.make_initializable_iterator()	

	ds1_iter_s = sess.run(ds1_iter.string_handle())
	print('ds1_iter_s', ds1_iter_s, type(ds1_iter_s))
	ds2_iter_s = sess.run(ds2_iter.string_handle())
	print('ds2_iter_s', ds2_iter_s, type(ds2_iter_s))

	print('*ds1')
	for _ in range(7):
		b = sess.run(iter_get_next, feed_dict = {p: ds1_iter_s})
		print(b, type(b))

	for _ in range(2):
		print('*ds2')
		sess.run(ds2_iter.initializer)
		for _ in range(3):
			b = sess.run(iter_get_next, feed_dict = {p: ds2_iter_s})
			print(b, type(b))
