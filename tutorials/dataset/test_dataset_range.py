import tensorflow as tf

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
	ds = tf.data.Dataset.range(5)
	print_dataset(ds)

	ds = tf.data.Dataset.range(2, 5)
	print_dataset(ds)

	ds = tf.data.Dataset.range(5, 0, -1)
	print_dataset(ds)