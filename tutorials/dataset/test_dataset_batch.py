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
	print('*batch')
	ds = tf.data.Dataset.range(10)
	print_dataset(ds.batch(3))

	print('*padded batch 1')
	ds = tf.data.Dataset.range(10).map(lambda e: tf.fill([tf.cast(e, 'int32')], e))
	print_dataset(ds.padded_batch(3, [None]))

	print('*padded batch 2')
	ds = tf.data.Dataset.from_tensor_slices(tf.ones([7, 2, 2]))
	print_dataset(ds.padded_batch(3, [2,3], 2.0))

	print('*structured batch')
	def gen1():
		for i in range(6):
			yield({'a': i, 'b': i*2})
	ds = tf.data.Dataset.from_generator(gen1, {'a': tf.int32, 'b': tf.int32})
	print_dataset(ds.batch(3))

	print('*structured batch2')
	def gen2():
		for i in range(6):
			yield({'a': [i, i], 'b': [i*2, i*2+1]})
	ds = tf.data.Dataset.from_generator(gen2, {'a': tf.int32, 'b': tf.int32})
	print_dataset(ds.batch(3))
