import numpy
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
	print('*gen1')
	data = numpy.array([[1,2,3], [4,5,6]])
	def gen1():
		for i in range(2):
			yield {'a': data[i, 0], 'b': data[i, 1]}
	print_dataset(tf.data.Dataset.from_generator(gen1, {'a': tf.float32, 'b': tf.float32}))

	print('*gen2')
	def gen2():
		for i in range(6):
			yield({'a': [i, i], 'b': [i*2, i*2+1]})
	print_dataset(tf.data.Dataset.from_generator(gen2, {'a': tf.float32, 'b': tf.float32}))
