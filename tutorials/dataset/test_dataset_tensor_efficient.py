import numpy
import tensorflow as tf
#from tensorflow.python.util import nest
#from tensorflow.contrib.framework import nest

def print_iter(iter):
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = sess.run(a)
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

with tf.Session() as sess:
	data = numpy.array([[1,2,3], [4,5,6]])

	p = tf.placeholder(data.dtype, data.shape)
	ds = tf.data.Dataset.from_tensor_slices(p)
	iter = ds.make_initializable_iterator()
	sess.run(iter.initializer, feed_dict={p: data})
	print_iter(iter)


	