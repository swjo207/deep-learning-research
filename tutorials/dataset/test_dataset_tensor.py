import tensorflow as tf
#from tensorflow.python.util import nest
#from tensorflow.contrib.framework import nest

with tf.Session() as sess:
	print('*slices')
	ds = tf.data.Dataset.from_tensor_slices([[1,2,3],[4,5,6]])
	print(ds.output_shapes)
	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = a.eval()
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

	print('*slices, tuple')
	ds = tf.data.Dataset.from_tensor_slices(([[1,2,3],[4,5,6]], [[1,1,1],[0,0,0]]))
	print(ds.output_shapes)
	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			# a[0].eval() and a[1].eval() will iterate twice!
			b = sess.run(a)
			print(b);
	except tf.errors.OutOfRangeError:	
		pass

	print('*tensors, tuple')
	ds = tf.data.Dataset.from_tensors(([1,2,3],[4,5,6]))
	for s in ds.output_shapes: print(s)

	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = sess.run(a)
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

	print('*tensors, list')
	ds = tf.data.Dataset.from_tensors([[1,2,3],[4,5,6]])
	print(ds.output_shapes)
	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = sess.run(a)
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass
