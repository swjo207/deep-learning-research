import tensorflow as tf
from tensorflow.python.util import nest

with tf.Session() as sess:
	ds = tf.data.Dataset.range(5)
	print("*one shot")
	iter = ds.make_one_shot_iterator()
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = a.eval()
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

	print("*initializable")
	iter = ds.make_initializable_iterator()
	print(type(iter.initializer))
	sess.run(iter.initializer)
	a = iter.get_next()
	print(a, type(a))
	try:
		while True:
			b = a.eval()
			print(b, type(b))
	except tf.errors.OutOfRangeError:	
		pass

