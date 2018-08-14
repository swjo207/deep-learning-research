import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.data.util import nest as data_nest

with tf.Session() as sess:
	a = tf.eye(3)
	b = tf.ones(3, 4)
	print(nest.flatten([a, b]))
	print(data_nest.flatten([a, b]))
	print(data_nest.flatten(dict(a=a, b=b)))
