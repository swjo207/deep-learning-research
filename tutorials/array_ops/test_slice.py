import tensorflow as tf

with tf.Session() as sess:
	print('*slice')
	#print(a.eval())
	a = tf.slice([1, 2, 3, 4], [1], [2])
	print(a.eval())
	a = tf.slice([[1,2], [3,4], [5,6], [7,8]], [1, 0], [2, -1])
	print(a.eval())

