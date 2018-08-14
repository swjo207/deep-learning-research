import tensorflow as tf

with tf.Session() as sess:
	a = tf.constant([[1, 2], [3, 4]], dtype='float32')
	print(a.op, type(a.op))
	print(a.op.inputs)
	print(a.op.outputs)
	print(sess.run(a.op))
	print(a.op.outputs)
	