import tensorflow as tf

with tf.Session() as sess:
	f = tf.float32
	print(f, type(f))

	t = tf.constant([1], dtype=tf.float32)
	print(t, type(t), t.dtype)
	r = t.eval()
	print(r, type(r))
