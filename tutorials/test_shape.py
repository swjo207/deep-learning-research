import tensorflow as tf

with tf.Session() as sess:
	t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
	print(t, type(t), t.shape, type(t.shape), type(t.shape[0]))
	print(t.shape)
	r = t.eval()
	print(r, type(r), r.shape, type(r.shape), type(r.shape[0]))
